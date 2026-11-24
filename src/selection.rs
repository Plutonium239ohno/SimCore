//! Selecting part of a scene: copy, cut, delete, and placing a copy back
//! moved, rotated or mirrored (walls, matter and reactor parts alike).
//!
//! A `Clip` holds copies of everything in a region, with positions relative
//! to the middle of what it holds, so it can be turned and flipped about
//! its centre and dropped anywhere.

use crate::detector::{DetShape, Detector};
use crate::field::Field;
use crate::reactor::{Drain, Pump, Rod, Site, Spawner};
use crate::rng::HashMap;
use crate::sim::Sim;
use crate::walls::Seg;
use macroquad::math::{Rect, Vec2, vec2};

/// What a selection takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kinds {
    pub walls: bool,
    /// Nuclei, nucleons, electrons, positrons...
    pub matter: bool,
    /// Spawners, fields, detectors, drains, pumps, rods, fuel channels,
    /// neutron sources.
    pub parts: bool,
}

impl Default for Kinds {
    fn default() -> Self {
        Self { walls: true, matter: true, parts: true }
    }
}

/// One particle: position, velocity, kind, nucleus tag, packed (Z, N).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub pos: Vec2,
    pub vel: Vec2,
    pub kind: u8,
    pub tag: u32,
    pub zn: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Clip {
    pub walls: Vec<Seg>,
    pub particles: Vec<Particle>,
    pub spawners: Vec<Spawner>,
    pub fields: Vec<Field>,
    pub detectors: Vec<Detector>,
    pub drains: Vec<Drain>,
    pub pumps: Vec<Pump>,
    pub rods: Vec<Rod>,
    pub sites: Vec<Site>,
    pub sources: Vec<Vec2>,
}

/// Bounding box of points.
fn bounds_of(pts: impl Iterator<Item = Vec2>) -> Option<Rect> {
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    let mut any = false;
    for p in pts {
        lo = lo.min(p);
        hi = hi.max(p);
        any = true;
    }
    any.then(|| Rect::new(lo.x, lo.y, hi.x - lo.x, hi.y - lo.y))
}

fn corners(r: Rect) -> [Vec2; 2] {
    [vec2(r.x, r.y), vec2(r.x + r.w, r.y + r.h)]
}

fn shift_rect(r: Rect, d: Vec2) -> Rect {
    Rect::new(r.x + d.x, r.y + d.y, r.w, r.h)
}

fn rod_rect(r: &Rod) -> Rect {
    Rect::new(r.x - r.half_w, r.top, 2.0 * r.half_w, r.len)
}

/// A linear map (rotation and/or mirror) of relative coordinates.
#[derive(Clone, Copy)]
struct Lin {
    x: Vec2,
    y: Vec2,
}

impl Lin {
    fn rotation(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self { x: vec2(c, s), y: vec2(-s, c) }
    }
    fn apply(self, p: Vec2) -> Vec2 {
        self.x * p.x + self.y * p.y
    }
    fn mirrors(self) -> bool {
        self.x.perp_dot(self.y) < 0.0
    }
    /// An axis-aligned rectangle under the map: its centre moves, it stays
    /// axis-aligned (turned on its side when the map is closer to a quarter
    /// turn).
    fn rect(self, r: Rect) -> Rect {
        let c = self.apply(r.center());
        let ex = self.apply(Vec2::X);
        let (w, h) = if ex.x.abs() >= ex.y.abs() { (r.w, r.h) } else { (r.h, r.w) };
        Rect::new(c.x - w * 0.5, c.y - h * 0.5, w, h)
    }
}

impl Clip {
    pub fn is_empty(&self) -> bool {
        self.walls.is_empty()
            && self.particles.is_empty()
            && self.spawners.is_empty()
            && self.fields.is_empty()
            && self.detectors.is_empty()
            && self.drains.is_empty()
            && self.pumps.is_empty()
            && self.rods.is_empty()
            && self.sites.is_empty()
            && self.sources.is_empty()
    }

    /// Bodies (nuclei, free nucleons) and other particles.
    pub fn bodies(&self) -> (usize, usize) {
        let mut tags: Vec<u32> = self.particles.iter().filter(|p| p.tag != u32::MAX).map(|p| p.tag).collect();
        tags.sort_unstable();
        tags.dedup();
        (tags.len(), self.particles.iter().filter(|p| p.tag == u32::MAX).count())
    }

    /// "12 walls, 3 nuclei, 1 field..."
    pub fn summary(&self) -> String {
        let (bodies, others) = self.bodies();
        let parts: [(usize, &str, &str); 11] = [
            (self.walls.len(), "wall", "walls"),
            (bodies, "nucleus", "nuclei"),
            (others, "other particle", "other particles"),
            (self.spawners.len(), "spawner", "spawners"),
            (self.fields.len(), "field", "fields"),
            (self.detectors.len(), "detector", "detectors"),
            (self.drains.len(), "drain", "drains"),
            (self.pumps.len(), "pump", "pumps"),
            (self.rods.len(), "rod", "rods"),
            (self.sites.len(), "fuel channel", "fuel channels"),
            (self.sources.len(), "n source", "n sources"),
        ];
        let v: Vec<String> = parts.iter().filter(|p| p.0 > 0).map(|(n, one, many)| format!("{n} {}", if *n == 1 { one } else { many })).collect();
        if v.is_empty() { "nothing".into() } else { v.join(", ") }
    }

    /// Every point that outlines the contents.
    fn points(&self) -> Vec<Vec2> {
        let mut v: Vec<Vec2> = Vec::new();
        for s in &self.walls {
            v.extend([s.a, s.b]);
        }
        v.extend(self.particles.iter().map(|p| p.pos));
        for s in &self.spawners {
            v.push(s.pos);
            if let Some(a) = s.area {
                v.extend(corners(a));
            }
        }
        for f in &self.fields {
            v.extend(corners(f.rect));
        }
        for d in &self.detectors {
            match d.shape {
                DetShape::Box(r) => v.extend(corners(r)),
                DetShape::Line(a, b) => v.extend([a, b]),
            }
        }
        for d in &self.drains {
            v.extend(corners(d.rect));
        }
        for p in &self.pumps {
            v.extend(corners(p.rect));
        }
        for r in &self.rods {
            v.extend(corners(rod_rect(r)));
        }
        v.extend(self.sites.iter().map(|s| s.pos));
        v.extend(self.sources.iter().copied());
        v
    }

    /// Bounding box (relative to the clip's centre).
    pub fn bounds(&self) -> Rect {
        bounds_of(self.points().into_iter()).unwrap_or(Rect::new(0.0, 0.0, 0.0, 0.0))
    }

    /// Move every position by d.
    fn shift(&mut self, d: Vec2) {
        for s in &mut self.walls {
            s.a += d;
            s.b += d;
        }
        for p in &mut self.particles {
            p.pos += d;
        }
        for s in &mut self.spawners {
            s.pos += d;
            s.area = s.area.map(|a| shift_rect(a, d));
        }
        for f in &mut self.fields {
            f.rect = shift_rect(f.rect, d);
        }
        for det in &mut self.detectors {
            det.shape = match det.shape {
                DetShape::Box(r) => DetShape::Box(shift_rect(r, d)),
                DetShape::Line(a, b) => DetShape::Line(a + d, b + d),
            };
        }
        for dr in &mut self.drains {
            dr.rect = shift_rect(dr.rect, d);
        }
        for p in &mut self.pumps {
            p.rect = shift_rect(p.rect, d);
        }
        for r in &mut self.rods {
            r.x += d.x;
            r.top += d.y;
        }
        for s in &mut self.sites {
            s.pos += d;
        }
        for s in &mut self.sources {
            *s += d;
        }
    }

    /// Re-centre so the bounding box's middle is the origin.
    fn recentre(&mut self) {
        let c = self.bounds().center();
        self.shift(-c);
    }

    fn map(&mut self, m: Lin) {
        for s in &mut self.walls {
            s.a = m.apply(s.a);
            s.b = m.apply(s.b);
        }
        for p in &mut self.particles {
            p.pos = m.apply(p.pos);
            p.vel = m.apply(p.vel);
        }
        for s in &mut self.spawners {
            s.pos = m.apply(s.pos);
            s.vel = m.apply(s.vel);
            s.area = s.area.map(|a| m.rect(a));
        }
        for f in &mut self.fields {
            f.rect = m.rect(f.rect);
            f.e = m.apply(f.e);
            // B is out of the plane: a mirror image turns the other way.
            if m.mirrors() {
                f.b = -f.b;
            }
        }
        for d in &mut self.detectors {
            d.shape = match d.shape {
                DetShape::Box(r) => DetShape::Box(m.rect(r)),
                DetShape::Line(a, b) => DetShape::Line(m.apply(a), m.apply(b)),
            };
        }
        for d in &mut self.drains {
            d.rect = m.rect(d.rect);
        }
        for p in &mut self.pumps {
            p.rect = m.rect(p.rect);
            p.dir = m.apply(p.dir).normalize_or(p.dir);
        }
        // Rods run vertically (they drop from the top): only their centres move.
        for r in &mut self.rods {
            let c = m.apply(vec2(r.x, r.top + r.len * 0.5));
            r.x = c.x;
            r.top = c.y - r.len * 0.5;
        }
        for s in &mut self.sites {
            s.pos = m.apply(s.pos);
        }
        for s in &mut self.sources {
            *s = m.apply(*s);
        }
        self.recentre();
    }

    /// Turn by `angle` (radians, counter-clockwise on screen) about the
    /// centre. Boxes (fields, drains, detector boxes, pumps) stay
    /// axis-aligned: exact for quarter turns.
    pub fn rotate(&mut self, angle: f32) {
        // Screen y points down: counter-clockwise on screen is a negative
        // maths angle.
        self.map(Lin::rotation(-angle));
    }

    /// Mirror left-right.
    pub fn flip_x(&mut self) {
        self.map(Lin { x: vec2(-1.0, 0.0), y: vec2(0.0, 1.0) });
    }

    /// Mirror top-bottom.
    pub fn flip_y(&mut self) {
        self.map(Lin { x: vec2(1.0, 0.0), y: vec2(0.0, -1.0) });
    }

    /// Where to drop it near `at` so its bounding box's corner lands on the
    /// grid (the shape kept as it is).
    pub fn grid_anchor(&self, at: Vec2, grid: f32) -> Vec2 {
        if grid <= 0.5 {
            return at;
        }
        let corner = at + vec2(self.bounds().x, self.bounds().y);
        at + ((corner / grid).round() * grid - corner)
    }

    /// Put every wall end, box corner and part position, dropped at `at`,
    /// onto the grid (cleans up freehand drawing). Matter only moves with
    /// the whole.
    pub fn snap_points(&mut self, at: Vec2, grid: f32) {
        if grid <= 0.5 {
            return;
        }
        let snap = |p: Vec2| ((p + at) / grid).round() * grid - at;
        let snap_rect = |r: Rect| {
            let a = snap(vec2(r.x, r.y));
            let b = snap(vec2(r.x + r.w, r.y + r.h));
            let (lo, hi) = (a.min(b), a.max(b));
            Rect::new(lo.x, lo.y, (hi.x - lo.x).max(grid), (hi.y - lo.y).max(grid))
        };
        for s in &mut self.walls {
            s.a = snap(s.a);
            s.b = snap(s.b);
        }
        self.walls.retain(|s| s.a.distance(s.b) > 0.1);
        for s in &mut self.spawners {
            s.pos = snap(s.pos);
            s.area = s.area.map(snap_rect);
        }
        for f in &mut self.fields {
            f.rect = snap_rect(f.rect);
        }
        for d in &mut self.detectors {
            d.shape = match d.shape {
                DetShape::Box(r) => DetShape::Box(snap_rect(r)),
                DetShape::Line(a, b) => DetShape::Line(snap(a), snap(b)),
            };
        }
        for d in &mut self.drains {
            d.rect = snap_rect(d.rect);
        }
        for p in &mut self.pumps {
            p.rect = snap_rect(p.rect);
        }
        for r in &mut self.rods {
            let c = snap(vec2(r.x, r.top));
            r.x = c.x;
            r.top = c.y;
        }
        for s in &mut self.sites {
            s.pos = snap(s.pos);
        }
        for s in &mut self.sources {
            *s = snap(*s);
        }
    }
}

/// What lies in `region`: walls with both ends in it, parts whose centre
/// is in it, nuclei whose centre is in it.
struct Picked {
    walls: Vec<usize>,
    particles: Vec<usize>,
    spawners: Vec<usize>,
    fields: Vec<usize>,
    detectors: Vec<usize>,
    drains: Vec<usize>,
    pumps: Vec<usize>,
    rods: Vec<usize>,
    sites: Vec<usize>,
    sources: Vec<usize>,
}

fn pick(sim: &Sim, region: Rect, kinds: Kinds) -> Picked {
    let inside = |p: Vec2| region.contains(p);
    let idx = |n: usize, f: &dyn Fn(usize) -> bool| -> Vec<usize> { (0..n).filter(|&k| f(k)).collect() };
    let re = &sim.reactor;
    let mut particles = Vec::new();
    if kinds.matter {
        // A nucleus made of several particles goes whole, by its centre.
        let mut centre: HashMap<u32, (Vec2, f32)> = HashMap::default();
        for i in 0..sim.len() {
            if sim.tag[i] != u32::MAX {
                let e = centre.entry(sim.tag[i]).or_insert((Vec2::ZERO, 0.0));
                e.0 += sim.pos[i];
                e.1 += 1.0;
            }
        }
        for i in 0..sim.len() {
            let p = match centre.get(&sim.tag[i]) {
                Some(&(s, n)) if sim.tag[i] != u32::MAX => s / n,
                _ => sim.pos[i],
            };
            if inside(p) {
                particles.push(i);
            }
        }
    }
    let parts = kinds.parts;
    Picked {
        walls: if kinds.walls { idx(sim.walls.segs.len(), &|k| inside(sim.walls.segs[k].a) && inside(sim.walls.segs[k].b)) } else { Vec::new() },
        particles,
        spawners: if parts { idx(re.spawners.len(), &|k| inside(re.spawners[k].pos)) } else { Vec::new() },
        fields: if parts { idx(re.fields.len(), &|k| inside(re.fields[k].rect.center())) } else { Vec::new() },
        detectors: if parts { idx(re.detectors.len(), &|k| inside(re.detectors[k].center())) } else { Vec::new() },
        drains: if parts { idx(re.drains.len(), &|k| inside(re.drains[k].rect.center())) } else { Vec::new() },
        pumps: if parts { idx(re.pumps.len(), &|k| inside(re.pumps[k].rect.center())) } else { Vec::new() },
        rods: if parts { idx(re.rods.len(), &|k| inside(rod_rect(&re.rods[k]).center())) } else { Vec::new() },
        sites: if parts { idx(re.sites.len(), &|k| inside(re.sites[k].pos)) } else { Vec::new() },
        sources: if parts { idx(re.sources.len(), &|k| inside(re.sources[k])) } else { Vec::new() },
    }
}

/// Copy what's in `region` (see `pick`), centred on its own middle.
/// Returns the clip and where its centre was.
pub fn copy(sim: &Sim, region: Rect, kinds: Kinds) -> (Clip, Vec2) {
    let p = pick(sim, region, kinds);
    let re = &sim.reactor;
    let mut c = Clip {
        walls: p.walls.iter().map(|&k| sim.walls.segs[k]).collect(),
        particles: p.particles.iter().map(|&i| Particle { pos: sim.pos[i], vel: sim.vel[i], kind: sim.kind[i], tag: sim.tag[i], zn: sim.zn[i] }).collect(),
        spawners: p.spawners.iter().map(|&k| re.spawners[k]).collect(),
        fields: p.fields.iter().map(|&k| re.fields[k]).collect(),
        detectors: p.detectors.iter().map(|&k| re.detectors[k].clone()).collect(),
        drains: p.drains.iter().map(|&k| re.drains[k].clone()).collect(),
        pumps: p.pumps.iter().map(|&k| re.pumps[k]).collect(),
        rods: p.rods.iter().map(|&k| re.rods[k]).collect(),
        sites: p.sites.iter().map(|&k| re.sites[k]).collect(),
        sources: p.sources.iter().map(|&k| re.sources[k]).collect(),
    };
    let centre = c.bounds().center();
    c.shift(-centre);
    (c, centre)
}

/// Remove what's in `region` (see `pick`). Returns whether anything went.
pub fn delete(sim: &mut Sim, region: Rect, kinds: Kinds) -> bool {
    let p = pick(sim, region, kinds);
    fn drop_idx<T>(v: &mut Vec<T>, idx: &[usize]) {
        let mut k = 0;
        v.retain(|_| {
            let keep = idx.binary_search(&k).is_err();
            k += 1;
            keep
        });
    }
    let any = !(p.walls.is_empty()
        && p.particles.is_empty()
        && p.spawners.is_empty()
        && p.fields.is_empty()
        && p.detectors.is_empty()
        && p.drains.is_empty()
        && p.pumps.is_empty()
        && p.rods.is_empty()
        && p.sites.is_empty()
        && p.sources.is_empty());
    if !p.walls.is_empty() {
        drop_idx(&mut sim.walls.segs, &p.walls);
        sim.walls.touch();
    }
    if !p.particles.is_empty() {
        let gone: crate::rng::HashSet<usize> = p.particles.iter().copied().collect();
        sim.retain(|i| !gone.contains(&i));
        sim.index_nuclei();
    }
    let re = &mut sim.reactor;
    drop_idx(&mut re.spawners, &p.spawners);
    drop_idx(&mut re.fields, &p.fields);
    drop_idx(&mut re.detectors, &p.detectors);
    drop_idx(&mut re.drains, &p.drains);
    drop_idx(&mut re.pumps, &p.pumps);
    drop_idx(&mut re.rods, &p.rods);
    drop_idx(&mut re.sites, &p.sites);
    drop_idx(&mut re.sources, &p.sources);
    any
}

/// Drop a copy of `clip` with its centre at `at`. Matter outside the world
/// is left out; nuclei get new identities; counters start from zero.
/// Returns the region it now covers.
pub fn place(sim: &mut Sim, clip: &Clip, at: Vec2) -> Rect {
    for s in &clip.walls {
        let mut s = *s;
        s.a += at;
        s.b += at;
        sim.walls.segs.push(s);
    }
    if !clip.walls.is_empty() {
        sim.walls.touch();
    }
    if !clip.particles.is_empty() {
        let (w, h) = (sim.width, sim.height);
        let out = |p: Vec2| p.x < 0.0 || p.y < 0.0 || p.x > w || p.y > h;
        // A nucleus partly outside is left out whole.
        let mut outside: crate::rng::HashSet<u32> = Default::default();
        for p in &clip.particles {
            if p.tag != u32::MAX && out(p.pos + at) {
                outside.insert(p.tag);
            }
        }
        let mut new_tag: HashMap<u32, u32> = HashMap::default();
        for p in &clip.particles {
            let pos = p.pos + at;
            if out(pos) || outside.contains(&p.tag) {
                continue;
            }
            sim.add(pos, p.vel, p.kind);
            let last = sim.len() - 1;
            sim.zn[last] = p.zn;
            sim.tag[last] = if p.tag == u32::MAX { u32::MAX } else { *new_tag.entry(p.tag).or_insert_with(|| sim.new_tag()) };
        }
        sim.index_nuclei();
    }
    let re = &mut sim.reactor;
    for s in &clip.spawners {
        let mut s = *s;
        s.pos += at;
        s.area = s.area.map(|a| shift_rect(a, at));
        s.spawned = 0;
        re.spawners.push(s);
    }
    for f in &clip.fields {
        re.fields.push(Field::new(shift_rect(f.rect, at), f.e, f.b));
    }
    for d in &clip.detectors {
        let shape = match d.shape {
            DetShape::Box(r) => DetShape::Box(shift_rect(r, at)),
            DetShape::Line(a, b) => DetShape::Line(a + at, b + at),
        };
        re.detectors.push(Detector::new(shape, d.filter, d.light));
    }
    for d in &clip.drains {
        let mut n = Drain::new(shift_rect(d.rect, at), d.filter);
        n.per_second = d.per_second;
        re.drains.push(n);
    }
    for p in &clip.pumps {
        re.pumps.push(Pump { rect: shift_rect(p.rect, at), ..*p });
    }
    for r in &clip.rods {
        re.rods.push(Rod { x: r.x + at.x, top: r.top + at.y, ..*r });
    }
    for s in &clip.sites {
        re.sites.push(Site { pos: s.pos + at, ..*s });
    }
    for s in &clip.sources {
        re.sources.push(*s + at);
    }
    let b = clip.bounds();
    Rect::new(b.x + at.x - 2.0, b.y + at.y - 2.0, b.w + 4.0, b.h + 4.0)
}
