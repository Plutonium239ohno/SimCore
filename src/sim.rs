use crate::bh::Tree;
use crate::grid::Grid;
use crate::nuclear::NuclearState;
use crate::types::{ParticleType, RuleMatrix, default_types, ids};
use crate::walls::Walls;
use macroquad::math::{Vec2, vec2};
use crate::rng::gen_range;
use rayon::prelude::*;

/// Every force/feature is toggleable and tunable from here.
#[derive(Clone, Debug)]
pub struct Config {
    pub substeps: u32,
    /// Collision solver iterations per substep (more = stiffer deep piles).
    pub iterations: u32,

    pub collisions: bool,
    /// 0 = perfectly inelastic, 1 = perfectly elastic.
    pub restitution: f32,

    /// Uniform downward pull (px/s^2).
    pub uniform_gravity: bool,
    pub g: f32,

    /// Every particle attracts every other: a = G*m_j / r^2.
    pub mutual_gravity: bool,
    pub big_g: f32,

    /// Coulomb: a = -k*q_i*q_j / (m_i r^2) (like charges repel).
    pub coulomb: bool,
    pub k: f32,

    /// Plummer softening length for long-range forces (avoids 1/r^2 blow-ups).
    pub softening: f32,
    /// Use the Barnes–Hut tree (O(N log N)) instead of exact all-pairs (O(N^2)).
    pub barnes_hut: bool,
    /// Barnes–Hut accuracy: smaller = more accurate, slower.
    pub theta: f32,

    /// Short-range type-vs-type rules (see RuleMatrix).
    pub rules: bool,
    pub rule_range: f32,
    pub rule_strength: f32,

    /// Nuclear mode: strong force between nucleons, radioactive decay,
    /// electron capture and pair annihilation. Off = p/n are plain particles.
    pub nuclear: bool,
    /// Strong-force pull between touching nucleons (px/s^2).
    pub strong: f32,
    /// Strong-force reach beyond contact, as a fraction of the contact distance.
    pub strong_range: f32,
    /// Pull of every nucleon towards its nucleus' centre of mass (1/s^2):
    /// the liquid-drop "surface tension" that keeps nuclei round.
    pub roundness: f32,
    /// Damping (1/s) of nucleon motion *relative to their nucleus*: excited
    /// nuclei shed vibration/rotation (by gamma emission in reality) while
    /// the nucleus as a whole keeps its momentum.
    pub nuclear_damping: f32,
    /// The world is dense matter (a reactor, a sample) rather than vacuum or
    /// a thin plasma: charged particles are slowed by ionizing the material
    /// (towards its temperature), atoms carry their own electrons (electron
    /// capture works, positrons annihilate, stopped beta electrons join the
    /// material). Neutrons are neutral and unaffected, which is why they
    /// carry chain reactions.
    pub matter: bool,
    /// Stopping rate (1/s) of charged particles in matter.
    pub stopping: f32,
    /// Temperature of the material (K).
    pub temperature: f32,
    /// Coupling (1/s) of charged matter to a heat bath at `temperature`
    /// (0 = adiabatic: heat stays where it is deposited and only walls,
    /// drains and collisions move it).
    pub cooling: f32,
    /// Fraction of the energy fast ions lose while stopping that heats the
    /// atoms around them. (Real fission heat spreads over ~10^20 atoms;
    /// a sim nucleus stands for many, so only a fraction is kept.)
    pub heat_fraction: f32,
    /// Freeze plugs melt above this temperature (K).
    pub melt_temp: f32,
    /// Point-nucleus mode: a free neutron interacts with a nucleus when it
    /// passes within xs_scale * sigma^xs_exponent px (sigma = total
    /// cross-section in barns). The power compresses real cross-sections
    /// (3 b for deuterium to ~700 b for U-235) like energy and time are.
    pub xs_scale: f32,
    pub xs_exponent: f32,
    /// Multiplier on the quantum tunnelling probability through the Coulomb
    /// barrier (1 = real physics; larger makes fusion easier).
    pub tunneling: f32,
    /// Speed (px/s) of a 1 MeV neutron; sets the energy scale.
    pub mev_speed: f32,
    /// Energy compression: sim kinetic energy ~ E_real^energy_exponent, so
    /// speed ~ E^(exponent/2). Real energies span 0.025 eV (thermal neutrons)
    /// to 200 MeV (fission): 10 decades that have to fit on one screen.
    pub energy_exponent: f32,
    /// Half-life compression: t_sim = decay_scale * t_real^time_exponent.
    pub decay_scale: f32,
    pub time_exponent: f32,
    /// Speed of photons and neutrinos (px/s).
    pub light_speed: f32,

    /// Linear drag per second (0 = none).
    pub drag: f32,
    pub max_speed: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            substeps: 4,
            iterations: 4,
            collisions: true,
            restitution: 0.5,
            uniform_gravity: false,
            g: 400.0,
            mutual_gravity: false,
            big_g: 200.0,
            coulomb: true,
            k: 20000.0,
            softening: 6.0,
            barnes_hut: true,
            theta: 0.8,
            rules: false,
            rule_range: 60.0,
            rule_strength: 300.0,
            nuclear: false,
            strong: 6000.0,
            strong_range: 0.4,
            roundness: 120.0,
            nuclear_damping: 8.0,
            matter: false,
            stopping: 25.0,
            temperature: 293.0,
            cooling: 25.0,
            heat_fraction: 0.02,
            melt_temp: 1500.0,
            xs_scale: 3.8,
            xs_exponent: 0.3,
            tunneling: 1.0,
            mev_speed: 450.0,
            energy_exponent: 0.3,
            decay_scale: 3.0,
            time_exponent: 0.3,
            light_speed: 1200.0,
            drag: 0.1,
            max_speed: 1500.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MouseForce {
    pub pos: Vec2,
    /// Positive attracts, negative repels (px/s^2 at the centre).
    pub strength: f32,
    pub radius: f32,
}

/// Flattened per-type properties, rebuilt each step (tiny, stays in L1).
struct Table {
    n: usize,
    radius: Vec<f32>,
    mass: Vec<f32>,
    inv_mass: Vec<f32>,
    charge: Vec<f32>,
    rules: Vec<f32>,
    ghost: Vec<bool>,
    collides: Vec<bool>,
    max_radius: f32,
}

#[derive(Default)]
struct Sources {
    x: Vec<f32>,
    y: Vec<f32>,
    m: Vec<f32>,
    q: Vec<f32>,
}

/// Timing breakdown of the last step, in milliseconds.
#[derive(Default, Clone, Copy, Debug)]
pub struct Timings {
    pub far: f32,
    pub rules: f32,
    pub total: f32,
    /// Substep breakdown: sort, short-range, integrate, positions, velocities.
    pub sub: [f32; 5],
}

pub struct Sim {
    pub width: f32,
    pub height: f32,

    // Particle state (struct-of-arrays). Reordered by grid cell every substep,
    // so particle indices are NOT stable identities.
    pub pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    pub kind: Vec<u8>,
    /// Nucleus membership for nucleons (same tag = same nucleus); u32::MAX for
    /// everything else. Maintained by nuclear.rs.
    pub tag: Vec<u32>,
    /// Point nuclei (kind `ids::NUCLEUS`): packed (Z << 16 | N); 0 for
    /// every other particle.
    pub zn: Vec<u32>,
    /// Point-nucleus mode: every nucleus heavier than a nucleon is a single
    /// particle (see nuclear.rs); free neutrons don't collide, their
    /// interactions are rolled from cross-sections.
    pub point_nuclei: bool,
    pub(crate) next_tag: u32,
    /// Steps taken (with `seed`, decides every random draw of the next step).
    pub step_count: u64,
    /// Random seed of this scene: the same scene, stepped the same way,
    /// plays out identically (see rng.rs).
    pub seed: u64,
    /// Index of the particle's nucleus (u32::MAX: none / not yet indexed),
    /// rebuilt by nuclear.rs every frame; used for the roundness force.
    pub nuc: Vec<u32>,
    pub(crate) n_nuclei: usize,
    nuc_center: Vec<Vec2>,
    nuc_count: Vec<u32>,
    nuc_core: Vec<f32>,
    nuc_vel: Vec<Vec2>,
    /// Per-particle roundness acceleration (net zero over each nucleus).
    round_acc: Vec<Vec2>,
    /// Position correction from contacts inside the particle's own nucleus.
    int_corr: Vec<Vec2>,
    /// Mass of each nucleus (sum of its nucleons).
    nuc_mass: Vec<f32>,
    /// Nucleus contains protons (and is more than a lone neutron).
    nuc_charged: Vec<bool>,
    /// Element (Z) of each nucleus, for filter walls.
    nuc_z: Vec<u32>,
    /// What each particle looks like to the walls this substep.
    hits: Vec<crate::walls::Hit>,
    /// External correction (position or velocity) and contact count per
    /// particle, summed per nucleus and applied to it as a rigid body.
    ext: Vec<(Vec2, f32)>,
    body_pre: Vec<Vec2>,
    body_now: Vec<Vec2>,
    body_pos: Vec<Vec2>,
    buf_nuc: Vec<u32>,
    far: Vec<Vec2>,

    // Scratch buffers (reused every substep, never reallocated in steady state).
    prev_pos: Vec<Vec2>,
    pred_pos: Vec<Vec2>,
    vel_pre: Vec<Vec2>,
    near: Vec<Vec2>,
    buf_pos: Vec<Vec2>,
    buf_vel: Vec<Vec2>,
    buf_kind: Vec<u8>,
    buf_tag: Vec<u32>,
    buf_zn: Vec<u32>,
    buf_far: Vec<Vec2>,

    pub(crate) grid: Grid,
    /// User-drawn static walls.
    pub walls: Walls,
    sources: Sources,
    tree: Tree,

    pub types: Vec<ParticleType>,
    pub rules: RuleMatrix,
    pub cfg: Config,
    pub timings: Timings,
    pub nuclear: NuclearState,
    /// Control rods and fuel channels (Nuclear Game reactors).
    pub reactor: crate::reactor::Reactor,
    /// Kinetic energy of multi-nucleon nuclei before this frame's solve,
    /// when it should be conserved (see thermal.rs).
    pub(crate) ke_before: Option<f32>,
}

impl Sim {
    pub fn new(width: f32, height: f32) -> Self {
        let types = default_types();
        let rules = RuleMatrix::zero(types.len());
        Self {
            width,
            height,
            pos: Vec::new(),
            vel: Vec::new(),
            kind: Vec::new(),
            tag: Vec::new(),
            zn: Vec::new(),
            point_nuclei: false,
            next_tag: 0,
            step_count: 0,
            seed: 1,
            nuc: Vec::new(),
            n_nuclei: 0,
            nuc_center: Vec::new(),
            nuc_count: Vec::new(),
            nuc_core: Vec::new(),
            nuc_vel: Vec::new(),
            round_acc: Vec::new(),
            int_corr: Vec::new(),
            nuc_mass: Vec::new(),
            nuc_charged: Vec::new(),
            nuc_z: Vec::new(),
            hits: Vec::new(),
            ext: Vec::new(),
            body_pre: Vec::new(),
            body_now: Vec::new(),
            body_pos: Vec::new(),
            buf_nuc: Vec::new(),
            far: Vec::new(),
            prev_pos: Vec::new(),
            pred_pos: Vec::new(),
            vel_pre: Vec::new(),
            near: Vec::new(),
            buf_pos: Vec::new(),
            buf_vel: Vec::new(),
            buf_kind: Vec::new(),
            buf_tag: Vec::new(),
            buf_zn: Vec::new(),
            buf_far: Vec::new(),
            grid: Grid::default(),
            walls: Walls::default(),
            sources: Sources::default(),
            tree: Tree::default(),
            types,
            rules,
            cfg: Config::default(),
            timings: Timings::default(),
            nuclear: NuclearState::default(),
            reactor: crate::reactor::Reactor::default(),
            ke_before: None,
        }
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Long-range (gravity + Coulomb) acceleration from the last step.
    pub fn far_accel(&self) -> &[Vec2] {
        &self.far
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    pub fn add(&mut self, p: Vec2, v: Vec2, kind: u8) {
        self.pos.push(p);
        self.vel.push(v);
        self.kind.push(kind);
        let tag = if ParticleType::is_nucleon(kind) { self.new_tag() } else { u32::MAX };
        self.tag.push(tag);
        self.zn.push(0);
        self.nuc.push(u32::MAX);
        self.far.push(Vec2::ZERO);
    }

    /// Add a body of z protons and n neutrons as one particle: a free
    /// neutron or proton, or a point nucleus. Returns its tag.
    pub fn add_point(&mut self, p: Vec2, v: Vec2, z: u32, n: u32) -> u32 {
        match (z, n) {
            (0, 1) => self.add(p, v, ids::NEUTRON),
            (1, 0) => self.add(p, v, ids::PROTON),
            _ => {
                self.add(p, v, ids::NUCLEUS);
                let last = self.len() - 1;
                self.zn[last] = pack_zn(z, n);
                self.tag[last] = self.new_tag();
            }
        }
        self.tag[self.len() - 1]
    }

    /// Turn particle i into the body (z, n) (see `add_point`).
    pub fn set_point(&mut self, i: usize, z: u32, n: u32) {
        match (z, n) {
            (0, 1) => {
                self.kind[i] = ids::NEUTRON;
                self.zn[i] = 0;
            }
            (1, 0) => {
                self.kind[i] = ids::PROTON;
                self.zn[i] = 0;
            }
            _ => {
                self.kind[i] = ids::NUCLEUS;
                self.zn[i] = pack_zn(z, n);
            }
        }
    }

    /// Protons and neutrons particle i holds: (1, 0) for a proton, (0, 1)
    /// for a neutron, the whole nucleus for a point nucleus, else (0, 0).
    #[inline]
    pub fn zn_of(&self, i: usize) -> (u32, u32) {
        match self.kind[i] {
            ids::PROTON => (1, 0),
            ids::NEUTRON => (0, 1),
            _ => unpack_zn(self.zn[i]),
        }
    }

    /// Particle i is (part of) a nucleus: a nucleon or a point nucleus.
    #[inline]
    pub fn is_nuclear(&self, i: usize) -> bool {
        ParticleType::is_nucleon(self.kind[i]) || self.zn[i] != 0
    }

    /// Mass of particle i (nucleon masses).
    #[inline]
    pub fn mass_of(&self, i: usize) -> f32 {
        if self.zn[i] != 0 {
            let (z, n) = unpack_zn(self.zn[i]);
            (z + n) as f32
        } else {
            self.types[self.kind[i] as usize].mass
        }
    }

    /// Every nucleus by tag: centre, velocity, Z, N and member count.
    pub fn bodies(&self) -> crate::rng::HashMap<u32, BodyInfo> {
        let mut m: crate::rng::HashMap<u32, BodyInfo> = crate::rng::HashMap::with_capacity_and_hasher(self.n_nuclei + 8, Default::default());
        for i in 0..self.len() {
            if !self.is_nuclear(i) {
                continue;
            }
            let (z, n) = self.zn_of(i);
            let e = m.entry(self.tag[i]).or_default();
            e.pos += self.pos[i];
            e.vel += self.vel[i];
            e.z += z;
            e.n += n;
            e.parts += 1;
        }
        for b in m.values_mut() {
            b.pos /= b.parts as f32;
            b.vel /= b.parts as f32;
        }
        m
    }

    /// Group nucleons into nuclei by tag (for the roundness force).
    pub fn index_nuclei(&mut self) {
        let mut map: crate::rng::HashMap<u32, u32> = crate::rng::HashMap::with_capacity_and_hasher(self.len() / 4, Default::default());
        for i in 0..self.len() {
            self.nuc[i] = if self.is_nuclear(i) {
                let next = map.len() as u32;
                *map.entry(self.tag[i]).or_insert(next)
            } else {
                u32::MAX
            };
        }
        self.n_nuclei = map.len();
    }

    fn update_centroids(&mut self) {
        let m = self.n_nuclei;
        self.nuc_center.clear();
        self.nuc_center.resize(m, Vec2::ZERO);
        self.nuc_count.clear();
        self.nuc_count.resize(m, 0);
        self.nuc_vel.clear();
        self.nuc_vel.resize(m, Vec2::ZERO);
        for ((p, v), &k) in self.pos.iter().zip(&self.vel).zip(&self.nuc) {
            if (k as usize) < m {
                self.nuc_center[k as usize] += *p;
                self.nuc_vel[k as usize] += *v;
                self.nuc_count[k as usize] += 1;
            }
        }
        for ((c, v), &k) in self.nuc_center.iter_mut().zip(self.nuc_vel.iter_mut()).zip(&self.nuc_count) {
            *c /= k.max(1) as f32;
            *v /= k.max(1) as f32;
        }
        self.nuc_mass.clear();
        self.nuc_mass.resize(m, 0.0);
        self.nuc_charged.clear();
        self.nuc_charged.resize(m, false);
        self.nuc_z.clear();
        self.nuc_z.resize(m, 0);
        for i in 0..self.pos.len() {
            let k = self.nuc[i] as usize;
            if k < m {
                self.nuc_mass[k] += self.mass_of(i);
                let z = self.zn_of(i).0;
                self.nuc_z[k] += z;
                if z > 0 {
                    self.nuc_charged[k] = true;
                }
            }
        }
        // Radius of an ideally packed disc of A nucleons (hex packing
        // fraction 0.9069): nucleons inside it feel no pull.
        let r = self.types[ids::NEUTRON as usize].radius;
        self.nuc_core.clear();
        self.nuc_core.extend(self.nuc_count.iter().map(|&k| r * (k as f32 / 0.9069).sqrt()));

        // Liquid-drop surface tension: nucleons sticking out beyond the ideal
        // disc are pulled towards the centre (only the excess: a constant
        // squeeze would keep every contact loaded). The nucleus' mean pull is
        // subtracted so the net force is zero: no self-propulsion, and no
        // torque since the pulls are central.
        let k_round = self.cfg.roundness * 4.0;
        self.round_acc.clear();
        self.round_acc.resize(self.pos.len(), Vec2::ZERO);
        if k_round > 0.0 {
            let mut mean = vec![Vec2::ZERO; m];
            for i in 0..self.pos.len() {
                let k = self.nuc[i] as usize;
                if k < m {
                    let d = self.nuc_center[k] - self.pos[i];
                    let dist = d.length();
                    if dist > self.nuc_core[k] {
                        let f = d * (k_round * (dist - self.nuc_core[k]) / dist);
                        self.round_acc[i] = f;
                        mean[k] += f;
                    }
                }
            }
            for i in 0..self.pos.len() {
                let k = self.nuc[i] as usize;
                if k < m && self.nuc_count[k] > 1 {
                    self.round_acc[i] -= mean[k] / self.nuc_count[k] as f32;
                }
            }
        }
    }

    /// A fresh nucleus id.
    pub fn new_tag(&mut self) -> u32 {
        self.next_tag = self.next_tag.wrapping_add(1) % (u32::MAX - 1);
        self.next_tag
    }

    pub fn clear(&mut self) {
        self.pos.clear();
        self.vel.clear();
        self.kind.clear();
        self.tag.clear();
        self.zn.clear();
        self.nuc.clear();
        self.n_nuclei = 0;
        self.far.clear();
    }

    /// Keep only particles for which `keep(i)` is true (order preserved).
    pub fn retain(&mut self, keep: impl Fn(usize) -> bool) {
        let mut w = 0;
        for i in 0..self.pos.len() {
            if keep(i) {
                self.pos[w] = self.pos[i];
                self.vel[w] = self.vel[i];
                self.kind[w] = self.kind[i];
                self.tag[w] = self.tag[i];
                self.zn[w] = self.zn[i];
                self.nuc[w] = self.nuc[i];
                self.far[w] = self.far[i];
                w += 1;
            }
        }
        self.pos.truncate(w);
        self.vel.truncate(w);
        self.kind.truncate(w);
        self.tag.truncate(w);
        self.zn.truncate(w);
        self.nuc.truncate(w);
        self.far.truncate(w);
    }

    /// Add a new custom species (and grow the rule matrix). Returns its id.
    pub fn add_species(&mut self) -> Option<u8> {
        if self.types.len() >= crate::types::MAX_TYPES {
            return None;
        }
        let n_custom = self.types.iter().filter(|t| t.custom).count();
        let name = format!("Species {}", (b'A' + n_custom as u8) as char);
        let hue = (n_custom as f32 * 137.5 + 130.0) % 360.0;
        self.types.push(ParticleType::custom(&name, hue, 4.0, 1.0, 0.0));
        self.rules.resize(self.types.len());
        Some(self.types.len() as u8 - 1)
    }

    /// Remove the last custom species and all particles of it.
    pub fn remove_last_species(&mut self) {
        if let Some(last) = self.types.last() {
            if last.custom && self.types.len() > ids::FIRST_CUSTOM as usize {
                let k = (self.types.len() - 1) as u8;
                let kind = self.kind.clone();
                self.retain(|i| kind[i] != k);
                self.types.pop();
                self.rules.resize(self.types.len());
            }
        }
    }

    /// Remove custom species `k` and all its particles; later species move
    /// down one id (particles and the rule matrix follow).
    pub fn remove_species(&mut self, k: u8) {
        let ku = k as usize;
        if ku >= self.types.len() || !self.types[ku].custom {
            return;
        }
        let kind = self.kind.clone();
        self.retain(|i| kind[i] != k);
        for x in self.kind.iter_mut() {
            if *x > k {
                *x -= 1;
            }
        }
        self.types.remove(ku);
        let n = self.types.len();
        let mut r = crate::types::RuleMatrix::zero(n);
        for a in 0..n {
            for b in 0..n {
                let (oa, ob) = (a + (a >= ku) as usize, b + (b >= ku) as usize);
                r.set(a, b, self.rules.get(oa, ob));
            }
        }
        self.rules = r;
    }

    /// Copy of species `k` (same rules towards and from everyone) as a new
    /// species. Returns its id.
    pub fn duplicate_species(&mut self, k: u8) -> Option<u8> {
        if self.types.len() >= crate::types::MAX_TYPES || k as usize >= self.types.len() {
            return None;
        }
        let mut t = self.types[k as usize].clone();
        t.custom = true;
        t.name = format!("{} copy", t.name);
        t.hue = (t.hue + 40.0) % 360.0;
        t.color = crate::types::hsv(t.hue, 0.65, 0.95);
        self.types.push(t);
        let n = self.types.len();
        self.rules.resize(n);
        let new = n - 1;
        for j in 0..n - 1 {
            let (a, b) = (self.rules.get(k as usize, j), self.rules.get(j, k as usize));
            self.rules.set(new, j, a);
            self.rules.set(j, new, b);
        }
        let own = self.rules.get(k as usize, k as usize);
        self.rules.set(new, new, own);
        Some(new as u8)
    }

    fn max_radius(&self) -> f32 {
        self.types.iter().map(|t| t.radius).fold(1.0, f32::max)
    }

    /// Fill the box with up to `count` random particles without overlaps
    /// (overlaps would be resolved violently and look like explosions).
    /// Returns how many were placed; fewer if the box is too full.
    pub fn spawn_random(&mut self, count: usize) -> usize {
        use crate::rng::HashMap;
        let cell = self.max_radius() * 2.0;
        let key = |p: Vec2| ((p.x / cell) as i32, (p.y / cell) as i32);
        let mut occ: HashMap<(i32, i32), Vec<(Vec2, f32)>> = HashMap::default();
        for (p, k) in self.pos.iter().zip(&self.kind) {
            occ.entry(key(*p)).or_default().push((*p, self.types[*k as usize].radius));
        }
        let pool: Vec<u8> = (0..self.types.len() as u8)
            .filter(|&k| self.types[k as usize].in_random_spawn)
            .collect();
        if pool.is_empty() {
            return 0;
        }
        let mut placed = 0;
        for _ in 0..count * 20 {
            if placed == count {
                break;
            }
            let kind = pool[gen_range(0, pool.len())];
            let r = self.types[kind as usize].radius;
            let p = vec2(gen_range(r, self.width - r), gen_range(r, self.height - r));
            let (kx, ky) = key(p);
            if self.walls.blocked(p, r) {
                continue;
            }
            let free = (-1..=1).all(|dy| {
                (-1..=1).all(|dx| {
                    occ.get(&(kx + dx, ky + dy))
                        .is_none_or(|v| v.iter().all(|(q, rq)| p.distance_squared(*q) >= (r + rq) * (r + rq)))
                })
            });
            if free {
                occ.entry((kx, ky)).or_default().push((p, r));
                let v = vec2(gen_range(-50.0, 50.0), gen_range(-50.0, 50.0));
                self.add(p, v, kind);
                placed += 1;
            }
        }
        placed
    }

    /// Brush: add up to `count` particles of `kind` inside a circle, only in free space.
    pub fn spawn_in_circle(&mut self, center: Vec2, radius: f32, count: usize, kind: u8) {
        let r = self.types[kind as usize].radius;
        let reach = radius + r + self.max_radius();
        let mut near: Vec<(Vec2, f32)> = self
            .pos
            .iter()
            .zip(&self.kind)
            .filter(|(p, _)| p.distance_squared(center) < reach * reach)
            .map(|(p, k)| (*p, self.types[*k as usize].radius))
            .collect();
        let mut placed = 0;
        for _ in 0..count * 8 {
            if placed == count {
                break;
            }
            let a = gen_range(0.0, std::f32::consts::TAU);
            let p = center + vec2(a.cos(), a.sin()) * (radius * gen_range(0.0f32, 1.0).sqrt());
            if p.x < r || p.y < r || p.x > self.width - r || p.y > self.height - r {
                continue;
            }
            if near.iter().all(|(q, rq)| p.distance_squared(*q) >= (r + rq) * (r + rq)) && !self.walls.blocked(p, r) {
                near.push((p, r));
                self.add(p, Vec2::ZERO, kind);
                placed += 1;
            }
        }
    }

    fn table(&self) -> Table {
        let n = self.types.len();
        let radius: Vec<f32> = self.types.iter().map(|t| t.radius).collect();
        let max_radius = radius.iter().cloned().fold(1.0, f32::max);
        Table {
            n,
            mass: self.types.iter().map(|t| t.mass).collect(),
            inv_mass: self.types.iter().map(|t| 1.0 / t.mass.max(1e-6)).collect(),
            charge: self.types.iter().map(|t| t.charge).collect(),
            rules: self.rules.strength.clone(),
            ghost: self.types.iter().map(|t| t.ghost).collect(),
            // Point-nucleus mode: free neutrons fly through everything but
            // walls; their scattering and absorption are rolled in nuclear.rs.
            collides: self
                .types
                .iter()
                .enumerate()
                .map(|(k, t)| t.collides && !t.ghost && !(self.point_nuclei && k == ids::NEUTRON as usize))
                .collect(),
            radius,
            max_radius,
        }
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32, mouse: Option<MouseForce>) {
        self.step_count = self.step_count.wrapping_add(1);
        crate::rng::seed(self.seed ^ crate::rng::mix(self.step_count));
        if self.pos.is_empty() {
            // Nothing to move, but spawners and sources may add something.
            if self.cfg.nuclear && !self.reactor.is_empty() {
                crate::reactor::update(self, dt);
            }
            return;
        }
        let t0 = std::time::Instant::now();
        let t = self.table();
        // Cells slightly larger than the biggest particle diameter, so contacts
        // created by one substep of motion are still found in the 3x3 block.
        let cell = t.max_radius * 2.0 * 1.25;
        let n_sub = self.cfg.substeps.max(1);
        let h = dt / n_sub as f32;
        self.walls.prepare(self.width, self.height, t.max_radius + self.cfg.max_speed * h + 2.0);

        // Soft forces (long-range gravity/Coulomb and the short-range type rules)
        // are computed once per frame and held over the substeps: they vary
        // slowly compared with contacts, and they're the expensive part.
        self.sort_by_cell(cell);
        let tf = std::time::Instant::now();
        self.compute_far(&t);
        self.timings.far = tf.elapsed().as_secs_f32() * 1000.0;
        let tr = std::time::Instant::now();
        let mut far = std::mem::take(&mut self.far);
        self.compute_short(&t, true, false, &mut far);
        self.far = far;
        self.timings.rules = tr.elapsed().as_secs_f32() * 1000.0;

        // Collisions between nuclei are elastic: in neutral matter (no
        // field doing work) their kinetic energy should come out unchanged.
        self.ke_before = if self.cfg.nuclear && self.cfg.matter && !self.cfg.coulomb && mouse.is_none() {
            Some(crate::thermal::body_ke(self))
        } else {
            None
        };
        self.timings.sub = [0.0; 5];
        for s in 0..n_sub {
            let mut tk = std::time::Instant::now();
            let mut lap = |k: usize, tm: &mut Timings| {
                tm.sub[k] += tk.elapsed().as_secs_f32() * 1000.0;
                tk = std::time::Instant::now();
            };
            if s > 0 {
                self.sort_by_cell(cell);
            }
            if self.cfg.nuclear {
                self.update_centroids();
            } else {
                self.nuc_center.clear();
                self.nuc_vel.clear();
                self.nuc_count.clear();
                self.nuc_mass.clear();
                self.nuc_charged.clear();
            }
            self.compute_hits();
            lap(0, &mut self.timings);
            // Stiff short-range strong force: fresh every substep.
            let mut near = std::mem::take(&mut self.near);
            near.clear();
            near.resize(self.pos.len(), Vec2::ZERO);
            self.compute_short(&t, false, true, &mut near);
            self.near = near;
            lap(1, &mut self.timings);
            self.integrate(&t, h, mouse);
            lap(2, &mut self.timings);
            self.solve_positions(&t, h);
            lap(3, &mut self.timings);
            self.solve_velocities(&t, h);
            lap(4, &mut self.timings);
        }

        // Decays and reactions. Runs while the grid still matches particle order.
        if self.cfg.nuclear {
            crate::thermal::apply(self, dt);
            crate::nuclear::process(self, dt);
            crate::reactor::update(self, dt);
        }

        // Photons/neutrinos that left the box are gone.
        if t.ghost.iter().any(|&g| g) {
            let (w, hgt) = (self.width, self.height);
            let dead: Vec<bool> = self
                .pos
                .iter()
                .zip(&self.kind)
                .map(|(p, &k)| t.ghost[k as usize] && (p.x < -10.0 || p.y < -10.0 || p.x > w + 10.0 || p.y > hgt + 10.0))
                .collect();
            if dead.iter().any(|&d| d) {
                self.retain(|i| !dead[i]);
            }
        }
        self.timings.total = t0.elapsed().as_secs_f32() * 1000.0;
    }

    fn sort_by_cell(&mut self, cell: f32) {
        self.grid.build(&self.pos, self.width, self.height, cell);
        let order = &self.grid.order;
        permute(&self.pos, &mut self.buf_pos, order);
        permute(&self.vel, &mut self.buf_vel, order);
        permute(&self.kind, &mut self.buf_kind, order);
        permute(&self.tag, &mut self.buf_tag, order);
        permute(&self.zn, &mut self.buf_zn, order);
        permute(&self.nuc, &mut self.buf_nuc, order);
        permute(&self.far, &mut self.buf_far, order);
        std::mem::swap(&mut self.pos, &mut self.buf_pos);
        std::mem::swap(&mut self.vel, &mut self.buf_vel);
        std::mem::swap(&mut self.kind, &mut self.buf_kind);
        std::mem::swap(&mut self.tag, &mut self.buf_tag);
        std::mem::swap(&mut self.zn, &mut self.buf_zn);
        std::mem::swap(&mut self.nuc, &mut self.buf_nuc);
        std::mem::swap(&mut self.far, &mut self.buf_far);
    }

    /// Mutual gravity + Coulomb.
    fn compute_far(&mut self, t: &Table) {
        let g = if self.cfg.mutual_gravity { self.cfg.big_g } else { 0.0 };
        let k = if self.cfg.coulomb { self.cfg.k } else { 0.0 };
        self.far.resize(self.pos.len(), Vec2::ZERO);
        if g == 0.0 && k == 0.0 {
            self.far.fill(Vec2::ZERO);
            return;
        }

        // Only particles that actually source a field (e.g. only charged ones
        // when Coulomb is the only long-range force) are included.
        let src = &mut self.sources;
        src.x.clear();
        src.y.clear();
        src.m.clear();
        src.q.clear();
        for ((p, &kd), &zn) in self.pos.iter().zip(&self.kind).zip(&self.zn) {
            let (m, q) = if zn != 0 {
                let (z, n) = unpack_zn(zn);
                ((z + n) as f32, z as f32)
            } else {
                (t.mass[kd as usize], t.charge[kd as usize])
            };
            if g * m != 0.0 || k * q != 0.0 {
                src.x.push(p.x);
                src.y.push(p.y);
                src.m.push(m);
                src.q.push(q);
            }
        }

        let eps2 = self.cfg.softening * self.cfg.softening;
        let (pos, kind, zn) = (&self.pos, &self.kind, &self.zn);
        let target_kq = |i: usize| {
            if zn[i] != 0 {
                let (z, n) = unpack_zn(zn[i]);
                return k * z as f32 / (z + n) as f32;
            }
            let ti = kind[i] as usize;
            k * t.charge[ti] * t.inv_mass[ti]
        };

        if self.cfg.barnes_hut {
            self.tree.build(&src.x, &src.y, &src.m, &src.q);
            let tree = &self.tree;
            let theta = self.cfg.theta;
            self.far.par_iter_mut().enumerate().for_each(|(i, a)| {
                let kq = target_kq(i);
                *a = if g == 0.0 && kq == 0.0 {
                    Vec2::ZERO
                } else {
                    let (ax, ay) = tree.accel(pos[i].x, pos[i].y, g, kq, eps2, theta);
                    vec2(ax, ay)
                };
            });
            return;
        }

        // Exact all-pairs: multithreaded, 8-wide chunks so LLVM vectorizes it.
        while src.x.len() % 8 != 0 {
            src.x.push(0.0);
            src.y.push(0.0);
            src.m.push(0.0);
            src.q.push(0.0);
        }
        let (sx, sy, sm, sq) = (&src.x, &src.y, &src.m, &src.q);
        self.far.par_iter_mut().enumerate().for_each(|(i, a)| {
            let p = pos[i];
            let kq = target_kq(i);
            if g == 0.0 && kq == 0.0 {
                *a = Vec2::ZERO;
                return;
            }
            let mut ax = [0.0f32; 8];
            let mut ay = [0.0f32; 8];
            for c in (0..sx.len()).step_by(8) {
                let xs: &[f32; 8] = sx[c..c + 8].try_into().unwrap();
                let ys: &[f32; 8] = sy[c..c + 8].try_into().unwrap();
                let ms: &[f32; 8] = sm[c..c + 8].try_into().unwrap();
                let qs: &[f32; 8] = sq[c..c + 8].try_into().unwrap();
                for l in 0..8 {
                    let dx = xs[l] - p.x;
                    let dy = ys[l] - p.y;
                    let r2 = dx * dx + dy * dy + eps2;
                    let s = (g * ms[l] - kq * qs[l]) / (r2 * r2.sqrt());
                    ax[l] += s * dx;
                    ay[l] += s * dy;
                }
            }
            *a = vec2(ax.iter().sum(), ay.iter().sum());
        });
    }

    /// Short-range forces, added into the per-frame force buffer:
    /// type-vs-type rules and (in nuclear mode) the strong force.
    ///
    /// Adds into `out`. The rules are soft and evaluated once per frame; the
    /// strong force is stiff and must be evaluated every substep (a force held
    /// across substeps points where the partner *was*, which pumps energy into
    /// any rotating bound pair).
    fn compute_short(&self, t: &Table, rules_on: bool, strong_on: bool, out: &mut [Vec2]) {
        let cfg = &self.cfg;
        let rules_on = rules_on && cfg.rules && cfg.rule_range > 0.0;
        let strong_on = strong_on && cfg.nuclear && cfg.strong > 0.0;
        if !rules_on && !strong_on {
            return;
        }
        let range = cfg.rule_range;
        let nucleon_r = t.radius[ids::NEUTRON as usize].max(t.radius[ids::PROTON as usize]);
        let strong_reach = 2.0 * nucleon_r * (1.0 + cfg.strong_range);
        let mut search = if rules_on { range } else { 0.0 };
        if strong_on {
            search = search.max(strong_reach);
        }

        let search2 = search * search;
        let reach = (search * self.grid.inv).ceil() as i32;
        let collisions = cfg.collisions;
        let rule_k = cfg.rule_strength;
        let (strong, strong_frac) = (cfg.strong, cfg.strong_range.max(1e-3));
        let tag = &self.tag;
        let round_acc = if strong_on { &self.round_acc[..] } else { &[] };
        let n = t.n;
        let (pos, kind, grid) = (&self.pos, &self.kind, &self.grid);

        out.par_iter_mut().enumerate().for_each(|(i, a)| {
            let ti = kind[i] as usize;
            if t.ghost[ti] || (!rules_on && !(strong_on && ParticleType::is_nucleon(ti as u8))) {
                return;
            }
            let p = pos[i];
            let ri = t.radius[ti];
            let row = &t.rules[ti * n..ti * n + n];
            let i_nucleon = strong_on && ParticleType::is_nucleon(ti as u8);
            let mut f_rule = Vec2::ZERO;
            let mut f_strong = Vec2::ZERO;
            grid.for_each_near(p, reach, |j| {
                let d = pos[j] - p;
                let d2 = d.length_squared();
                if d2 >= search2 || d2 == 0.0 {
                    return;
                }
                let tj = kind[j] as usize;
                if t.ghost[tj] {
                    return;
                }
                let dist = d2.sqrt();
                let rs = ri + t.radius[tj];

                // Only nucleons of the *same nucleus* bind. Separate nuclei just
                // collide and repel until they fuse (see nuclear.rs).
                if i_nucleon && ParticleType::is_nucleon(tj as u8) && tag[i] == tag[j] {
                    // Strong force: short-ranged, charge-independent attraction.
                    // Shaped as a smooth well that is zero at contact and at the
                    // range edge (peak in between), so nucleons settle *into*
                    // contact instead of being driven into overlap every substep.
                    let x = (dist - rs) / (rs * strong_frac);
                    if x > 0.0 && x < 1.0 {
                        f_strong += d * (strong * 4.0 * x * (1.0 - x) / dist);
                    }
                }

                if rules_on && dist < range {
                    let s = if dist > rs {
                        if range <= rs {
                            0.0
                        } else {
                            // Triangle profile: 0 at contact, peak mid-range, 0 at range.
                            let x = (dist - rs) / (range - rs);
                            row[tj] * (1.0 - (2.0 * x - 1.0).abs())
                        }
                    } else if !collisions {
                        // Soft core repulsion so overlapping particles separate.
                        dist / rs - 1.0
                    } else {
                        0.0
                    };
                    f_rule += d * (s / dist);
                }
            });
            *a += (f_rule * rule_k + f_strong) * t.inv_mass[ti];
            if let Some(r) = round_acc.get(i) {
                *a += *r;
            }
        });
    }

    /// Apply forces, update velocity, predict positions (semi-implicit Euler).
    /// Afterwards: pos = predicted, prev_pos = start-of-substep, vel_pre = velocity
    /// before contacts (used for restitution).
    fn integrate(&mut self, t: &Table, h: f32, mouse: Option<MouseForce>) {
        let cfg = &self.cfg;
        let gvec = if cfg.uniform_gravity { vec2(0.0, cfg.g) } else { Vec2::ZERO };
        let damp = (1.0 - cfg.drag * h).max(0.0);
        let vmax = cfg.max_speed;
        let (pos, vel, far, near, kind) = (&self.pos, &self.vel, &self.far, &self.near, &self.kind);
        let (nuc, nuc_vel) = (&self.nuc, &self.nuc_vel);
        // (Stopping and the heat bath act once per frame: see thermal.rs.)
        let int_damp = (-cfg.nuclear_damping * h).exp();

        self.buf_pos.resize(pos.len(), Vec2::ZERO);
        self.vel_pre.resize(pos.len(), Vec2::ZERO);
        self.buf_pos
            .par_iter_mut()
            .zip(self.vel_pre.par_iter_mut())
            .enumerate()
            .for_each(|(i, (np, nv))| {
                let p = pos[i];
                if t.ghost[kind[i] as usize] {
                    // Photons/neutrinos: straight lines, no forces.
                    *nv = vel[i];
                    *np = p + vel[i] * h;
                    return;
                }
                let mut a = far[i] + near[i] + gvec;
                if let Some(m) = mouse {
                    let d = m.pos - p;
                    let dist = d.length();
                    if dist < m.radius && dist > 1e-3 {
                        a += d * (m.strength * (1.0 - dist / m.radius) / dist);
                    }
                }
                let mut v = clamp_speed((vel[i] + a * h) * damp, vmax);
                if let Some(&vm) = nuc_vel.get(nuc[i] as usize) {
                    v = vm + (v - vm) * int_damp;
                }
                // Restitution judges impacts by the velocity *before* this
                // substep's forces: contacts merely pressed together by a force
                // (gravity, strong force, Coulomb) must not "bounce".
                *nv = vel[i];
                *np = p + v * h;
            });

        std::mem::swap(&mut self.prev_pos, &mut self.pos);
        std::mem::swap(&mut self.pos, &mut self.buf_pos);
    }

    /// Resolve overlaps and walls on positions, then derive
    /// velocity from the actual displacement (position-based dynamics). This is
    /// what lets dense piles come to rest instead of jittering.
    ///
    /// Jacobi style: each particle computes its own correction from all its
    /// contacts and writes only itself, so it's embarrassingly parallel and
    /// deterministic. Contact corrections are averaged over the contact count
    /// so a particle squeezed from k sides doesn't overshoot.
    fn solve_positions(&mut self, t: &Table, h: f32) {
        let collisions = self.cfg.collisions;
        let (w, hgt) = (self.width, self.height);
        let iters = if collisions { self.cfg.iterations.max(1) } else { 1 };
        const OMEGA: f32 = 1.5; // over-relaxation; Jacobi averaging under-corrects otherwise
        // Inserted control rods: solid to everything but free neutrons.
        let rods = if self.cfg.nuclear { self.reactor.rects() } else { Vec::new() };
        let rods = &rods[..];
        let rod_solid = |k: u8, bound: bool| bound || (k != ids::NEUTRON && k != ids::PHOTON && k != ids::NEUTRINO);

        // Keep the predicted (pre-solve) positions: the restitution pass needs
        // to know which pairs were in contact before they got pushed apart.
        self.pred_pos.clone_from(&self.pos);

        for _ in 0..iters {
            let (pos, prev, kind, grid, walls) = (&self.pos, &self.prev_pos, &self.kind, &self.grid, &self.walls);
            let nuc = &self.nuc;
            // Nuclei act as rigid bodies in collisions with anything outside
            // them (other nuclei, walls): the contact moves the whole nucleus,
            // with its whole mass. Contacts inside a nucleus only shape it.
            let bodies = self.cfg.nuclear && self.n_nuclei > 0 && self.nuc_count.len() == self.n_nuclei;
            let (nuc_count, nuc_mass) = (&self.nuc_count, &self.nuc_mass);
            let multi = |i: usize| bodies && nuc_count.get(nuc[i] as usize).is_some_and(|&c| c > 1);
            let zn = &self.zn;
            let inv_body = |i: usize, ti: usize| {
                if multi(i) {
                    1.0 / nuc_mass[nuc[i] as usize].max(1e-3)
                } else if zn[i] != 0 {
                    let (z, n) = unpack_zn(zn[i]);
                    1.0 / (z + n) as f32
                } else {
                    t.inv_mass[ti]
                }
            };
            let hits = &self.hits;
            let hit = |i: usize| hits[i];
            self.int_corr.resize(pos.len(), Vec2::ZERO);
            self.ext.resize(pos.len(), (Vec2::ZERO, 0.0));
            self.buf_pos.resize(pos.len(), Vec2::ZERO);
            self.buf_pos
                .par_iter_mut()
                .zip(self.int_corr.par_iter_mut())
                .zip(self.ext.par_iter_mut())
                .enumerate()
                .with_min_len(64)
                .for_each(|(i, ((np, ic), ext))| {
                let mut p = pos[i];
                let ti = kind[i] as usize;
                *ic = Vec2::ZERO;
                *ext = (Vec2::ZERO, 0.0);
                if t.ghost[ti] {
                    *np = p;
                    return;
                }
                let i_multi = multi(i);
                let my_nuc = if bodies { nuc[i] } else { u32::MAX };
                let ri = t.radius[ti];
                let im_i = t.inv_mass[ti];
                let wb_i = inv_body(i, ti);
                let i_collides = collisions && t.collides[ti];

                let mut dp_int = Vec2::ZERO;
                let mut n_int = 0u32;
                let mut dp_ext = Vec2::ZERO;
                let mut n_ext = 0u32;
                let mut groups = Groups::new();
                if i_collides {
                    grid.for_each_near(p, 1, |j| {
                        if j == i {
                            return;
                        }
                        let tj = kind[j] as usize;
                        if !t.collides[tj] {
                            return;
                        }
                        let d = p - pos[j];
                        let d2 = d.length_squared();
                        let rs = ri + t.radius[tj];
                        if d2 >= rs * rs {
                            return;
                        }
                        let (nrm, dist) = if d2 > 1e-12 {
                            let dist = d2.sqrt();
                            (d / dist, dist)
                        } else {
                            // Exactly coincident: separate along x, deterministically.
                            (vec2(if i < j { -1.0 } else { 1.0 }, 0.0), 0.0)
                        };
                        if my_nuc != u32::MAX && nuc[j] == my_nuc {
                            let im_j = t.inv_mass[tj];
                            dp_int += nrm * ((rs - dist) * im_i / (im_i + im_j));
                            n_int += 1;
                        } else {
                            let wb_j = inv_body(j, tj);
                            let c = nrm * ((rs - dist) * wb_i / (wb_i + wb_j));
                            dp_ext += c;
                            n_ext += 1;
                            groups.add(if multi(j) { nuc[j] } else { u32::MAX - j as u32 }, c);
                        }
                    });
                }
                // Over-relax only shared contacts; a lone contact is solved
                // exactly (overshooting it injects energy).
                let relax = |n: u32| (OMEGA / n.max(1) as f32).min(1.0);
                let mut corr = dp_int * relax(n_int);
                if !i_multi {
                    let (g, nb) = groups.total();
                    corr += g * relax(nb);
                }
                // Never move more than one radius per iteration (e.g. the brush
                // spawns particles on top of each other).
                let c2 = corr.length_squared();
                let mut cap = 1.0;
                if c2 > ri * ri {
                    cap = ri / c2.sqrt();
                    corr *= cap;
                }
                p += corr;
                *ic = dp_int * (relax(n_int) * cap);

                let mut q = walls.resolve(p, prev[i], ri, hit(i));
                if !rods.is_empty() && rod_solid(kind[i], i_multi) {
                    q = crate::reactor::push_out(rods, q, ri);
                }
                q.x = q.x.clamp(ri, (w - ri).max(ri));
                q.y = q.y.clamp(ri, (hgt - ri).max(ri));
                if i_multi {
                    // Hand external pushes (contacts + walls) to the nucleus.
                    let wall = q - p;
                    if wall != Vec2::ZERO {
                        dp_ext += wall;
                        n_ext += 1;
                    }
                    *ext = (dp_ext, n_ext as f32);
                    *np = p;
                } else {
                    *np = q;
                }
            });
            if bodies {
                // Jacobi corrections scaled per particle aren't equal and
                // opposite, so contacts *inside* a nucleus could push it around
                // (it would propel itself): remove the net internal correction.
                // External pushes move the whole nucleus by their average.
                let m = self.n_nuclei;
                let mut net = vec![Vec2::ZERO; m];
                let mut ext_sum = vec![(Vec2::ZERO, 0.0f32); m];
                for i in 0..self.buf_pos.len() {
                    let k = self.nuc[i] as usize;
                    if k < m {
                        net[k] += self.int_corr[i];
                        ext_sum[k].0 += self.ext[i].0;
                        ext_sum[k].1 += self.ext[i].1;
                    }
                }
                for (i, p) in self.buf_pos.iter_mut().enumerate() {
                    let k = self.nuc[i] as usize;
                    if k < m && self.nuc_count[k] > 1 {
                        *p -= net[k] / self.nuc_count[k] as f32;
                        if ext_sum[k].1 > 0.0 {
                            *p += ext_sum[k].0 / ext_sum[k].1;
                        }
                    }
                }
            }
            std::mem::swap(&mut self.pos, &mut self.buf_pos);
        }

        // Nuclei were moved by averaged pushes: make sure no nucleon is left
        // inside a wall.
        // The push is not turned into velocity (prev moves along): bouncing
        // is the restitution pass's job, and a wall drawn through a nucleus
        // must shove it aside, not fire its nucleons off.
        if self.cfg.nuclear && !self.nuc_count.is_empty() {
            let (kind, walls, nuc, nuc_count, hits) = (&self.kind, &self.walls, &self.nuc, &self.nuc_count, &self.hits);
            self.pos.par_iter_mut().zip(self.prev_pos.par_iter_mut()).enumerate().for_each(|(i, (p, pv))| {
                if nuc_count.get(nuc[i] as usize).is_some_and(|&c| c > 1) {
                    let ri = t.radius[kind[i] as usize];
                    let mut q = walls.resolve(*p, *pv, ri, hits[i]);
                    if !rods.is_empty() {
                        q = crate::reactor::push_out(rods, q, ri);
                    }
                    q.x = q.x.clamp(ri, (w - ri).max(ri));
                    q.y = q.y.clamp(ri, (hgt - ri).max(ri));
                    *pv += q - *p;
                    *p = q;
                }
            });
        }

        let inv_h = 1.0 / h;
        let vmax = self.cfg.max_speed;
        let (pos, prev, kind) = (&self.pos, &self.prev_pos, &self.kind);
        self.vel.resize(pos.len(), Vec2::ZERO);
        self.vel.par_iter_mut().enumerate().for_each(|(i, v)| {
            if !t.ghost[kind[i] as usize] {
                *v = clamp_speed((pos[i] - prev[i]) * inv_h, vmax);
            }
        });
    }

    /// Restitution: for contacts that were approaching fast before the solve,
    /// set the separating normal speed to e * approach speed. Slow (resting)
    /// contacts are left inelastic so piles settle.
    /// Nuclei bounce as rigid bodies (their velocity and total mass), so
    /// nucleus-nucleus and neutron-nucleus collisions are properly elastic.
    fn solve_velocities(&mut self, t: &Table, h: f32) {
        let e = self.cfg.restitution;
        if e <= 0.0 {
            return;
        }
        let collisions = self.cfg.collisions;
        let (w, hgt) = (self.width, self.height);
        let _ = h;
        // Slower contacts are left inelastic so piles settle under gravity.
        // A nuclear world has no gravity and its cold heavy gases move at
        // a few px/s: their collisions must stay elastic or heat leaks away.
        let rest_speed = if self.cfg.nuclear { 0.2 } else { 2.0 };

        let bodies = self.cfg.nuclear && self.n_nuclei > 0 && self.nuc_count.len() == self.n_nuclei;
        let m = if bodies { self.n_nuclei } else { 0 };
        // Body (nucleus) velocities before the substep and now.
        self.body_pre.clear();
        self.body_pre.resize(m, Vec2::ZERO);
        self.body_now.clear();
        self.body_now.resize(m, Vec2::ZERO);
        self.body_pos.clear();
        self.body_pos.resize(m, Vec2::ZERO);
        for i in 0..self.pos.len() {
            let k = self.nuc[i] as usize;
            if k < m {
                self.body_pre[k] += self.vel_pre[i];
                self.body_now[k] += self.vel[i];
                self.body_pos[k] += self.pos[i];
            }
        }
        for k in 0..m {
            let c = self.nuc_count[k].max(1) as f32;
            self.body_pre[k] /= c;
            self.body_now[k] /= c;
            self.body_pos[k] /= c;
        }

        let (pos, pred, vel, vpre, kind, grid, walls) =
            (&self.pos, &self.pred_pos, &self.vel, &self.vel_pre, &self.kind, &self.grid, &self.walls);
        let (nuc, nuc_count, nuc_mass, body_pre, body_now, body_pos) =
            (&self.nuc, &self.nuc_count, &self.nuc_mass, &self.body_pre, &self.body_now, &self.body_pos);
        let multi = |i: usize| bodies && nuc_count.get(nuc[i] as usize).is_some_and(|&c| c > 1);
        let zn = &self.zn;
        let inv_body = |i: usize, ti: usize| {
            if multi(i) {
                1.0 / nuc_mass[nuc[i] as usize].max(1e-3)
            } else if zn[i] != 0 {
                let (z, n) = unpack_zn(zn[i]);
                1.0 / (z + n) as f32
            } else {
                t.inv_mass[ti]
            }
        };
        let hits = &self.hits;
        let v_pre = |i: usize| if multi(i) { body_pre[nuc[i] as usize] } else { vpre[i] };
        let v_now = |i: usize| if multi(i) { body_now[nuc[i] as usize] } else { vel[i] };
        // Nuclei are round: they bounce along the line between centres.
        let center = |i: usize| if multi(i) { body_pos[nuc[i] as usize] } else { pos[i] };

        self.ext.resize(pos.len(), (Vec2::ZERO, 0.0));
        self.buf_vel.resize(pos.len(), Vec2::ZERO);
        self.buf_vel.par_iter_mut().zip(self.ext.par_iter_mut()).enumerate().for_each(|(i, (nv, ext))| {
            let p = pos[i];
            let mut v = vel[i];
            *ext = (Vec2::ZERO, 0.0);
            let ti = kind[i] as usize;
            if t.ghost[ti] {
                *nv = v;
                return;
            }
            let ri = t.radius[ti];
            let i_multi = multi(i);
            let (vp_i, vn_i) = (v_pre(i), v_now(i));
            let wb_i = inv_body(i, ti);
            let mut dv = Vec2::ZERO;
            let mut count = 0u32;
            let mut groups = Groups::new();

            if collisions && t.collides[ti] {
                grid.for_each_near(p, 1, |j| {
                    if j == i {
                        return;
                    }
                    let tj = kind[j] as usize;
                    if !t.collides[tj] || (bodies && nuc[i] != u32::MAX && nuc[j] == nuc[i]) {
                        return;
                    }
                    // A contact = a pair that overlapped before the position solve.
                    let rs = ri + t.radius[tj];
                    let dpred = pred[i] - pred[j];
                    if dpred.length_squared() >= rs * rs {
                        return;
                    }
                    let d = center(i) - center(j);
                    let d2 = d.length_squared();
                    if d2 < 1e-12 {
                        return;
                    }
                    let nrm = d / d2.sqrt();
                    let v_before = (vp_i - v_pre(j)).dot(nrm);
                    if v_before >= -rest_speed {
                        return;
                    }
                    let v_rel = (vn_i - v_now(j)).dot(nrm);
                    let target = -e * v_before;
                    let wb_j = inv_body(j, tj);
                    let d = nrm * ((target - v_rel) * wb_i / (wb_i + wb_j));
                    dv += d;
                    count += 1;
                    groups.add(if multi(j) { nuc[j] } else { u32::MAX - j as u32 }, d);
                });
            }
            if !i_multi {
                dv = groups.total().0;
            }

            // Drawn walls.
            if let Some(n) = walls.contact(p, ri, hits[i]) {
                let vn = vp_i.dot(n);
                if vn < -rest_speed {
                    let d = n * (-e * vn - vn_i.dot(n));
                    if i_multi {
                        dv += d;
                        count += 1;
                    } else {
                        v += d;
                    }
                }
            }

            // Box walls.
            const EPS: f32 = 1e-3;
            let mut bounce = |axis: Vec2, hit: bool| {
                let vn = vp_i.dot(axis);
                if hit && vn < -rest_speed {
                    let d = axis * (-e * vn - vn_i.dot(axis));
                    if i_multi {
                        dv += d;
                        count += 1;
                    } else {
                        // Single particles: set the normal speed directly.
                        v += axis * (-e * vn - v.dot(axis));
                    }
                }
            };
            bounce(Vec2::X, p.x <= ri + EPS);
            bounce(-Vec2::X, p.x >= w - ri - EPS);
            bounce(Vec2::Y, p.y <= ri + EPS);
            bounce(-Vec2::Y, p.y >= hgt - ri - EPS);

            if i_multi {
                *ext = (dv, count as f32);
            } else if count > 0 {
                v += dv;
            }
            *nv = v;
        });

        if bodies {
            // A nucleus bounces as a whole: average its members' impulses.
            let mut sum = vec![(Vec2::ZERO, 0.0f32); m];
            for i in 0..self.ext.len() {
                let k = self.nuc[i] as usize;
                if k < m {
                    sum[k].0 += self.ext[i].0;
                    sum[k].1 += self.ext[i].1;
                }
            }
            for (i, v) in self.buf_vel.iter_mut().enumerate() {
                let k = self.nuc[i] as usize;
                if k < m && self.nuc_count[k] > 1 && sum[k].1 > 0.0 {
                    *v += sum[k].0 / sum[k].1;
                }
            }
        }

        std::mem::swap(&mut self.vel, &mut self.buf_vel);
    }
}


/// Contacts of one particle grouped by the body they touch, so touching
/// several nucleons of the same nucleus counts as one contact with it.
struct Groups {
    key: [u32; 8],
    sum: [Vec2; 8],
    cnt: [u32; 8],
    len: usize,
}

impl Groups {
    fn new() -> Self {
        Self { key: [0; 8], sum: [Vec2::ZERO; 8], cnt: [0; 8], len: 0 }
    }
    #[inline]
    fn add(&mut self, key: u32, v: Vec2) {
        for k in 0..self.len {
            if self.key[k] == key {
                self.sum[k] += v;
                self.cnt[k] += 1;
                return;
            }
        }
        let k = if self.len < 8 { self.len += 1; self.len - 1 } else { 7 };
        self.key[k] = key;
        self.sum[k] += v;
        self.cnt[k] += 1;
    }
    /// (sum over bodies of the per-body average, number of bodies)
    fn total(&self) -> (Vec2, u32) {
        let mut t = Vec2::ZERO;
        for k in 0..self.len {
            t += self.sum[k] / self.cnt[k] as f32;
        }
        (t, self.len as u32)
    }
}

#[inline(always)]
fn clamp_speed(v: Vec2, vmax: f32) -> Vec2 {
    let s2 = v.length_squared();
    if s2 > vmax * vmax { v * (vmax / s2.sqrt()) } else { v }
}

fn permute<T: Copy + Default + Send + Sync>(src: &[T], dst: &mut Vec<T>, order: &[u32]) {
    dst.resize(src.len(), T::default());
    dst.par_iter_mut()
        .zip(order.par_iter())
        .for_each(|(d, &o)| *d = src[o as usize]);
}

/// Pack/unpack a point nucleus' (Z, N).
#[inline(always)]
pub fn pack_zn(z: u32, n: u32) -> u32 {
    (z << 16) | (n & 0xffff)
}

#[inline(always)]
pub fn unpack_zn(v: u32) -> (u32, u32) {
    (v >> 16, v & 0xffff)
}

/// One nucleus as seen from outside (`Sim::bodies`).
#[derive(Clone, Copy, Default, Debug)]
pub struct BodyInfo {
    pub pos: Vec2,
    pub vel: Vec2,
    pub z: u32,
    pub n: u32,
    /// Particles it is made of (1 for a point nucleus).
    pub parts: u32,
}

impl BodyInfo {
    pub fn a(&self) -> u32 {
        self.z + self.n
    }
}

/// What a wall sees hitting it (for filters and neutron-transparent walls).
/// Whole nuclei are judged as one body, so a filter never splits one.
impl Sim {
    fn compute_hits(&mut self) {
        use crate::walls::{Hit, NOT_NUCLEAR};
        let n = self.pos.len();
        self.hits.resize(n, Hit::OTHER);
        if self.walls.is_empty() {
            return;
        }
        let energy = self.walls.needs_energy;
        let cfg = &self.cfg;
        let mev = |v: Vec2, m: f32| if energy { crate::nuclear::energy_of(cfg, v.length(), m) as f32 } else { 0.0 };
        let (kind, nuc, zn, vel) = (&self.kind, &self.nuc, &self.zn, &self.vel);
        let (nuc_z, nuc_count, nuc_vel, nuc_mass) = (&self.nuc_z, &self.nuc_count, &self.nuc_vel, &self.nuc_mass);
        let types = &self.types;
        let nuclear = self.cfg.nuclear;
        self.hits.par_iter_mut().enumerate().with_min_len(256).for_each(|(i, h)| {
            let k = nuc[i] as usize;
            let kd = kind[i];
            *h = if nuclear && nuc_count.get(k).is_some_and(|&c| c > 1) {
                let z = nuc_z[k];
                Hit { z, a: nuc_count[k], charged: z > 0, neutron: false, mev: mev(nuc_vel[k], nuc_mass[k]) }
            } else if zn[i] != 0 {
                let (z, nn) = unpack_zn(zn[i]);
                Hit { z, a: z + nn, charged: z > 0, neutron: false, mev: mev(vel[i], (z + nn) as f32) }
            } else if kd == ids::NEUTRON {
                Hit { mev: mev(vel[i], 1.0), ..Hit::NEUTRON }
            } else if kd == ids::PROTON {
                Hit { z: 1, a: 1, charged: true, neutron: false, mev: mev(vel[i], 1.0) }
            } else {
                let t = &types[kd as usize];
                Hit { z: NOT_NUCLEAR, a: 0, charged: t.charge != 0.0, neutron: false, mev: mev(vel[i], t.mass.max(1e-3)) }
            };
        });
    }
}
