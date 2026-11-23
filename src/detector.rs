//! Detectors: a box or a line that counts what passes through it, sorted by
//! species, with an energy spectrum. Each can be set to count only neutral
//! or charged particles, only positive or negative ones, one element or
//! one isotope.

use crate::nucdata;
use crate::nuclear::{energy_of, format_energy};
use crate::rng::HashMap;
use crate::sim::Sim;
use crate::types::ids;
use macroquad::math::{Rect, Vec2};

/// What a detector looks at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DetShape {
    /// Counts everything that enters the box.
    Box(Rect),
    /// Counts everything that crosses the line (either way).
    Line(Vec2, Vec2),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectFilter {
    All,
    Neutral,
    Charged,
    Positive,
    Negative,
    /// One element, every isotope (Z; 0 = free neutrons).
    Element(u32),
    /// One isotope (Z, N).
    Isotope(u32, u32),
}

impl DetectFilter {
    /// The kinds offered in the panel (the element and isotope keep their
    /// own Z, N).
    pub const NAMES: [&str; 7] = ["Everything", "Neutral", "Charged", "Positive", "Negative", "One element", "One isotope"];

    pub fn index(self) -> usize {
        match self {
            DetectFilter::All => 0,
            DetectFilter::Neutral => 1,
            DetectFilter::Charged => 2,
            DetectFilter::Positive => 3,
            DetectFilter::Negative => 4,
            DetectFilter::Element(_) => 5,
            DetectFilter::Isotope(..) => 6,
        }
    }

    pub fn from_index(k: usize, z: u32, n: u32) -> Self {
        match k {
            1 => DetectFilter::Neutral,
            2 => DetectFilter::Charged,
            3 => DetectFilter::Positive,
            4 => DetectFilter::Negative,
            5 => DetectFilter::Element(z),
            6 => DetectFilter::Isotope(z, n),
            _ => DetectFilter::All,
        }
    }

    pub fn describe(self) -> String {
        match self {
            DetectFilter::All => "everything".into(),
            DetectFilter::Neutral => "neutral particles only".into(),
            DetectFilter::Charged => "charged particles only".into(),
            DetectFilter::Positive => "positive particles only".into(),
            DetectFilter::Negative => "negative particles only".into(),
            DetectFilter::Element(0) => "free neutrons only".into(),
            DetectFilter::Element(z) => format!("{} (any isotope) only", nucdata::SYMBOLS.get(z as usize).copied().unwrap_or("?")),
            DetectFilter::Isotope(z, n) => format!("{} only", nucdata::nuclide_name(z, n)),
        }
    }

    pub fn takes(self, s: Species) -> bool {
        let q = s.charge();
        match self {
            DetectFilter::All => true,
            DetectFilter::Neutral => q == 0,
            DetectFilter::Charged => q != 0,
            DetectFilter::Positive => q > 0,
            DetectFilter::Negative => q < 0,
            DetectFilter::Element(z) => matches!(s, Species::Nuclide(sz, _) if sz == z),
            DetectFilter::Isotope(z, n) => s == Species::Nuclide(z, n),
        }
    }
}

/// Something a detector can count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Species {
    Nuclide(u32, u32),
    Electron,
    Positron,
    Photon,
    Neutrino,
}

impl Species {
    pub fn charge(self) -> i32 {
        match self {
            Species::Nuclide(z, _) => z as i32,
            Species::Electron => -1,
            Species::Positron => 1,
            Species::Photon | Species::Neutrino => 0,
        }
    }

    pub fn name(self) -> String {
        match self {
            Species::Nuclide(z, n) => nucdata::nuclide_name(z, n),
            Species::Electron => "e-".into(),
            Species::Positron => "e+".into(),
            Species::Photon => "γ".into(),
            Species::Neutrino => "ν".into(),
        }
    }

    /// Photons and neutrinos (counted only when asked for).
    pub fn light(self) -> bool {
        matches!(self, Species::Photon | Species::Neutrino)
    }
}

/// Energy spectrum: half-decade bins from 1 meV (1e-9 MeV) to 10 TeV.
pub const BIN_LO: f64 = -9.0;
pub const BINS_PER_DECADE: f64 = 2.0;
pub const BINS: usize = 34;

pub fn bin_of(mev: f64) -> usize {
    let x = (mev.max(1e-30).log10() - BIN_LO) * BINS_PER_DECADE;
    (x.floor().max(0.0) as usize).min(BINS - 1)
}

/// Lower edge of bin k (MeV).
pub fn bin_edge(k: usize) -> f64 {
    10f64.powf(BIN_LO + k as f64 / BINS_PER_DECADE)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detector {
    pub shape: DetShape,
    pub filter: DetectFilter,
    /// Also count photons and neutrinos.
    pub light: bool,
    /// Everything counted since the last reset.
    pub count: u64,
    pub species: HashMap<Species, u64>,
    /// Energy spectrum (kinetic energy; photons and neutrinos have none here).
    pub hist: [u64; BINS],
    pub energy_sum: f64,
    pub with_energy: u64,
    /// Smoothed count rate (per sim second).
    pub rate: f32,
    /// Sim time of the last reset.
    pub since: f64,
    acc: (u64, f32),
    /// Box: nuclei (by tag) inside it last frame.
    inside: crate::rng::HashSet<u32>,
}

impl Detector {
    pub fn new(shape: DetShape, filter: DetectFilter, light: bool) -> Self {
        Self { shape, filter, light, count: 0, species: HashMap::default(), hist: [0; BINS], energy_sum: 0.0, with_energy: 0, rate: 0.0, since: 0.0, acc: (0, 0.0), inside: Default::default() }
    }

    pub fn reset(&mut self, now: f64) {
        let inside = std::mem::take(&mut self.inside);
        *self = Self { since: now, inside, ..Self::new(self.shape, self.filter, self.light) };
    }

    pub fn mean_energy(&self) -> Option<f64> {
        (self.with_energy > 0).then(|| self.energy_sum / self.with_energy as f64)
    }

    /// Species counted, most first.
    pub fn top_species(&self) -> Vec<(Species, u64)> {
        let mut v: Vec<(Species, u64)> = self.species.iter().map(|(s, c)| (*s, *c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    pub fn center(&self) -> Vec2 {
        match self.shape {
            DetShape::Box(r) => r.center(),
            DetShape::Line(a, b) => (a + b) * 0.5,
        }
    }

    /// Does the path p0 -> p1 count? (Box: nuclei count when first seen
    /// inside - entering, or made there; leptons when they enter.)
    fn crossed(&self, tag: Option<u32>, p0: Vec2, p1: Vec2) -> bool {
        match self.shape {
            DetShape::Box(r) => match tag {
                Some(t) => r.contains(p1) && !self.inside.contains(&t),
                None => r.contains(p1) && !r.contains(p0),
            },
            DetShape::Line(a, b) => segments_cross(p0, p1, a, b),
        }
    }

    /// Hover text.
    pub fn summary(&self, now: f64) -> Vec<String> {
        let kind = match self.shape {
            DetShape::Box(_) => "Detector (box: counts what enters it or is made in it)",
            DetShape::Line(..) => "Detector (line: counts what crosses)",
        };
        let mut l = vec![kind.to_string(), format!("counts {}{}", self.filter.describe(), if self.light { ", photons and neutrinos too" } else { "" })];
        l.push(format!("{} in {:.0} s   {:.1}/s now", self.count, (now - self.since).max(0.0), self.rate));
        if let Some(e) = self.mean_energy() {
            l.push(format!("mean energy {}", format_energy(e)));
        }
        let top: Vec<String> = self.top_species().iter().take(5).map(|(s, c)| format!("{} {c}", s.name())).collect();
        if !top.is_empty() {
            l.push(top.join("   "));
        }
        l
    }
}

fn segments_cross(p0: Vec2, p1: Vec2, a: Vec2, b: Vec2) -> bool {
    let d = p1 - p0;
    let e = b - a;
    let den = d.perp_dot(e);
    if den.abs() < 1e-9 {
        return false;
    }
    let t = (a - p0).perp_dot(e) / den;
    let u = (a - p0).perp_dot(d) / den;
    // Half-open along the path, so landing exactly on the line counts once.
    t > 0.0 && t <= 1.0 && (0.0..=1.0).contains(&u)
}

/// Kinetic energy (MeV) of an electron or positron moving at `speed`: they
/// are relativistic here, with c = light_speed.
fn lepton_energy(sim: &Sim, speed: f32) -> f64 {
    let beta = (speed / sim.cfg.light_speed).clamp(0.0, 0.999) as f64;
    (1.0 / (1.0 - beta * beta).sqrt() - 1.0) * nucdata::M_E
}

/// Count this frame's crossings. Call once per frame, after the nuclear
/// processing (positions and velocities are this frame's).
pub fn update(sim: &mut Sim, dt: f32) {
    if sim.reactor.detectors.is_empty() || dt <= 0.0 {
        return;
    }
    let mut dets = std::mem::take(&mut sim.reactor.detectors);
    // (species, tag, now, before, energy)
    let mut items: Vec<(Species, Option<u32>, Vec2, Vec2, Option<f64>)> = Vec::new();
    for (&tag, b) in sim.bodies().iter() {
        let s = Species::Nuclide(b.z, b.n);
        let e = energy_of(&sim.cfg, b.vel.length(), b.a() as f32);
        items.push((s, Some(tag), b.pos, b.pos - b.vel * dt, Some(e)));
    }
    for i in 0..sim.len() {
        if sim.is_nuclear(i) {
            continue;
        }
        let (s, e) = match sim.kind[i] {
            ids::ELECTRON => (Species::Electron, Some(lepton_energy(sim, sim.vel[i].length()))),
            ids::POSITRON => (Species::Positron, Some(lepton_energy(sim, sim.vel[i].length()))),
            ids::PHOTON => (Species::Photon, None),
            ids::NEUTRINO => (Species::Neutrino, None),
            _ => continue,
        };
        items.push((s, None, sim.pos[i], sim.pos[i] - sim.vel[i] * dt, e));
    }
    for d in dets.iter_mut() {
        for &(s, tag, p1, p0, e) in &items {
            if (s.light() && !d.light) || !d.filter.takes(s) || !d.crossed(tag, p0, p1) {
                continue;
            }
            d.count += 1;
            *d.species.entry(s).or_default() += 1;
            d.acc.0 += 1;
            if let Some(e) = e {
                d.hist[bin_of(e)] += 1;
                d.energy_sum += e;
                d.with_energy += 1;
            }
        }
        if let DetShape::Box(r) = d.shape {
            d.inside = items.iter().filter_map(|&(_, tag, p1, _, _)| tag.filter(|_| r.contains(p1))).collect();
        }
        d.acc.1 += dt;
        if d.acc.1 >= 0.5 {
            let r = d.acc.0 as f32 / d.acc.1;
            d.rate = if d.rate == 0.0 { r } else { d.rate * 0.6 + r * 0.4 };
            d.acc = (0, 0.0);
        }
    }
    sim.reactor.detectors = dets;
}
