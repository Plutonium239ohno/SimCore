use macroquad::color::Color;
use crate::rng::gen_range;

/// Fixed indices of the built-in particles. Custom species start at FIRST_CUSTOM.
pub mod ids {
    pub const NEUTRON: u8 = 0;
    pub const PROTON: u8 = 1;
    pub const ELECTRON: u8 = 2;
    pub const POSITRON: u8 = 3;
    pub const PHOTON: u8 = 4;
    pub const NEUTRINO: u8 = 5;
    pub const FIRST_CUSTOM: u8 = 6;
    /// Point nucleus (Nuclear Game only, where there are no custom species):
    /// one particle carrying a whole nucleus, its Z and N in `Sim::zn`.
    pub const NUCLEUS: u8 = 6;
}

pub const MAX_TYPES: usize = 16;

/// A particle "species". Particles only store a type id; all properties live here.
#[derive(Clone, Debug)]
pub struct ParticleType {
    pub name: String,
    pub color: Color,
    pub radius: f32,
    pub mass: f32,
    pub charge: f32,
    /// Massless, chargeless, no collisions or forces; flies straight and is
    /// removed when it leaves the box (photons, neutrinos).
    pub ghost: bool,
    /// Takes part in contact collisions. Electrons/positrons don't: they are
    /// point-like, so they pass through other particles (only forces act).
    pub collides: bool,
    /// Editable in the species panel.
    pub custom: bool,
    /// Included by "Spawn random".
    pub in_random_spawn: bool,
    /// Hue, only meaningful for custom species (drives `color`).
    pub hue: f32,
}

impl ParticleType {
    fn builtin(name: &str, color: Color, radius: f32, mass: f32, charge: f32) -> Self {
        Self {
            name: name.to_string(),
            color,
            radius,
            mass,
            charge,
            ghost: false,
            collides: true,
            custom: false,
            in_random_spawn: true,
            hue: 0.0,
        }
    }

    pub fn custom(name: &str, hue: f32, radius: f32, mass: f32, charge: f32) -> Self {
        Self {
            name: name.to_string(),
            color: hsv(hue, 0.65, 0.95),
            radius,
            mass,
            charge,
            ghost: false,
            collides: true,
            custom: true,
            in_random_spawn: true,
            hue,
        }
    }

    pub fn is_nucleon(kind: u8) -> bool {
        kind == ids::NEUTRON || kind == ids::PROTON
    }
}

/// Real electron/proton mass ratio is 1/1836; that is too stiff for a real-time
/// integrator, so electrons are "only" 20x lighter than nucleons here.
pub const ELECTRON_MASS: f32 = 0.05;

pub fn default_types() -> Vec<ParticleType> {
    let mut positron = ParticleType::builtin("Positron e+", Color::from_rgba(255, 150, 60, 255), 1.6, ELECTRON_MASS, 1.0);
    positron.in_random_spawn = false;
    positron.collides = false;
    let mut photon = ParticleType::builtin("Photon", Color::from_rgba(255, 250, 200, 255), 1.6, 0.0, 0.0);
    photon.ghost = true;
    photon.collides = false;
    photon.in_random_spawn = false;
    let mut neutrino = ParticleType::builtin("Neutrino", Color::from_rgba(120, 255, 230, 140), 1.2, 0.0, 0.0);
    neutrino.ghost = true;
    neutrino.collides = false;
    neutrino.in_random_spawn = false;

    let mut electron = ParticleType::builtin("Electron e-", Color::from_rgba(80, 160, 255, 255), 1.6, ELECTRON_MASS, -1.0);
    electron.collides = false;

    vec![
        ParticleType::builtin("Neutron n", Color::from_rgba(175, 178, 190, 255), 4.0, 1.0, 0.0),
        ParticleType::builtin("Proton p+", Color::from_rgba(240, 75, 75, 255), 4.0, 1.0, 1.0),
        electron,
        positron,
        photon,
        neutrino,
        ParticleType::custom("Species A", 130.0, 4.0, 1.0, 0.0),
        ParticleType::custom("Species B", 280.0, 4.0, 1.0, 0.0),
        ParticleType::custom("Species C", 185.0, 3.0, 0.5, 0.0),
        ParticleType::custom("Species D", 40.0, 6.0, 3.0, 0.0),
    ]
}

pub fn hsv(h: f32, s: f32, v: f32) -> Color {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    Color::new(r + m, g + m, b + m, 1.0)
}

/// Type-vs-type interaction rules ("particle life").
/// `get(a, b)` = how strongly type `b` pulls on type `a` (negative = push).
/// Deliberately allowed to be asymmetric: A can chase B while B flees A.
/// Each cell also has its own range (a multiple of the global rule range)
/// and an inner repulsion zone (`core`, a fraction of that range where the
/// pair pushes apart whatever the strength: short-range repel, longer-range
/// attract).
#[derive(Clone, Debug)]
pub struct RuleMatrix {
    pub n: usize,
    pub strength: Vec<f32>,
    pub range: Vec<f32>,
    pub core: Vec<f32>,
}

/// Limits of the per-cell settings.
pub const RANGE_MIN: f32 = 0.25;
pub const RANGE_MAX: f32 = 3.0;
pub const CORE_MAX: f32 = 0.6;

#[allow(dead_code)]
impl RuleMatrix {
    pub fn zero(n: usize) -> Self {
        Self { n, strength: vec![0.0; n * n], range: vec![1.0; n * n], core: vec![0.0; n * n] }
    }

    #[inline]
    pub fn get(&self, a: usize, b: usize) -> f32 {
        self.strength[a * self.n + b]
    }

    pub fn set(&mut self, a: usize, b: usize, v: f32) {
        self.strength[a * self.n + b] = v;
    }

    pub fn get_mut(&mut self, a: usize, b: usize) -> &mut f32 {
        &mut self.strength[a * self.n + b]
    }

    #[inline]
    pub fn range_of(&self, a: usize, b: usize) -> f32 {
        self.range[a * self.n + b]
    }

    pub fn set_range(&mut self, a: usize, b: usize, v: f32) {
        self.range[a * self.n + b] = v.clamp(RANGE_MIN, RANGE_MAX);
    }

    #[inline]
    pub fn core_of(&self, a: usize, b: usize) -> f32 {
        self.core[a * self.n + b]
    }

    pub fn set_core(&mut self, a: usize, b: usize, v: f32) {
        self.core[a * self.n + b] = v.clamp(0.0, CORE_MAX);
    }

    /// Copy every setting of cell (a, b) from cell (c, d) of `src`.
    fn copy_cell(&mut self, a: usize, b: usize, src: &RuleMatrix, c: usize, d: usize) {
        let (i, j) = (a * self.n + b, c * src.n + d);
        self.strength[i] = src.strength[j];
        self.range[i] = src.range[j];
        self.core[i] = src.core[j];
    }

    /// Grow/shrink to `n` types, keeping existing entries.
    pub fn resize(&mut self, n: usize) {
        let old = self.clone();
        *self = Self::zero(n);
        let m = old.n.min(n);
        for a in 0..m {
            for b in 0..m {
                self.copy_cell(a, b, &old, a, b);
            }
        }
    }

    /// Remove type `k` (later types move down one).
    pub fn remove(&mut self, k: usize) {
        let old = self.clone();
        *self = Self::zero(old.n - 1);
        for a in 0..self.n {
            for b in 0..self.n {
                self.copy_cell(a, b, &old, a + (a >= k) as usize, b + (b >= k) as usize);
            }
        }
    }

    /// Give type `to` the same rules as type `from`, both ways.
    pub fn copy_type(&mut self, from: usize, to: usize) {
        let old = self.clone();
        for j in 0..self.n {
            let jj = if j == to { from } else { j };
            self.copy_cell(to, j, &old, from, jj);
            self.copy_cell(j, to, &old, jj, from);
        }
    }

    /// Largest range factor in use.
    pub fn max_range(&self) -> f32 {
        self.range.iter().copied().fold(RANGE_MIN, f32::max)
    }

    /// Randomize the strengths where both types satisfy `pick` (ranges and
    /// cores are left alone).
    pub fn randomize_where(&mut self, pick: impl Fn(usize) -> bool) {
        for a in 0..self.n {
            for b in 0..self.n {
                if pick(a) && pick(b) {
                    self.set(a, b, gen_range(-1.0f32, 1.0f32));
                }
            }
        }
    }

    pub fn randomize(&mut self) {
        self.randomize_where(|_| true);
    }
}

