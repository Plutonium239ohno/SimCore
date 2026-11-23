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

/// Type-vs-type interaction strengths ("particle life" rules).
/// `get(a, b)` = how strongly type `b` pulls on type `a` (negative = push).
/// Deliberately allowed to be asymmetric: A can chase B while B flees A.
#[derive(Clone, Debug)]
pub struct RuleMatrix {
    pub n: usize,
    pub strength: Vec<f32>,
}

#[allow(dead_code)]
impl RuleMatrix {
    pub fn zero(n: usize) -> Self {
        Self { n, strength: vec![0.0; n * n] }
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

    /// Grow/shrink to `n` types, keeping existing entries.
    pub fn resize(&mut self, n: usize) {
        let mut s = vec![0.0; n * n];
        let m = self.n.min(n);
        for a in 0..m {
            for b in 0..m {
                s[a * n + b] = self.get(a, b);
            }
        }
        self.n = n;
        self.strength = s;
    }

    /// Randomize only the entries where both types satisfy `pick`.
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
