//! What the scene is made of, sorted the way a reactor physicist or a
//! fusion engineer would sort it (fuel, bred fuel, fission products, fusion
//! fuel and ash...), and the decay heat each part gives off: the heat that
//! keeps coming after a fission reactor is shut down, from spent fuel in the
//! pool, from tritium and activated walls in a fusion machine.

use crate::nucdata::{self, Mode, info, mass_excess};
use crate::nuclear::sim_half_life;
use crate::reactor::JOULES_PER_MEV;
use crate::sim::Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    /// U-235, Pu-239, U-233, Pu-241...: fissile with slow neutrons.
    FissileFuel,
    /// U-238, Th-232: make fissile fuel when they catch a neutron.
    Fertile,
    /// Other actinides (Np, Am, Cm, U-236, Pu-240...): the long-lived waste.
    MinorActinide,
    /// Fission products still radioactive.
    FissionProductHot,
    /// Xe-135, Sm-149: neutron poisons.
    Poison,
    /// Fission products that have settled down (stable or very long-lived).
    FissionProductStable,
    /// D, T, He-3, Li-6, Li-7, B-11: what fusion burns or breeds from.
    FusionFuel,
    /// He-4: the ash of fusion (and of alpha decay).
    FusionAsh,
    /// Anything else radioactive (activation of moderator, walls, coolant).
    Activated,
    /// Anything else stable (moderator, coolant, structure).
    Stable,
    /// Free neutrons and protons.
    Nucleons,
}

impl Category {
    pub const ALL: [Category; 11] = [
        Category::FissileFuel,
        Category::Fertile,
        Category::MinorActinide,
        Category::FissionProductHot,
        Category::Poison,
        Category::FissionProductStable,
        Category::FusionFuel,
        Category::FusionAsh,
        Category::Activated,
        Category::Stable,
        Category::Nucleons,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Category::FissileFuel => "Fissile fuel",
            Category::Fertile => "Fertile (U-238, Th-232)",
            Category::MinorActinide => "Other actinides",
            Category::FissionProductHot => "Fission products, radioactive",
            Category::Poison => "Poisons (Xe-135, Sm-149)",
            Category::FissionProductStable => "Fission products, settled",
            Category::FusionFuel => "Fusion fuel (D, T, He-3, Li, B-11)",
            Category::FusionAsh => "Helium-4 (fusion ash, alphas)",
            Category::Activated => "Other radioactive (activation)",
            Category::Stable => "Other stable (moderator, coolant)",
            Category::Nucleons => "Free neutrons and protons",
        }
    }
}

pub fn category(z: u32, n: u32) -> Category {
    let a = z + n;
    if a <= 1 {
        return Category::Nucleons;
    }
    if (z, n) == (2, 2) {
        return Category::FusionAsh;
    }
    if matches!((z, n), (1, 1) | (1, 2) | (2, 1) | (3, 3) | (3, 4) | (5, 6)) {
        return Category::FusionFuel;
    }
    if z >= 90 {
        if crate::reactions::fissile(z, n) {
            return Category::FissileFuel;
        }
        if matches!((z, n), (92, 146) | (90, 142)) {
            return Category::Fertile;
        }
        return Category::MinorActinide;
    }
    if matches!((z, n), (54, 81) | (62, 87)) {
        return Category::Poison;
    }
    let hot = info(z, n).half_life.is_some_and(|t| t > 1e-12);
    if (66..=172).contains(&a) && z >= 28 {
        // Long-lived (over ~30 years) products count as settled.
        return if hot && info(z, n).half_life.is_some_and(|t| t < 1e9) { Category::FissionProductHot } else { Category::FissionProductStable };
    }
    if hot { Category::Activated } else { Category::Stable }
}

/// Heat (MeV) a decay leaves in matter: the neutrino's share of a beta
/// decay (about a third) escapes.
pub fn deposited(mode: Mode, q: f64) -> f64 {
    let q = q.max(0.0);
    match mode {
        Mode::BetaMinus | Mode::BetaPlus | Mode::DoubleBetaMinus => 0.67 * q,
        Mode::ElectronCapture => 0.1 * q,
        _ => q,
    }
}

/// Mean heat (MeV) deposited per decay of (z, n).
pub fn heat_per_decay(z: u32, n: u32) -> f64 {
    let nuc = info(z, n);
    nuc.branches
        .iter()
        .map(|&(m, f)| {
            let q = match m {
                Mode::BetaMinus => nucdata::q_beta_minus(z, n),
                // (A positron's annihilation photons stay in the matter too.)
                Mode::BetaPlus | Mode::ElectronCapture => nucdata::q_ec(z, n),
                Mode::DoubleBetaMinus => mass_excess(z, n) - mass_excess(z + 2, n.saturating_sub(2)),
                Mode::Alpha => nucdata::q_alpha(z, n),
                Mode::Fission => 200.0,
                Mode::Neutron => -nucdata::separation(z, n, 0, 1),
                Mode::TwoNeutron => -nucdata::separation(z, n, 0, 2),
                Mode::Proton => -nucdata::separation(z, n, 1, 0),
                Mode::TwoProton => -nucdata::separation(z, n, 2, 0),
            };
            f * deposited(m, if q.is_finite() { q } else { 0.0 })
        })
        .sum()
}

/// Decay heat (W, per second of sim time) of `count` nuclei of (z, n) on the
/// sim's clock.
pub fn decay_heat_w(cfg: &Config, z: u32, n: u32, count: f64) -> f64 {
    let Some(t) = info(z, n).half_life else { return 0.0 };
    if t < 1e-12 {
        return 0.0;
    }
    let th = sim_half_life(t, cfg);
    if th <= 0.0 {
        return 0.0;
    }
    count * std::f64::consts::LN_2 / th * heat_per_decay(z, n) * JOULES_PER_MEV
}

/// One category's share.
#[derive(Clone, Debug, Default)]
pub struct Row {
    pub count: u32,
    pub nucleons: u64,
    pub heat_w: f64,
}

/// A breakdown of a list of nuclides.
#[derive(Clone, Debug, Default)]
pub struct Inventory {
    pub rows: Vec<(Category, Row)>,
    /// The nuclides giving off the most decay heat: ((z, n), count, W).
    pub hottest: Vec<((u32, u32), u32, f64)>,
    pub heat_w: f64,
    pub count: u32,
}

/// Sort `nuclides` ((z, n), count) into categories with their decay heat.
pub fn take(cfg: &Config, nuclides: impl IntoIterator<Item = ((u32, u32), f64)>) -> Inventory {
    let mut rows: Vec<(Category, Row)> = Category::ALL.iter().map(|&c| (c, Row::default())).collect();
    let mut hottest = Vec::new();
    let mut inv = Inventory::default();
    for ((z, n), c) in nuclides {
        if c <= 0.0 {
            continue;
        }
        let cat = category(z, n);
        let w = decay_heat_w(cfg, z, n, c);
        let row = &mut rows.iter_mut().find(|r| r.0 == cat).unwrap().1;
        row.count += c.round() as u32;
        row.nucleons += (c * (z + n) as f64).round() as u64;
        row.heat_w += w;
        inv.heat_w += w;
        inv.count += c.round() as u32;
        if w > 0.0 {
            hottest.push(((z, n), c.round() as u32, w));
        }
    }
    hottest.sort_by(|a, b| b.2.total_cmp(&a.2));
    hottest.truncate(8);
    rows.retain(|r| r.1.count > 0);
    inv.rows = rows;
    inv.hottest = hottest;
    inv
}

/// Spent fuel taken out by the fuelling machine: what each discharged
/// bundle had turned into, decaying on in the pool (each nuclide decays at
/// its own rate; its daughters are not followed).
#[derive(Clone, Debug, Default)]
pub struct Pool {
    /// (z, n, sim time discharged).
    pub items: Vec<(u32, u32, f64)>,
}

impl Pool {
    /// What is left now, as ((z, n), count).
    pub fn now(&self, cfg: &Config, t: f64) -> Vec<((u32, u32), f64)> {
        let mut m: crate::rng::HashMap<(u32, u32), f64> = crate::rng::HashMap::default();
        for &(z, n, t0) in &self.items {
            let left = match info(z, n).half_life {
                None => 1.0,
                Some(h) if h < 1e-12 => 0.0,
                Some(h) => 0.5f64.powf((t - t0).max(0.0) / sim_half_life(h, cfg).max(1e-9)),
            };
            *m.entry((z, n)).or_default() += left;
        }
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}
