//! Nuclear reaction physics, independent of the particle engine.
//!
//! * Charged nuclei fuse by quantum tunnelling through the Coulomb barrier
//!   (WKB penetrability of a point-charge barrier outside a nuclear radius).
//! * Neutrons have no barrier; whether one is absorbed or bounces off is the
//!   ratio of absorption to scattering cross-section (thermal values from
//!   measurements, falling as 1/v with energy).
//! * Once two nuclei merge they form an excited *compound nucleus*, which
//!   then decays through whatever channels energy allows: emitting n, p, d,
//!   t, He-3 or alpha (charged ones must tunnel out), a gamma ray, or
//!   fission. This one mechanism gives D+T -> He-4+n, D+D -> He-3+n / T+p,
//!   Li-6+n -> T+alpha, Li-7+p -> 2 alpha, U-235+n -> fission...

use crate::nucdata::{self, mass_excess, separation};
use crate::rng::gen_range;

/// e^2 / (4 pi eps0) in MeV fm.
const E2: f64 = 1.44;
/// Barrier radius parameter: R = R0 (A1^1/3 + A2^1/3).
const R0: f64 = 1.4;
/// Thermal neutron energy (MeV), 2200 m/s.
pub const E_THERMAL: f64 = 2.53e-8;

/// Height of the Coulomb barrier between two nuclei (MeV).
pub fn coulomb_barrier(z1: u32, a1: u32, z2: u32, a2: u32) -> f64 {
    if z1 * z2 == 0 {
        return 0.0;
    }
    E2 * (z1 * z2) as f64 / (R0 * ((a1 as f64).cbrt() + (a2 as f64).cbrt()))
}

/// Probability of tunnelling through the Coulomb barrier at centre-of-mass
/// energy `e` (MeV). Exact WKB for a Coulomb barrier with a sharp nuclear
/// radius; reduces to the Gamow factor exp(-sqrt(E_G/E)) far below the top.
pub fn penetrability(e: f64, z1: u32, a1: u32, z2: u32, a2: u32) -> f64 {
    if z1 * z2 == 0 {
        return 1.0;
    }
    if e <= 0.0 {
        return 0.0;
    }
    let vc = coulomb_barrier(z1, a1, z2, a2);
    if e >= vc {
        return 1.0;
    }
    let mu = (a1 * a2) as f64 / (a1 + a2) as f64; // amu
    let e_gamow = 0.979_15 * mu * ((z1 * z2) as f64).powi(2);
    let x = e / vc;
    let g = std::f64::consts::FRAC_2_PI * (x.sqrt().acos() - (x * (1.0 - x)).sqrt());
    (-(e_gamow / e).sqrt() * g).exp()
}

/// Measured thermal (2200 m/s) neutron cross-sections in barns:
/// (Z, N, absorption = capture + fission + (n,p) + (n,alpha), elastic scattering).
/// Mughabghab, Atlas of Neutron Resonances (rounded).
const THERMAL_XS: &[(u32, u32, f64, f64)] = &[
    (1, 0, 0.332, 20.5),
    (1, 1, 0.000519, 3.39),
    (1, 2, 6e-6, 1.7),
    (2, 1, 5333.0, 3.1),
    (2, 2, 0.0, 0.76),
    (3, 3, 940.0, 0.75),
    (3, 4, 0.0454, 0.97),
    (4, 5, 0.0076, 6.15),
    (5, 5, 3840.0, 2.1),
    (5, 6, 0.0055, 5.0),
    (6, 6, 0.00353, 4.75),
    (6, 7, 0.00137, 4.2),
    (7, 7, 1.91, 10.0),
    (7, 8, 2.4e-5, 4.6),
    (8, 8, 0.00019, 3.76),
    (8, 9, 0.236, 3.6),
    (9, 10, 0.0096, 3.6),
    (10, 10, 0.037, 2.6),
    (11, 12, 0.53, 3.0),
    (12, 12, 0.053, 3.4),
    (13, 14, 0.231, 1.4),
    (14, 14, 0.177, 2.0),
    (16, 16, 0.53, 1.0),
    (17, 18, 43.6, 21.0),
    (20, 20, 0.41, 3.0),
    (26, 30, 2.59, 12.4),
    (27, 32, 37.2, 6.0),
    (28, 30, 4.6, 26.0),
    (29, 34, 4.5, 5.2),
    (47, 60, 37.6, 7.6),
    (47, 62, 91.0, 2.3),
    (48, 65, 20600.0, 12.0),
    (49, 66, 202.0, 2.5),
    (53, 74, 6.2, 3.6),
    (54, 81, 2.65e6, 5.0),
    (62, 87, 40140.0, 200.0),
    (64, 91, 60900.0, 60.0),
    (64, 93, 254000.0, 1000.0),
    (72, 105, 375.0, 0.0),
    (79, 118, 98.65, 7.8),
    (80, 119, 2150.0, 66.0),
    (82, 126, 0.00048, 11.5),
    (83, 126, 0.034, 9.2),
    (90, 142, 7.37, 13.0),
    (92, 141, 578.0, 12.5),
    (92, 143, 683.0, 15.0),
    (92, 146, 2.68, 9.4),
    (93, 144, 175.9, 14.0),
    (94, 145, 1017.0, 7.9),
    (94, 146, 289.5, 1.6),
    (94, 147, 1375.0, 11.0),
    (95, 146, 587.0, 11.0),
];

/// Absorption resonance integrals (barns) over the epithermal region, for
/// the nuclides where resonances dominate (Mughabghab, rounded):
/// (Z, N, RI). U-238's 275 b is the "resonance escape" problem of reactors.
const RESONANCE_INTEGRALS: &[(u32, u32, f64)] = &[
    (27, 32, 75.0),
    (47, 62, 1470.0),
    (49, 66, 3300.0),
    (79, 118, 1550.0),
    (90, 142, 85.0),
    (92, 141, 900.0),
    (92, 143, 419.0),
    (92, 146, 275.0),
    (94, 145, 480.0),
    (94, 146, 8100.0),
];

/// Thermal (absorption, scattering) cross-sections in barns, measured where
/// known, otherwise rough systematics (magic numbers absorb little).
pub fn thermal_xs(z: u32, n: u32) -> (f64, f64) {
    if let Some(&(_, _, a, s)) = THERMAL_XS.iter().find(|e| e.0 == z && e.1 == n) {
        return (a, s.max(1.0));
    }
    let a = (z + n) as f64;
    let mut abs = if a <= 20.0 { 0.01 } else { 10f64.powf(-1.2 + 0.013 * a) };
    if z % 2 == 0 && n % 2 == 0 {
        abs /= 3.0;
    }
    if matches!(n, 20 | 28 | 50 | 82 | 126) || matches!(z, 20 | 28 | 50 | 82) {
        abs /= 30.0;
    }
    (abs, 3.0 + 0.25 * a.powf(2.0 / 3.0))
}

/// Largest 1/v enhancement over the thermal cross-section (see below).
pub const SUBTHERMAL_CAP: f64 = 3.0;

/// Probability that a neutron touching nucleus (z, n) with centre-of-mass
/// energy `e` (MeV) is absorbed rather than scattered. The low-energy part
/// of the absorption falls as 1/v; above that a fast-neutron compound
/// cross-section (~ nuclear area) remains.
pub fn neutron_absorption(z: u32, n: u32, e: f64) -> f64 {
    let (abs, scat) = neutron_xs(z, n, e);
    if abs + scat <= 0.0 { 0.0 } else { abs / (abs + scat) }
}

/// Neutron cross-sections (absorption incl. compound formation, elastic
/// scattering) in barns on nucleus (z, n) at centre-of-mass energy `e` MeV.
pub fn neutron_xs(z: u32, n: u32, e: f64) -> (f64, f64) {
    let (abs_th, scat) = thermal_xs(z, n);
    if z + n <= 4 && abs_th == 0.0 {
        return (0.0, scat);
    }
    // 1/v law. Below ~E_th/10 it is held level: the sim's speed compression
    // spreads a thermal Maxwellian over orders of magnitude in energy, and
    // real thermal neutrons almost never sit at those sub-meV energies.
    let slow = abs_th * (E_THERMAL / e.max(1e-12)).sqrt().min(SUBTHERMAL_CAP);
    // Resonances: flat from 1 eV to 1 keV (resolved resonances), then an
    // unresolved-resonance tail falling as E^-1/2 for fissile nuclei (whose
    // fission cross-section stays at barns through the keV range) and as
    // 1/E for the rest, gone by 1 MeV. Normalised so that the integral of
    // sigma dE/E is the measured resonance integral.
    let res = match RESONANCE_INTEGRALS.iter().find(|r| r.0 == z && r.1 == n) {
        Some(r) if (1e-6..1.0).contains(&e) => {
            let fis = fissile(z, n);
            let flat = r.2 / ((1e3f64).ln() + if fis { 2.0 } else { 1.0 });
            if e <= 1e-3 {
                flat
            } else if fis {
                flat * (1e-3 / e).sqrt()
            } else {
                flat * 1e-3 / e
            }
        }
        _ => 0.0,
    };
    // Fast neutrons (toward MeV) form compound nuclei on any nucleus
    // (inelastic scattering); negligible at low energy.
    let fast = 0.1 * ((z + n) as f64).powf(2.0 / 3.0) * e / (e + 0.5);
    (slow + res + fast, scat)
}

/// Fission barrier height (MeV). Liquid-drop barrier (Myers–Swiatecki style
/// fissility fit), floored at the ~6 MeV actinide plateau that shell effects
/// produce. Gives the textbook split: thermal neutrons fission U-233/235,
/// Pu-239/241 (compound E* above the barrier) but not U-238, Th-232, Pu-240.
pub fn fission_barrier(z: u32, n: u32) -> f64 {
    let a = (z + n) as f64;
    let zf = z as f64;
    let i = (n as f64 - zf) / a;
    let k = 1.0 - 1.7826 * i * i;
    let x = zf * zf / a / (50.88 * k);
    let es = 17.9439 * k * a.powf(2.0 / 3.0);
    let ld = if x >= 2.0 / 3.0 {
        0.83 * (1.0 - x).max(0.0).powi(3) * es
    } else {
        0.38 * (0.75 - x).max(0.0) * es
    };
    if z >= 88 && z <= 100 {
        ld.max(6.0)
    } else if z > 100 {
        ld.max(6.0 - 0.4 * (z - 100) as f64).max(0.5)
    } else {
        ld
    }
}

/// Probability to go over/through the fission barrier (Hill–Wheeler).
fn hill_wheeler(e_star: f64, barrier: f64) -> f64 {
    1.0 / (1.0 + (std::f64::consts::TAU * (barrier - e_star) / 0.5).exp())
}

/// What a compound nucleus does next.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Emit (z, n); `e` = kinetic energy of the relative motion (MeV).
    Emit { z: u32, n: u32, e: f64 },
    /// De-excite by a gamma ray of energy `e` (MeV).
    Gamma { e: f64 },
    Fission(FissionSplit),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FissionSplit {
    pub light: (u32, u32),
    pub heavy: (u32, u32),
    /// Kinetic energies of the prompt neutrons (MeV).
    pub neutrons: Vec<f64>,
    /// Total kinetic energy of the two fragments (MeV).
    pub tke: f64,
}

/// Light particles a compound nucleus can evaporate: (z, n, relative weight).
/// Composite particles must pre-form, so they're less likely than n/p/alpha.
const EJECTILES: [(u32, u32, f64); 6] = [(0, 1, 1.0), (1, 0, 1.0), (2, 2, 1.0), (1, 1, 0.3), (1, 2, 0.3), (2, 1, 0.3)];

fn gaussian() -> f64 {
    let u1 = gen_range(1e-9f64, 1.0);
    let u2 = gen_range(0.0f64, 1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Decay chain of compound nucleus (z, n) with excitation energy `e_star`
/// (MeV, may be negative for an unbound system). Returns the steps in order;
/// the remaining nucleus after all `Emit`s is the final residue.
pub fn compound_decay(z: u32, n: u32, e_star: f64) -> Vec<Step> {
    let mut steps = Vec::new();
    let (mut z, mut n, mut ex) = (z, n, e_star);
    for _ in 0..8 {
        let a = z + n;
        if a <= 1 {
            break;
        }
        let mut options: Vec<(Step, f64)> = Vec::new();
        for &(zx, nx, pre) in &EJECTILES {
            if zx > z || nx > n || (z - zx) + (n - nx) == 0 {
                continue;
            }
            let avail = ex - separation(z, n, zx, nx);
            if avail <= 0.0 {
                continue;
            }
            let pen = penetrability(avail, zx, zx + nx, z - zx, a - zx - nx);
            let w = 10.0 * pre * avail * pen;
            if w > 0.0 {
                options.push((Step::Emit { z: zx, n: nx, e: avail }, w));
            }
        }
        if ex > 0.0 {
            // Electromagnetic de-excitation is slow next to particle emission.
            options.push((Step::Gamma { e: ex }, 0.02 * (a as f64).powf(2.0 / 3.0)));
        }
        if z >= 80 && ex > 0.0 {
            let w = 5.0 * hill_wheeler(ex, fission_barrier(z, n));
            if w > 1e-12 {
                options.push((Step::Fission(sample_fission(z, n, ex)), w));
            }
        }
        if options.is_empty() {
            break;
        }
        let total: f64 = options.iter().map(|o| o.1).sum();
        let mut r = gen_range(0.0, total);
        let mut pick = options.len() - 1;
        for (k, o) in options.iter().enumerate() {
            if r < o.1 {
                pick = k;
                break;
            }
            r -= o.1;
        }
        let step = options.swap_remove(pick).0;
        match step {
            Step::Emit { z: zx, n: nx, e: avail } => {
                // Light nuclei: two-body breakup to the ground state.
                // Heavy nuclei: evaporation spectrum (temperature from the
                // Fermi-gas level density a = A/8 per MeV); the residue keeps
                // the rest as excitation and may emit again.
                let eps = if a <= 20 {
                    avail
                } else {
                    let t = (avail / (a as f64 / 8.0)).sqrt().max(0.05);
                    let mut eps;
                    loop {
                        eps = -t * (gen_range(1e-9f64, 1.0) * gen_range(1e-9f64, 1.0)).ln();
                        if eps <= avail {
                            break;
                        }
                    }
                    eps
                };
                steps.push(Step::Emit { z: zx, n: nx, e: eps });
                z -= zx;
                n -= nx;
                ex = avail - eps;
                if ex < 0.1 {
                    break;
                }
            }
            Step::Gamma { e } => {
                steps.push(Step::Gamma { e });
                break;
            }
            f @ Step::Fission(_) => {
                steps.push(f);
                break;
            }
        }
    }
    steps
}

/// Sample a fission event of (z, n) with excitation `e_star`:
/// asymmetric mass split (heavy peak at A~139.5), unchanged-charge-density
/// charge split with the usual ~0.5 polarization, Viola's total kinetic
/// energy, and prompt neutrons from a Maxwellian (T = 1.32 MeV).
pub fn sample_fission(z: u32, n: u32, e_star: f64) -> FissionSplit {
    let a = z + n;
    // Prompt neutron multiplicity: 2.43 for U-235 + thermal n, 3.76 for Cf-252 SF.
    let nu_bar = (1.585 + 0.22 * (z as f64 - 92.0) + 0.13 * e_star.max(0.0) + 0.06 * (a as f64 - 236.0)).max(0.5);
    let nu = ((nu_bar + 1.08 * gaussian()).round().max(0.0) as u32).min(n.saturating_sub(4)).min(8);
    let ap = (a - nu) as f64;
    let peak = if ap > 220.0 { 139.5f64.max(ap / 2.0) } else { ap / 2.0 };
    let a_h = ((peak + 5.0 * gaussian()).round()).clamp((ap / 2.0).ceil(), ap - 20.0) as u32;
    let a_l = a - nu - a_h;
    let ucd = z as f64 * a_h as f64 / ap;
    let z_h = ((ucd - 0.5 + 0.56 * gaussian()).round() as u32).clamp(1, z - 1).min(a_h - 1);
    let z_l = z - z_h;
    let (n_h, n_l) = (a_h - z_h, a_l.saturating_sub(z_l));
    let tke = 0.1189 * (z * z) as f64 / (a as f64).cbrt() + 7.3 + 8.0 * gaussian();
    let neutrons = (0..nu)
        .map(|_| {
            let t = 1.32;
            let (x1, x2, x3) = (gen_range(1e-9f64, 1.0), gen_range(1e-9f64, 1.0), gen_range(0.0f64, 1.0));
            -t * (x1.ln() + x2.ln() * (std::f64::consts::FRAC_PI_2 * x3).cos().powi(2))
        })
        .collect();
    FissionSplit { light: (z_l, n_l), heavy: (z_h, n_h), neutrons, tke: tke.max(50.0) }
}

/// Q-value of a + b -> compound (MeV): the compound's excitation at rest.
pub fn capture_q(za: u32, na: u32, zb: u32, nb: u32) -> f64 {
    mass_excess(za, na) + mass_excess(zb, nb) - mass_excess(za + zb, na + nb)
}

/// Is the nucleus fissile by thermal neutrons (compound above barrier)?
pub fn fissile(z: u32, n: u32) -> bool {
    z >= 88 && separation(z, n + 1, 0, 1) > fission_barrier(z, n + 1)
}

pub use nucdata::nuclide_name;

#[cfg(test)]
mod tests {
    use super::*;

    /// Outcome counts of `trials` reactions a + b at energy e.
    fn outcomes(za: u32, na: u32, zb: u32, nb: u32, e: f64, trials: usize) -> crate::rng::HashMap<String, usize> {
        let mut m = crate::rng::HashMap::default();
        for _ in 0..trials {
            let (z, n) = (za + zb, na + nb);
            let steps = compound_decay(z, n, capture_q(za, na, zb, nb) + e);
            let mut parts: Vec<String> = Vec::new();
            let (mut rz, mut rn) = (z, n);
            for s in &steps {
                match s {
                    Step::Emit { z: x, n: y, .. } => {
                        parts.push(nuclide_name(*x, *y));
                        rz -= x;
                        rn -= y;
                    }
                    Step::Gamma { .. } => parts.push("g".into()),
                    Step::Fission(_) => parts.push("fission".into()),
                }
            }
            if !steps.iter().any(|s| matches!(s, Step::Fission(_))) {
                parts.push(nuclide_name(rz, rn));
            }
            parts.sort();
            let key = parts.join("+");
            *m.entry(key).or_default() += 1;
        }
        m
    }

    #[test]
    fn barrier_and_tunnelling() {
        assert!((coulomb_barrier(1, 1, 1, 1) - 0.514).abs() < 0.01);
        let p10 = penetrability(0.010, 1, 2, 1, 3);
        let p50 = penetrability(0.050, 1, 2, 1, 3);
        assert!(p10 < p50 && p50 < 1.0 && p10 > 1e-6, "{p10} {p50}");
        assert_eq!(penetrability(1.0, 0, 1, 92, 235), 1.0);
    }

    #[test]
    fn classic_reactions() {
        let dt = outcomes(1, 1, 1, 2, 0.05, 200);
        assert!(dt.get("He-4+n").copied().unwrap_or(0) > 190, "D+T {dt:?}");
        let dd = outcomes(1, 1, 1, 1, 0.05, 400);
        let (n, p) = (dd.get("He-3+n").copied().unwrap_or(0), dd.get("H-3+p").copied().unwrap_or(0));
        assert!(n > 100 && p > 100, "D+D {dd:?}");
        let li6 = outcomes(0, 1, 3, 3, E_THERMAL, 200);
        let ta = li6.get("H-3+He-4").copied().unwrap_or(0);
        assert!(ta > 190, "Li-6+n {li6:?}");
        let li7p = outcomes(1, 0, 3, 4, 0.3, 200);
        assert!(li7p.get("He-4+He-4").copied().unwrap_or(0) > 190, "Li-7+p {li7p:?}");
        let np = outcomes(0, 1, 1, 0, E_THERMAL, 100);
        assert_eq!(np.get("H-2+g").copied().unwrap_or(0), 100, "n+p {np:?}");
        let he4n = outcomes(0, 1, 2, 2, 0.01, 50);
        assert_eq!(he4n.get("He-4+n").copied().unwrap_or(0), 50, "He-5 is unbound {he4n:?}");
    }

    #[test]
    fn thermal_fission() {
        assert!(fissile(92, 143), "U-235");
        assert!(fissile(94, 145), "Pu-239");
        assert!(fissile(92, 141), "U-233");
        assert!(!fissile(92, 146), "U-238");
        assert!(!fissile(90, 142), "Th-232");
        assert!(!fissile(94, 146), "Pu-240");
        let u = outcomes(0, 1, 92, 143, E_THERMAL, 400);
        let f = u.get("fission").copied().unwrap_or(0) as f64 / 400.0;
        assert!((0.75..0.95).contains(&f), "U-235 fission fraction {f} {u:?}");
        let u8 = outcomes(0, 1, 92, 146, E_THERMAL, 100);
        assert_eq!(u8.get("U-239+g").copied().unwrap_or(0), 100, "{u8:?}");
        // Mean prompt neutrons for U-235 + thermal n ~ 2.43.
        let mut sum = 0;
        let trials = 2000;
        for _ in 0..trials {
            sum += sample_fission(92, 144, 6.545).neutrons.len();
        }
        let nu = sum as f64 / trials as f64;
        assert!((nu - 2.43).abs() < 0.15, "nu = {nu}");
    }

    #[test]
    fn absorption() {
        assert!(neutron_absorption(92, 143, E_THERMAL) > 0.95);
        assert!(neutron_absorption(1, 0, E_THERMAL) < 0.03);
        assert!(neutron_absorption(5, 5, E_THERMAL) > 0.99);
        assert!(neutron_absorption(5, 5, 1.0) < 0.5);
        // Deuterium barely absorbs at any energy (heavy water's virtue).
        assert!(neutron_absorption(1, 1, 1e-3) < 1e-3);
        // U-238 resonance capture in the epithermal range.
        assert!(neutron_absorption(92, 146, 1e-4) > 0.6);
        assert_eq!(neutron_absorption(2, 2, E_THERMAL), 0.0);
    }
}
