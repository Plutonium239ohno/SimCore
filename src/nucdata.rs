//! Nuclide data: masses, half-lives and decay branches for every known
//! ground state (NUBASE2020, ~3500 nuclides), with a liquid-drop fallback
//! for nuclei nobody has measured.

use crate::rng::HashMap;
use std::sync::OnceLock;

const RAW: &str = include_str!("../data/nuclides.tsv");

/// Mass excess of the neutron and of the hydrogen atom (MeV).
pub const DELTA_N: f64 = 8.071_318;
pub const DELTA_H: f64 = 7.288_971;
/// Electron mass (MeV).
pub const M_E: f64 = 0.510_999;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mode {
    /// n -> p + e- + anti-neutrino
    BetaMinus,
    /// p -> n + e+ + neutrino
    BetaPlus,
    /// p + e- -> n + neutrino (needs an electron at the nucleus)
    ElectronCapture,
    Alpha,
    Neutron,
    TwoNeutron,
    Proton,
    TwoProton,
    /// Spontaneous fission.
    Fission,
    /// Two neutrons -> two protons at once (Ca-48, Te-130, U-238 tiny branch...).
    DoubleBetaMinus,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::BetaMinus => "beta-",
            Mode::BetaPlus => "beta+",
            Mode::ElectronCapture => "electron capture",
            Mode::Alpha => "alpha",
            Mode::Neutron => "neutron emission",
            Mode::TwoNeutron => "2-neutron emission",
            Mode::Proton => "proton emission",
            Mode::TwoProton => "2-proton emission",
            Mode::Fission => "spontaneous fission",
            Mode::DoubleBetaMinus => "double beta-",
        }
    }
    pub fn short(self) -> &'static str {
        match self {
            Mode::BetaMinus => "β-",
            Mode::BetaPlus => "β+",
            Mode::ElectronCapture => "EC",
            Mode::Alpha => "α",
            Mode::Neutron => "n",
            Mode::TwoNeutron => "2n",
            Mode::Proton => "p",
            Mode::TwoProton => "2p",
            Mode::Fission => "SF",
            Mode::DoubleBetaMinus => "2β-",
        }
    }
}

/// Particle emitted right after a beta decay (from an excited daughter state).
#[derive(Clone, Copy, Debug, Default)]
pub struct Delayed {
    /// Probability per beta- decay of emitting 1 / 2 neutrons, an alpha.
    pub n: f64,
    pub n2: f64,
    pub alpha_minus: f64,
    /// Probability per beta+/EC decay of emitting a proton / an alpha.
    pub p: f64,
    pub alpha_plus: f64,
}

#[derive(Clone, Debug)]
pub struct Nuclide {
    pub z: u32,
    pub n: u32,
    /// Atomic mass excess (MeV).
    pub mass_excess: f64,
    /// Half-life in seconds; None = stable; Some(0) = falls apart at once.
    pub half_life: Option<f64>,
    /// Primary decay branches, fractions summing to 1.
    pub branches: Vec<(Mode, f64)>,
    pub delayed: Delayed,
    /// True when this came from measured/evaluated data (not the fallback).
    pub measured: bool,
}

struct Db {
    list: Vec<Nuclide>,
    index: HashMap<(u32, u32), usize>,
}

fn db() -> &'static Db {
    static DB: OnceLock<Db> = OnceLock::new();
    DB.get_or_init(|| {
        let mut list = Vec::new();
        let mut modes = Vec::new();
        for line in RAW.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let (z, n): (u32, u32) = (f[0].parse().unwrap(), f[1].parse().unwrap());
            let mass_excess = f[2].parse::<f64>().unwrap() * 1e-3;
            let half_life = if f[3] == "stable" { None } else { Some(f[3].parse::<f64>().unwrap_or(0.0)) };
            list.push(Nuclide { z, n, mass_excess, half_life, branches: Vec::new(), delayed: Delayed::default(), measured: true });
            modes.push(f.get(4).copied().unwrap_or(""));
        }
        // Second pass: the EC/beta+ split needs daughter masses.
        let masses: HashMap<(u32, u32), f64> = list.iter().map(|x| ((x.z, x.n), x.mass_excess)).collect();
        let mass = |z: u32, n: u32| masses.get(&(z, n)).copied().unwrap_or_else(|| semf_mass_excess(z, n));
        for (nuc, m) in list.iter_mut().zip(modes) {
            let q = if nuc.z > 0 { mass(nuc.z, nuc.n) - mass(nuc.z - 1, nuc.n + 1) } else { 0.0 };
            parse_modes(nuc, m, q);
        }
        let index = list.iter().enumerate().map(|(i, x)| ((x.z, x.n), i)).collect();
        Db { list, index }
    })
}

/// Turn NUBASE's "B-=100;B-n=50.5;A=?" into normalized primary branches plus
/// conditional delayed-particle probabilities.
fn parse_modes(nuc: &mut Nuclide, s: &str, q_ec: f64) {
    let mut raw: Vec<(&str, Option<f64>)> = Vec::new();
    for tok in s.split(';').filter(|t| !t.is_empty()) {
        let (k, v) = tok.split_once('=').unwrap_or((tok, "?"));
        raw.push((k, v.parse::<f64>().ok()));
    }
    let get = |k: &str| raw.iter().find(|(m, _)| *m == k).map(|(_, v)| *v);
    let primary_keys = ["B-", "B+", "EC", "A", "n", "2n", "p", "2p", "SF", "2B-"];
    let known_total: f64 = raw.iter().filter(|(k, _)| primary_keys.contains(k)).filter_map(|(_, v)| *v).sum();
    let n_primary = raw.iter().filter(|(k, _)| primary_keys.contains(k)).count();
    // Unknown intensities ('?'): the whole decay if alone, otherwise a token 1%.
    let val = |v: Option<f64>| v.unwrap_or(if known_total > 0.0 || n_primary > 1 { 1.0 } else { 100.0 });

    let mut br: Vec<(Mode, f64)> = Vec::new();
    for (k, v) in &raw {
        let mode = match *k {
            "B-" => Mode::BetaMinus,
            "A" => Mode::Alpha,
            "n" => Mode::Neutron,
            "2n" => Mode::TwoNeutron,
            "p" => Mode::Proton,
            "2p" => Mode::TwoProton,
            "SF" => Mode::Fission,
            "2B-" => Mode::DoubleBetaMinus,
            "EC" if get("B+").is_none() => Mode::ElectronCapture,
            "B+" => {
                // NUBASE's B+ means beta+ and electron capture together. Use the
                // measured e+/EC split when given, otherwise split it later
                // from the decay energy (`ec_fraction`).
                let total = val(*v);
                match (get("e+").flatten(), get("EC").flatten()) {
                    (Some(ep), Some(ec)) if ep + ec > 0.0 => {
                        br.push((Mode::BetaPlus, total * ep / (ep + ec)));
                        br.push((Mode::ElectronCapture, total * ec / (ep + ec)));
                    }
                    _ => {
                        let f = ec_fraction_q(nuc.z, nuc.n, q_ec);
                        br.push((Mode::BetaPlus, total * (1.0 - f)));
                        br.push((Mode::ElectronCapture, total * f));
                    }
                }
                continue;
            }
            _ => continue,
        };
        br.push((mode, val(*v)));
    }
    br.retain(|(_, f)| *f > 0.0);
    let sum: f64 = br.iter().map(|(_, f)| f).sum();
    for b in br.iter_mut() {
        b.1 /= sum;
    }
    let beta_m = get("B-").flatten().unwrap_or(100.0).max(1e-9);
    let beta_p = get("B+").flatten().or(get("EC").flatten()).unwrap_or(100.0).max(1e-9);
    let cond = |k: &str, of: f64| get(k).flatten().map_or(0.0, |v| (v / of).min(1.0));
    nuc.delayed = Delayed {
        n: cond("B-n", beta_m),
        n2: cond("B-2n", beta_m),
        alpha_minus: cond("B-A", beta_m),
        p: cond("B+p", beta_p),
        alpha_plus: cond("B+A", beta_p),
    };
    nuc.branches = br;
}

/// Fraction of beta+ / EC decays that go by electron capture, from the decay
/// energy and Z (EC wins at low energy and high Z). Fitted to F-18, Cu-64,
/// Ga-68 and Zr-89; used only where NUBASE doesn't give the split.
pub fn ec_fraction(z: u32, n: u32) -> f64 {
    ec_fraction_q(z, n, q_ec(z, n))
}

fn ec_fraction_q(z: u32, n: u32, q: f64) -> f64 {
    if z == 0 {
        return 0.0;
    }
    // K-40: the beta+ branch is third-forbidden (0.001 %).
    if (z, n) == (19, 21) {
        return 0.999_9;
    }
    if q <= 2.0 * M_E {
        return 1.0;
    }
    let r = (-3.44 + 0.218 * (z as f64 - 9.0)).exp() * (0.634 / (q - 2.0 * M_E)).powf(3.22);
    r / (1.0 + r)
}

/// Bethe–Weizsäcker binding energy (MeV), used only off the data table.
pub fn semf_binding(z: i64, n: i64) -> f64 {
    let a = z + n;
    if z < 0 || n < 0 || a <= 1 {
        return 0.0;
    }
    let (af, zf, nf) = (a as f64, z as f64, n as f64);
    let mut b = 15.75 * af - 17.8 * af.powf(2.0 / 3.0) - 0.711 * zf * (zf - 1.0) / af.cbrt()
        - 23.7 * (nf - zf).powi(2) / af;
    let pairing = 11.18 / af.sqrt();
    if z % 2 == 0 && n % 2 == 0 {
        b += pairing;
    } else if z % 2 == 1 && n % 2 == 1 {
        b -= pairing;
    }
    b
}

pub fn lookup(z: u32, n: u32) -> Option<&'static Nuclide> {
    let d = db();
    d.index.get(&(z, n)).map(|&i| &d.list[i])
}

/// All nuclides in the table.
pub fn all() -> &'static [Nuclide] {
    &db().list
}

/// Atomic mass excess in MeV (data, else liquid drop).
pub fn mass_excess(z: u32, n: u32) -> f64 {
    if let Some(x) = lookup(z, n) {
        return x.mass_excess;
    }
    semf_mass_excess(z, n)
}

fn semf_mass_excess(z: u32, n: u32) -> f64 {
    z as f64 * DELTA_H + n as f64 * DELTA_N - semf_binding(z as i64, n as i64)
}

/// Total binding energy (MeV).
pub fn binding(z: u32, n: u32) -> f64 {
    z as f64 * DELTA_H + n as f64 * DELTA_N - mass_excess(z, n)
}

/// Energy needed to remove (zx, nx) from (z, n), MeV (negative = unbound).
pub fn separation(z: u32, n: u32, zx: u32, nx: u32) -> f64 {
    if zx > z || nx > n {
        return f64::INFINITY;
    }
    mass_excess(z - zx, n - nx) + mass_excess(zx, nx) - mass_excess(z, n)
}

/// Q-values (MeV) of the basic decays.
pub fn q_beta_minus(z: u32, n: u32) -> f64 {
    if n == 0 { f64::NEG_INFINITY } else { mass_excess(z, n) - mass_excess(z + 1, n - 1) }
}
pub fn q_ec(z: u32, n: u32) -> f64 {
    if z == 0 { f64::NEG_INFINITY } else { mass_excess(z, n) - mass_excess(z - 1, n + 1) }
}
pub fn q_alpha(z: u32, n: u32) -> f64 {
    -separation(z, n, 2, 2)
}

/// What a nuclide does: measured data, or a liquid-drop guess for nuclei
/// outside the table (beyond the drip lines, superheavies...).
pub fn info(z: u32, n: u32) -> Nuclide {
    if let Some(x) = lookup(z, n) {
        if !(x.branches.is_empty() && x.half_life.is_some()) {
            return x.clone();
        }
    }
    guess(z, n)
}

/// Fallback for unmeasured nuclei: particle emission if unbound, otherwise
/// the fastest of beta (Sargent's rule), alpha (Viola–Seaborg) and fission.
fn guess(z: u32, n: u32) -> Nuclide {
    let mut nuc = Nuclide {
        z,
        n,
        mass_excess: mass_excess(z, n),
        half_life: None,
        branches: Vec::new(),
        delayed: Delayed::default(),
        measured: false,
    };
    let instant = |nuc: &mut Nuclide, m: Mode| {
        nuc.half_life = Some(0.0);
        nuc.branches = vec![(m, 1.0)];
    };
    if z + n <= 1 {
        return nuc;
    }
    if z == 0 || (n > 0 && separation(z, n, 0, 1) < 0.0) {
        instant(&mut nuc, Mode::Neutron);
        return nuc;
    }
    if n == 0 || (z > 0 && separation(z, n, 1, 0) < 0.0) {
        instant(&mut nuc, Mode::Proton);
        return nuc;
    }
    let mut best: Option<(Mode, f64)> = None;
    let mut consider = |m: Mode, t: f64| {
        if best.is_none_or(|(_, bt)| t < bt) {
            best = Some((m, t));
        }
    };
    // Sargent's rule t ~ 1/Q^5, calibrated on the free neutron.
    let sargent = |q: f64| 178.0 / q.max(0.02).powi(5);
    let qb = q_beta_minus(z, n);
    if qb > 0.0 {
        consider(Mode::BetaMinus, sargent(qb));
    }
    let qe = q_ec(z, n);
    if qe > 0.0 {
        consider(if qe > 2.0 * M_E { Mode::BetaPlus } else { Mode::ElectronCapture }, sargent(qe));
    }
    let qa = q_alpha(z, n);
    if z > 2 && qa > 0.5 {
        let zf = (z - 2) as f64;
        let log_t = (1.66175 * zf - 8.5166) / qa.sqrt() - 0.20228 * zf - 33.9069;
        if log_t < 25.0 {
            consider(Mode::Alpha, 10f64.powf(log_t));
        }
    }
    if z >= 90 {
        // Fit to measured spontaneous-fission partial half-lives vs Z^2/A.
        let x = (z * z) as f64 / (z + n) as f64;
        let log_t = 218.3 - 5.48 * x;
        if log_t < 30.0 {
            consider(Mode::Fission, 10f64.powf(log_t));
        }
    }
    if let Some((m, t)) = best {
        nuc.half_life = Some(t);
        nuc.branches = vec![(m, 1.0)];
    }
    nuc
}

pub const SYMBOLS: [&str; 119] = [
    "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

pub fn nuclide_name(z: u32, n: u32) -> String {
    match (z, n) {
        (0, 1) => "n".into(),
        (1, 0) => "p".into(),
        (0, _) => format!("{n}n"),
        _ => {
            let sym = SYMBOLS.get(z as usize).copied().unwrap_or("?");
            format!("{sym}-{}", z + n)
        }
    }
}

/// Parse "U-235", "u235", "235U", "He4", "n", "p", "d", "t", "alpha".
pub fn parse_nuclide(s: &str) -> Option<(u32, u32)> {
    let s = s.trim();
    match s.to_ascii_lowercase().as_str() {
        "n" | "neutron" => return Some((0, 1)),
        "p" | "proton" => return Some((1, 0)),
        "d" | "deuteron" | "deuterium" => return Some((1, 1)),
        "t" | "triton" | "tritium" => return Some((1, 2)),
        "a" | "alpha" => return Some((2, 2)),
        _ => {}
    }
    let letters: String = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    let a: u32 = digits.parse().ok()?;
    let z = SYMBOLS.iter().position(|sym| sym.eq_ignore_ascii_case(&letters))? as u32;
    if z == 0 || a < z {
        return None;
    }
    Some((z, a - z))
}

/// "12.3 y", "4.5 Gy", "81 as"...
pub fn format_time(t: f64) -> String {
    const Y: f64 = 31_556_926.0;
    let units: [(f64, &str); 12] = [
        (1e9 * Y, "Gy"),
        (1e6 * Y, "My"),
        (1e3 * Y, "ky"),
        (Y, "y"),
        (86400.0, "d"),
        (3600.0, "h"),
        (60.0, "min"),
        (1.0, "s"),
        (1e-3, "ms"),
        (1e-6, "µs"),
        (1e-9, "ns"),
        (1e-12, "ps"),
    ];
    if t <= 0.0 {
        return "instant".into();
    }
    for (u, name) in units {
        if t >= u {
            let v = t / u;
            return if v >= 100.0 { format!("{v:.0} {name}") } else { format!("{v:.3} {name}").replace(".000", "") };
        }
    }
    format!("{t:.1e} s")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_loads() {
        assert!(all().len() > 3000);
        let n = info(0, 1);
        assert!((n.half_life.unwrap() - 609.8).abs() < 1.0);
        assert_eq!(n.branches[0].0, Mode::BetaMinus);
        assert!(info(2, 2).half_life.is_none());
        assert!(info(26, 30).half_life.is_none(), "Fe-56 stable");
        assert!((mass_excess(2, 2) - 2.42492).abs() < 1e-4);
    }

    #[test]
    fn branches() {
        let k40 = info(19, 21);
        let bm = k40.branches.iter().find(|b| b.0 == Mode::BetaMinus).unwrap().1;
        assert!((bm - 0.8928).abs() < 1e-3);
        let na22 = info(11, 11);
        let ec = na22.branches.iter().find(|b| b.0 == Mode::ElectronCapture).unwrap().1;
        assert!((ec - 0.0943).abs() < 1e-3);
        assert_eq!(info(4, 3).branches[0].0, Mode::ElectronCapture);
        let cf = info(98, 154);
        assert!(cf.branches.iter().any(|b| b.0 == Mode::Fission && (b.1 - 0.031).abs() < 1e-3));
        // Li-9: beta-delayed neutrons half the time.
        assert!((info(3, 6).delayed.n - 0.505).abs() < 1e-3);
        // F-18 split from systematics: mostly beta+.
        let f18 = info(9, 9);
        let bp = f18.branches.iter().find(|b| b.0 == Mode::BetaPlus).unwrap().1;
        assert!(bp > 0.9, "{bp}");
    }

    #[test]
    fn energetics() {
        // D-T fusion releases 17.59 MeV.
        let q = mass_excess(1, 1) + mass_excess(1, 2) - mass_excess(2, 2) - mass_excess(0, 1);
        assert!((q - 17.589).abs() < 0.01, "{q}");
        assert!((q_alpha(92, 146) - 4.27).abs() < 0.02);
        assert!((separation(92, 144, 0, 1) - 6.545).abs() < 0.01, "U-236 Sn");
        assert!(separation(2, 3, 0, 1) < 0.0, "He-5 unbound");
    }

    #[test]
    fn names() {
        assert_eq!(nuclide_name(6, 8), "C-14");
        assert_eq!(parse_nuclide("U-235"), Some((92, 143)));
        assert_eq!(parse_nuclide("235u"), Some((92, 143)));
        assert_eq!(parse_nuclide("he4"), Some((2, 2)));
        assert_eq!(parse_nuclide("d"), Some((1, 1)));
        assert!(format_time(3.888e8).starts_with("12.32"), "{}", format_time(3.888e8));
    }
}
