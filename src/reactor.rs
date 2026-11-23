//! Reactor hardware for the Nuclear Game: control rods, fuel channels with
//! on-power refuelling (as in a CANDU heavy-water reactor) and the power
//! instrumentation.
//!
//! Control rods are boron-carbide absorbers that slide into vertical guide
//! channels from the top. They are solid (nuclei are pushed out of the
//! inserted part) and absorb neutrons through 10B(n,alpha)7Li with the 1/v
//! cross-section: B4C is black to thermal neutrons (a fraction of a mm of
//! it stops them) but nearly transparent to MeV neutrons, so a rod mostly
//! catches neutrons that the moderator has already slowed down - which is why
//! moderated reactors are easy to control.

use crate::nucdata;
use crate::nuclear::{energy_of, place_nucleus, speed_for};
use crate::reactions::E_THERMAL;
use crate::sim::Sim;
use crate::types::ids;
use macroquad::math::{Rect, Vec2};
use crate::rng::HashMap;
use std::collections::VecDeque;

/// Q value of 10B(n,alpha)7Li (MeV), ground-state branch.
pub const Q_BORON: f64 = 2.79;

/// Optical depth of the rod per px of path for a thermal neutron (game
/// scale, 2 px nucleons): a 5 px rod is ~80 mean free paths thick, like a
/// real B4C rod. It falls as 1/v: ~6 at 1 keV, ~0.4 at 100 keV.
pub const ROD_MU_THERMAL: f32 = 16.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rod {
    /// Channel centre line.
    pub x: f32,
    /// Top of the channel (the rod enters here).
    pub top: f32,
    /// Channel length (full insertion).
    pub len: f32,
    pub half_w: f32,
}

impl Rod {
    /// The part of the rod inside the core at insertion `f` (0..1).
    pub fn inserted(&self, f: f32) -> Rect {
        Rect::new(self.x - self.half_w, self.top, 2.0 * self.half_w, self.len * f.clamp(0.0, 1.0))
    }
}

/// A fuel position in the lattice: what to load there when it is spent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Site {
    pub pos: Vec2,
    pub z: u32,
    pub n: u32,
    /// Seconds the site has been without fresh fuel.
    pub empty_for: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    pub t: f64,
    pub neutrons: u32,
    /// Fissions per sim second over the last sample interval.
    pub fission_rate: f32,
    pub insertion: f32,
    /// Thermal power from fission (W, real units: ~200 MeV per fission).
    pub power_w: f64,
    /// Mean kinetic temperature of the matter (K): every nucleus counts,
    /// pumped flow included.
    pub temperature_k: f64,
    /// Hottest conducting wall (K), 0 if there is none.
    pub hottest_wall_k: f64,
    /// Events per sim second over the last interval.
    pub fusion_rate: f32,
    pub breakup_rate: f32,
    pub capture_rate: f32,
    pub decay_rate: f32,
    /// Power (W) given out by exothermic reactions and taken in by
    /// endothermic ones (fission, fusion, breakup, capture; not decays).
    pub released_w: f64,
    pub absorbed_w: f64,
    /// Kinetic power (W) put in by spawners and beams.
    pub beam_w: f64,
}

impl Sample {
    /// Net reaction power (W): released minus absorbed.
    pub fn net_w(&self) -> f64 {
        self.released_w - self.absorbed_w
    }
}

/// Running totals the samples are differences of.
#[derive(Clone, Copy, Default)]
struct Totals {
    fissions: u64,
    fusions: u64,
    breakups: u64,
    captures: u64,
    decays: u64,
    q_released: f64,
    q_absorbed: f64,
    beam_mev: f64,
}

impl Totals {
    fn now(sim: &Sim, beam_mev: f64) -> Self {
        let st = &sim.nuclear;
        Totals {
            fissions: st.fissions,
            fusions: st.fusions,
            breakups: st.breakups,
            captures: st.captures,
            decays: st.counts.values().sum(),
            q_released: st.q_released,
            q_absorbed: st.q_absorbed,
            beam_mev,
        }
    }
}

/// Real energy per fission (MeV) and joules per MeV.
pub const MEV_PER_FISSION: f64 = 200.0;
pub const JOULES_PER_MEV: f64 = 1.602e-13;

/// Write samples as CSV (time, neutrons, fission rate, power, rods, temperatures).
pub fn write_samples_csv<'a>(path: &std::path::Path, samples: impl IntoIterator<Item = &'a Sample>) -> std::io::Result<usize> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        f,
        "time_s,free_neutrons,fissions_per_s,fusions_per_s,breakups_per_s,captures_per_s,decays_per_s,fission_power_w,reaction_power_released_w,reaction_power_absorbed_w,net_reaction_power_w,beam_power_w,rod_insertion,temperature_k,hottest_wall_k"
    )?;
    let mut n = 0;
    for s in samples {
        writeln!(
            f,
            "{:.2},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.3},{:.1},{:.1}",
            s.t,
            s.neutrons,
            s.fission_rate,
            s.fusion_rate,
            s.breakup_rate,
            s.capture_rate,
            s.decay_rate,
            s.power_w,
            s.released_w,
            s.absorbed_w,
            s.net_w(),
            s.beam_w,
            s.insertion,
            s.temperature_k,
            s.hottest_wall_k
        )?;
        n += 1;
    }
    f.flush()?;
    Ok(n)
}

pub struct Reactor {
    pub rods: Vec<Rod>,
    /// Actual rod-bank insertion, 0 = withdrawn, 1 = fully in.
    pub insertion: f32,
    /// Where the rod drive is taking it.
    pub target: f32,
    /// Rod drive speed (fraction of the full stroke per sim second).
    pub drive: f32,
    /// Emergency shutdown: rods drop under gravity, much faster than the drive.
    pub scram: bool,
    pub sites: Vec<Site>,
    /// Startup neutron sources (Am-Be) and the neutrons per sim second each
    /// emits. Without one a shut-down core has no neutrons to restart from.
    pub sources: Vec<Vec2>,
    pub source_rate: f32,
    /// Injectors: fuel lines, propellant feeds, anything that adds nuclei.
    pub spawners: Vec<Spawner>,
    /// Outlets: exhaust nozzles, drain tanks, off-gas and reprocessing.
    pub drains: Vec<Drain>,
    /// Pumps: push matter (not neutrons) through an area.
    pub pumps: Vec<Pump>,
    pub auto_refuel: bool,
    /// Reload every spent channel at the next check, whatever the settings.
    pub reload_now: bool,
    /// Seconds a site must stay spent before the fuelling machine reloads it.
    pub refuel_delay: f32,
    pub refuels: u64,
    pub rod_captures: u64,
    pub samples: VecDeque<Sample>,
    /// Every sample since the scene was loaded or cleared (the graphs only
    /// keep the last `MAX_SAMPLES`): what "Export graphs" writes.
    pub recording: Vec<Sample>,
    sample_timer: f32,
    last: Totals,
    /// Kinetic energy (MeV) spawners have put in so far.
    pub beam_mev: f64,
    site_timer: f32,
}

impl Default for Reactor {
    fn default() -> Self {
        Self {
            rods: Vec::new(),
            insertion: 1.0,
            target: 1.0,
            drive: 0.08,
            scram: false,
            sites: Vec::new(),
            sources: Vec::new(),
            source_rate: 0.5,
            spawners: Vec::new(),
            drains: Vec::new(),
            pumps: Vec::new(),
            auto_refuel: true,
            reload_now: false,
            refuel_delay: 2.0,
            refuels: 0,
            rod_captures: 0,
            samples: VecDeque::new(),
            recording: Vec::new(),
            sample_timer: 0.0,
            last: Totals::default(),
            beam_mev: 0.0,
            site_timer: 0.0,
        }
    }
}

pub const SAMPLE_DT: f32 = 0.25;
/// Spawners pause above this many particles.
pub const MAX_PARTICLES: usize = 40_000;

/// Adds a nuclide at a rate, with a launch velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spawner {
    pub pos: Vec2,
    pub z: u32,
    pub n: u32,
    pub vel: Vec2,
    /// Nuclei per sim second.
    pub rate: f32,
    pub acc: f32,
    pub spawned: u64,
    /// "Keep" mode: add nuclei anywhere in `area` (at `rate` at most)
    /// whenever fewer than `keep` of them are inside it - a fuel
    /// concentration controller. `keep == 0`: plain spawner at `pos`.
    pub keep: u32,
    pub area: Option<Rect>,
}

impl Spawner {
    pub fn new(pos: Vec2, z: u32, n: u32, vel: Vec2, rate: f32) -> Self {
        Self { pos, z, n, vel, rate, acc: 0.0, spawned: 0, keep: 0, area: None }
    }
}

/// What a drain takes out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrainFilter {
    /// Everything (exhaust nozzle, dump tank).
    All,
    /// Fission products only (A 66-172): on-line reprocessing.
    FissionProducts,
    /// Noble gases (Kr, Xe) and helium: the off-gas system that strips
    /// Xe-135 poison from a molten-salt reactor.
    OffGas,
    /// Everything except actinides (Z >= 90): keeps the fuel, drops the rest.
    NotFuel,
    /// Fission products that have stopped being radioactive (stable or
    /// half-life over a day): reprocessing that leaves the short-lived
    /// ones (delayed-neutron precursors, I-135 -> Xe-135) in place.
    SpentProducts,
}

impl DrainFilter {
    pub const ALL: [DrainFilter; 5] = [DrainFilter::All, DrainFilter::FissionProducts, DrainFilter::SpentProducts, DrainFilter::OffGas, DrainFilter::NotFuel];

    pub fn name(self) -> &'static str {
        match self {
            DrainFilter::All => "Everything",
            DrainFilter::FissionProducts => "Fission products",
            DrainFilter::OffGas => "Off-gas (Kr, Xe, He)",
            DrainFilter::NotFuel => "All but fuel",
            DrainFilter::SpentProducts => "Spent fission products",
        }
    }

    pub fn code(self) -> u8 {
        match self {
            DrainFilter::All => 0,
            DrainFilter::FissionProducts => 1,
            DrainFilter::OffGas => 2,
            DrainFilter::NotFuel => 3,
            DrainFilter::SpentProducts => 4,
        }
    }

    pub fn from_code(c: u8) -> Self {
        match c {
            1 => DrainFilter::FissionProducts,
            2 => DrainFilter::OffGas,
            3 => DrainFilter::NotFuel,
            4 => DrainFilter::SpentProducts,
            _ => DrainFilter::All,
        }
    }

    /// Does a body (z, n) go down this drain? (`z + n == 0`: an electron.)
    pub fn takes(self, z: u32, n: u32) -> bool {
        let a = z + n;
        match self {
            DrainFilter::All => true,
            DrainFilter::FissionProducts => (66..=172).contains(&a),
            DrainFilter::OffGas => matches!(z, 2 | 36 | 54) && n > 0,
            DrainFilter::NotFuel => z < 90 && !(z == 0 && n == 1),
            DrainFilter::SpentProducts => spent_product(z, n),
        }
    }
}

/// An outlet: removes what enters its area and measures the outflow.
#[derive(Clone, Debug, PartialEq)]
pub struct Drain {
    pub rect: Rect,
    pub filter: DrainFilter,
    /// Processing rate: chance per second that something inside is taken
    /// (0 = at once, like a nozzle). Real MSR processing treats a fraction
    /// of the salt per day.
    pub per_second: f32,
    pub removed: u64,
    /// Smoothed outflow: bodies/s, nucleons/s, momentum flux (thrust, in
    /// nucleon masses x px/s^2), mean speed (px/s), mean energy (MeV/u).
    pub rate: f32,
    pub mass_rate: f32,
    pub thrust: f32,
    pub speed: f32,
    pub energy: f64,
    acc: (f32, f32, Vec2, f32, f64, f32),
}

impl Drain {
    pub fn new(rect: Rect, filter: DrainFilter) -> Self {
        Self { rect, filter, per_second: 0.0, removed: 0, rate: 0.0, mass_rate: 0.0, thrust: 0.0, speed: 0.0, energy: 0.0, acc: Default::default() }
    }
}

/// Pushes matter through its area along `dir` up to `speed` (px/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pump {
    pub rect: Rect,
    pub dir: Vec2,
    pub speed: f32,
}
pub const MAX_SAMPLES: usize = 960; // 4 minutes of history

impl Reactor {
    pub fn is_empty(&self) -> bool {
        self.rods.is_empty()
            && self.sites.is_empty()
            && self.sources.is_empty()
            && self.spawners.is_empty()
            && self.drains.is_empty()
            && self.pumps.is_empty()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Absorbing rectangles at the current insertion (empty when withdrawn).
    pub fn rects(&self) -> Vec<Rect> {
        self.rods.iter().map(|r| r.inserted(self.insertion)).filter(|r| r.h > 0.5).collect()
    }

    pub fn scram(&mut self) {
        self.scram = true;
        self.target = 1.0;
    }

    /// Smoothed fission rate (per sim second) over the last ~2 s.
    pub fn power(&self) -> f32 {
        let k = self.samples.len().min(8);
        if k == 0 {
            return 0.0;
        }
        self.samples.iter().rev().take(k).map(|s| s.fission_rate).sum::<f32>() / k as f32
    }

    /// Reactor period (s): e-folding time of the neutron population over the
    /// last ~5 s. None when there are too few neutrons to tell.
    pub fn period(&self) -> Option<f32> {
        let k = 20;
        if self.samples.len() < k {
            return None;
        }
        let s: Vec<&Sample> = self.samples.iter().rev().take(k).collect();
        let (newer, older) = s.split_at(k / 2);
        let a = newer.iter().map(|s| s.neutrons as f32).sum::<f32>() / (k / 2) as f32;
        let b = older.iter().map(|s| s.neutrons as f32).sum::<f32>() / (k / 2) as f32;
        if a < 3.0 && b < 3.0 {
            return None;
        }
        let growth = (a.max(0.5) / b.max(0.5)).ln() / (SAMPLE_DT * (k / 2) as f32);
        if growth.abs() < 1e-3 { Some(f32::INFINITY) } else { Some(1.0 / growth) }
    }
}

/// Push p (radius r) horizontally out of the inserted part of any rod: the
/// rods are vertical, so moderator is moved aside rather than ploughed down.
#[inline]
pub fn push_out(rects: &[Rect], p: Vec2, r: f32) -> Vec2 {
    let mut q = p;
    for rc in rects {
        if q.y + r <= rc.y || q.y - r >= rc.y + rc.h || q.x + r <= rc.x || q.x - r >= rc.x + rc.w {
            continue;
        }
        let mid = rc.x + rc.w * 0.5;
        q.x = if q.x < mid { rc.x - r } else { rc.x + rc.w + r };
    }
    q
}

/// Nucleons per nucleus, indexed like `sim.nuc` (valid after index_nuclei).
fn body_sizes(sim: &Sim) -> Vec<u32> {
    let mut c = vec![0u32; sim.n_nuclei];
    for &k in &sim.nuc {
        if let Some(x) = c.get_mut(k as usize) {
            *x += 1;
        }
    }
    c
}

/// Per-frame reactor update: rod drive, neutron absorption in the rods,
/// on-power refuelling, instruments. Called after the nuclear processing.
pub fn update(sim: &mut Sim, dt: f32) {
    if sim.reactor.is_empty() {
        return;
    }
    let mut re = std::mem::take(&mut sim.reactor);

    // ---- Rod drive ----
    let speed = if re.scram { 1.5 } else { re.drive };
    let d = re.target - re.insertion;
    let step = speed * dt;
    re.insertion = if d.abs() <= step { re.target } else { re.insertion + step * d.signum() };
    if re.scram && re.insertion >= 1.0 {
        re.scram = false;
    }

    // ---- Startup sources: 9Be(alpha,n)12C, neutrons of a few MeV ----
    let mut added = false;
    for &at in &re.sources {
        if re.source_rate > 0.0 && 1.0 - crate::rng::gen_range(0.0f32, 1.0) <= re.source_rate * dt {
            let a = crate::rng::gen_range(0.0, std::f32::consts::TAU);
            let u = Vec2::new(a.cos(), a.sin());
            let e = crate::rng::gen_range(1.0f64, 8.0);
            let v = crate::nuclear::speed_for(&sim.cfg, e, 1.0).min(sim.cfg.max_speed);
            sim.add(at + u * 2.0, u * v, ids::NEUTRON);
            added = true;
        }
    }

    // ---- Spawners ----
    if sim.len() < MAX_PARTICLES {
        for sp in re.spawners.iter_mut() {
            sp.acc = (sp.acc + sp.rate * dt).min(3.0);
            if sp.acc < 1.0 {
                continue;
            }
            if sp.keep > 0 {
                let Some(area) = sp.area else { continue };
                if count_in(sim, area, sp.z, sp.n) >= sp.keep {
                    sp.acc = sp.acc.min(1.0);
                    continue;
                }
                // Somewhere in the area with room, or where light salt can
                // be displaced.
                for _ in 0..12 {
                    let p = Vec2::new(
                        crate::rng::gen_range(area.x, area.x + area.w),
                        crate::rng::gen_range(area.y, area.y + area.h),
                    );
                    let ok = place_nucleus(sim, p, sp.z, sp.n, sp.vel)
                        || (sp.z + sp.n >= 20 && displace_light(sim, p, sp.z + sp.n) && place_nucleus(sim, p, sp.z, sp.n, sp.vel));
                    if ok {
                        sp.acc -= 1.0;
                        sp.spawned += 1;
                        re.beam_mev += energy_of(&sim.cfg, sp.vel.length(), (sp.z + sp.n) as f32);
                        added = true;
                        break;
                    }
                }
                continue;
            }
            while sp.acc >= 1.0 {
                // Somewhere in a small disc, so a stream doesn't block itself;
                // a few tries per frame in crowded places.
                let rn = sim.types[ids::NEUTRON as usize].radius;
                let spread = if sim.point_nuclei {
                    4.0 * crate::nuclear::POINT_RADIUS
                } else {
                    2.0 * rn * (((sp.z + sp.n) as f32 / 0.9069).sqrt() + 2.0)
                };
                let mut ok = false;
                for _ in 0..6 {
                    let a = crate::rng::gen_range(0.0, std::f32::consts::TAU);
                    let d = spread * crate::rng::gen_range(0.0f32, 1.0).sqrt();
                    if place_nucleus(sim, sp.pos + Vec2::new(a.cos(), a.sin()) * d, sp.z, sp.n, sp.vel) {
                        ok = true;
                        break;
                    }
                }
                // Heavy nuclei (fuel salt) displace light carrier salt.
                if !ok && sp.z + sp.n >= 20 && displace_light(sim, sp.pos, sp.z + sp.n) {
                    ok = place_nucleus(sim, sp.pos, sp.z, sp.n, sp.vel);
                }
                if ok {
                    sp.acc -= 1.0;
                    sp.spawned += 1;
                    re.beam_mev += energy_of(&sim.cfg, sp.vel.length(), (sp.z + sp.n) as f32);
                    added = true;
                } else {
                    sp.acc = sp.acc.min(1.5); // blocked: try again next frame
                    break;
                }
            }
        }
    }
    if added {
        sim.index_nuclei();
    }

    // ---- Drains ----
    if !re.drains.is_empty() {
        drain(sim, &mut re, dt);
    }

    // ---- Neutron absorption in the rods ----
    let rects = re.rects();
    if !rects.is_empty() {
        let n = sim.len();
        let sizes = body_sizes(sim);
        let mut dead = vec![false; n];
        let mut absorbed = 0u32;
        for i in 0..n {
            if sim.kind[i] != ids::NEUTRON {
                continue;
            }
            if sizes[sim.nuc[i] as usize] > 1 {
                continue; // bound in a nucleus
            }
            let p = sim.pos[i];
            if !rects.iter().any(|r| r.contains(p)) {
                continue;
            }
            let v = sim.vel[i].length();
            let e = energy_of(&sim.cfg, v, 1.0).max(1e-12);
            let mu = ROD_MU_THERMAL * (E_THERMAL / e).sqrt().min(crate::reactions::SUBTHERMAL_CAP) as f32;
            let p_abs = 1.0 - (-(mu * v * dt)).exp();
            if 1.0 - crate::rng::gen_range(0.0f32, 1.0) <= p_abs {
                dead[i] = true;
                absorbed += 1;
            }
        }
        if absorbed > 0 {
            re.rod_captures += absorbed as u64;
            let st = &mut sim.nuclear;
            st.captures += absorbed as u64;
            st.energy_released += Q_BORON * absorbed as f64;
            for _ in 0..absorbed {
                st.record(
                    crate::nuclear::EventKind::Capture,
                    format!("B-10 + n → Li-7 + α  (control rod capture, Q {:+.2} MeV)", Q_BORON),
                );
            }
            sim.retain(|i| !dead[i]);
            sim.index_nuclei();
        }
    }

    // ---- Fuelling machine ----
    re.site_timer += dt;
    if !re.sites.is_empty() && re.site_timer >= 0.25 {
        let elapsed = re.site_timer;
        re.site_timer = 0.0;
        refuel(sim, &mut re, elapsed);
    }

    sim.reactor = re;
}

/// Instruments: every `SAMPLE_DT` of sim time, record rates, powers and
/// temperatures. Runs in every nuclear scene (not only reactors).
pub fn sample(sim: &mut Sim, dt: f32) {
    let re = &mut sim.reactor;
    re.sample_timer += dt;
    if re.sample_timer < SAMPLE_DT {
        return;
    }
    let span = re.sample_timer;
    re.sample_timer = 0.0;
    let now = Totals::now(sim, sim.reactor.beam_mev);
    let last = sim.reactor.last;
    sim.reactor.last = now;
    let rate = |a: u64, b: u64| a.saturating_sub(b) as f32 / span;
    let watts = |mev: f64| mev.max(0.0) / span as f64 * JOULES_PER_MEV;

    let sizes = body_sizes(sim);
    let neutrons = (0..sim.len()).filter(|&i| sim.kind[i] == ids::NEUTRON && sizes[sim.nuc[i] as usize] == 1).count() as u32;
    // Temperature: random motion of the matter. Fast bodies (beams,
    // fragments still being stopped: above ~0.1 MeV per nucleon) are not
    // heat yet.
    let (mut ke, mut count) = (0.0f64, 0u32);
    for b in sim.bodies().values() {
        if b.a() > 1 || b.z > 0 {
            let a = b.a().max(1);
            if b.vel.length() > speed_for(&sim.cfg, 0.1 * a as f64, a as f32) {
                continue;
            }
            ke += 0.5 * a as f64 * b.vel.length_squared() as f64;
            count += 1;
        }
    }
    let theta = if count > 0 { (ke / count as f64) as f32 } else { 0.0 };
    let hottest = sim.walls.segs.iter().filter(|w| w.kind.conducts()).map(|w| w.theta).fold(None, |m: Option<f32>, t| Some(m.map_or(t, |m| m.max(t))));
    let fission_rate = rate(now.fissions, last.fissions);
    let re = &mut sim.reactor;
    let sample = Sample {
        t: sim.nuclear.time,
        neutrons,
        fission_rate,
        insertion: re.insertion,
        power_w: fission_rate as f64 * MEV_PER_FISSION * JOULES_PER_MEV,
        temperature_k: if count > 0 { crate::thermal::kelvin_of(&sim.cfg, theta) } else { 0.0 },
        hottest_wall_k: hottest.map_or(0.0, |t| crate::thermal::kelvin_of(&sim.cfg, t)),
        fusion_rate: rate(now.fusions, last.fusions),
        breakup_rate: rate(now.breakups, last.breakups),
        capture_rate: rate(now.captures, last.captures),
        decay_rate: rate(now.decays, last.decays),
        released_w: watts(now.q_released - last.q_released),
        absorbed_w: watts(now.q_absorbed - last.q_absorbed),
        beam_w: watts(now.beam_mev - last.beam_mev),
    };
    re.recording.push(sample);
    re.samples.push_back(sample);
    while re.samples.len() > MAX_SAMPLES {
        re.samples.pop_front();
    }
}

/// Centre of the core (mean of the fuel sites), for labels.
pub fn core_center(re: &Reactor) -> Option<Vec2> {
    if re.sites.is_empty() {
        return None;
    }
    Some(re.sites.iter().map(|s| s.pos).sum::<Vec2>() / re.sites.len() as f32)
}


/// Does a nucleus at a fuel site still count as fuel? The loaded nuclide,
/// anything fissile, and short-lived actinides on their way to becoming
/// fissile (U-239 -> Np-239 -> Pu-239 breeding) do; long-lived non-fissile
/// leftovers (U-236) are spent.
fn still_fuel(z: u32, n: u32, site: &Site) -> bool {
    if (z, n) == (site.z, site.n) || crate::reactions::fissile(z, n) {
        return true;
    }
    z >= 90 && nucdata::info(z, n).half_life.is_some_and(|t| t < 3.2e7)
}

/// Fission products that have stopped being radioactive (stable, or
/// half-life over a day) leave with the spent fuel; the short-lived ones -
/// delayed-neutron precursors (Br-87, I-137), the I-135 -> Xe-135 poison
/// chain - stay in the core while they matter.
fn spent_product(z: u32, n: u32) -> bool {
    let a = z + n;
    (66..=172).contains(&a) && nucdata::info(z, n).half_life.is_none_or(|t| t > 86_400.0)
}

fn refuel(sim: &mut Sim, re: &mut Reactor, elapsed: f32) {
    // Nucleus centres: tag -> (sum pos, nucleons, protons).
    let bodies = mass_bodies(sim);
    let r_n = sim.types[ids::NEUTRON as usize].radius;
    // The core: fuel sites plus half a lattice pitch around them.
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for s in &re.sites {
        lo = lo.min(s.pos);
        hi = hi.max(s.pos);
    }
    let pad = r_n * (240.0f32 / 0.9069).sqrt() * 2.0;
    let (lo, hi) = (lo - Vec2::splat(pad), hi + Vec2::splat(pad));

    let mut discharge: Vec<u32> = Vec::new();
    let mut load: Vec<usize> = Vec::new();
    for (si, s) in re.sites.iter_mut().enumerate() {
        let r_site = (r_n * ((s.z + s.n) as f32 / 0.9069).sqrt()).max(4.0 * crate::nuclear::POINT_RADIUS);
        let fuelled = bodies.values().any(|&(sum, c, z)| {
            c >= 200 && still_fuel(z, c - z, s) && (sum / c as f32).distance(s.pos) < r_site
        });
        if fuelled {
            s.empty_for = 0.0;
            continue;
        }
        s.empty_for += elapsed;
        if re.reload_now || (re.auto_refuel && s.empty_for >= re.refuel_delay) {
            for (&tag, &(sum, c, _)) in &bodies {
                if c >= 200 && (sum / c as f32).distance(s.pos) < r_site * 1.5 {
                    discharge.push(tag); // the spent bundle
                }
            }
            load.push(si);
        }
    }
    re.reload_now = false;
    if !load.is_empty() {
        for (&tag, &(sum, c, z)) in &bodies {
            let p = sum / c as f32;
            if p.x > lo.x && p.y > lo.y && p.x < hi.x && p.y < hi.y && spent_product(z, c - z) {
                discharge.push(tag);
            }
        }
    }
    if !discharge.is_empty() {
        let (kind, tag) = (&sim.kind, &sim.tag);
        let dead: Vec<bool> = (0..sim.len()).map(|i| sim.is_nuclear(i) && discharge.contains(&tag[i])).collect();
        let _ = kind;
        sim.retain(|i| !dead[i]);
        sim.index_nuclei();
    }
    for si in load {
        let s = re.sites[si];
        clear_spot(sim, s.pos, s.z + s.n);
        if place_nucleus(sim, s.pos, s.z, s.n, Vec2::ZERO) {
            re.sites[si].empty_for = 0.0;
            re.refuels += 1;
            sim.nuclear.record(
                crate::nuclear::EventKind::Other,
                format!("fresh {} → channel {}  (refuelled by the fuelling machine)", nucdata::nuclide_name(s.z, s.n), si + 1),
            );
        }
    }
    sim.index_nuclei();
}

/// Make room where a nucleus of mass number `a` is about to be loaded: free
/// neutrons are slid just outside, anything else still in the way (a stray
/// deuteron, a stopped fragment) is displaced out of the simulation - a
/// fresh fuel bundle pushes the moderator aside.
fn clear_spot(sim: &mut Sim, at: Vec2, a: u32) {
    let disc = placed_radius(sim, a);
    let mut gone: Vec<(bool, u32)> = Vec::new();
    for i in 0..sim.len() {
        let t = &sim.types[sim.kind[i] as usize];
        if !t.collides {
            continue;
        }
        let d = sim.pos[i].distance(at);
        if d >= disc + t.radius {
            continue;
        }
        let nucleon = sim.is_nuclear(i);
        let lone = sim.kind[i] == ids::NEUTRON && !(0..sim.len()).any(|j| j != i && sim.is_nuclear(j) && sim.tag[j] == sim.tag[i]);
        if lone {
            let u = if d > 1e-3 { (sim.pos[i] - at) / d } else { Vec2::X };
            sim.pos[i] = at + u * (disc + t.radius + 0.5);
        } else {
            gone.push(if nucleon { (true, sim.tag[i]) } else { (false, i as u32) });
        }
    }
    if !gone.is_empty() {
        let (nuclear, tag): (Vec<bool>, Vec<u32>) = ((0..sim.len()).map(|i| sim.is_nuclear(i)).collect(), sim.tag.clone());
        sim.retain(|i| {
            let key = if nuclear[i] { (true, tag[i]) } else { (false, i as u32) };
            !gone.contains(&key)
        });
        sim.index_nuclei();
    }
}

/// Remove what has entered a drain and update its outflow readings.
fn drain(sim: &mut Sim, re: &mut Reactor, dt: f32) {
    let bodies = sim.bodies();
    let mut gone: Vec<u32> = Vec::new();
    for d in re.drains.iter_mut() {
        for (&tag, b) in &bodies {
            let (p, v, a, z) = (b.pos, b.vel, b.a(), b.z);
            let roll = d.per_second <= 0.0 || 1.0 - crate::rng::gen_range(0.0f32, 1.0) <= d.per_second * dt;
            if d.rect.contains(p) && d.filter.takes(z, a - z) && roll && !gone.contains(&tag) {
                gone.push(tag);
                d.removed += 1;
                let speed = v.length();
                d.acc.0 += 1.0;
                d.acc.1 += a as f32;
                d.acc.2 += v * a as f32;
                d.acc.3 += speed;
                d.acc.4 += energy_of(&sim.cfg, speed, a as f32) / a as f64;
            }
        }
        d.acc.5 += dt;
        if d.acc.5 >= 0.5 {
            let (k, span) = (d.acc.0, d.acc.5);
            let w = 0.5;
            d.rate = d.rate * (1.0 - w) + w * k / span;
            d.mass_rate = d.mass_rate * (1.0 - w) + w * d.acc.1 / span;
            d.thrust = d.thrust * (1.0 - w) + w * d.acc.2.length() / span;
            if k > 0.0 {
                d.speed = d.speed * (1.0 - w) + w * d.acc.3 / k;
                d.energy = d.energy * (1.0 - w as f64) + w as f64 * d.acc.4 / k as f64;
            }
            d.acc = Default::default();
        }
    }
    // Loose electrons in an "everything" drain go too.
    let all_rects: Vec<Rect> = re.drains.iter().filter(|d| d.filter == DrainFilter::All).map(|d| d.rect).collect();
    if !gone.is_empty() || !all_rects.is_empty() {
        let (kind, tag, pos) = (&sim.kind, &sim.tag, &sim.pos);
        let dead: Vec<bool> = (0..sim.len())
            .map(|i| {
                if sim.is_nuclear(i) {
                    gone.contains(&tag[i])
                } else {
                    (kind[i] == ids::ELECTRON || kind[i] == ids::POSITRON) && all_rects.iter().any(|r| r.contains(pos[i]))
                }
            })
            .collect();
        if dead.iter().any(|&d| d) {
            sim.retain(|i| !dead[i]);
            sim.index_nuclei();
        }
    }
}

/// Remove light nuclei (A < 20) and stray leptons where a nucleus of mass
/// number `a` is to go, so it can be placed. Gives up (false) if anything
/// heavy is in the way. Free neutrons are slid aside, not removed.
fn displace_light(sim: &mut Sim, at: Vec2, a: u32) -> bool {
    let disc = placed_radius(sim, a);
    let size: HashMap<u32, u32> = sim.bodies().into_iter().map(|(t, b)| (t, b.a())).collect();
    let mut gone: Vec<u32> = Vec::new();
    let mut gone_leptons: Vec<usize> = Vec::new();
    for i in 0..sim.len() {
        let t = &sim.types[sim.kind[i] as usize];
        if !t.collides || sim.pos[i].distance(at) >= disc + t.radius {
            continue;
        }
        if !sim.is_nuclear(i) {
            gone_leptons.push(i);
            continue;
        }
        let s = size[&sim.tag[i]];
        if s == 1 && sim.kind[i] == ids::NEUTRON {
            let d = sim.pos[i] - at;
            let u = if d.length() > 1e-3 { d.normalize() } else { Vec2::X };
            sim.pos[i] = at + u * (disc + t.radius + 0.5);
        } else if s >= 20 {
            return false;
        } else if !gone.contains(&sim.tag[i]) {
            gone.push(sim.tag[i]);
        }
    }
    let (nuclear, tag): (Vec<bool>, Vec<u32>) = ((0..sim.len()).map(|i| sim.is_nuclear(i)).collect(), sim.tag.clone());
    sim.retain(|i| {
        if nuclear[i] { !gone.contains(&tag[i]) } else { !gone_leptons.contains(&i) }
    });
    sim.index_nuclei();
    true
}

/// How many nuclei (z, n) have their centre inside `area`.
fn count_in(sim: &Sim, area: Rect, z: u32, n: u32) -> u32 {
    sim.bodies().values().filter(|b| b.z == z && b.n == n && area.contains(b.pos)).count() as u32
}

/// Bodies as (sum of positions, A, Z) such that sum / A is the centre (the
/// shape `refuel` works with).
fn mass_bodies(sim: &Sim) -> HashMap<u32, (Vec2, u32, u32)> {
    sim.bodies().into_iter().map(|(t, b)| (t, (b.pos * b.a() as f32, b.a(), b.z))).collect()
}

/// Radius of what `place_nucleus` puts down for mass number `a`, plus a
/// contact.
fn placed_radius(sim: &Sim, a: u32) -> f32 {
    let r = sim.types[ids::NEUTRON as usize].radius.max(sim.types[ids::PROTON as usize].radius);
    if sim.point_nuclei && a >= 2 {
        crate::nuclear::POINT_RADIUS + r
    } else {
        1.06 * r * (a as f32).sqrt() + r
    }
}
