//! Nuclear mode: nucleus bookkeeping, reactions and radioactive decay inside
//! the particle engine. The physics lives in `nucdata` (measured nuclide
//! data) and `reactions` (tunnelling, cross-sections, compound nucleus);
//! this module finds nuclei among the particles and carries the outcomes out.
//!
//! A *nucleus* is a connected cluster of nucleons sharing a tag (held
//! together by the strong force and pulled round by `roundness`, see sim.rs).
//!
//! Scales (real physics spans too many orders of magnitude to show 1:1):
//! * Time: t_sim = decay_scale * t_real^time_exponent. With the defaults
//!   (3, 0.3): free neutron 20 s, N-13 20 s, tritium 19 min, U-235 3 days.
//!   time_exponent = 1, decay_scale = 1 is real time.
//! * Energy: kinetic energy in the sim is E_sim = 1/2 v_MeV^2 (E/MeV)^x per
//!   nucleon mass, where v_MeV = `mev_speed` is the speed of a 1 MeV neutron
//!   and x = `energy_exponent` (0.3): speed ~ E^0.15, so thermal neutrons
//!   (0.025 eV) drift at ~32 px/s while 14 MeV fusion neutrons fly at ~670.

use crate::nucdata::{self, Mode, info, mass_excess};
use crate::reactions::{self, Step};
use crate::sim::{Config, Sim};
use crate::types::{ParticleType, ids};
use macroquad::math::{Vec2, vec2};
use crate::rng::gen_range;
use crate::rng::{HashMap, HashSet};
use std::collections::VecDeque;

pub use crate::nucdata::format_time;
pub use crate::nucdata::nuclide_name;

/// Settings for a nuclear-physics world chosen to be as close to the real
/// world as the engine allows: no artificial friction, elastic collisions (so
/// neutrons are moderated by real momentum transfer), real tunnelling.
///
/// Lengths are those of `GAME_NUCLEON_RADIUS` nucleons: everything with a
/// length in it (speeds, force strengths) is scaled from the sandbox's
/// radius-4 nucleons by the same factor, so the dynamics look the same.
pub fn game_config() -> Config {
    let s = GAME_NUCLEON_RADIUS / 4.0;
    let d = Config::default();
    Config {
        nuclear: true,
        coulomb: true,
        rules: false,
        uniform_gravity: false,
        mutual_gravity: false,
        drag: 0.0,
        restitution: 1.0,
        mev_speed: d.mev_speed * s,
        light_speed: d.light_speed * s,
        max_speed: d.max_speed * s,
        strong: d.strong * s,
        // Coulomb acceleration k q^2 / (m d^2) must scale like a length.
        k: d.k * s * s * s,
        softening: d.softening * s,
        ..d
    }
    .with_medium(true)
}

/// Nucleon radius (px) in the Nuclear Game (the sandbox uses 4).
pub const GAME_NUCLEON_RADIUS: f32 = 2.0;

/// Radius (px) of a point nucleus: a little bigger than a free nucleon so
/// both stay visible.
pub const POINT_RADIUS: f32 = 3.0;

/// Upper limit of the point-mode neutron interaction radius (px; see
/// `Config::xs_scale`).
pub const POINT_XS_MAX: f32 = 60.0;

/// A world for the Nuclear Game: nuclear particle types only, at game size,
/// with `game_config()`.
pub fn game_sim(width: f32, height: f32) -> Sim {
    let mut sim = Sim::new(width, height);
    sim.types.truncate(ids::FIRST_CUSTOM as usize);
    let s = GAME_NUCLEON_RADIUS / 4.0;
    for t in sim.types.iter_mut() {
        t.radius *= s;
    }
    // Point nuclei (mass and charge come from each particle's Z and N).
    let mut point = ParticleType::custom("Nucleus", 0.6, POINT_RADIUS, 1.0, 0.0);
    point.custom = false;
    point.in_random_spawn = false;
    point.color = macroquad::color::Color::from_rgba(150, 170, 255, 255);
    sim.types.push(point);
    debug_assert_eq!(sim.types.len() - 1, ids::NUCLEUS as usize);
    sim.rules = crate::types::RuleMatrix::zero(sim.types.len());
    sim.cfg = game_config();
    sim.walls.ambient_theta = crate::thermal::theta_of(&sim.cfg, sim.cfg.temperature);
    sim
}

impl Config {
    /// Matter: atoms are neutral (Coulomb screened by their electrons) and
    /// charged particles stop. Vacuum: bare ions repel, nothing stops them.
    pub fn with_medium(mut self, matter: bool) -> Self {
        self.matter = matter;
        self.coulomb = !matter;
        self
    }
}

/// Compressed, watchable half-life in sim seconds.
pub fn sim_half_life(real: f64, cfg: &Config) -> f64 {
    if real < 1e-12 {
        0.0
    } else {
        cfg.decay_scale as f64 * real.powf(cfg.time_exponent as f64)
    }
}

/// Speed (px/s) of a body of `mass` nucleon masses (or the relative speed of
/// a pair with reduced mass `mass`) carrying `mev` of kinetic energy.
///
/// The compression acts on speed (energy per unit mass), so it is the same
/// for light and heavy bodies: v_sim = v_MeV * (E/m in MeV/u)^(x/2).
pub fn speed_for(cfg: &Config, mev: f64, mass: f32) -> f32 {
    let specific = (mev.max(0.0) / mass.max(1e-3) as f64) as f32;
    cfg.mev_speed * specific.powf(0.5 * cfg.energy_exponent)
}

/// Real kinetic energy (MeV) of a body of `mass` nucleon masses moving at
/// `speed` px/s: the inverse of `speed_for`.
pub fn energy_of(cfg: &Config, speed: f32, mass: f32) -> f64 {
    let x = (speed.max(0.0) / cfg.mev_speed) as f64;
    mass as f64 * x.powf(2.0 / cfg.energy_exponent.max(0.05) as f64)
}

/// Real velocity (in sqrt(MeV per nucleon mass) units, so E = m |u|^2) of
/// something moving at sim velocity `v`.
pub fn real_velocity(cfg: &Config, v: Vec2) -> (f64, f64) {
    let s = v.length();
    if s < 1e-9 {
        return (0.0, 0.0);
    }
    let k = energy_of(cfg, s, 1.0).sqrt() / s as f64;
    (v.x as f64 * k, v.y as f64 * k)
}

/// Electrons/positrons are relativistic: speed from kinetic energy with c = light_speed.
fn lepton_speed(cfg: &Config, t_mev: f64) -> f32 {
    let gamma = 1.0 + t_mev / nucdata::M_E;
    let beta = (1.0 - 1.0 / (gamma * gamma)).sqrt();
    cfg.light_speed * (beta as f32).clamp(0.05, 0.97)
}

/// What sort of event a log entry records (for filtering and export).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum EventKind {
    Fission,
    Fusion,
    /// Nucleus-nucleus reaction where nothing heavier than the reactants
    /// comes out: knock-out, charge exchange, spallation.
    Breakup,
    Capture,
    Scatter,
    Decay,
    Annihilation,
    Other,
}

impl EventKind {
    pub const ALL: [EventKind; 8] = [
        EventKind::Fission,
        EventKind::Fusion,
        EventKind::Breakup,
        EventKind::Capture,
        EventKind::Scatter,
        EventKind::Decay,
        EventKind::Annihilation,
        EventKind::Other,
    ];

    pub fn name(self) -> &'static str {
        match self {
            EventKind::Fission => "fission",
            EventKind::Fusion => "fusion",
            EventKind::Breakup => "breakup",
            EventKind::Capture => "capture",
            EventKind::Scatter => "scattering",
            EventKind::Decay => "decay",
            EventKind::Annihilation => "annihilation",
            EventKind::Other => "other",
        }
    }
}

/// One logged event.
#[derive(Clone, Debug)]
pub struct Event {
    /// Sim time (s) since the scene started.
    pub t: f64,
    pub kind: EventKind,
    pub text: String,
}

impl Event {
    /// The reaction equation ("D + T → He-4 + n").
    pub fn equation(&self) -> &str {
        self.text.split("  (").next().unwrap_or(&self.text)
    }

    /// The bracketed details ("fusion, Q +17.59 MeV, at 52 keV").
    pub fn details(&self) -> &str {
        match self.text.find("  (") {
            Some(k) => self.text[k + 3..].trim_end_matches(')'),
            None => "",
        }
    }

    /// Q value (MeV) if the entry states one.
    pub fn q(&self) -> Option<f64> {
        let d = self.details();
        let k = d.find("Q ")?;
        d[k + 2..].split_whitespace().next()?.parse().ok()
    }

    /// Collision energy (MeV, centre of mass) if the entry states one
    /// ("..., at 26.07 GeV").
    pub fn energy(&self) -> Option<f64> {
        let d = self.details();
        let k = d.rfind("at ")?;
        let mut it = d[k + 3..].split_whitespace();
        let x: f64 = it.next()?.parse().ok()?;
        let unit = match it.next()?.trim_end_matches(')') {
            "meV" => 1e-9,
            "eV" => 1e-6,
            "keV" => 1e-3,
            "MeV" => 1.0,
            "GeV" => 1e3,
            "TeV" => 1e6,
            _ => return None,
        };
        Some(x * unit)
    }
}

/// Events kept in the history (the oldest are dropped beyond this).
pub const HISTORY_MAX: usize = 100_000;

/// Births remembered (nucleus lineage).
pub const BORN_MAX: usize = 300_000;

/// How a nucleus came to be: the event (global history index; None for an
/// unlogged instant breakup) and its parents as (tag, packed Z/N; tag
/// u32::MAX = none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Birth {
    pub event: Option<u64>,
    pub t: f64,
    pub parents: [(u32, u32); 2],
}

impl NuclearState {
    /// The logged event with global index `g`, if it is still kept.
    pub fn event(&self, g: u64) -> Option<&Event> {
        g.checked_sub(self.dropped).and_then(|k| self.history.get(k as usize))
    }

    /// Nuclei made from `tag` (by a reaction or decay it took part in).
    pub fn children(&self, tag: u32) -> Vec<(u32, Birth)> {
        let mut v: Vec<(u32, Birth)> = self.born.iter().filter(|(_, b)| b.parents.iter().any(|p| p.0 == tag)).map(|(t, b)| (*t, *b)).collect();
        v.sort_by_key(|c| c.0);
        v
    }

    /// Record that the tags made since `from` in this frame came from
    /// `parents` in the event just logged (if `logged`).
    fn link(&mut self, fr: &Frame, from: usize, parents: [(u32, u32); 2], logged: bool) {
        let event = logged.then(|| (self.dropped + self.history.len() as u64).saturating_sub(1));
        for &(t, _) in &fr.new_grace[from.min(fr.new_grace.len())..] {
            if self.born.insert(t, Birth { event, t: self.time, parents }).is_none() {
                self.born_order.push_back(t);
            }
        }
        while self.born_order.len() > BORN_MAX {
            if let Some(t) = self.born_order.pop_front() {
                self.born.remove(&t);
            }
        }
    }
}

/// Persistent nuclear bookkeeping shown in the UI.
#[derive(Default)]
pub struct NuclearState {
    pub counts: HashMap<Mode, u64>,
    pub captures: u64,
    /// Nuclei merging into something heavier than either.
    pub fusions: u64,
    /// Nucleus-nucleus reactions that only rearrange or chip pieces off.
    pub breakups: u64,
    /// Energy given out by exothermic reactions (Q > 0) and taken in by
    /// endothermic ones (Q < 0, from the collision's kinetic energy), MeV.
    pub q_released: f64,
    pub q_absorbed: f64,
    pub fissions: u64,
    pub annihilations: u64,
    /// Total energy released by reactions and decays (MeV).
    pub energy_released: f64,
    /// Most recent events, newest first.
    pub log: VecDeque<String>,
    /// Every event since the scene started (up to HISTORY_MAX), oldest first.
    pub history: VecDeque<Event>,
    /// Events dropped from the front of `history`.
    pub dropped: u64,
    /// Sim seconds since the scene started.
    pub time: f64,
    /// Current nuclide census, most common first: ((Z, N), count).
    pub census: Vec<((u32, u32), u32)>,
    /// Also log instantaneous breakups of unbound nuclei.
    pub log_breakups: bool,
    /// Heat left in the matter by radioactive decays so far (MeV; the
    /// neutrinos' share is gone).
    pub decay_heat: f64,
    /// Where each nucleus came from: tag -> the event that made it and its
    /// parents (oldest dropped beyond BORN_MAX).
    pub born: HashMap<u32, Birth>,
    born_order: VecDeque<u32>,
    /// Products of one event can't react with each other for a few frames:
    /// tag -> (event id, frames left).
    grace: HashMap<u32, (u64, u8)>,
    /// Nucleus pairs touching last frame (reactions are rolled once per contact).
    touching: HashSet<(u32, u32)>,
    /// Point-nucleus mode: (neutron, nucleus) pairs within interaction
    /// range last frame (one interaction per pass).
    near: HashSet<(u32, u32)>,
    /// Products of nuclear events still moving fast (being stopped).
    pub fast: HashSet<u32>,
    /// Nuclei fired by a beam (they keep their energy through thin matter
    /// like reaction products; a nucleus merely shoved fast by a collision
    /// does not).
    pub beam: HashSet<u32>,
    /// Products of fragmentation (they may fragment more: spallation showers).
    pub hot: HashSet<u32>,
    /// Fast charged nuclei at the start of this frame: (tag, centre,
    /// velocity, radius). Their whole path through the frame is checked for
    /// contacts: a fast ion that hit something mid-frame has bounced away
    /// again by the end of it.
    pub(crate) sweep: Vec<(u32, Vec2, Vec2, f32)>,
    /// Velocities at the start of the frame of the nuclei in `sweep`: a
    /// swept collision happens at the speed they came in with.
    sweep_vel: HashMap<u32, Vec2>,
    /// Velocity of every charged nucleus at the start of the frame (by
    /// tag): a fast elastic bounce is redone from these with real kinematics.
    frame_vel: HashMap<u32, Vec2>,
    /// (nucleus tag, wall) contacts last frame, for conductor walls.
    pub(crate) wall_touch: HashSet<(u32, u32)>,
    /// Pairs of connected conductor walls, for the walls' `version`.
    pub(crate) wall_pairs: Vec<(u32, u32)>,
    pub(crate) wall_version: u64,
    next_event: u64,
}

impl NuclearState {
    pub fn record(&mut self, kind: EventKind, text: String) {
        self.log.push_front(text.clone());
        self.log.truncate(14);
        self.history.push_back(Event { t: self.time, kind, text });
        if self.history.len() > HISTORY_MAX {
            self.history.pop_front();
            self.dropped += 1;
        }
    }

    pub fn push_log(&mut self, s: String) {
        self.record(EventKind::Other, s);
    }

    /// Forget everything (new scene).
    pub fn reset(&mut self) {
        let keep = self.log_breakups;
        *self = Self::default();
        self.log_breakups = keep;
    }
}

/// True with probability p. (Uses 1-u in (0,1] so an exact 0 from the
/// generator, e.g. the first draw of an unseeded one, never fires.)
fn chance(p: f64) -> bool {
    p >= 1.0 || 1.0 - gen_range(0.0f64, 1.0) <= p
}

fn random_unit() -> Vec2 {
    let a = gen_range(0.0, std::f32::consts::TAU);
    vec2(a.cos(), a.sin())
}

/// Electron's share of the beta decay energy, from the allowed spectrum
/// shape ~ sqrt(x) (1-x)^2 (the neutrino takes the rest).
fn beta_fraction() -> f64 {
    loop {
        let x = gen_range(0.0f64, 1.0);
        if gen_range(0.0f64, 0.29) < x.sqrt() * (1.0 - x).powi(2) {
            return x;
        }
    }
}

/// One nucleus found this frame.
struct Comp {
    members: Vec<usize>,
    z: u32,
    n: u32,
    tag: u32,
    consumed: bool,
}

/// Work queued during a frame (indices stay valid until the end).
struct Frame {
    spawn: Vec<(Vec2, Vec2, u8)>,
    dead: Vec<bool>,
    event: u64,
    /// Tags created this frame by `event`.
    new_grace: Vec<(u32, u64)>,
}

fn com(sim: &Sim, members: &[usize]) -> Vec2 {
    members.iter().map(|&i| sim.pos[i]).sum::<Vec2>() / members.len().max(1) as f32
}

fn mean_vel(sim: &Sim, members: &[usize]) -> Vec2 {
    members.iter().map(|&i| sim.vel[i]).sum::<Vec2>() / members.len().max(1) as f32
}

fn count_z(sim: &Sim, members: &[usize]) -> u32 {
    members.iter().filter(|&&i| sim.kind[i] == ids::PROTON).count() as u32
}

/// Take `zx` protons and `nx` neutrons from the side of `members` facing `dir`.
/// Returns (taken, rest), or None if there aren't enough.
fn split_side(sim: &Sim, members: &[usize], zx: u32, nx: u32, dir: Vec2) -> Option<(Vec<usize>, Vec<usize>)> {
    let c = com(sim, members);
    let proj = |i: usize| (sim.pos[i] - c).dot(dir);
    let mut ps: Vec<usize> = members.iter().copied().filter(|&i| sim.kind[i] == ids::PROTON).collect();
    let mut ns: Vec<usize> = members.iter().copied().filter(|&i| sim.kind[i] == ids::NEUTRON).collect();
    if (ps.len() as u32) < zx || (ns.len() as u32) < nx {
        return None;
    }
    ps.sort_by(|&a, &b| proj(b).total_cmp(&proj(a)));
    ns.sort_by(|&a, &b| proj(b).total_cmp(&proj(a)));
    let mut taken: Vec<usize> = ps[..zx as usize].to_vec();
    taken.extend_from_slice(&ns[..nx as usize]);
    let mut rest: Vec<usize> = ps[zx as usize..].to_vec();
    rest.extend_from_slice(&ns[nx as usize..]);
    Some((taken, rest))
}

fn set_vel(sim: &mut Sim, members: &[usize], v: Vec2) {
    let v = v.clamp_length_max(sim.cfg.max_speed);
    for &i in members {
        sim.vel[i] = v;
    }
}

fn retag(sim: &mut Sim, fr: &mut Frame, members: &[usize]) -> u32 {
    let t = sim.new_tag();
    for &i in members {
        sim.tag[i] = t;
    }
    fr.new_grace.push((t, fr.event));
    t
}

/// Two-body split with real kinematics: pieces of mass mx and mr fly apart
/// along `dir` with relative kinetic energy `e_mev`, about a centre of mass
/// moving at sim velocity `v`. Sim speeds are a compressed (non-linear) map
/// of real ones, so the boost must be added to real velocities: adding sim
/// velocities turns a 3 MeV fusion proton from a moving pair into a 100 MeV
/// one. Returns the sim velocities (x, rest).
fn split_vel(cfg: &Config, v: Vec2, dir: Vec2, e_mev: f64, mx: f32, mr: f32) -> (Vec2, Vec2) {
    let u = real_velocity(cfg, v);
    let (mx, mr) = (mx.max(1e-3) as f64, mr.max(1e-3) as f64);
    let w = (e_mev.max(0.0) * (mx + mr) / (mx * mr)).sqrt();
    let (d0, d1) = (dir.x as f64 * w, dir.y as f64 * w);
    let (fx, fr) = (mr / (mx + mr), mx / (mx + mr));
    (sim_velocity(cfg, (u.0 + d0 * fx, u.1 + d1 * fx)), sim_velocity(cfg, (u.0 - d0 * fr, u.1 - d1 * fr)))
}

/// Sim velocity of a particle with `e_per_nucleon` MeV per nucleon of
/// kinetic energy along `dir`, in a frame moving at sim velocity `v`.
fn boosted(cfg: &Config, v: Vec2, dir: Vec2, e_per_nucleon: f64) -> Vec2 {
    let u = real_velocity(cfg, v);
    let w = e_per_nucleon.max(0.0).sqrt();
    sim_velocity(cfg, (u.0 + dir.x as f64 * w, u.1 + dir.y as f64 * w))
}

/// Centre-of-mass sim velocity of masses ma, mb at sim velocities va, vb
/// (real momentum is what is conserved).
fn cm_velocity(cfg: &Config, va: Vec2, ma: f32, vb: Vec2, mb: f32) -> Vec2 {
    let (ua, ub) = (real_velocity(cfg, va), real_velocity(cfg, vb));
    let (ma, mb) = (ma as f64, mb as f64);
    sim_velocity(cfg, ((ua.0 * ma + ub.0 * mb) / (ma + mb), (ua.1 * ma + ub.1 * mb) / (ma + mb)))
}

/// Two-body breakup: `x` (zx, nx) leaves `members` with relative kinetic
/// energy `e_mev`, momentum conserved about velocity `v`. Returns the rest.
fn emit(sim: &mut Sim, fr: &mut Frame, members: &[usize], zx: u32, nx: u32, e_mev: f64, v: Vec2) -> Option<Vec<usize>> {
    let dir = random_unit();
    let (x, rest) = split_side(sim, members, zx, nx, dir)?;
    if rest.is_empty() {
        return None;
    }
    let (vx, vr) = split_vel(&sim.cfg, v, dir, e_mev, x.len() as f32, rest.len() as f32);
    set_vel(sim, &x, vx);
    set_vel(sim, &rest, vr);
    retag(sim, fr, &x);
    retag(sim, fr, &rest);
    Some(rest)
}

fn photon(sim: &Sim, fr: &mut Frame, at: Vec2) {
    let u = random_unit();
    fr.spawn.push((at + u * 3.0, u * sim.cfg.light_speed, ids::PHOTON));
}

/// Split `members` into fission fragments. Returns a description.
fn fission(sim: &mut Sim, fr: &mut Frame, members: &[usize], split: &reactions::FissionSplit, v: Vec2) -> Option<String> {
    let ((zl, nl), (zh, nh)) = (split.light, split.heavy);
    let nu = split.neutrons.len() as u32;
    let z = count_z(sim, members);
    let n = members.len() as u32 - z;
    if zl + zh != z || nl + nh + nu != n {
        return None;
    }
    let axis = random_unit();
    let c = com(sim, members);
    let proj = |i: usize| (sim.pos[i] - c).dot(axis);
    let mut ps: Vec<usize> = members.iter().copied().filter(|&i| sim.kind[i] == ids::PROTON).collect();
    let mut ns: Vec<usize> = members.iter().copied().filter(|&i| sim.kind[i] == ids::NEUTRON).collect();
    ps.sort_by(|&a, &b| proj(b).total_cmp(&proj(a)));
    ns.sort_by(|&a, &b| proj(b).total_cmp(&proj(a)));
    let mut light: Vec<usize> = ps[..zl as usize].to_vec();
    light.extend_from_slice(&ns[..nl as usize]);
    let mut heavy: Vec<usize> = ps[zl as usize..].to_vec();
    heavy.extend_from_slice(&ns[(nl + nu) as usize..]);
    let prompt: Vec<usize> = ns[nl as usize..(nl + nu) as usize].to_vec();

    let (vl, vh) = split_vel(&sim.cfg, v, axis, split.tke, light.len() as f32, heavy.len() as f32);
    set_vel(sim, &light, vl);
    set_vel(sim, &heavy, vh);
    retag(sim, fr, &light);
    retag(sim, fr, &heavy);
    // Prompt neutrons leave from the surface (taken from the middle of the
    // nucleus they would start buried inside a fragment).
    let reach = members.iter().map(|&i| sim.pos[i].distance(c)).fold(0.0, f32::max) + 6.0;
    let rn = sim.types[ids::NEUTRON as usize].radius;
    for (&i, &e) in prompt.iter().zip(&split.neutrons) {
        let u = random_unit();
        let out = c + u * reach;
        if out.x > rn && out.y > rn && out.x < sim.width - rn && out.y < sim.height - rn && !sim.walls.blocked(out, rn) {
            sim.pos[i] = out;
        }
        sim.vel[i] = boosted(&sim.cfg, v, u, e);
        retag(sim, fr, &[i]);
    }
    // Prompt gammas (~7 MeV total).
    for _ in 0..3 {
        photon(sim, fr, c);
    }
    let neutrons = match nu {
        0 => String::new(),
        1 => " + n".to_string(),
        k => format!(" + {k}n"),
    };
    Some(format!("{} + {}{neutrons}", nuclide_name(zl, nl), nuclide_name(zh, nh)))
}

/// Carry out a compound-nucleus decay chain on `members` (moving at `v`).
/// Returns the product names and the final residue.
fn run_steps(sim: &mut Sim, fr: &mut Frame, members: Vec<usize>, steps: &[Step], v: Vec2) -> (Vec<String>, Vec<usize>) {
    let mut cur = members;
    let mut vcur = v;
    let mut products = Vec::new();
    let mut emitted_any = false;
    for s in steps {
        match s {
            Step::Emit { z, n, e } => {
                if let Some(rest) = emit(sim, fr, &cur, *z, *n, *e, vcur) {
                    products.push(nuclide_name(*z, *n));
                    vcur = mean_vel(sim, &rest);
                    cur = rest;
                    emitted_any = true;
                }
            }
            Step::Gamma { .. } => {
                let c = com(sim, &cur);
                photon(sim, fr, c);
                products.push("γ".into());
            }
            Step::Fission(split) => {
                if let Some(desc) = fission(sim, fr, &cur, split, vcur) {
                    products.push(desc);
                    return (products, Vec::new());
                }
            }
        }
    }
    if !emitted_any {
        set_vel(sim, &cur, vcur);
    }
    let z = count_z(sim, &cur);
    let n = cur.len() as u32 - z;
    if !cur.is_empty() {
        products.insert(0, nuclide_name(z, n));
    }
    (products, cur)
}

/// Note the fast charged nuclei at the start of a frame (see `sweep`).
pub fn note_fast(sim: &mut Sim, dt: f32) {
    let r_n = sim.types[ids::NEUTRON as usize].radius;
    let mut v = std::mem::take(&mut sim.nuclear.sweep);
    v.clear();
    let mut fv = std::mem::take(&mut sim.nuclear.frame_vel);
    fv.clear();
    for (tag, b) in sim.bodies() {
        if b.z > 0 {
            fv.insert(tag, b.vel);
        }
        if b.z > 0 && b.vel.length() * dt > 2.0 {
            let r = if b.parts == 1 { sim.types[if b.a() == 1 { ids::PROTON } else { ids::NUCLEUS } as usize].radius } else { r_n * (b.a() as f32 / 0.9069).sqrt() };
            v.push((tag, b.pos, b.vel, r));
        }
    }
    sim.nuclear.sweep = v;
    sim.nuclear.frame_vel = fv;
}

/// Run one frame of nuclear physics. Must be called right after the
/// substeps, while `sim.grid` still matches particle order.
pub fn process(sim: &mut Sim, dt: f32) {
    let n = sim.len();
    let mut st = std::mem::take(&mut sim.nuclear);
    st.time += dt as f64;
    st.next_event += 1;
    let mut fr = Frame { spawn: Vec::new(), dead: vec![false; n], event: st.next_event, new_grace: Vec::new() };
    st.grace.retain(|_, g| {
        g.1 = g.1.saturating_sub(1);
        g.1 > 0
    });

    let r_nuc = sim.types[ids::NEUTRON as usize].radius.max(sim.types[ids::PROTON as usize].radius);
    let touch = 2.0 * r_nuc * 1.1;

    // ---- 1. Nuclei = nucleons sharing a tag ----
    // Identity comes only from the tag: nuclei split only through reactions
    // and decays (MeV binding energies), never by being jostled apart; the
    // roundness pull brings displaced nucleons back.
    let mut comp_of = vec![u32::MAX; n];
    let mut comps: Vec<Comp> = Vec::new();
    {
        let mut by_tag: HashMap<u32, u32> = HashMap::default();
        for i in 0..n {
            if !sim.is_nuclear(i) {
                continue;
            }
            let c = *by_tag.entry(sim.tag[i]).or_insert_with(|| {
                comps.push(Comp { members: Vec::new(), z: 0, n: 0, tag: sim.tag[i], consumed: false });
                comps.len() as u32 - 1
            });
            comp_of[i] = c;
            let (z, nn) = sim.zn_of(i);
            let cp = &mut comps[c as usize];
            cp.members.push(i);
            cp.z += z;
            cp.n += nn;
        }
    }
    // Contacts between different nuclei. (Point mode: free neutrons don't
    // touch anything, their interactions are rolled below.)
    let points = sim.point_nuclei;
    let mut cross: Vec<(usize, usize)> = Vec::new();
    {
        let (pos, vel, kind, tag, grid) = (&sim.pos, &sim.vel, &sim.kind, &sim.tag, &sim.grid);
        let radius = |i: usize| sim.types[kind[i] as usize].radius;
        let charged = |i: usize| sim.zn_of(i).0 > 0;
        // A fast ion that hit something during the substeps has usually
        // bounced off again by the end of the frame: charged pairs closing
        // or parting fast count as touching within the distance they
        // cover in half a frame.
        let cell = 1.0 / sim.grid.inv.max(1e-6);
        for i in 0..n {
            if !sim.is_nuclear(i) || (points && kind[i] == ids::NEUTRON) {
                continue;
            }
            let fast = charged(i) && vel[i].length() * dt > 2.0;
            let reach = if fast { (((vel[i].length() * dt * 0.5).min(12.0) + 8.0) / cell).ceil() as i32 } else { 1 };
            grid.for_each_near(pos[i], reach.max(1), |j| {
                if j != i && (j > i || fast) && sim.is_nuclear(j) && tag[i] != tag[j] && !(points && kind[j] == ids::NEUTRON) {
                    let mut t = ((radius(i) + radius(j)) * 1.1).max(touch);
                    if charged(i) && charged(j) {
                        t += ((vel[i] - vel[j]).length() * dt * 0.5).min(12.0);
                    }
                    if pos[i].distance_squared(pos[j]) < t * t {
                        cross.push((i.min(j), i.max(j)));
                    }
                }
            });
        }
    }

    // Swept contacts of fast charged nuclei.
    if !st.sweep.is_empty() {
        let mut first: HashMap<u32, usize> = HashMap::default();
        for (k, c) in comps.iter().enumerate() {
            first.insert(c.tag, k);
        }
        let r_nuc = sim.types[ids::NEUTRON as usize].radius;
        let cell = 1.0 / sim.grid.inv.max(1e-6);
        for &(tag, p0, v0, rad) in &st.sweep {
            let Some(&ci) = first.get(&tag) else { continue };
            let Some(&i0) = comps[ci].members.first() else { continue };
            let before = cross.len();
            let p1 = p0 + v0 * dt;
            let seg = p1 - p0;
            let l2 = seg.length_squared().max(1e-9);
            let mid = (p0 + p1) * 0.5;
            let reach = ((seg.length() * 0.5 + rad + 2.0 * r_nuc) / cell).ceil() as i32;
            let (pos, kind, tg) = (&sim.pos, &sim.kind, &sim.tag);
            sim.grid.for_each_near(mid, reach.max(1), |j| {
                if j >= n || !sim.is_nuclear(j) || tg[j] == tag || (points && kind[j] == ids::NEUTRON) {
                    return;
                }
                let q = pos[j];
                let t = ((q - p0).dot(seg) / l2).clamp(0.0, 1.0);
                let rj = sim.types[kind[j] as usize].radius;
                if (p0 + seg * t).distance(q) < (rad + rj) * 1.05 {
                    cross.push((i0.min(j), i0.max(j)));
                }
            });
            if cross.len() > before {
                st.sweep_vel.insert(tag, v0);
            }
        }
        st.sweep.clear();
    }

    // ---- 2. Reactions between touching nuclei (rolled once per contact) ----
    let mut pairs: Vec<(u32, u32)> = cross
        .iter()
        .map(|&(i, j)| (comp_of[i].min(comp_of[j]), comp_of[i].max(comp_of[j])))
        .filter(|(a, b)| a != b)
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    let mut touching_now = HashSet::default();
    let contact_pairs = pairs.clone();
    for (a, b) in pairs {
        let (ta, tb) = (comps[a as usize].tag, comps[b as usize].tag);
        let key = (ta.min(tb), ta.max(tb));
        touching_now.insert(key);
        if comps[a as usize].consumed || comps[b as usize].consumed {
            continue;
        }
        if let (Some(ga), Some(gb)) = (st.grace.get(&ta), st.grace.get(&tb)) {
            if ga.0 == gb.0 {
                continue;
            }
        }
        if !st.touching.contains(&key) && react(sim, &mut st, &mut fr, &mut comps, a as usize, b as usize) {
            continue;
        }
        elastic_bounce(sim, &st, &comps, a as usize, b as usize);
    }
    st.touching = touching_now;
    cap_pileups(sim, &st, &comps, &contact_pairs);
    if points {
        neutron_encounters(sim, &mut st, &mut fr, &mut comps, &comp_of, n, dt);
    }

    // ---- 3. Census + decays ----
    let mut census: HashMap<(u32, u32), u32> = HashMap::default();
    for k in 0..comps.len() {
        if comps[k].consumed {
            continue;
        }
        let (z, nn) = (comps[k].z, comps[k].n);
        *census.entry((z, nn)).or_default() += 1;
        let nuc = info(z, nn);
        let Some(t) = nuc.half_life else { continue };
        let th = sim_half_life(t, &sim.cfg);
        let p = if th <= 0.0 { 1.0 } else { 1.0 - 0.5f64.powf(dt as f64 / th) };
        if !chance(p) || nuc.branches.is_empty() {
            continue;
        }
        let mut r = gen_range(0.0f64, 1.0);
        let mut mode = nuc.branches[0].0;
        for &(m, f) in &nuc.branches {
            if r < f {
                mode = m;
                break;
            }
            r -= f;
        }
        st.next_event += 1;
        fr.event = st.next_event;
        let members = std::mem::take(&mut comps[k].members);
        let from = fr.new_grace.len();
        let parent = (comps[k].tag, crate::sim::pack_zn(z, nn));
        if let Some(msg) = decay(sim, &mut fr, &members, z, nn, mode, &nuc, &mut st) {
            comps[k].consumed = true;
            *st.counts.entry(mode).or_default() += 1;
            let logged = st.log_breakups || t > 1e-12;
            if logged {
                st.record(if mode == Mode::Fission { EventKind::Fission } else { EventKind::Decay }, msg);
            }
            st.link(&fr, from, [parent, (u32::MAX, 0)], logged);
        }
    }
    let mut census: Vec<_> = census.into_iter().collect();
    census.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    st.census = census;

    // ---- 4. Pair annihilation: e+ + e- -> 2 gamma ----
    let c = sim.cfg.light_speed;
    for i in 0..n {
        if sim.kind[i] != ids::POSITRON || fr.dead[i] {
            continue;
        }
        // In matter a positron that has slowed down meets an atomic electron.
        if sim.cfg.matter && sim.vel[i].length() < 0.15 * c {
            fr.dead[i] = true;
            let u = random_unit();
            fr.spawn.push((sim.pos[i] + u * 2.0, u * c, ids::PHOTON));
            fr.spawn.push((sim.pos[i] - u * 2.0, -u * c, ids::PHOTON));
            st.annihilations += 1;
            st.energy_released += 2.0 * nucdata::M_E;
            st.record(EventKind::Annihilation, "e+ + e- → 2γ  (annihilation, Q +1.022 MeV)".into());
            continue;
        }
        let ri = sim.types[ids::POSITRON as usize].radius;
        let mut partner = None;
        sim.grid.for_each_near(sim.pos[i], 1, |j| {
            if partner.is_none() && j < n && sim.kind[j] == ids::ELECTRON && !fr.dead[j] {
                let rs = (ri + sim.types[ids::ELECTRON as usize].radius) * 1.5;
                if sim.pos[i].distance_squared(sim.pos[j]) < rs * rs {
                    partner = Some(j);
                }
            }
        });
        if let Some(j) = partner {
            fr.dead[i] = true;
            fr.dead[j] = true;
            let mid = (sim.pos[i] + sim.pos[j]) * 0.5;
            let u = random_unit();
            fr.spawn.push((mid + u * 2.0, u * c, ids::PHOTON));
            fr.spawn.push((mid - u * 2.0, -u * c, ids::PHOTON));
            st.annihilations += 1;
            st.energy_released += 2.0 * nucdata::M_E;
            st.record(EventKind::Annihilation, "e+ + e- → 2γ  (annihilation, Q +1.022 MeV)".into());
        }
    }

    // Stopped beta electrons become part of the material.
    if sim.cfg.matter {
        for i in 0..n {
            if sim.kind[i] == ids::ELECTRON && sim.vel[i].length() < 0.08 * c {
                fr.dead[i] = true;
            }
        }
    }

    st.sweep_vel.clear();
    if st.beam.len() + st.hot.len() > 20_000 {
        let alive: HashSet<u32> = sim.tag.iter().copied().collect();
        st.beam.retain(|t| alive.contains(t));
        st.hot.retain(|t| alive.contains(t));
    }

    // ---- 5. Apply ----
    for (t, ev) in fr.new_grace.drain(..) {
        st.grace.insert(t, (ev, 20));
        st.fast.insert(t);
    }
    if fr.dead.iter().any(|&d| d) {
        let dead = std::mem::take(&mut fr.dead);
        sim.retain(|i| !dead[i]);
    }
    for (p, v, k) in fr.spawn.drain(..) {
        sim.add(p, v, k);
    }
    sim.nuclear = st;
    sim.index_nuclei();
}

/// A fast charged nucleus bouncing off another without reacting: the
/// solver bounces them in sim velocities, which (sim speeds being a
/// compressed map of real ones) hands the lighter one far more energy than
/// a real collision could - a 15 MeV deuteron would kick a proton to 50
/// MeV, and trapped protons would pump themselves up to GeV. Keep the
/// direction the solver scattered them in, but redo the energies as a real
/// elastic collision (real momentum and energy conserved) from the
/// velocities they came in with. Vacuum only; pairs the solver didn't
/// deflect (a fast ion passing through) are left alone.
const BOUNCE_MIN: f64 = 0.5;

fn elastic_bounce(sim: &mut Sim, st: &NuclearState, comps: &[Comp], a: usize, b: usize) {
    // In matter the stopping (applied during the frame too) is what decides
    // where a fast ion's energy goes; redoing the bounce from the incoming
    // velocities would undo it.
    if sim.cfg.matter {
        return;
    }
    let (ca, cb) = (&comps[a], &comps[b]);
    if ca.z == 0 || cb.z == 0 || ca.members.is_empty() || cb.members.is_empty() {
        return;
    }
    let moving = |t: &u32| st.fast.contains(t) || st.beam.contains(t) || st.hot.contains(t);
    if !moving(&ca.tag) && !moving(&cb.tag) {
        return;
    }
    let (Some(&va), Some(&vb)) = (st.frame_vel.get(&ca.tag), st.frame_vel.get(&cb.tag)) else { return };
    let (w0, w1) = (va - vb, mean_vel(sim, &ca.members) - mean_vel(sim, &cb.members));
    if (w1 - w0).length() < 0.1 * w0.length() {
        return; // not deflected this frame
    }
    let dir = w1.normalize_or_zero();
    if dir == Vec2::ZERO {
        return;
    }
    let cfg = &sim.cfg;
    let (ua, ub) = (real_velocity(cfg, va), real_velocity(cfg, vb));
    let (ma, mb) = ((ca.z + ca.n) as f64, (cb.z + cb.n) as f64);
    let w = ((ua.0 - ub.0).powi(2) + (ua.1 - ub.1).powi(2)).sqrt();
    // Below BOUNCE_MIN (CM MeV): a slow knock (lattice neighbours, a recoil
    // settling). Beyond FRAG_FULL per nucleon a real nucleus is
    // shattered, not bounced back as a hard sphere: the
    // solver's handling is left alone there.
    if ma * mb / (ma + mb) * w * w < BOUNCE_MIN || w * w > FRAG_FULL {
        return;
    }
    let w2 = (dir.x as f64 * w, dir.y as f64 * w);
    let cm = ((ma * ua.0 + mb * ub.0) / (ma + mb), (ma * ua.1 + mb * ub.1) / (ma + mb));
    let (fa, fb) = (mb / (ma + mb), ma / (ma + mb));
    let na = sim_velocity(cfg, (cm.0 + w2.0 * fa, cm.1 + w2.1 * fa));
    let nb = sim_velocity(cfg, (cm.0 - w2.0 * fb, cm.1 - w2.1 * fb));
    let (ma_, mb_) = (ca.members.clone(), cb.members.clone());
    set_vel(sim, &ma_, na);
    set_vel(sim, &mb_, nb);
}

/// Backstop for pile-ups (three or more nuclei pressed together in one
/// frame - a fast ion driven into two others): the solver pushes them
/// apart in sim velocities, which can turn a 15 MeV hit into a 100 MeV
/// proton. In vacuum nothing else adds energy to a touching cluster in a
/// frame (bar an electric field), so if its real kinetic energy went up,
/// its motion about its centre of mass is scaled back to what came in.
fn cap_pileups(sim: &mut Sim, st: &NuclearState, comps: &[Comp], pairs: &[(u32, u32)]) {
    if sim.cfg.matter || pairs.is_empty() {
        return;
    }
    // Union-find over the touching comps.
    let mut root: HashMap<u32, u32> = HashMap::default();
    fn find(root: &mut HashMap<u32, u32>, x: u32) -> u32 {
        let mut r = x;
        while let Some(&p) = root.get(&r) {
            if p == r {
                break;
            }
            r = p;
        }
        root.insert(x, r);
        r
    }
    for &(a, b) in pairs {
        root.entry(a).or_insert(a);
        root.entry(b).or_insert(b);
        let (ra, rb) = (find(&mut root, a), find(&mut root, b));
        if ra != rb {
            root.insert(ra, rb);
        }
    }
    let mut groups: HashMap<u32, Vec<usize>> = HashMap::default();
    let keys: Vec<u32> = root.keys().copied().collect();
    for k in keys {
        let r = find(&mut root, k);
        groups.entry(r).or_default().push(k as usize);
    }
    let cfg = sim.cfg.clone();
    for (_, g) in groups {
        if g.len() < 2 {
            continue;
        }
        let mut pre = 0.0f64;
        let mut post: Vec<(f64, (f64, f64))> = Vec::with_capacity(g.len());
        let mut ok = true;
        for &k in &g {
            let c = &comps[k];
            let Some(&v0) = st.frame_vel.get(&c.tag) else {
                ok = false; // newborn (a reaction's energy is real) or neutral
                break;
            };
            if c.consumed || c.members.is_empty() || sim.reactor.fields.iter().any(|f| f.e != Vec2::ZERO && f.rect.contains(com(sim, &c.members))) {
                ok = false;
                break;
            }
            let m = (c.z + c.n) as f64;
            let u0 = real_velocity(&cfg, v0);
            pre += m * (u0.0 * u0.0 + u0.1 * u0.1);
            post.push((m, real_velocity(&cfg, mean_vel(sim, &c.members))));
        }
        if !ok {
            continue;
        }
        let mt: f64 = post.iter().map(|p| p.0).sum();
        let cm = (post.iter().map(|p| p.0 * p.1 .0).sum::<f64>() / mt, post.iter().map(|p| p.0 * p.1 .1).sum::<f64>() / mt);
        let e_cm = mt * (cm.0 * cm.0 + cm.1 * cm.1);
        let e_int: f64 = post.iter().map(|(m, u)| m * ((u.0 - cm.0).powi(2) + (u.1 - cm.1).powi(2))).sum();
        if e_cm + e_int <= pre * 1.02 + 0.05 {
            continue;
        }
        // Scale the internal motion (and, if even the drift is too much, all of it).
        let (cm, k) = if e_cm < pre { (cm, ((pre - e_cm) / e_int.max(1e-12)).sqrt()) } else { ((0.0, 0.0), (pre / (e_cm + e_int)).sqrt()) };
        for (&kk, &(_, u)) in g.iter().zip(&post) {
            let un = (cm.0 + (u.0 - cm.0) * k, cm.1 + (u.1 - cm.1) * k);
            let v = sim_velocity(&cfg, un);
            let members = comps[kk].members.clone();
            set_vel(sim, &members, v);
        }
    }
}

/// Try a reaction between touching nuclei a and b.
/// Returns whether anything happened.
fn react(sim: &mut Sim, st: &mut NuclearState, fr: &mut Frame, comps: &mut [Comp], a: usize, b: usize) -> bool {
    let (za, na, zb, nb) = (comps[a].z, comps[a].n, comps[b].z, comps[b].n);
    if za + zb == 0 {
        return false; // neutron + neutron: nothing binds
    }
    let (aa, ab) = ((za + na) as f32, (zb + nb) as f32);
    let vel_of = |k: usize| {
        // Velocities from before the solver bounced them apart this frame.
        st.sweep_vel.get(&comps[k].tag).or_else(|| st.frame_vel.get(&comps[k].tag)).copied().unwrap_or_else(|| mean_vel(sim, &comps[k].members))
    };
    let (va, vb) = (vel_of(a), vel_of(b));
    let (ca, cb) = (com(sim, &comps[a].members), com(sim, &comps[b].members));
    let mu = aa * ab / (aa + ab);
    // Collision energy from the *real* relative velocity: sim speeds are a
    // compressed (non-linear) map of real ones, so velocities must be turned
    // back into real ones before they are subtracted. (Subtracting sim
    // velocities turns a slow neutron meeting a fast fission fragment into a
    // GeV collision.)
    let (ua, ub) = (real_velocity(&sim.cfg, va), real_velocity(&sim.cfg, vb));
    let mut e = mu as f64 * ((ua.0 - ub.0).powi(2) + (ua.1 - ub.1).powi(2));
    // Speed "at infinity": add back what the sim's Coulomb repulsion took
    // away on the way in, so tunnelling sees the true collision energy.
    if sim.cfg.coulomb {
        let v2 = (va - vb).length_squared();
        let d2 = ca.distance_squared(cb) + sim.cfg.softening * sim.cfg.softening;
        let v2_inf = v2 + 2.0 * sim.cfg.k * (za * zb) as f32 / (mu * d2.sqrt());
        e += energy_of(&sim.cfg, v2_inf.sqrt(), mu) - energy_of(&sim.cfg, v2.sqrt(), mu);
    }

    let p = if (za, na) == (0, 1) {
        reactions::neutron_absorption(zb, nb, e)
    } else if (zb, nb) == (0, 1) {
        reactions::neutron_absorption(za, na, e)
    } else if za * zb == 0 {
        1.0
    } else if (aa > 20.0 && st.fast.contains(&comps[a].tag)) || (ab > 20.0 && st.fast.contains(&comps[b].tag)) {
        // A heavy ion still being stopped (a fission fragment, a recoil)
        // loses its energy to electrons long before it could get through
        // another nucleus's Coulomb barrier: no reaction.
        0.0
    } else {
        // In dense matter a fast ion makes ~1e5 ionizing collisions for each
        // close nuclear encounter, so reactions while it slows down are rare
        // (thick-target yields ~1e-6..1e-3). Sim contacts are far too
        // frequent, so they're scaled down. Neutrons are unaffected.
        // Fast ions (beyond ~100 MeV/u) go further than a nuclear mean
        // free path before they stop: no suppression at ~1 GeV/u.
        let e_lab = e / mu as f64;
        // In a plasma (vacuum here) a fresh fast product - a fusion alpha, a
        // triton - glances off thousands of ions by Coulomb scattering for
        // every nuclear reaction it makes: one sim contact is not one try.
        let product = st.fast.contains(&comps[a].tag) || st.fast.contains(&comps[b].tag);
        let thick = if sim.cfg.matter {
            (e_lab / 1000.0).powf(1.5).clamp(1e-3, 1.0)
        } else if product && e_lab < FRAG_START {
            0.02
        } else {
            1.0
        };
        (reactions::penetrability(e, za, za + na, zb, zb + nb) * sim.cfg.tunneling as f64 * thick).min(1.0)
    };
    if !chance(p) {
        return false;
    }
    let from = fr.new_grace.len();
    let parents = [(comps[a].tag, crate::sim::pack_zn(za, na)), (comps[b].tag, crate::sim::pack_zn(zb, nb))];
    // Fast collisions shatter the nuclei instead of fusing them.
    let e_lab = e / mu as f64;
    // Only genuinely fast particles shatter nuclei: beam nuclei and the debris
    // of earlier fragmentation. (In nucleon-cluster mode a light particle
    // shoved by a fission fragment gets far too much energy in the sim's
    // compressed units; it shouldn't set off showers.)
    let genuine = |t: u32| st.beam.contains(&t) || st.hot.contains(&t);
    if za + zb > 0 && (genuine(comps[a].tag) || genuine(comps[b].tag)) && chance(fragmentation_chance(e_lab)) && fragment(sim, st, fr, comps, a, b, e) {
        for &(t, _) in &fr.new_grace[from..] {
            st.hot.insert(t);
        }
        st.link(fr, from, parents, true);
        return true;
    }
    let (z, n) = (za + zb, na + nb);
    let e_star = reactions::capture_q(za, na, zb, nb) + e;
    let steps = reactions::compound_decay(z, n, e_star);
    // Compound elastic: it just spat the projectile back out.
    let elastic = steps.is_empty()
        || (steps.len() == 1
            && matches!(steps[0], Step::Emit { z: x, n: y, .. } if (x, y) == (za, na) || (x, y) == (zb, nb)));
    if elastic {
        return false;
    }
    st.next_event += 1;
    fr.event = st.next_event;
    let v = cm_velocity(&sim.cfg, va, aa, vb, ab);
    let single = comps[a].members.len() == 1 && comps[b].members.len() == 1;
    let (products, residue) = if sim.point_nuclei && single {
        // The compound nucleus is the heavier particle; the lighter is gone.
        let (ia, ib) = (comps[a].members[0], comps[b].members[0]);
        let (keep, gone) = if aa >= ab { (ia, ib) } else { (ib, ia) };
        fr.dead[gone] = true;
        sim.set_point(keep, z, n);
        comps[a].members.clear();
        comps[b].members.clear();
        run_steps_point(sim, fr, keep, &steps, v)
    } else {
        let mut members = std::mem::take(&mut comps[a].members);
        members.extend(std::mem::take(&mut comps[b].members));
        let t = sim.new_tag();
        for &i in &members {
            sim.tag[i] = t;
        }
        run_steps(sim, fr, members, &steps, v)
    };
    comps[a].consumed = true;
    comps[b].consumed = true;
    if !residue.is_empty() {
        retag(sim, fr, &residue);
    }

    // Q-value from the masses of everything that came out.
    let mut m_out = 0.0;
    let mut fissioned = false;
    let (mut rz, mut rn) = (z, n);
    for s in &steps {
        match s {
            Step::Emit { z: x, n: y, .. } => {
                m_out += mass_excess(*x, *y);
                rz -= x;
                rn -= y;
            }
            Step::Fission(f) => {
                fissioned = true;
                m_out += mass_excess(f.light.0, f.light.1)
                    + mass_excess(f.heavy.0, f.heavy.1)
                    + f.neutrons.len() as f64 * nucdata::DELTA_N;
            }
            Step::Gamma { .. } => {}
        }
    }
    if !fissioned {
        m_out += mass_excess(rz, rn);
    }
    let q = mass_excess(za, na) + mass_excess(zb, nb) - m_out;
    st.energy_released += q.max(0.0);
    if q > 0.0 {
        st.q_released += q;
    } else {
        st.q_absorbed -= q;
    }

    // Heaviest thing that came out: fusion makes something heavier than
    // either reactant; otherwise it was a breakup.
    let mut heaviest = if fissioned { 0 } else { rz + rn };
    for s in &steps {
        if let Step::Emit { z: x, n: y, .. } = s {
            heaviest = heaviest.max(x + y);
        }
    }
    let mut products = products;
    sort_products(&mut products);

    let (big, small) = if aa >= ab { ((za, na), (zb, nb)) } else { ((zb, nb), (za, na)) };
    let (kind, ek) = if fissioned {
        st.fissions += 1;
        ("induced fission", EventKind::Fission)
    } else if steps.iter().any(|s| matches!(s, Step::Emit { z: x, n: y, .. } if (*x, *y) == small)) && (rz, rn) == big {
        ("inelastic scattering", EventKind::Scatter)
    } else if (za, na) == (0, 1) || (zb, nb) == (0, 1) {
        st.captures += 1;
        ("neutron capture", EventKind::Capture)
    } else if heaviest > (za + na).max(zb + nb) || (q > 0.0 && za > 0 && zb > 0 && (za + na).max(zb + nb) <= 20) {
        // Something heavier than either was made - or light nuclei reacted
        // and gave out energy (p + B-11 → 3 He-4, Li-6 + D → 2 He-4).
        st.fusions += 1;
        ("fusion", EventKind::Fusion)
    } else {
        st.breakups += 1;
        ("breakup", EventKind::Breakup)
    };
    st.record(ek, format!(
        "{} + {} → {}  ({kind}, Q {:+.2} MeV, at {})",
        nuclide_name(big.0, big.1),
        nuclide_name(small.0, small.1),
        products.join(" + "),
        q,
        format_energy(e),
    ));
    st.link(fr, from, parents, true);
    true
}

/// Beam energy per nucleon (MeV/u, a projectile on a target at rest) where
/// collisions start to shatter nuclei instead of fusing them, and where
/// they all do. Real heavy-ion collisions go from fusion and deep-inelastic
/// reactions below ~10 MeV/u to fragmentation above ~50-100 MeV/u.
pub const FRAG_START: f64 = 15.0;
pub const FRAG_FULL: f64 = 80.0;
/// Above this (MeV/u) pions are made and relativity matters: beyond the model.
pub const FRAG_RELATIVISTIC: f64 = 300.0;
/// Excitation energy (MeV) a spectator remnant gets per nucleon sheared off
/// (Gaimard-Schmidt).
const E_PER_ABRADED: f64 = 13.3;

/// Chance that a collision at `e_lab` MeV per nucleon fragments.
pub fn fragmentation_chance(e_lab: f64) -> f64 {
    let x = ((e_lab - FRAG_START) / (FRAG_FULL - FRAG_START)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Fraction of a disc of radius r1 that a disc of radius r2 at distance d
/// overlaps.
fn overlap_fraction(r1: f64, r2: f64, d: f64) -> f64 {
    if d >= r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        return if r1 <= r2 { 1.0 } else { (r2 * r2) / (r1 * r1) };
    }
    let a1 = r1 * r1 * ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).clamp(-1.0, 1.0).acos();
    let a2 = r2 * r2 * ((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).clamp(-1.0, 1.0).acos();
    let k = 0.5 * ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)).max(0.0).sqrt();
    ((a1 + a2 - k) / (std::f64::consts::PI * r1 * r1)).clamp(0.0, 1.0)
}

/// Draw `k` nucleons from z protons and n neutrons: how many are protons.
fn draw_protons(z: u32, n: u32, k: u32) -> u32 {
    let (mut zr, mut nr, mut got) = (z, n, 0);
    for _ in 0..k.min(z + n) {
        if chance(zr as f64 / (zr + nr).max(1) as f64) {
            zr -= 1;
            got += 1;
        } else {
            nr -= 1;
        }
    }
    got
}

/// Light clusters the hot overlap zone condenses into: free nucleons when
/// it is hot, more d, t, He-3 and alphas when it is cooler.
fn coalesce(mut z: u32, mut n: u32, t_mev: f64) -> Vec<(u32, u32)> {
    let c = 0.6 * (-t_mev / 15.0).exp();
    let mut out = Vec::new();
    while z + n > 0 {
        if chance(c) {
            let options: [((u32, u32), f64); 4] = [((2, 2), 0.35), ((1, 1), 0.35), ((1, 2), 0.15), ((2, 1), 0.15)];
            let ok: Vec<_> = options.iter().filter(|((cz, cn), _)| *cz <= z && *cn <= n).collect();
            if !ok.is_empty() {
                let total: f64 = ok.iter().map(|o| o.1).sum();
                let mut r = gen_range(0.0, total);
                let mut pick = ok[0].0;
                for o in &ok {
                    if r < o.1 {
                        pick = o.0;
                        break;
                    }
                    r -= o.1;
                }
                z -= pick.0;
                n -= pick.1;
                out.push(pick);
                continue;
            }
        }
        if chance(z as f64 / (z + n) as f64) {
            z -= 1;
            out.push((1, 0));
        } else {
            n -= 1;
            out.push((0, 1));
        }
    }
    out
}

/// Fragmentation (abrasion-ablation): where the two nuclei overlap
/// (participants, from a random impact parameter) they are sheared off into
/// a hot zone that flies apart as nucleons and light clusters; the rest of
/// each nucleus (the spectators) carries on at its own velocity, excited by
/// what it lost, and boils off nucleons (or fissions, if heavy). Energy and
/// momentum are conserved. Returns false (nothing done) if the collision
/// can't pay for it.
fn fragment(sim: &mut Sim, st: &mut NuclearState, fr: &mut Frame, comps: &mut [Comp], a: usize, b: usize, e: f64) -> bool {
    let (za, na, zb, nb) = (comps[a].z, comps[a].n, comps[b].z, comps[b].n);
    let (aa, ab) = (za + na, zb + nb);
    let vel_of = |k: usize| {
        // Velocities from before the solver bounced them apart this frame.
        st.sweep_vel.get(&comps[k].tag).or_else(|| st.frame_vel.get(&comps[k].tag)).copied().unwrap_or_else(|| mean_vel(sim, &comps[k].members))
    };
    let (va, vb) = (vel_of(a), vel_of(b));
    let (ca, cb) = (com(sim, &comps[a].members), com(sim, &comps[b].members));
    let (ua, ub) = (real_velocity(&sim.cfg, va), real_velocity(&sim.cfg, vb));
    // Geometry (nuclear radii, fm) and a random impact parameter.
    let (ra, rb) = (1.2 * (aa as f64).cbrt(), 1.2 * (ab as f64).cbrt());
    let bimp = (ra + rb) * gen_range(0.0f64, 1.0).sqrt();
    let pa = ((aa as f64 * overlap_fraction(ra, rb, bimp)).round() as u32).clamp(1, aa);
    let pb = ((ab as f64 * overlap_fraction(rb, ra, bimp)).round() as u32).clamp(1, ab);
    let zpa = draw_protons(za, na, pa);
    let zpb = draw_protons(zb, nb, pb);
    // Spectators; tiny or impossible remnants join the hot zone.
    let mut spec = [(za - zpa, na - (pa - zpa)), (zb - zpb, nb - (pb - zpb))];
    let mut part = [pa, pb];
    let (mut zf, mut nf) = (zpa + zpb, (pa - zpa) + (pb - zpb));
    for k in 0..2 {
        let (z, n) = spec[k];
        if z + n < 2 || z == 0 || (n == 0 && z > 1) {
            zf += z;
            nf += n;
            part[k] += z + n;
            spec[k] = (0, 0);
        }
    }
    let (pa, pb) = (part[0] as f64, part[1] as f64);
    let du2 = (ua.0 - ub.0).powi(2) + (ua.1 - ub.1).powi(2);
    let e_rel = pa * pb / (pa + pb) * du2;
    let pieces = coalesce(zf, nf, e_rel / (pa + pb));
    if pieces.len() + spec.iter().filter(|s| s.0 + s.1 > 0).count() < 2 {
        return false; // everything ended up in one piece: that's fusion, not this
    }
    // Nothing changed hands (two nucleons, say): just a scattering.
    {
        let mut out: Vec<(u32, u32)> = pieces.iter().copied().chain(spec.iter().copied().filter(|s| s.0 + s.1 > 0)).collect();
        let mut inp = vec![(za, na), (zb, nb)];
        out.sort_unstable();
        inp.sort_unstable();
        if out == inp {
            return false;
        }
    }
    let m_in = mass_excess(za, na) + mass_excess(zb, nb);
    let m_mid: f64 = pieces.iter().map(|&(z, n)| mass_excess(z, n)).sum::<f64>()
        + spec.iter().filter(|s| s.0 + s.1 > 0).map(|&(z, n)| mass_excess(z, n)).sum::<f64>();
    let avail = e_rel + (m_in - m_mid);
    if avail <= 0.0 {
        return false;
    }
    // Excitation of the spectators, then what's left heats the hot zone.
    let want = [
        if spec[0].0 + spec[0].1 > 0 { E_PER_ABRADED * (aa - spec[0].0 - spec[0].1) as f64 } else { 0.0 },
        if spec[1].0 + spec[1].1 > 0 { E_PER_ABRADED * (ab - spec[1].0 - spec[1].1) as f64 } else { 0.0 },
    ];
    let k = if want[0] + want[1] > 0.6 * avail { 0.6 * avail / (want[0] + want[1]) } else { 1.0 };
    let mut ex = [want[0] * k, want[1] * k];
    let mut e_th = avail - ex[0] - ex[1];
    if pieces.len() < 2 {
        // One piece can't share out heat by itself: the spectators take it.
        let s = if spec[0].0 + spec[0].1 > 0 { 0 } else { 1 };
        ex[s] += e_th;
        e_th = 0.0;
    }
    // Hot-zone pieces: random directions and shares, zero total momentum in
    // the zone's frame, total kinetic energy e_th (E = m u^2 in these units).
    let uf = ((pa * ua.0 + pb * ub.0) / (pa + pb), (pa * ua.1 + pb * ub.1) / (pa + pb));
    let masses: Vec<f64> = pieces.iter().map(|&(z, n)| (z + n) as f64).collect();
    let mut us: Vec<(f64, f64)> = masses
        .iter()
        .map(|&m| {
            let w = -(1.0 - gen_range(0.0f64, 1.0)).ln();
            let d = random_unit();
            let s = (w / m).sqrt();
            (d.x as f64 * s, d.y as f64 * s)
        })
        .collect();
    let mtot: f64 = masses.iter().sum();
    let p = us.iter().zip(&masses).fold((0.0, 0.0), |acc, (u, m)| (acc.0 + m * u.0, acc.1 + m * u.1));
    for u in us.iter_mut() {
        u.0 -= p.0 / mtot;
        u.1 -= p.1 / mtot;
    }
    let ke: f64 = us.iter().zip(&masses).map(|(u, m)| m * (u.0 * u.0 + u.1 * u.1)).sum();
    let scale = if ke > 1e-30 { (e_th / ke).sqrt() } else { 0.0 };
    let vel_of = |cfg: &Config, u: (f64, f64)| sim_velocity(cfg, (uf.0 + u.0 * scale, uf.1 + u.1 * scale));

    st.next_event += 1;
    fr.event = st.next_event;
    let mut names: Vec<String> = Vec::new();
    let mut out_zn: Vec<(u32, u32)> = Vec::new();
    let steps: [Vec<Step>; 2] = [
        if spec[0].0 + spec[0].1 > 0 { reactions::compound_decay(spec[0].0, spec[0].1, ex[0]) } else { Vec::new() },
        if spec[1].0 + spec[1].1 > 0 { reactions::compound_decay(spec[1].0, spec[1].1, ex[1]) } else { Vec::new() },
    ];
    let single = comps[a].members.len() == 1 && comps[b].members.len() == 1;
    if sim.point_nuclei && single {
        let ids_ = [comps[a].members[0], comps[b].members[0]];
        let vs = [va, vb];
        let c = (ca + cb) * 0.5;
        for (k, &(z, n)) in pieces.iter().enumerate() {
            let u = random_unit();
            let at = spot_near(sim, c, u, gen_range(2.0, 7.0), 2.0);
            let v = vel_of(&sim.cfg, us[k]);
            spawn_body(sim, fr, at, v, z, n);
            names.push(nuclide_name(z, n));
            out_zn.push((z, n));
        }
        for s in 0..2 {
            let i = ids_[s];
            let (z, n) = spec[s];
            if z + n == 0 {
                fr.dead[i] = true;
                continue;
            }
            sim.set_point(i, z, n);
            sim.vel[i] = vs[s].clamp_length_max(sim.cfg.max_speed);
            retag(sim, fr, &[i]);
            let (prods, _) = run_steps_point(sim, fr, i, &steps[s], vs[s]);
            out_zn.extend(step_products(z, n, &steps[s]));
            names.extend(prods);
        }
        comps[a].members.clear();
        comps[b].members.clear();
    } else {
        let mut pool: Vec<usize> = Vec::new();
        let mut spect: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        let dirs = [(ca - cb).normalize_or(Vec2::X), (cb - ca).normalize_or(Vec2::X)];
        for s in 0..2 {
            let members = std::mem::take(&mut comps[if s == 0 { a } else { b }].members);
            let (z, n) = spec[s];
            if z + n == 0 {
                pool.extend(members);
                continue;
            }
            match split_side(sim, &members, z, n, dirs[s]) {
                Some((keep, rest)) => {
                    spect[s] = keep;
                    pool.extend(rest);
                }
                None => pool.extend(members),
            }
        }
        let mut ps: Vec<usize> = pool.iter().copied().filter(|&i| sim.kind[i] == ids::PROTON).collect();
        let mut ns: Vec<usize> = pool.iter().copied().filter(|&i| sim.kind[i] == ids::NEUTRON).collect();
        for (k, &(z, n)) in pieces.iter().enumerate() {
            if ps.len() < z as usize || ns.len() < n as usize {
                break;
            }
            let mut m: Vec<usize> = ps.split_off(ps.len() - z as usize);
            m.extend(ns.split_off(ns.len() - n as usize));
            let v = vel_of(&sim.cfg, us[k]);
            set_vel(sim, &m, v);
            retag(sim, fr, &m);
            names.push(nuclide_name(z, n));
            out_zn.push((z, n));
        }
        let vs = [va, vb];
        for s in 0..2 {
            let m = std::mem::take(&mut spect[s]);
            if m.is_empty() {
                continue;
            }
            set_vel(sim, &m, vs[s]);
            retag(sim, fr, &m);
            let (prods, _) = run_steps(sim, fr, m, &steps[s], vs[s]);
            out_zn.extend(step_products(spec[s].0, spec[s].1, &steps[s]));
            names.extend(prods);
        }
    }
    comps[a].consumed = true;
    comps[b].consumed = true;

    let q = m_in - out_zn.iter().map(|&(z, n)| mass_excess(z, n)).sum::<f64>();
    st.energy_released += q.max(0.0);
    if q > 0.0 {
        st.q_released += q;
    } else {
        st.q_absorbed -= q;
    }
    st.breakups += 1;
    let mut names = group_products(names);
    sort_products(&mut names);
    let (big, small) = if aa >= ab { ((za, na), (zb, nb)) } else { ((zb, nb), (za, na)) };
    let e_lab = e / (aa as f64 * ab as f64 / (aa + ab) as f64);
    let note = if e_lab > FRAG_RELATIVISTIC { ", pions not modelled" } else { "" };
    st.record(
        EventKind::Breakup,
        format!(
            "{} + {} → {}  (fragmentation{note}, Q {:+.2} MeV, at {})",
            nuclide_name(big.0, big.1),
            nuclide_name(small.0, small.1),
            names.join(" + "),
            q,
            format_energy(e),
        ),
    );
    true
}

/// What a compound-decay chain on (z, n) leaves: emitted pieces, fission
/// fragments and the residue.
fn step_products(z: u32, n: u32, steps: &[Step]) -> Vec<(u32, u32)> {
    let (mut rz, mut rn) = (z, n);
    let mut out = Vec::new();
    for s in steps {
        match s {
            Step::Emit { z: x, n: y, .. } if *x <= rz && *y <= rn && x + y < rz + rn => {
                out.push((*x, *y));
                rz -= x;
                rn -= y;
            }
            Step::Fission(f) => {
                out.push(f.light);
                out.push(f.heavy);
                out.extend(std::iter::repeat_n((0, 1), f.neutrons.len()));
                return out;
            }
            _ => {}
        }
    }
    out.push((rz, rn));
    out
}

/// "n", "n", "p", "He-4", "He-4" -> "2n", "p", "2×He-4".
fn group_products(names: Vec<String>) -> Vec<String> {
    let mut counts: Vec<(String, u32)> = Vec::new();
    for s in names {
        if s.is_empty() || s == "γ" {
            continue;
        }
        match counts.iter_mut().find(|c| c.0 == s) {
            Some(c) => c.1 += 1,
            None => counts.push((s, 1)),
        }
    }
    counts
        .into_iter()
        .map(|(s, k)| match (k, s.as_str()) {
            (1, _) => s,
            (k, "n" | "p") => format!("{k}{s}"),
            (k, _) => format!("{k}×{s}"),
        })
        .collect()
}

/// Heaviest first, so the same reaction always reads the same way.
fn sort_products(p: &mut [String]) {
    // "He-3" -> 3, "n"/"p" -> 1, "2n" -> 2, "γ" -> 0.
    let mass = |s: &str| -> u32 {
        if let Some((_, a)) = s.rsplit_once('-') {
            return a.parse().unwrap_or(0);
        }
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        let k: u32 = digits.parse().unwrap_or(1);
        match &s[digits.len()..] {
            "n" | "p" => k,
            _ => 0,
        }
    };
    p.sort_by(|a, b| mass(b).cmp(&mass(a)).then_with(|| a.cmp(b)));
}

/// "25 meV", "3.4 keV", "14.1 MeV".
pub fn format_energy(mev: f64) -> String {
    if mev >= 1e6 {
        format!("{:.2} TeV", mev * 1e-6)
    } else if mev >= 1e3 {
        format!("{:.2} GeV", mev * 1e-3)
    } else if mev >= 1.0 {
        format!("{mev:.2} MeV")
    } else if mev >= 1e-3 {
        format!("{:.1} keV", mev * 1e3)
    } else if mev >= 1e-6 {
        format!("{:.1} eV", mev * 1e6)
    } else {
        format!("{:.0} meV", mev * 1e9)
    }
}

/// Carry out a decay of nucleus `members` = (z, nn). Returns a log line, or
/// None if it couldn't happen (e.g. electron capture with no electron).
#[allow(clippy::too_many_arguments)]
fn decay(
    sim: &mut Sim,
    fr: &mut Frame,
    members: &[usize],
    z: u32,
    nn: u32,
    mode: Mode,
    nuc: &nucdata::Nuclide,
    st: &mut NuclearState,
) -> Option<String> {
    if members.len() == 1 && (sim.point_nuclei || sim.zn[members[0]] != 0) {
        return decay_point(sim, fr, members[0], z, nn, mode, nuc, st);
    }
    let parent = nuclide_name(z, nn);
    let v = mean_vel(sim, members);
    let c = com(sim, members);
    let cfg = sim.cfg.clone();
    let outermost = |sim: &Sim, k: u8| {
        members
            .iter()
            .copied()
            .filter(|&i| sim.kind[i] == k)
            .max_by(|&x, &y| sim.pos[x].distance_squared(c).total_cmp(&sim.pos[y].distance_squared(c)))
    };
    // Nucleus radius, for "is an electron at the nucleus?".
    let radius = members.iter().map(|&i| sim.pos[i].distance(c)).fold(0.0, f32::max) + 4.0;

    let beta = |sim: &mut Sim, fr: &mut Frame, minus: bool, q: f64| -> Option<()> {
        let (from, to, lepton) = if minus {
            (ids::NEUTRON, ids::PROTON, ids::ELECTRON)
        } else {
            (ids::PROTON, ids::NEUTRON, ids::POSITRON)
        };
        let i = outermost(sim, from)?;
        sim.kind[i] = to;
        let dir = (sim.pos[i] - c).normalize_or(random_unit());
        let t = beta_fraction() * q.max(0.001);
        let r = sim.types[to as usize].radius + sim.types[lepton as usize].radius + 0.5;
        fr.spawn.push((sim.pos[i] + dir * r, v + dir * lepton_speed(&cfg, t), lepton));
        let a = gen_range(-1.2f32, 1.2);
        let nu_dir = vec2(-dir.x * a.cos() + dir.y * a.sin(), -dir.x * a.sin() - dir.y * a.cos());
        fr.spawn.push((sim.pos[i], nu_dir * cfg.light_speed, ids::NEUTRINO));
        Some(())
    };

    // Beta-delayed particle emission from the excited daughter.
    let delayed = |sim: &mut Sim, fr: &mut Frame, members: &[usize], probs: &[(f64, u32, u32, f64)]| -> String {
        let mut cur = members.to_vec();
        let mut extra = String::new();
        for &(p, zx, nx, e) in probs {
            if p > 0.0 && chance(p) {
                if let Some(rest) = emit(sim, fr, &cur, zx, nx, e, v) {
                    extra += &format!(" + {}", nuclide_name(zx, nx));
                    cur = rest;
                }
            }
        }
        if extra.is_empty() {
            retag(sim, fr, &cur);
        }
        extra
    };

    let q;
    let text = match mode {
        Mode::BetaMinus => {
            q = nucdata::q_beta_minus(z, nn);
            beta(sim, fr, true, q)?;
            let d = nuc.delayed;
            let extra = delayed(sim, fr, members, &[(d.n, 0, 1, 0.45), (d.n2, 0, 2, 0.6), (d.alpha_minus, 2, 2, 1.5)]);
            let what = if z + nn == 1 { "  free neutron" } else { "" };
            format!("{parent} → {} + e- + ν̄{extra}  (β-, Q {q:.3} MeV){what}", nuclide_name(z + 1, nn - 1))
        }
        Mode::BetaPlus => {
            q = nucdata::q_ec(z, nn) - 2.0 * nucdata::M_E;
            beta(sim, fr, false, q)?;
            let d = nuc.delayed;
            let extra = delayed(sim, fr, members, &[(d.p, 1, 0, 1.0), (d.alpha_plus, 2, 2, 1.5)]);
            format!("{parent} → {} + e+ + ν{extra}  (β+, Q {q:.3} MeV)", nuclide_name(z - 1, nn + 1))
        }
        Mode::ElectronCapture => {
            q = nucdata::q_ec(z, nn);
            // Needs an electron at the nucleus: bare nuclei can't do this.
            let mut best: Option<(usize, f32)> = None;
            for j in 0..sim.len() {
                if sim.kind[j] == ids::ELECTRON && !fr.dead[j] {
                    let d = sim.pos[j].distance(c);
                    if d < radius && best.is_none_or(|b| d < b.1) {
                        best = Some((j, d));
                    }
                }
            }
            // In matter, the atom's own electrons are there.
            if best.is_none() && !cfg.matter {
                return None;
            }
            let p = outermost(sim, ids::PROTON)?;
            sim.kind[p] = ids::NEUTRON;
            if let Some((e_idx, _)) = best {
                fr.dead[e_idx] = true;
            }
            fr.spawn.push((c, random_unit() * cfg.light_speed, ids::NEUTRINO));
            let d = nuc.delayed;
            let extra = delayed(sim, fr, members, &[(d.p, 1, 0, 1.0), (d.alpha_plus, 2, 2, 1.5)]);
            format!("{parent} + e- → {} + ν{extra}  (EC, Q {q:.3} MeV)", nuclide_name(z - 1, nn + 1))
        }
        Mode::DoubleBetaMinus => {
            q = mass_excess(z, nn) - mass_excess(z + 2, nn.saturating_sub(2));
            beta(sim, fr, true, q / 2.0)?;
            beta(sim, fr, true, q / 2.0)?;
            retag(sim, fr, members);
            format!("{parent} → {} + 2e- + 2ν̄  (2β-, Q {q:.3} MeV)", nuclide_name(z + 2, nn - 2))
        }
        Mode::Alpha | Mode::Neutron | Mode::TwoNeutron | Mode::Proton | Mode::TwoProton => {
            let (zx, nx) = match mode {
                Mode::Alpha => (2, 2),
                Mode::Neutron => (0, 1),
                Mode::TwoNeutron => (0, 2),
                Mode::Proton => (1, 0),
                _ => (2, 0),
            };
            if zx > z || nx > nn || zx + nx >= z + nn {
                return None;
            }
            q = -nucdata::separation(z, nn, zx, nx);
            emit(sim, fr, members, zx, nx, q.max(0.01), v)?;
            let ej = if (zx, nx) == (2, 0) { "2p".to_string() } else if (zx, nx) == (0, 2) { "2n".to_string() } else { nuclide_name(zx, nx) };
            format!("{parent} → {} + {ej}  ({}, Q {q:.2} MeV)", nuclide_name(z - zx, nn - nx), mode.name())
        }
        Mode::Fission => {
            let split = reactions::sample_fission(z, nn, 0.0);
            let desc = fission(sim, fr, members, &split, v)?;
            q = mass_excess(z, nn)
                - mass_excess(split.light.0, split.light.1)
                - mass_excess(split.heavy.0, split.heavy.1)
                - split.neutrons.len() as f64 * nucdata::DELTA_N;
            format!("{parent} → {desc}  (spontaneous fission, Q {q:.0} MeV)")
        }
    };
    st.energy_released += q.max(0.0);
    st.decay_heat += crate::inventory::deposited(mode, if mode == Mode::Fission { q } else { q.max(0.0) });
    Some(text)
}

/// Lay out a nucleus of z protons and n neutrons as a hexagonally packed
/// disc centred on `at` (protons spread evenly through it). Returns false
/// (adding nothing) if the space is occupied.
pub fn place_nucleus(sim: &mut Sim, at: Vec2, z: u32, n: u32, vel: Vec2) -> bool {
    let fast = energy_of(&sim.cfg, vel.length(), 1.0) > 1.0;
    match place_nucleus_tag(sim, at, z, n, vel) {
        Some(t) => {
            // Launched fast (over 1 MeV per nucleon): a genuine fast
            // particle, like a beam's (see `NuclearState::beam`).
            if fast {
                sim.nuclear.beam.insert(t);
            }
            true
        }
        None => false,
    }
}

/// `place_nucleus`, returning the new nucleus's tag.
pub fn place_nucleus_tag(sim: &mut Sim, at: Vec2, z: u32, n: u32, vel: Vec2) -> Option<u32> {
    let first = sim.len();
    (place_nucleus_inner(sim, at, z, n, vel) && sim.len() > first).then(|| sim.tag[first])
}

fn place_nucleus_inner(sim: &mut Sim, at: Vec2, z: u32, n: u32, vel: Vec2) -> bool {
    let a = (z + n) as usize;
    if a == 0 {
        return false;
    }
    if sim.point_nuclei && a >= 2 {
        return place_point(sim, at, z, n, vel);
    }
    let r = sim.types[ids::NEUTRON as usize].radius.max(sim.types[ids::PROTON as usize].radius);
    let d = 2.0 * r;
    let mut spots: Vec<Vec2> = Vec::new();
    let rings = ((a as f32).sqrt() as i32) + 3;
    for q in -rings..=rings {
        for s in -rings..=rings {
            let p = vec2(d * (q as f32 + 0.5 * s as f32), d * 0.866_025_4 * s as f32);
            spots.push(p);
        }
    }
    spots.sort_by(|p, q| p.length_squared().total_cmp(&q.length_squared()));
    spots.truncate(a);
    let pts: Vec<Vec2> = spots.iter().map(|p| at + *p).collect();
    // Room check against everything that collides.
    let reach = pts.iter().map(|p| p.distance(at)).fold(0.0, f32::max) + 3.0 * r;
    for (i, q) in sim.pos.iter().enumerate() {
        let t = &sim.types[sim.kind[i] as usize];
        if !t.collides || q.distance(at) > reach + t.radius {
            continue;
        }
        if pts.iter().any(|p| p.distance(*q) < r + t.radius) {
            return false;
        }
    }
    if pts.iter().any(|p| p.x < r || p.y < r || p.x > sim.width - r || p.y > sim.height - r || sim.walls.blocked(*p, r)) {
        return false;
    }
    if in_rods(sim, &pts, r) {
        return false; // the rods would shove it out violently
    }
    // Spread protons evenly: Bresenham-style interleave from the centre out.
    let tag = sim.new_tag();
    let mut placed_p = 0u32;
    for (k, p) in pts.iter().enumerate() {
        let want = ((k + 1) as u64 * z as u64).div_ceil(a as u64) as u32;
        let kind = if placed_p < want { ids::PROTON } else { ids::NEUTRON };
        if kind == ids::PROTON {
            placed_p += 1;
        }
        sim.add(*p, vel, kind);
        let last = sim.len() - 1;
        sim.tag[last] = tag;
    }
    true
}

// ---------------------------------------------------------------------------
// Point-nucleus mode: every nucleus is one particle (kind NUCLEUS, its Z and
// N in `sim.zn`). Reactions and decays change that particle and add the
// ejectiles as new particles. Free neutrons don't collide: each time one
// passes within the interaction radius of a nucleus (from the total
// cross-section) it scatters or is absorbed, with the real branching.
// ---------------------------------------------------------------------------

/// Add a body (neutron, proton or point nucleus) during a frame: a product
/// of event `fr.event`. Returns its index.
fn spawn_body(sim: &mut Sim, fr: &mut Frame, p: Vec2, v: Vec2, z: u32, n: u32) -> usize {
    let v = v.clamp_length_max(sim.cfg.max_speed);
    let t = sim.add_point(p, v, z, n);
    fr.dead.push(false);
    fr.new_grace.push((t, fr.event));
    sim.len() - 1
}

/// A spot `dist` px from `c` along `u`, inside the box and out of walls
/// (else `c` itself).
fn spot_near(sim: &Sim, c: Vec2, u: Vec2, dist: f32, r: f32) -> Vec2 {
    let p = c + u * dist;
    if p.x > r && p.y > r && p.x < sim.width - r && p.y < sim.height - r && !sim.walls.blocked(p, r) {
        p
    } else {
        c
    }
}

fn body_radius(sim: &Sim, z: u32, n: u32) -> f32 {
    let k = match (z, n) {
        (0, 1) => ids::NEUTRON,
        (1, 0) => ids::PROTON,
        _ => ids::NUCLEUS,
    };
    sim.types[k as usize].radius
}

/// Two-body breakup of point nucleus i: (zx, nx) leaves with relative
/// kinetic energy `e_mev`, momentum conserved about `v`.
fn emit_point(sim: &mut Sim, fr: &mut Frame, i: usize, zx: u32, nx: u32, e_mev: f64, v: Vec2) -> Option<()> {
    let (z, n) = sim.zn_of(i);
    if zx > z || nx > n || zx + nx >= z + n {
        return None;
    }
    let (mx, mr) = ((zx + nx) as f32, (z + n - zx - nx) as f32);
    let dir = random_unit();
    let (vx, vr) = split_vel(&sim.cfg, v, dir, e_mev, mx, mr);
    sim.set_point(i, z - zx, n - nx);
    sim.vel[i] = vr;
    let gap = body_radius(sim, zx, nx) + sim.types[sim.kind[i] as usize].radius + 0.5;
    let at = spot_near(sim, sim.pos[i], dir, gap, 2.0);
    spawn_body(sim, fr, at, vx, zx, nx);
    retag(sim, fr, &[i]);
    Some(())
}

/// Split point nucleus i into fission fragments and prompt neutrons.
fn fission_point(sim: &mut Sim, fr: &mut Frame, i: usize, split: &reactions::FissionSplit, v: Vec2) -> Option<String> {
    let ((zl, nl), (zh, nh)) = (split.light, split.heavy);
    let nu = split.neutrons.len() as u32;
    let (z, n) = sim.zn_of(i);
    if zl + zh != z || nl + nh + nu != n {
        return None;
    }
    let axis = random_unit();
    let c = sim.pos[i];
    let (vl, vh) = split_vel(&sim.cfg, v, axis, split.tke, (zl + nl) as f32, (zh + nh) as f32);
    let r = POINT_RADIUS + 0.5;
    sim.set_point(i, zl, nl);
    sim.pos[i] = spot_near(sim, c, axis, r, r);
    sim.vel[i] = vl;
    retag(sim, fr, &[i]);
    let at = spot_near(sim, c, -axis, r, r);
    spawn_body(sim, fr, at, vh, zh, nh);
    for &e in &split.neutrons {
        let u = random_unit();
        let at = spot_near(sim, c, u, 2.0 * r, 2.0);
        spawn_body(sim, fr, at, boosted(&sim.cfg, v, u, e), 0, 1);
    }
    for _ in 0..3 {
        photon(sim, fr, c);
    }
    let neutrons = match nu {
        0 => String::new(),
        1 => " + n".to_string(),
        k => format!(" + {k}n"),
    };
    Some(format!("{} + {}{neutrons}", nuclide_name(zl, nl), nuclide_name(zh, nh)))
}

/// `run_steps` for a point nucleus. Returns the product names and the
/// residue (empty if it fissioned).
fn run_steps_point(sim: &mut Sim, fr: &mut Frame, i: usize, steps: &[Step], v: Vec2) -> (Vec<String>, Vec<usize>) {
    let mut vcur = v;
    let mut products = Vec::new();
    let mut emitted_any = false;
    for s in steps {
        match s {
            Step::Emit { z, n, e } => {
                if emit_point(sim, fr, i, *z, *n, *e, vcur).is_some() {
                    products.push(nuclide_name(*z, *n));
                    vcur = sim.vel[i];
                    emitted_any = true;
                }
            }
            Step::Gamma { .. } => {
                photon(sim, fr, sim.pos[i]);
                products.push("γ".into());
            }
            Step::Fission(split) => {
                if let Some(desc) = fission_point(sim, fr, i, split, vcur) {
                    products.push(desc);
                    return (products, Vec::new());
                }
            }
        }
    }
    if !emitted_any {
        sim.vel[i] = vcur.clamp_length_max(sim.cfg.max_speed);
    }
    let (z, n) = sim.zn_of(i);
    products.insert(0, nuclide_name(z, n));
    (products, vec![i])
}

/// `decay` for a point nucleus (or a free neutron in point mode).
#[allow(clippy::too_many_arguments)]
fn decay_point(
    sim: &mut Sim,
    fr: &mut Frame,
    i: usize,
    z: u32,
    nn: u32,
    mode: Mode,
    nuc: &nucdata::Nuclide,
    st: &mut NuclearState,
) -> Option<String> {
    let parent = nuclide_name(z, nn);
    let v = sim.vel[i];
    let c = sim.pos[i];
    let cfg = sim.cfg.clone();
    let r = sim.types[sim.kind[i] as usize].radius;
    let lepton_out = |fr: &mut Frame, lepton: u8, q: f64| {
        let dir = random_unit();
        let t = beta_fraction() * q.max(0.001);
        fr.spawn.push((c + dir * (r + 1.5), v + dir * lepton_speed(&cfg, t), lepton));
        let a = gen_range(-1.2f32, 1.2);
        let nu_dir = vec2(-dir.x * a.cos() + dir.y * a.sin(), -dir.x * a.sin() - dir.y * a.cos());
        fr.spawn.push((c, nu_dir * cfg.light_speed, ids::NEUTRINO));
    };
    // Beta-delayed particle emission from the excited daughter.
    let delayed = |sim: &mut Sim, fr: &mut Frame, probs: &[(f64, u32, u32, f64)]| -> String {
        let mut extra = String::new();
        for &(p, zx, nx, e) in probs {
            if p > 0.0 && chance(p) && emit_point(sim, fr, i, zx, nx, e, v).is_some() {
                extra += &format!(" + {}", nuclide_name(zx, nx));
            }
        }
        if extra.is_empty() {
            retag(sim, fr, &[i]);
        }
        extra
    };
    let q;
    let text = match mode {
        Mode::BetaMinus => {
            if nn == 0 {
                return None;
            }
            q = nucdata::q_beta_minus(z, nn);
            sim.set_point(i, z + 1, nn - 1);
            lepton_out(fr, ids::ELECTRON, q);
            let d = nuc.delayed;
            let extra = delayed(sim, fr, &[(d.n, 0, 1, 0.45), (d.n2, 0, 2, 0.6), (d.alpha_minus, 2, 2, 1.5)]);
            let what = if z + nn == 1 { "  free neutron" } else { "" };
            format!("{parent} → {} + e- + ν̄{extra}  (β-, Q {q:.3} MeV){what}", nuclide_name(z + 1, nn - 1))
        }
        Mode::BetaPlus => {
            if z == 0 {
                return None;
            }
            q = nucdata::q_ec(z, nn) - 2.0 * nucdata::M_E;
            sim.set_point(i, z - 1, nn + 1);
            lepton_out(fr, ids::POSITRON, q);
            let d = nuc.delayed;
            let extra = delayed(sim, fr, &[(d.p, 1, 0, 1.0), (d.alpha_plus, 2, 2, 1.5)]);
            format!("{parent} → {} + e+ + ν{extra}  (β+, Q {q:.3} MeV)", nuclide_name(z - 1, nn + 1))
        }
        Mode::ElectronCapture => {
            if z == 0 {
                return None;
            }
            q = nucdata::q_ec(z, nn);
            let mut best: Option<(usize, f32)> = None;
            for j in 0..sim.len() {
                if sim.kind[j] == ids::ELECTRON && !fr.dead[j] {
                    let d = sim.pos[j].distance(c);
                    if d < r + 4.0 && best.is_none_or(|b| d < b.1) {
                        best = Some((j, d));
                    }
                }
            }
            if best.is_none() && !cfg.matter {
                return None;
            }
            if let Some((e_idx, _)) = best {
                fr.dead[e_idx] = true;
            }
            sim.set_point(i, z - 1, nn + 1);
            fr.spawn.push((c, random_unit() * cfg.light_speed, ids::NEUTRINO));
            let d = nuc.delayed;
            let extra = delayed(sim, fr, &[(d.p, 1, 0, 1.0), (d.alpha_plus, 2, 2, 1.5)]);
            format!("{parent} + e- → {} + ν{extra}  (EC, Q {q:.3} MeV)", nuclide_name(z - 1, nn + 1))
        }
        Mode::DoubleBetaMinus => {
            if nn < 2 {
                return None;
            }
            q = mass_excess(z, nn) - mass_excess(z + 2, nn - 2);
            sim.set_point(i, z + 2, nn - 2);
            lepton_out(fr, ids::ELECTRON, q / 2.0);
            lepton_out(fr, ids::ELECTRON, q / 2.0);
            retag(sim, fr, &[i]);
            format!("{parent} → {} + 2e- + 2ν̄  (2β-, Q {q:.3} MeV)", nuclide_name(z + 2, nn - 2))
        }
        Mode::Alpha | Mode::Neutron | Mode::TwoNeutron | Mode::Proton | Mode::TwoProton => {
            let (zx, nx) = match mode {
                Mode::Alpha => (2, 2),
                Mode::Neutron => (0, 1),
                Mode::TwoNeutron => (0, 2),
                Mode::Proton => (1, 0),
                _ => (2, 0),
            };
            q = -nucdata::separation(z, nn, zx, nx);
            if matches!(mode, Mode::TwoNeutron | Mode::TwoProton) {
                // Two separate nucleons, not a bound pair.
                let (one_z, one_n) = (zx / 2, nx / 2);
                emit_point(sim, fr, i, one_z, one_n, (q / 2.0).max(0.01), v)?;
                let v2 = sim.vel[i];
                emit_point(sim, fr, i, one_z, one_n, (q / 2.0).max(0.01), v2)?;
            } else {
                emit_point(sim, fr, i, zx, nx, q.max(0.01), v)?;
            }
            let ej = if (zx, nx) == (2, 0) { "2p".to_string() } else if (zx, nx) == (0, 2) { "2n".to_string() } else { nuclide_name(zx, nx) };
            format!("{parent} → {} + {ej}  ({}, Q {q:.2} MeV)", nuclide_name(z - zx, nn - nx), mode.name())
        }
        Mode::Fission => {
            let split = reactions::sample_fission(z, nn, 0.0);
            let desc = fission_point(sim, fr, i, &split, v)?;
            q = mass_excess(z, nn)
                - mass_excess(split.light.0, split.light.1)
                - mass_excess(split.heavy.0, split.heavy.1)
                - split.neutrons.len() as f64 * nucdata::DELTA_N;
            format!("{parent} → {desc}  (spontaneous fission, Q {q:.0} MeV)")
        }
    };
    st.energy_released += q.max(0.0);
    st.decay_heat += crate::inventory::deposited(mode, if mode == Mode::Fission { q } else { q.max(0.0) });
    Some(text)
}

/// Interaction radius (px) of a free neutron with nucleus (z, n) at
/// centre-of-mass energy `e` (point mode).
pub fn interaction_radius(cfg: &Config, z: u32, n: u32, e: f64) -> f32 {
    let (abs, scat) = reactions::neutron_xs(z, n, e);
    (cfg.xs_scale * ((abs + scat).max(0.0) as f32).powf(cfg.xs_exponent)).min(POINT_XS_MAX)
}

/// Collision energy (MeV, centre of mass) of masses ma, mb at sim
/// velocities va, vb.
fn cm_energy(cfg: &Config, va: Vec2, ma: f32, vb: Vec2, mb: f32) -> f64 {
    let (ua, ub) = (real_velocity(cfg, va), real_velocity(cfg, vb));
    let mu = (ma * mb / (ma + mb)) as f64;
    mu * ((ua.0 - ub.0).powi(2) + (ua.1 - ub.1).powi(2))
}

/// Sim velocity of something with real velocity u (sqrt(MeV/u) units).
fn sim_velocity(cfg: &Config, u: (f64, f64)) -> Vec2 {
    let s2 = u.0 * u.0 + u.1 * u.1;
    if s2 < 1e-30 {
        return Vec2::ZERO;
    }
    let speed = speed_for(cfg, s2, 1.0).min(cfg.max_speed);
    let k = speed as f64 / s2.sqrt();
    vec2((u.0 * k) as f32, (u.1 * k) as f32)
}

/// Elastic neutron scattering, isotropic in the centre-of-mass frame (s
/// wave), done with real velocities so the energy lost per collision is
/// the real one (half on average on hydrogen, ~1/7 on carbon...).
/// Returns the recoil energy (MeV).
fn scatter(sim: &mut Sim, ni: usize, members: &[usize], mass: f32) -> f64 {
    let cfg = &sim.cfg;
    let vb = mean_vel(sim, members);
    let (un, ub) = (real_velocity(cfg, sim.vel[ni]), real_velocity(cfg, vb));
    let m = mass as f64;
    let cm = ((un.0 + m * ub.0) / (1.0 + m), (un.1 + m * ub.1) / (1.0 + m));
    let rel = ((un.0 - cm.0).powi(2) + (un.1 - cm.1).powi(2)).sqrt();
    let d = random_unit();
    let (dx, dy) = (d.x as f64 * rel, d.y as f64 * rel);
    let vn = sim_velocity(cfg, (cm.0 + dx, cm.1 + dy));
    let vt = sim_velocity(cfg, (cm.0 - dx / m, cm.1 - dy / m));
    sim.vel[ni] = vn;
    for &j in members {
        sim.vel[j] = vt;
    }
    let ut = (cm.0 - dx / m, cm.1 - dy / m);
    m * (ut.0 * ut.0 + ut.1 * ut.1)
}

/// A knock-on nucleus above this energy (MeV) is a fast ion: it is slowed
/// down by the matter like a fission fragment, depositing only
/// `heat_fraction` of its energy as heat (see thermal.rs).
const RECOIL_FAST: f64 = 1e-6;

/// Free neutrons meeting nuclei (point mode): one interaction each time a
/// neutron's path passes within the interaction radius of a nucleus.
fn neutron_encounters(sim: &mut Sim, st: &mut NuclearState, fr: &mut Frame, comps: &mut [Comp], comp_of: &[u32], n: usize, dt: f32) {
    let reach = (POINT_XS_MAX * sim.grid.inv).ceil() as i32;
    let mut near_now: HashSet<(u32, u32)> = HashSet::default();
    for i in 0..n {
        if sim.kind[i] != ids::NEUTRON || fr.dead[i] {
            continue;
        }
        let ci = comp_of[i] as usize;
        if ci >= comps.len() || comps[ci].consumed || comps[ci].members.len() != 1 {
            continue;
        }
        let p1 = sim.pos[i];
        let p0 = p1 - sim.vel[i] * dt;
        let seg = p1 - p0;
        let l2 = seg.length_squared();
        let vi = sim.vel[i];
        // Nearest new encounter on this frame's path.
        let mut best: Option<(usize, f32)> = None;
        let mut found: Vec<(u32, u32)> = Vec::new();
        sim.grid.for_each_near(p1, reach, |j| {
            if j >= n || j == i || sim.kind[j] == ids::NEUTRON || fr.dead[j] || !sim.is_nuclear(j) {
                return;
            }
            let cj = comp_of[j] as usize;
            if cj >= comps.len() || comps[cj].consumed {
                return;
            }
            let q = sim.pos[j];
            let t = if l2 > 1e-12 { ((q - p0).dot(seg) / l2).clamp(0.0, 1.0) } else { 1.0 };
            let d = (p0 + seg * t).distance(q);
            if d > POINT_XS_MAX {
                return;
            }
            let (z, nn) = (comps[cj].z, comps[cj].n);
            let a = (z + nn) as f32;
            let e = cm_energy(&sim.cfg, vi, 1.0, sim.vel[j], a);
            if d < interaction_radius(&sim.cfg, z, nn, e) {
                let key = (sim.tag[i], comps[cj].tag);
                found.push(key);
                if !st.near.contains(&key) && best.is_none_or(|b| d < b.1) {
                    best = Some((cj, d));
                }
            }
        });
        near_now.extend(found);
        let Some((cj, _)) = best else { continue };
        if st.grace.get(&sim.tag[i]).zip(st.grace.get(&comps[cj].tag)).is_some_and(|(a, b)| a.0 == b.0) {
            continue; // born in the same event
        }
        if !react(sim, st, fr, comps, ci, cj) {
            let members = comps[cj].members.clone();
            let a = (comps[cj].z + comps[cj].n) as f32;
            if scatter(sim, i, &members, a) > RECOIL_FAST {
                st.fast.insert(comps[cj].tag);
            }
        }
    }
    st.near = near_now;
}

/// `place_nucleus` in point mode.
fn place_point(sim: &mut Sim, at: Vec2, z: u32, n: u32, vel: Vec2) -> bool {
    let r = POINT_RADIUS;
    if at.x < r || at.y < r || at.x > sim.width - r || at.y > sim.height - r || sim.walls.blocked(at, r) || in_rods(sim, &[at], r) {
        return false;
    }
    for (i, q) in sim.pos.iter().enumerate() {
        let k = sim.kind[i];
        let t = &sim.types[k as usize];
        if !t.collides || k == ids::NEUTRON {
            continue;
        }
        if q.distance(at) < r + t.radius {
            return false;
        }
    }
    sim.add_point(at, vel, z, n);
    true
}

/// Switch between nucleon clusters and point nuclei, converting what is in
/// the scene. Returns how many nuclei didn't fit (going back to clusters,
/// which are much bigger) and were left out.
pub fn set_point_mode(sim: &mut Sim, on: bool) -> u32 {
    if sim.point_nuclei == on {
        return 0;
    }
    let mut lost = 0;
    if on {
        let bodies = sim.bodies();
        let big: HashMap<u32, crate::sim::BodyInfo> = bodies.into_iter().filter(|(_, b)| b.parts > 1).collect();
        let (kind, tag) = (sim.kind.clone(), sim.tag.clone());
        sim.retain(|i| !(ParticleType::is_nucleon(kind[i]) && big.contains_key(&tag[i])));
        sim.point_nuclei = true;
        for b in big.values() {
            sim.add_point(b.pos, b.vel, b.z, b.n);
        }
    } else {
        let pts: Vec<(Vec2, Vec2, u32, u32)> =
            (0..sim.len()).filter(|&i| sim.zn[i] != 0).map(|i| (sim.pos[i], sim.vel[i], sim.zn_of(i).0, sim.zn_of(i).1)).collect();
        let zn = sim.zn.clone();
        sim.retain(|i| zn[i] == 0);
        sim.point_nuclei = false;
        // Heaviest first: they need the most room.
        let mut pts = pts;
        pts.sort_by(|a, b| (b.2 + b.3).cmp(&(a.2 + a.3)));
        for (p, v, z, n) in pts {
            if !place_nucleus(sim, p, z, n, v) {
                lost += 1;
            }
        }
    }
    sim.nuclear.touching.clear();
    sim.nuclear.near.clear();
    sim.index_nuclei();
    lost
}

/// Would particles of radius r at `pts` overlap an inserted control rod?
fn in_rods(sim: &Sim, pts: &[Vec2], r: f32) -> bool {
    if sim.reactor.rods.is_empty() {
        return false;
    }
    let rects = sim.reactor.rects();
    pts.iter().any(|p| rects.iter().any(|c| p.x + r > c.x && p.x - r < c.x + c.w && p.y + r > c.y && p.y - r < c.y + c.h))
}
