//! Heat: stopping of fast ions, the ambient heat bath, thermal-conductor
//! walls, freeze plugs and pumps. Runs once per frame, on whole nuclei.
//!
//! Temperature in the sim is the mean kinetic energy per body, `theta` (sim
//! units, the same for every mass: elastic collisions share energy that
//! way). A real temperature T corresponds to the theta of a nucleon whose
//! (compressed) speed carries kT: theta(T) = 1/2 speed_for(kT, 1)^2.
//!
//! * Fast ions (fission fragments, alphas, recoils) lose speed to the
//!   electrons of the matter they cross ("stopping"); `heat_fraction` of
//!   what they lose heats the atoms around them.
//! * `cooling` couples charged matter to a bath at `temperature`; at 0 the
//!   matter is adiabatic and heat only moves by collisions, walls and flow.
//! * A conductor wall swaps energy with whatever touches it (on either
//!   side) and conducts heat to connected conductor walls: a heat exchanger.
//! * Freeze plugs are conductor walls that melt when hot.
//! * Pumps push matter (not neutrons) through their area.

use crate::nuclear::{energy_of, speed_for};
use crate::sim::Sim;
use crate::types::ids;
use crate::walls::WallKind;
use macroquad::math::{Vec2, vec2};
use crate::rng::gen_range;
use crate::rng::{HashMap, HashSet};

const K_B: f64 = 8.617e-11; // MeV/K

/// Sim temperature (mean KE per body) for real temperature T (K).
pub fn theta_of(cfg: &crate::sim::Config, t_kelvin: f32) -> f32 {
    let v = speed_for(cfg, K_B * t_kelvin.max(0.0) as f64, 1.0);
    0.5 * v * v
}

/// Real temperature (K) for sim temperature theta.
pub fn kelvin_of(cfg: &crate::sim::Config, theta: f32) -> f64 {
    energy_of(cfg, (2.0 * theta.max(0.0)).sqrt(), 1.0) / K_B
}

/// "293 K", "1.4 kK" -> "1400 K", "3.2 MK".
pub fn format_kelvin(t: f64) -> String {
    if t >= 1e9 {
        format!("{:.1} GK", t * 1e-9)
    } else if t >= 1e6 {
        format!("{:.1} MK", t * 1e-6)
    } else if t >= 1e4 {
        format!("{:.0} kK", t * 1e-3)
    } else {
        format!("{t:.0} K")
    }
}

/// Matter density (nucleons per 1000 px^2 at the game's nucleon radius 2)
/// above which fast ions get the full `stopping` rate; less matter around
/// them stops them proportionally less. Dense presets sit at 8-11.
pub const DENSE_MATTER: f32 = 4.0;
/// Radius (in nucleon radii) over which that density is measured.
pub const DENSITY_REACH: f32 = 20.0;

/// Heat capacity of a conducting wall per px of length, in bodies.
pub const WALL_CAPACITY: f32 = 0.25;
/// A freeze plug is a thick block of frozen salt, not a thin sheet.
pub const PLUG_CAPACITY: f32 = 20.0 * WALL_CAPACITY;

fn capacity(s: &crate::walls::Seg) -> f32 {
    let per_px = if s.kind == WallKind::FreezePlug { PLUG_CAPACITY } else { WALL_CAPACITY };
    per_px * (s.b - s.a).length().max(1.0)
}
/// Conduction rate between connected conductor walls (1/s).
pub const WALL_CONDUCTION: f32 = 3.0;
/// Share of the way to the wall's temperature a body goes on an impact.
pub const ACCOMMODATION: f32 = 0.5;
/// Relaxation rate (1/s) of a body resting against a conductor wall.
pub const CONTACT_RATE: f32 = 4.0;

#[derive(Clone, Copy, Default)]
struct Body {
    tag: u32,
    mass: f32,
    com: Vec2,
    vel: Vec2,
    z: u32,
    a: u32,
    parts: u32,
}

fn st_fast_contains(st: &crate::nuclear::NuclearState, tag: u32) -> bool {
    st.fast.contains(&tag)
}

fn gauss() -> Vec2 {
    let u1: f32 = 1.0 - gen_range(0.0f32, 1.0);
    let u2: f32 = gen_range(0.0f32, 1.0);
    let r = (-2.0 * u1.ln()).sqrt();
    let a = std::f32::consts::TAU * u2;
    vec2(r * a.cos(), r * a.sin())
}

fn random_unit() -> Vec2 {
    let a = gen_range(0.0, std::f32::consts::TAU);
    vec2(a.cos(), a.sin())
}

pub fn apply(sim: &mut Sim, dt: f32) {
    let n = sim.len();
    let m = sim.n_nuclei;
    let cfg = sim.cfg.clone();
    let r_n = sim.types[ids::NEUTRON as usize].radius;

    // ---- Bodies ----
    let mut b = vec![Body::default(); m];
    for i in 0..n {
        let k = sim.nuc[i] as usize;
        if k >= m {
            continue;
        }
        let x = &mut b[k];
        x.tag = sim.tag[i];
        x.mass += sim.mass_of(i);
        x.com += sim.pos[i];
        x.vel += sim.vel[i];
        let (z, nn) = sim.zn_of(i);
        x.a += z + nn;
        x.z += z;
        x.parts += 1;
    }
    for x in b.iter_mut() {
        if x.parts > 0 {
            x.com /= x.parts as f32;
            x.vel /= x.parts as f32;
        }
    }
    let mut dv = vec![Vec2::ZERO; m];
    let theta_amb = theta_of(&cfg, cfg.temperature);

    // ---- Rigid nuclei: no internal heat ----
    // Nuclei have no internal degrees of freedom at these energies, but in
    // the particle model collisions leave a little wobble inside them,
    // which internal damping would then remove: heat would leak away. Fold
    // the internal kinetic energy back into the motion of the whole.
    let mut e_int = vec![0.0f32; m];
    for i in 0..n {
        let k = sim.nuc[i] as usize;
        if k < m && b[k].parts > 1 {
            let u = sim.vel[i] - b[k].vel;
            e_int[k] += 0.5 * sim.types[sim.kind[i] as usize].mass * u.length_squared();
        }
    }
    // ---- Energy the contact solver lost ----
    // The particle solver's rigid-body contacts bleed a little energy
    // (simultaneous contacts, overlap resolution). Give it back to the
    // nuclei that collide, in proportion to their energy, so heat is
    // conserved in adiabatic matter.
    if let Some(before) = sim.ke_before.take() {
        // What the internal wobble holds is folded back below: count it.
        let now: f32 = b.iter().filter(|x| x.a > 1).map(|x| 0.5 * x.mass * x.vel.length_squared()).sum::<f32>() + e_int.iter().sum::<f32>();
        // Both ways: correcting only losses would ratchet the solver's
        // frame-to-frame noise into heating.
        if now > 1e-6 && before > 1e-6 {
            let s = (before / now).sqrt().clamp(0.975, 1.025);
            for (k, x) in b.iter().enumerate() {
                if x.a > 1 {
                    dv[k] += x.vel * (s - 1.0);
                }
            }
        }
    }

    for k in 0..m {
        let x = &b[k];
        if x.parts > 1 && e_int[k] > 0.0 && !st_fast_contains(&sim.nuclear, x.tag) {
            let e_com = 0.5 * x.mass * x.vel.length_squared();
            let v2 = if e_com > 1e-9 { x.vel * ((e_com + e_int[k]) / e_com).sqrt() } else { random_unit() * (2.0 * e_int[k] / x.mass).sqrt() };
            dv[k] += v2 - x.vel;
        }
    }
    sim.walls.ambient_theta = theta_amb;
    let mut st = std::mem::take(&mut sim.nuclear);

    // ---- Matter: stopping of fast ions, heat bath ----
    if cfg.matter {
        // Stopping power is proportional to the density of the matter an
        // ion crosses: full in dense matter, none in vacuum (an alpha that
        // escaped the vessel flies on). Matter behind a wall doesn't count.
        let dens_reach = DENSITY_REACH * r_n;
        let dens_cell = dens_reach;
        let mut dens_grid: HashMap<(i32, i32), Vec<usize>> = HashMap::default();
        for (k, x) in b.iter().enumerate() {
            if x.a > 0 && !(x.z == 0 && x.a == 1) {
                dens_grid.entry(((x.com.x / dens_cell).floor() as i32, (x.com.y / dens_cell).floor() as i32)).or_default().push(k);
            }
        }
        let walls = &sim.walls;
        let matter_around = |k: usize| -> f32 {
            let p = b[k].com;
            let (cx, cy) = ((p.x / dens_cell).floor() as i32, (p.y / dens_cell).floor() as i32);
            let mut a = 0u32;
            for gy in cy - 1..=cy + 1 {
                for gx in cx - 1..=cx + 1 {
                    for &j in dens_grid.get(&(gx, gy)).map_or(&[][..], |v| &v[..]) {
                        if j != k && b[j].com.distance_squared(p) < dens_reach * dens_reach && !walls.separates(p, b[j].com) {
                            a += b[j].a;
                        }
                    }
                }
            }
            let per_1000 = a as f32 / (std::f32::consts::PI * dens_reach * dens_reach) * 1000.0 * (2.0 / r_n).powi(2);
            (per_1000 / DENSE_MATTER).min(1.0)
        };
        let mut heat: Vec<(Vec2, f32, usize)> = Vec::new();
        let stop = (-cfg.stopping * dt).exp();
        let c = (-cfg.cooling * dt).exp();
        let kick = (1.0 - c * c).max(0.0).sqrt();
        for (k, x) in b.iter().enumerate() {
            if x.a == 0 || x.z == 0 {
                continue; // free neutrons (and neutron clusters) don't stop
            }
            let mut v = x.vel;
            // Equipartition: every body has mean KE theta (2 dof).
            let v_th = (2.0 * theta_amb / x.mass).sqrt();
            let speed = v.length();
            // Fast: born in a nuclear event, or above ~100 keV per nucleon.
            let v_fast = speed_for(&cfg, 0.1 * x.a as f64, x.a as f32);
            let fast = st.fast.contains(&x.tag) || speed > v_fast;
            if fast && cfg.stopping > 0.0 {
                let f = matter_around(k);
                let s2 = speed * if f >= 1.0 { stop } else { (-cfg.stopping * f * dt).exp() };
                let new_v = if speed > 1e-6 { v * (s2 / speed) } else { v };
                let lost = 0.5 * x.mass * (speed * speed - s2 * s2);
                v = new_v;
                if lost > 0.0 && cfg.heat_fraction > 0.0 {
                    heat.push((x.com, lost * cfg.heat_fraction, k));
                }
                if s2 < 2.0 * v_th.max(1.0) {
                    st.fast.remove(&x.tag);
                }
            } else if !fast {
                st.fast.remove(&x.tag);
            }
            if cfg.cooling > 0.0 {
                v = v * c + gauss() * (v_th * std::f32::consts::FRAC_1_SQRT_2 * kick);
            }
            dv[k] += v - x.vel;
        }
        // Electrons and positrons: Langevin towards the bath (they are
        // absorbed or annihilate once slow).
        let stop_e = (-cfg.stopping.max(cfg.cooling) * dt).exp();
        let kick_e = (1.0 - stop_e * stop_e).max(0.0).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
        let kt = K_B * cfg.temperature as f64;
        for i in 0..n {
            let kd = sim.kind[i];
            if kd == ids::ELECTRON || kd == ids::POSITRON {
                let me = sim.types[kd as usize].mass;
                let sigma = speed_for(&cfg, kt, me) * kick_e;
                sim.vel[i] = sim.vel[i] * stop_e + gauss() * sigma;
            }
        }

        // Deposit the stopping heat in the atoms around each fast ion.
        if !heat.is_empty() {
            let cell = 16.0 * r_n;
            let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::default();
            for (k, x) in b.iter().enumerate() {
                if x.a > 0 && !(x.z == 0 && x.a == 1) {
                    grid.entry(((x.com.x / cell) as i32, (x.com.y / cell) as i32)).or_default().push(k);
                }
            }
            let reach = 12.0 * r_n;
            for (p, e, src) in heat {
                let (cx, cy) = ((p.x / cell) as i32, (p.y / cell) as i32);
                let mut near = Vec::new();
                for gy in cy - 1..=cy + 1 {
                    for gx in cx - 1..=cx + 1 {
                        if let Some(l) = grid.get(&(gx, gy)) {
                            near.extend(l.iter().copied().filter(|&k| {
                                k != src
                                    && b[k].com.distance(p) < reach + r_n * (b[k].a as f32).sqrt()
                                    // Not through a wall: an ion slowing
                                    // down inside a vessel heats what is
                                    // inside it, not what is outside.
                                    && !sim.walls.separates(p, b[k].com)
                            }));
                        }
                    }
                }
                if near.is_empty() {
                    continue; // nothing around: the energy leaves as radiation
                }
                let share = e / near.len() as f32;
                for k in near {
                    dv[k] += random_unit() * (2.0 * share / b[k].mass).sqrt();
                }
            }
        }
    }

    // ---- Pumps ----
    for pump in &sim.reactor.pumps {
        let dir = pump.dir.normalize_or_zero();
        for (k, x) in b.iter().enumerate() {
            if x.a == 0 || (x.z == 0 && x.a == 1) || !pump.rect.contains(x.com) {
                continue;
            }
            let along = (x.vel + dv[k]).dot(dir);
            if along < pump.speed {
                dv[k] += dir * (pump.speed - along).min(4.0 * pump.speed * dt);
            }
        }
    }

    // ---- Conductor walls ----
    let walls = &mut sim.walls;
    if walls.segs.iter().any(|s| s.kind.conducts()) {
        // Local flow (mass-weighted mean velocity per 40 px cell): heat is
        // motion relative to it, so a pumped stream turning at a wall isn't
        // read as hot.
        let cell = 40.0;
        let mut flow: HashMap<(i32, i32), (Vec2, f32)> = HashMap::default();
        for (k, x) in b.iter().enumerate() {
            if x.a > 1 || x.z > 0 {
                let e = flow.entry(((x.com.x / cell) as i32, (x.com.y / cell) as i32)).or_insert((Vec2::ZERO, 0.0));
                e.0 += (x.vel + dv[k]) * x.mass;
                e.1 += x.mass;
            }
        }
        let mut touching: HashSet<(u32, u32)> = HashSet::default();
        let cont = 1.0 - (-CONTACT_RATE * dt).exp();
        for (k, x) in b.iter().enumerate() {
            // Fast ions being stopped give their energy to the matter (see
            // above), not to walls: real fission fragments stop within
            // micrometres of where they are born.
            if x.a == 0 || (x.z == 0 && x.a == 1) || st.fast.contains(&x.tag) {
                continue;
            }
            let radius = r_n * (x.a as f32 / 0.9069).sqrt();
            let v = x.vel + dv[k];
            let reach = radius + r_n + v.length() * dt + 1.0;
            let Some(s) = walls.nearest(x.com, reach) else { continue };
            if !walls.segs[s].kind.conducts() {
                continue;
            }
            let (dist, nrm) = walls.distance_to(s, x.com);
            let key = (x.tag, s as u32);
            let impact = !st.wall_touch.contains(&key);
            touching.insert(key);
            // A new contact counts once it is bouncing off (moving away).
            let half = walls.segs[s].half;
            if impact && v.dot(nrm) < 0.0 && dist > radius + half + 0.5 {
                touching.remove(&key);
                continue;
            }
            // Resting contact only when actually touching.
            if !impact && dist > radius + half + 1.0 {
                touching.remove(&key);
                continue;
            }
            let alpha = if impact { ACCOMMODATION } else { cont };
            // Energy passes through the wall-normal motion only: a salt
            // sliding along an exchanger wall doesn't turn its flow into
            // heat.
            // Flow from the *other* bodies in the cell, if there are enough
            // of them to define one (a lone atom has no flow but itself).
            let u = flow.get(&((x.com.x / cell) as i32, (x.com.y / cell) as i32)).map_or(Vec2::ZERO, |f| {
                let (p, m) = (f.0 - (x.vel + dv[k]) * x.mass, f.1 - x.mass);
                if m > 3.0 * x.mass { p / m } else { Vec2::ZERO }
            });
            let vn = (v - u).dot(nrm);
            let ke = 0.5 * x.mass * vn * vn;
            // Diffuse re-emission from a wall at theta_w: the normal energy
            // of what leaves is exponential with mean theta_w (flux
            // weighting); something resting against it has theta_w / 2.
            let theta_w = walls.segs[s].theta;
            let sample = if impact { theta_w * -(1.0 - gen_range(0.0f32, 1.0)).ln() } else { 0.5 * theta_w };
            let ke2 = ((1.0 - alpha) * ke + alpha * sample).max(0.0);
            // Keep the direction (into or out of the wall) of the normal motion.
            let sign = if vn.abs() > 1e-6 { vn.signum() } else { 1.0 };
            let vn2 = sign * (2.0 * ke2 / x.mass).sqrt();
            dv[k] += nrm * (vn2 - vn);
            let seg = &mut walls.segs[s];
            let cap = capacity(seg);
            seg.theta = (seg.theta - (ke2 - ke) / cap).max(0.0);
        }
        st.wall_touch = touching;

        // Conduction along connected conductor walls.
        if walls.version != st.wall_version {
            st.wall_version = walls.version;
            st.wall_pairs.clear();
            for i in 0..walls.segs.len() {
                for j in i + 1..walls.segs.len() {
                    let (a, bb) = (&walls.segs[i], &walls.segs[j]);
                    let gap = a.half + bb.half + 1.5;
                    if !a.kind.conducts() || !bb.kind.conducts() {
                        continue;
                    }
                    let close = [a.a, a.b].iter().any(|&p| walls.distance_to(j, p).0 < gap)
                        || [bb.a, bb.b].iter().any(|&p| walls.distance_to(i, p).0 < gap);
                    if close {
                        st.wall_pairs.push((i as u32, j as u32));
                    }
                }
            }
        }
        let rate = (WALL_CONDUCTION * dt).min(0.45);
        for &(i, j) in &st.wall_pairs {
            let (i, j) = (i as usize, j as usize);
            if i >= walls.segs.len() || j >= walls.segs.len() {
                continue;
            }
            let ci = capacity(&walls.segs[i]);
            let cj = capacity(&walls.segs[j]);
            let flow = rate * (walls.segs[j].theta - walls.segs[i].theta) * ci * cj / (ci + cj);
            walls.segs[i].theta += flow / ci;
            walls.segs[j].theta -= flow / cj;
        }
        // Walls also lose heat to the outside world at the bath rate when
        // the matter is cooled (a cooled matter world has cooled walls).
        if cfg.matter && cfg.cooling > 0.0 {
            let c = (-0.2 * cfg.cooling * dt).exp();
            for seg in walls.segs.iter_mut().filter(|s| s.kind.conducts()) {
                seg.theta = theta_amb + (seg.theta - theta_amb) * c;
            }
        }

        // Freeze plugs melt.
        let melt = theta_of(&cfg, cfg.melt_temp);
        let before = walls.segs.len();
        walls.segs.retain(|s| !(s.kind == WallKind::FreezePlug && s.theta > melt));
        if walls.segs.len() != before {
            walls.touch();
            st.record(crate::nuclear::EventKind::Other, format!("freeze plug melted  (hotter than {})", format_kelvin(cfg.melt_temp as f64)));
        }
    }

    // ---- Apply ----
    for i in 0..n {
        let k = sim.nuc[i] as usize;
        if k >= m || !sim.is_nuclear(i) {
            continue;
        }
        if b[k].parts > 1 && e_int[k] > 0.0 && !sim.nuclear.fast.contains(&b[k].tag) {
            // Rigid: every nucleon moves with the nucleus.
            sim.vel[i] = (b[k].vel + dv[k]).clamp_length_max(cfg.max_speed);
        } else if dv[k] != Vec2::ZERO {
            sim.vel[i] = (sim.vel[i] + dv[k]).clamp_length_max(cfg.max_speed);
        }
    }
    sim.nuclear = st;
}

/// Temperature (sim units) of the matter in a region: mean KE per body.
pub fn region_theta(sim: &Sim, inside: impl Fn(Vec2) -> bool) -> Option<f32> {
    let mut bodies: HashMap<u32, (Vec2, Vec2, f32, u32, bool)> = HashMap::default();
    for (tag, b) in sim.bodies() {
        bodies.insert(tag, (b.pos * b.parts as f32, b.vel * b.parts as f32, b.a() as f32, b.parts.max(1), b.z > 0 || b.a() > 1));
    }
    // Temperature is motion relative to the local flow: subtract the mean
    // velocity of the bodies around each one (a coarse flow field).
    let fast = &sim.nuclear.fast;
    let list: Vec<(Vec2, Vec2, f32)> = bodies
        .into_iter()
        .filter(|(tag, _)| !fast.contains(tag))
        .map(|(_, b)| b)
        .filter(|b| b.4)
        .map(|(p, v, mass, a, _)| (p / a as f32, v / a as f32, mass))
        .filter(|b| inside(b.0))
        .collect();
    if list.is_empty() {
        return None;
    }
    let cell = 40.0;
    let mut flow: HashMap<(i32, i32), (Vec2, f32)> = HashMap::default();
    for (p, v, m) in &list {
        let e = flow.entry(((p.x / cell) as i32, (p.y / cell) as i32)).or_insert((Vec2::ZERO, 0.0));
        e.0 += *v * *m;
        e.1 += *m;
    }
    let mut sum = 0.0;
    for (p, v, m) in &list {
        let (mv, mm) = flow[&((p.x / cell) as i32, (p.y / cell) as i32)];
        // Flow of the others in the cell (none if too few to define one).
        let (op, om) = (mv - *v * *m, mm - *m);
        let u = if om > 3.0 * *m { *v - op / om } else { *v };
        sum += 0.5 * m * u.length_squared();
    }
    Some(sum / list.len() as f32)
}

/// Give every nucleus (not free neutrons) a random thermal velocity at
/// temperature `t_kelvin`, on top of nothing: Maxwellian with mean KE
/// theta per body (equipartition).
pub fn thermalize(sim: &mut Sim, t_kelvin: f32) {
    let theta = theta_of(&sim.cfg, t_kelvin);
    let mut kick: HashMap<u32, Vec2> = HashMap::default();
    let mut mass: HashMap<u32, (f32, u32, bool)> = HashMap::default();
    for (tag, b) in sim.bodies() {
        mass.insert(tag, (b.a() as f32, b.a(), b.z > 0));
    }
    for (&tag, &(m, a, charged)) in &mass {
        if charged || a > 1 {
            // Two degrees of freedom share theta: sigma^2 = theta / m each.
            kick.insert(tag, gauss() * (theta / m.max(1e-3)).sqrt());
        }
    }
    for i in 0..sim.len() {
        if let Some(v) = sim.is_nuclear(i).then(|| kick.get(&sim.tag[i])).flatten() {
            sim.vel[i] = *v;
        }
    }
}

/// Kinetic energy of the multi-nucleon nuclei (centre-of-mass motion).
pub fn body_ke(sim: &Sim) -> f32 {
    let m = sim.n_nuclei;
    let mut acc = vec![(Vec2::ZERO, 0.0f32, 0u32); m];
    for i in 0..sim.len() {
        let k = sim.nuc[i] as usize;
        if k < m {
            let mass = sim.mass_of(i);
            acc[k].0 += sim.vel[i] * mass;
            acc[k].1 += mass;
            acc[k].2 += sim.zn_of(i).0 + sim.zn_of(i).1;
        }
    }
    acc.iter().filter(|a| a.2 > 1).map(|a| 0.5 * a.0.length_squared() / a.1.max(1e-6)).sum()
}
