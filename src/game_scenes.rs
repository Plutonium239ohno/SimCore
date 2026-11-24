//! Scene files and built-in scenarios for the Nuclear Game.
//! (Library code so the headless probes and tests can use them too.)

use macroquad::math::{Vec2, vec2};
use crate::rng::gen_range;
use crate::nuclear::{place_nucleus, speed_for};
use crate::reactor::{Drain, DrainFilter, Pump, Rod, Site, Spawner};
use crate::walls::WallKind;
use macroquad::math::Rect;
use crate::sim::Sim;
use crate::types::{ParticleType, ids};
use crate::rng::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

/// Where scenes are saved: $XDG_DATA_HOME/nuclear-game/scenes (default
/// ~/.local/share/nuclear-game/scenes; on Windows %APPDATA%\nuclear-game\scenes).
pub fn scene_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("NUCLEAR_GAME_SCENES") {
        return PathBuf::from(d);
    }
    // Windows: %APPDATA% (HOME is usually unset there).
    let windows = cfg!(windows).then(|| std::env::var_os("APPDATA")).flatten().map(PathBuf::from);
    let base = windows
        .or_else(|| std::env::var_os("XDG_DATA_HOME").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("nuclear-game").join("scenes")
}

/// Where event logs are exported: next to the scenes folder.
pub fn log_dir() -> PathBuf {
    scene_dir().parent().map(|p| p.join("logs")).unwrap_or_else(|| PathBuf::from("logs"))
}

/// UTC "YYYYMMDD-HHMMSS" for file names (no date crate needed).
pub fn timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, rem / 60 % 60, rem % 60)
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

/// Write events as CSV (UTF-8): index, sim time, type, reaction, Q, details.
pub fn write_events_csv<'a>(path: &std::path::Path, events: impl Iterator<Item = (u64, &'a crate::nuclear::Event)>) -> Result<usize, String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let mut s = String::from("index,sim_time_s,type,reaction,Q_MeV,details\n");
    let mut n = 0;
    for (k, ev) in events {
        let q = ev.q().map(|q| format!("{q}")).unwrap_or_default();
        let _ = writeln!(
            s,
            "{},{:.3},{},{},{},{}",
            k,
            ev.t,
            ev.kind.name(),
            csv_field(ev.equation()),
            q,
            csv_field(ev.details())
        );
        n += 1;
    }
    std::fs::write(path, s).map_err(|e| e.to_string())?;
    Ok(n)
}

/// Saved scene names, alphabetical.
pub fn list_scenes() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(scene_dir())
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str()?.strip_suffix(".scene").map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn safe_name(name: &str) -> String {
    let s: String = name.trim().chars().map(|c| if c.is_alphanumeric() || "-_ .".contains(c) { c } else { '_' }).collect();
    if s.is_empty() { "untitled".into() } else { s }
}

/// Settings stored with a scene (a macro keeps save and load in sync).
macro_rules! cfg_fields {
    ($m:ident) => {
        $m!(coulomb, bool);
        $m!(k, f32);
        $m!(softening, f32);
        $m!(collisions, bool);
        $m!(restitution, f32);
        $m!(drag, f32);
        $m!(strong, f32);
        $m!(strong_range, f32);
        $m!(roundness, f32);
        $m!(nuclear_damping, f32);
        $m!(matter, bool);
        $m!(stopping, f32);
        $m!(temperature, f32);
        $m!(tunneling, f32);
        $m!(mev_speed, f32);
        $m!(energy_exponent, f32);
        $m!(decay_scale, f32);
        $m!(time_exponent, f32);
        $m!(light_speed, f32);
        $m!(max_speed, f32);
        $m!(cooling, f32);
        $m!(heat_fraction, f32);
        $m!(melt_temp, f32);
        $m!(xs_scale, f32);
        $m!(xs_exponent, f32);
        $m!(substeps, u32);
        $m!(iterations, u32);
    };
}

pub fn save(sim: &Sim, name: &str) -> Result<PathBuf, String> {
    let dir = scene_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}.scene", safe_name(name)));
    std::fs::write(&path, to_text(sim)).map_err(|e| e.to_string())?;
    Ok(path)
}

/// The whole scene as text (the scene-file format; also used for undo).
pub fn to_text(sim: &Sim) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "nuclear-game-scene 2");
    let _ = writeln!(s, "size {} {}", sim.width, sim.height);
    let _ = writeln!(s, "seed {} {}", sim.seed, sim.step_count);
    macro_rules! put {
        ($f:ident, $t:ty) => {
            let _ = writeln!(s, "cfg {} {}", stringify!($f), sim.cfg.$f);
        };
    }
    cfg_fields!(put);
    let _ = writeln!(s, "points {}", sim.point_nuclei as u8);
    let _ = writeln!(s, "wallhalf {}", sim.walls.half);
    for w in &sim.walls.segs {
        let _ = writeln!(
            s,
            "wall {} {} {} {} {} {} {} {} {} {} {} {}",
            w.a.x,
            w.a.y,
            w.b.x,
            w.b.y,
            w.kind.code(),
            w.theta,
            w.filter_z,
            w.only as u8,
            w.half,
            w.reflect_neutrons as u8,
            w.rule as u8,
            w.threshold
        );
    }
    let re = &sim.reactor;
    if !re.is_empty() {
        let _ = writeln!(s, "reactor {} {} {} {} {}", re.insertion, re.target, re.drive, re.auto_refuel as u8, re.refuel_delay);
        for r in &re.rods {
            let _ = writeln!(s, "rod {} {} {} {}", r.x, r.top, r.len, r.half_w);
        }
        for f in &re.sites {
            let _ = writeln!(s, "site {} {} {} {}", f.pos.x, f.pos.y, f.z, f.n);
        }
        for p in &re.sources {
            let _ = writeln!(s, "source {} {} {}", p.x, p.y, re.source_rate);
        }
        for sp in &re.spawners {
            let _ = write!(s, "spawner {} {} {} {} {} {} {}", sp.pos.x, sp.pos.y, sp.z, sp.n, sp.vel.x, sp.vel.y, sp.rate);
            let a = sp.area.unwrap_or(Rect::new(0.0, 0.0, 0.0, 0.0));
            let _ = write!(s, " {} {} {} {} {}", sp.keep, a.x, a.y, a.w, a.h);
            if sp.beam || sp.pulse_period > 0.0 {
                let _ = write!(s, " beam {} {} {} {} {} {}", sp.beam as u8, sp.width, sp.energy_spread, sp.divergence, sp.pulse_period, sp.pulse_len);
            }
            if sp.is_mix() {
                let _ = write!(s, " mix {}", sp.mix_random as u8);
                for (z, n, w) in sp.nuclides() {
                    let _ = write!(s, " {z} {n} {w}");
                }
            }
            let _ = writeln!(s);
        }
        for f in &re.fields {
            let _ = writeln!(s, "field {} {} {} {} {} {} {}", f.rect.x, f.rect.y, f.rect.w, f.rect.h, f.e.x, f.e.y, f.b);
        }
        for d in &re.detectors {
            use crate::detector::{DetShape, DetectFilter};
            let (shape, a, b, c, e) = match d.shape {
                DetShape::Box(r) => ("box", r.x, r.y, r.w, r.h),
                DetShape::Line(p, q) => ("line", p.x, p.y, q.x, q.y),
            };
            let (fz, fnn) = match d.filter {
                DetectFilter::Element(z) => (z, 0),
                DetectFilter::Isotope(z, n) => (z, n),
                _ => (0, 0),
            };
            let _ = writeln!(s, "detector {shape} {a} {b} {c} {e} {} {fz} {fnn} {}", d.filter.index(), d.light as u8);
        }
        if re.controller != crate::control::Controller::default() {
            let _ = writeln!(s, "{}", re.controller.to_line());
        }
        for r in &re.rules {
            let _ = writeln!(s, "{}", r.to_line());
        }
        if re.throttle != 1.0 {
            let _ = writeln!(s, "throttle {}", re.throttle);
        }
        if !re.auto_on {
            let _ = writeln!(s, "auto off");
        }
        for d in &re.drains {
            let _ = writeln!(s, "drain {} {} {} {} {} {}", d.rect.x, d.rect.y, d.rect.w, d.rect.h, d.filter.code(), d.per_second);
        }
        for pm in &re.pumps {
            let _ = writeln!(s, "pump {} {} {} {} {} {} {}", pm.rect.x, pm.rect.y, pm.rect.w, pm.rect.h, pm.dir.x, pm.dir.y, pm.speed);
        }
    }
    for i in 0..sim.len() {
        let (p, v) = (sim.pos[i], sim.vel[i]);
        let _ = write!(s, "p {} {} {} {} {} {}", sim.kind[i], p.x, p.y, v.x, v.y, sim.tag[i]);
        if sim.zn[i] != 0 {
            let (z, n) = sim.zn_of(i);
            let _ = write!(s, " {z} {n}");
        }
        let _ = writeln!(s);
    }
    s
}

pub fn load(sim: &mut Sim, name: &str) -> Result<(), String> {
    let path = scene_dir().join(format!("{}.scene", safe_name(name)));
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    from_text(sim, &text)
}

/// Replace the scene with one saved by `to_text`.
pub fn from_text(sim: &mut Sim, text: &str) -> Result<(), String> {
    let mut lines = text.lines();
    let Some(version) = lines.next().and_then(|l| l.strip_prefix("nuclear-game-scene")).map(|v| v.trim().parse::<u32>().unwrap_or(1)) else {
        return Err("not a scene file".into());
    };
    // Version 1 scenes were made with nucleons twice as big (radius 4):
    // their nuclei are shrunk about their centres and the length-dependent
    // settings are left at today's defaults.
    let old_scale = version < 2;
    const LENGTH_SCALED: [&str; 6] = ["mev_speed", "light_speed", "k", "softening", "strong", "max_speed"];
    sim.clear();
    sim.walls.clear();
    sim.nuclear.reset();
    sim.reactor.clear();
    sim.point_nuclei = false;
    let mut scale = 1.0f32;
    let mut tags: HashMap<u32, u32> = HashMap::default();
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        let num = |k: usize| f.get(k).and_then(|x| x.parse::<f32>().ok()).unwrap_or(0.0);
        match f.first().copied() {
            Some("size") => {
                // A scene saved in a bigger window is shrunk to fit.
                scale = (sim.width / num(1).max(1.0)).min(sim.height / num(2).max(1.0)).min(1.0);
            }
            Some("cfg") if f.len() >= 3 => {
                let (key, val) = (f[1], f[2]);
                if old_scale && LENGTH_SCALED.contains(&key) {
                    continue;
                }
                macro_rules! get {
                    ($f:ident, $t:ty) => {
                        if key == stringify!($f) {
                            if let Ok(v) = val.parse::<$t>() {
                                sim.cfg.$f = v;
                            }
                        }
                    };
                }
                cfg_fields!(get);
            }
            Some("wallhalf") => sim.walls.half = num(1),
            Some("seed") => {
                sim.seed = f.get(1).and_then(|x| x.parse().ok()).unwrap_or(1);
                sim.step_count = f.get(2).and_then(|x| x.parse().ok()).unwrap_or(0);
            }
            Some("points") => sim.point_nuclei = num(1) != 0.0,
            Some("reactor") => {
                let re = &mut sim.reactor;
                re.insertion = num(1).clamp(0.0, 1.0);
                re.target = num(2).clamp(0.0, 1.0);
                re.drive = num(3).max(0.001);
                re.auto_refuel = num(4) != 0.0;
                re.refuel_delay = num(5).max(0.0);
            }
            Some("rod") => sim.reactor.rods.push(Rod { x: num(1) * scale, top: num(2) * scale, len: num(3) * scale, half_w: num(4) }),
            Some("source") => {
                sim.reactor.sources.push(vec2(num(1), num(2)) * scale);
                sim.reactor.source_rate = num(3);
            }
            Some("spawner") => {
                let mut sp = Spawner::new(vec2(num(1), num(2)) * scale, num(3) as u32, num(4) as u32, vec2(num(5), num(6)), num(7));
                if f.len() >= 13 && num(8) > 0.0 {
                    sp.keep = num(8) as u32;
                    sp.area = Some(Rect::new(num(9) * scale, num(10) * scale, num(11) * scale, num(12) * scale));
                }
                if let Some(k) = f.iter().position(|x| *x == "beam") {
                    sp.beam = num(k + 1) != 0.0;
                    sp.width = num(k + 2) * scale;
                    sp.energy_spread = num(k + 3);
                    sp.divergence = num(k + 4);
                    sp.pulse_period = num(k + 5);
                    sp.pulse_len = num(k + 6);
                }
                if let Some(k) = f.iter().position(|x| *x == "mix") {
                    let random = num(k + 1) != 0.0;
                    let g = |x: &str| x.parse::<f32>().unwrap_or(0.0);
                    let list: Vec<(u32, u32, u8)> =
                        f[(k + 2).min(f.len())..].chunks(3).filter(|c| c.len() == 3).map(|c| (g(c[0]) as u32, g(c[1]) as u32, g(c[2]).clamp(0.0, 255.0) as u8)).collect();
                    sp.set_mix(&list);
                    sp.mix_random = random;
                }
                sim.reactor.spawners.push(sp);
            }
            Some("field") => sim.reactor.fields.push(crate::field::Field::new(
                Rect::new(num(1) * scale, num(2) * scale, num(3) * scale, num(4) * scale),
                vec2(num(5), num(6)),
                num(7),
            )),
            Some("detector") if f.len() >= 10 => {
                use crate::detector::{DetShape, DetectFilter, Detector};
                let shape = if f[1] == "line" {
                    DetShape::Line(vec2(num(2), num(3)) * scale, vec2(num(4), num(5)) * scale)
                } else {
                    DetShape::Box(Rect::new(num(2) * scale, num(3) * scale, num(4) * scale, num(5) * scale))
                };
                let filter = DetectFilter::from_index(num(6) as usize, num(7) as u32, num(8) as u32);
                sim.reactor.detectors.push(Detector::new(shape, filter, num(9) != 0.0));
            }
            Some("controller") => {
                if let Some(c) = crate::control::Controller::from_fields(&f) {
                    sim.reactor.controller = c;
                }
            }
            Some("rule") => {
                if let Some(r) = crate::control::Rule::from_fields(&f) {
                    sim.reactor.rules.push(r);
                }
            }
            Some("throttle") => sim.reactor.throttle = num(1).clamp(0.0, 1.0),
            Some("auto") => sim.reactor.auto_on = f.get(1) != Some(&"off"),
            Some("drain") => {
                let mut d = Drain::new(Rect::new(num(1) * scale, num(2) * scale, num(3) * scale, num(4) * scale), DrainFilter::from_code(num(5) as u8));
                d.per_second = num(6);
                sim.reactor.drains.push(d);
            }
            Some("pump") => sim.reactor.pumps.push(Pump {
                rect: Rect::new(num(1) * scale, num(2) * scale, num(3) * scale, num(4) * scale),
                dir: vec2(num(5), num(6)),
                speed: num(7),
            }),
            Some("site") => sim.reactor.sites.push(Site {
                pos: vec2(num(1), num(2)) * scale,
                z: num(3) as u32,
                n: num(4) as u32,
                empty_for: 0.0,
            }),
            Some("wall") => {
                let kind = WallKind::from_code(num(5) as u8);
                sim.walls.add_kind(vec2(num(1), num(2)) * scale, vec2(num(3), num(4)) * scale, kind);
                let theta = if f.len() >= 7 { num(6) } else { crate::thermal::theta_of(&sim.cfg, sim.cfg.temperature) };
                let w = sim.walls.segs.last_mut().unwrap();
                w.theta = theta;
                if f.len() >= 9 {
                    w.filter_z = num(7) as u32;
                    w.only = num(8) != 0.0;
                }
                // Older saves: thickness from "wallhalf", default neutron
                // behaviour, element filters.
                if f.len() >= 13 {
                    w.half = num(9).max(0.25);
                    w.reflect_neutrons = num(10) != 0.0;
                    w.rule = crate::walls::FilterRule::from_code(num(11) as u8);
                    w.threshold = num(12);
                }
                sim.walls.touch();
            }
            Some("p") if f.len() >= 7 => {
                let kind = f[1].parse::<u8>().unwrap_or(0);
                if kind as usize >= sim.types.len() {
                    continue;
                }
                if kind == ids::NUCLEUS && f.len() >= 9 {
                    let (z, n) = (num(7) as u32, num(8) as u32);
                    if z + n >= 2 {
                        sim.add_point(vec2(num(2), num(3)) * scale, vec2(num(4), num(5)), z, n);
                    }
                    continue;
                }
                sim.add(vec2(num(2), num(3)) * scale, vec2(num(4), num(5)), kind);
                if ParticleType::is_nucleon(kind) {
                    let old = f[6].parse::<u32>().unwrap_or(0);
                    let t = *tags.entry(old).or_insert_with(|| sim.new_tag());
                    let last = sim.len() - 1;
                    sim.tag[last] = t;
                }
            }
            _ => {}
        }
    }
    if old_scale {
        // Shrink each nucleus about its centre by the size change.
        let k = sim.types[ids::NEUTRON as usize].radius / 4.0;
        let mut centre: HashMap<u32, (Vec2, u32)> = HashMap::default();
        for i in 0..sim.len() {
            if ParticleType::is_nucleon(sim.kind[i]) {
                let e = centre.entry(sim.tag[i]).or_insert((Vec2::ZERO, 0));
                e.0 += sim.pos[i];
                e.1 += 1;
            }
        }
        for i in 0..sim.len() {
            if ParticleType::is_nucleon(sim.kind[i]) {
                let (sum, c) = centre[&sim.tag[i]];
                let com = sum / c as f32;
                sim.pos[i] = com + (sim.pos[i] - com) * k;
            }
        }
        for v in sim.vel.iter_mut() {
            *v *= k; // speeds were scaled with sizes
        }
    }
    sim.index_nuclei();
    Ok(())
}

pub fn delete(name: &str) {
    let _ = std::fs::remove_file(scene_dir().join(format!("{}.scene", safe_name(name))));
}

// ---------------------------------------------------------------- presets

/// Index of the molten-salt reactor and rocket presets.
pub const MSR: usize = 6;
pub const ROCKET: usize = 7;

/// Width of the control panel: presets centre themselves in the rest.
pub const PANEL_W: f32 = 420.0;

/// Indices of the fusion and accelerator scenarios.
pub const DD: usize = 8;
pub const DHE3: usize = 9;
pub const PB11: usize = 10;
pub const BOTTLE: usize = 11;
pub const GENERATOR: usize = 12;
pub const SPALLATION: usize = 13;
pub const COLLIDER: usize = 14;

pub const PRESETS: [(&str, &str); 15] = [
    ("Nuclear soup", "protons, neutrons and electrons: watch deuterium form by n-capture"),
    ("D-T fusion plasma", "hot deuterium + tritium: D + T → He-4 + n (17.6 MeV)"),
    ("U-235 chain reaction", "packed U-235 inside a neutron reflector, three stray neutrons"),
    ("Heavy-water reactor", "U-235 fuel in heavy water with control rods and on-power refuelling: withdraw the rods to go critical"),
    ("Decay zoo", "radioactive nuclides in matter; clock 30x faster"),
    ("CNO cycle", "hot protons + C-12, tunnelling x1000: the cycle that powers big stars"),
    ("Molten-salt reactor", "fuel dissolved in salt pumped round a graphite column; on-line fuel feed and fission-product removal, heat exchanger to helium, freeze plug over a drain tank"),
    ("Nuclear thermal rocket", "a reactor heats a conductor plate; hydrogen pumped under it heats up and leaves through a nozzle"),
    ("D-D fusion plasma", "hot deuterium alone: D + D → He-3 + n or T + p, about half each (3-4 MeV); needs hotter plasma than D-T"),
    ("D-He-3 fusion plasma", "deuterium + helium-3: D + He-3 → He-4 + p (18.4 MeV), almost no neutrons - but even hotter"),
    ("p-B-11 fusion plasma", "protons + boron-11: p + B-11 → 3 He-4 (8.7 MeV), no neutrons at all; the hardest to light (tunnelling ×30)"),
    ("Magnetic bottle", "D-T plasma held in a magnetic field and topped up by fuel feeds: the ions circle and stay in, the fusion products heat it (watch it ignite), the neutrons fly straight out to the detectors; the automation on the Controls tab can throttle the feeds"),
    ("Neutron generator", "a 1 MeV deuteron beam into a solid tritium target (tunnelling ×300): D + T → He-4 + 14 MeV neutron; as deuterium builds up in the target, D + D joins in (real generators see this too)"),
    ("Spallation source", "a 1 GeV proton beam on a lead target: each proton shatters a nucleus and boils off a shower of neutrons"),
    ("Heavy-ion collider", "two Ca-40 beams at 30 MeV/u each, head-on (120 MeV/u as a beam on a target): the nuclei shatter (fragmentation); detectors count the charged and neutral debris"),
];

/// Random velocity whose kinetic energy is about `mev` for mass `a`.
fn thermal(sim: &Sim, mev: f64, a: f32) -> Vec2 {
    let ang = gen_range(0.0, std::f32::consts::TAU);
    let e = mev * gen_range(0.3, 1.7);
    vec2(ang.cos(), ang.sin()) * speed_for(&sim.cfg, e, a)
}

/// Place a nucleus, retrying nearby spots; adds `electrons` electrons around it.
fn put(sim: &mut Sim, at: Vec2, z: u32, n: u32, v: Vec2, electrons: u32) -> bool {
    for k in 0..12 {
        let s = sim.types[ids::NEUTRON as usize].radius / 4.0;
        let p = at + vec2(gen_range(-1.0, 1.0), gen_range(-1.0, 1.0)) * (k as f32 * 6.0 * s);
        if place_nucleus(sim, p, z, n, v) {
            add_electrons(sim, p, z + n, electrons);
            return true;
        }
    }
    false
}

/// `count` electrons on a ring outside a nucleus of mass number `a`.
pub fn add_electrons(sim: &mut Sim, at: Vec2, a: u32, count: u32) {
    let r_nuc = sim.types[ids::NEUTRON as usize].radius * (a as f32 / 0.9069).sqrt();
    let s = sim.types[ids::NEUTRON as usize].radius / 4.0;
    for k in 0..count {
        let ang = k as f32 / count.max(1) as f32 * std::f32::consts::TAU + gen_range(0.0, 0.3);
        let r = r_nuc + (10.0 + gen_range(0.0, 14.0)) * s;
        let dir = vec2(ang.cos(), ang.sin());
        let p = at + dir * r;
        if p.x > 2.0 && p.y > 2.0 && p.x < sim.width - 2.0 && p.y < sim.height - 2.0 {
            sim.add(p, vec2(-dir.y, dir.x) * gen_range(150.0, 300.0) * s, ids::ELECTRON);
        }
    }
}

fn fresh(sim: &mut Sim) {
    sim.clear();
    sim.walls.clear();
    sim.nuclear.reset();
    sim.reactor.clear();
}

pub fn load_preset(sim: &mut Sim, which: usize) {
    // Built from the scene's seed: the same seed gives the same layout (and,
    // stepped the same way, the same run).
    crate::rng::seed(sim.seed);
    sim.step_count = 0;
    // Scenes are laid out with nucleon clusters, then converted.
    let points = sim.point_nuclei;
    sim.point_nuclei = false;
    fresh(sim);
    let (w, h) = (sim.width, sim.height);
    let defaults = crate::nuclear::game_config();
    sim.cfg = defaults;
    sim.walls.ambient_theta = crate::thermal::theta_of(&sim.cfg, sim.cfg.temperature);
    // Lengths below were laid out for radius-4 nucleons.
    let s = sim.types[ids::NEUTRON as usize].radius / 4.0;
    match which {
        0 => {
            // Nucleon gas with electrons: a plasma in vacuum.
            sim.cfg = sim.cfg.clone().with_medium(false);
            let saved: Vec<bool> = sim.types.iter().map(|t| t.in_random_spawn).collect();
            for (k, t) in sim.types.iter_mut().enumerate() {
                t.in_random_spawn = matches!(k as u8, ids::NEUTRON | ids::PROTON | ids::ELECTRON);
            }
            sim.spawn_random(4000);
            for (t, s) in sim.types.iter_mut().zip(saved) {
                t.in_random_spawn = s;
            }
        }
        1 => {
            // ~30 keV plasma: hot enough to tunnel now and then (vacuum).
            // No slow neutrons to show, so energy isn't compressed: speeds
            // follow real kinematics (E ~ v^2) and the plasma is a true
            // Maxwellian.
            sim.cfg = sim.cfg.clone().with_medium(false);
            sim.cfg.energy_exponent = 1.0;
            for k in 0..500 {
                let at = vec2(gen_range(40.0, w - 40.0), gen_range(40.0, h - 40.0));
                let (z, n) = if k % 2 == 0 { (1, 1) } else { (1, 2) };
                let v = thermal(sim, 0.03, (z + n) as f32);
                put(sim, at, z, n, v, 0);
            }
        }
        2 => {
            // Fast-neutron chain reaction in densely packed U-235 inside a
            // reflector: walls bounce escaping neutrons back into the fuel
            // (real reflectors shrink the critical mass the same way).
            let r = sim.types[ids::NEUTRON as usize].radius * (235.0f32 / 0.9069).sqrt() + 4.0 * s;
            let step = 2.0 * r + 1.0 * s;
            let (cols, rows) = (8, 5);
            let x0 = w * 0.5 - step * (cols as f32 - 1.0) * 0.5;
            let y0 = h * 0.5 - step * (rows as f32 - 1.0) * 0.5;
            let (bx0, by0) = (x0 - r - 70.0 * s, y0 - r - 20.0 * s);
            let (bx1, by1) = (x0 + step * (cols as f32 - 1.0) + r + 20.0 * s, y0 + step * (rows as f32 - 1.0) + r + 20.0 * s);
            sim.walls.half = 4.0 * s;
            sim.walls.add(vec2(bx0, by0), vec2(bx1, by0));
            sim.walls.add(vec2(bx1, by0), vec2(bx1, by1));
            sim.walls.add(vec2(bx1, by1), vec2(bx0, by1));
            sim.walls.add(vec2(bx0, by1), vec2(bx0, by0));
            for i in 0..cols {
                for j in 0..rows {
                    place_nucleus(sim, vec2(x0 + i as f32 * step, y0 + j as f32 * step), 92, 143, Vec2::ZERO);
                }
            }
            // A few stray neutrons (one alone fizzles ~1 time in 6: it may
            // be captured, or its fission may release no neutrons).
            for row in [1, 2, 3] {
                sim.add(vec2(x0 - r - 40.0 * s, y0 + step * row as f32), vec2(speed_for(&sim.cfg, 1e-4, 1.0), 0.0), ids::NEUTRON);
            }
        }
        3 => reactor_preset(sim),
        4 => {
            let zoo: [(u32, u32); 15] = [
                (1, 2), (4, 3), (6, 8), (7, 6), (9, 9), (11, 11), (19, 21), (27, 33),
                (38, 52), (53, 78), (55, 82), (84, 126), (88, 138), (90, 140), (98, 154),
            ];
            // A faster clock so you can watch: half-lives here are 30x shorter.
            sim.cfg.decay_scale = 0.1;
            let cols = 5;
            for (k, &(z, n)) in zoo.iter().enumerate() {
                let at = vec2(
                    w * (0.12 + 0.19 * (k % cols) as f32),
                    h * (0.2 + 0.3 * (k / cols) as f32),
                );
                put(sim, at, z, n, Vec2::ZERO, 0);
            }
        }
        MSR => msr_preset(sim),
        ROCKET => rocket_preset(sim),
        DD => plasma(sim, &[((1, 1), 500)], 0.10, 1.0),
        DHE3 => plasma(sim, &[((1, 1), 250), ((2, 1), 250)], 0.20, 1.0),
        PB11 => plasma(sim, &[((1, 0), 450), ((5, 6), 90)], 0.50, 30.0),
        BOTTLE => bottle_preset(sim),
        GENERATOR => generator_preset(sim),
        SPALLATION => spallation_preset(sim),
        COLLIDER => collider_preset(sim),
        _ => {
            // ~150 keV protons around carbon nuclei. Stars do this at a few
            // keV over millions of years; here tunnelling is boosted x1000.
            sim.cfg = sim.cfg.clone().with_medium(false);
            sim.cfg.energy_exponent = 1.0;
            sim.cfg.tunneling = 1000.0;
            for k in 0..8 {
                let at = vec2(w * (0.15 + 0.1 * k as f32), h * if k % 2 == 0 { 0.35 } else { 0.65 });
                put(sim, at, 6, 6, Vec2::ZERO, 0);
            }
            for _ in 0..700 {
                let at = vec2(gen_range(10.0, w - 10.0), gen_range(10.0, h - 10.0));
                let v = thermal(sim, 0.15, 1.0);
                put(sim, at, 1, 0, v, 0);
            }
        }
    }
    sim.index_nuclei();
    if points {
        crate::nuclear::set_point_mode(sim, true);
        fill_holes(sim);
    }
}

/// A hot fusion plasma in vacuum: `fuel` ((Z, N), count) at about `mev`
/// per nucleus. Energy is not compressed (real kinematics, a true
/// Maxwellian), as in the D-T scene.
fn plasma(sim: &mut Sim, fuel: &[((u32, u32), u32)], mev: f64, tunnel: f32) {
    let (w, h) = (sim.width, sim.height);
    sim.cfg = sim.cfg.clone().with_medium(false);
    sim.cfg.energy_exponent = 1.0;
    sim.cfg.tunneling = tunnel;
    let left = if w > 1100.0 { PANEL_W } else { 0.0 };
    let total: u32 = fuel.iter().map(|f| f.1).sum();
    for k in 0..total {
        // Interleave the species.
        let mut r = k % total;
        let mut pick = fuel[0].0;
        for &(zn, c) in fuel {
            if r < c {
                pick = zn;
                break;
            }
            r -= c;
        }
        let at = vec2(gen_range(left + 40.0, w - 40.0), gen_range(40.0, h - 40.0));
        let v = thermal(sim, mev, (pick.0 + pick.1) as f32);
        put(sim, at, pick.0, pick.1, v, 0);
    }
}

/// A D-T plasma in a uniform magnetic field: the ions gyrate on circles a
/// few tens of px across and stay put; neutral fusion neutrons fly straight
/// out to neutron detectors lining the walls.
fn bottle_preset(sim: &mut Sim) {
    use crate::detector::{DetShape, DetectFilter, Detector};
    let c = view_center(sim);
    sim.cfg = sim.cfg.clone().with_medium(false);
    // A real plasma is neutral on average (its electrons cancel the ions'
    // charge), so it doesn't blow itself apart: no long-range Coulomb push.
    // Fusion still has to tunnel through the barrier.
    sim.cfg.coulomb = false;
    sim.cfg.energy_exponent = 1.0;
    sim.cfg.tunneling = 1.0;
    let (bw, bh) = ((sim.width - PANEL_W).clamp(400.0, 900.0) * 0.8, (sim.height * 0.75).min(560.0));
    let r = Rect::new(c.x - bw * 0.5, c.y - bh * 0.5, bw, bh);
    // A D at 12 keV circles at ~30 px.
    let v = speed_for(&sim.cfg, 0.012, 2.0);
    let b = v / 30.0 * 2.0; // proton turn rate: a deuteron (q/m = 1/2) turns half as fast
    sim.reactor.fields.push(crate::field::Field::new(r, Vec2::ZERO, b));
    sim.walls.half = 2.0;
    box_walls(sim, Rect::new(r.x - 6.0, r.y - 6.0, r.w + 12.0, r.h + 12.0), [WallKind::Conductor; 4]);
    // The first wall takes what reaches it (fast fusion products leave the
    // field on wide circles; neutrons fly straight out).
    let wall = 10.0;
    // (Neutrons pass: the detectors outside count them.)
    for d in [Rect::new(r.x, r.y, r.w, wall), Rect::new(r.x, r.bottom() - wall, r.w, wall), Rect::new(r.x, r.y, wall, r.h), Rect::new(r.right() - wall, r.y, wall, r.h)] {
        sim.reactor.drains.push(Drain::new(d, DrainFilter::NotFuel));
    }
    // Beyond the detectors, a shield takes everything (so nothing comes back
    // to be counted twice).
    let o = Rect::new(r.x - 40.0, r.y - 40.0, r.w + 80.0, r.h + 80.0);
    for d in [Rect::new(o.x, o.y, o.w, 18.0), Rect::new(o.x, o.bottom() - 18.0, o.w, 18.0), Rect::new(o.x, o.y, 18.0, o.h), Rect::new(o.right() - 18.0, o.y, 18.0, o.h)] {
        sim.reactor.drains.push(Drain::new(d, DrainFilter::All));
    }
    for (a, bb) in [(vec2(r.x - 14.0, r.y - 14.0), vec2(r.right() + 14.0, r.y - 14.0)), (vec2(r.x - 14.0, r.bottom() + 14.0), vec2(r.right() + 14.0, r.bottom() + 14.0))] {
        sim.reactor.detectors.push(Detector::new(DetShape::Line(a, bb), DetectFilter::Element(0), false));
    }
    let inner = Rect::new(r.x + r.w * 0.2, r.y + r.h * 0.2, r.w * 0.6, r.h * 0.6);
    // Fuel feeds (like pellet injection): keep the inner region topped up
    // with cold D and T as the plasma burns.
    for (z, n) in [(1, 1), (1, 2)] {
        let mut feed = Spawner::new(inner.center(), z, n, Vec2::ZERO, 20.0);
        feed.keep = 100;
        feed.area = Some(inner);
        sim.reactor.spawners.push(feed);
    }
    for k in 0..240 {
        let at = vec2(gen_range(inner.x, inner.right()), gen_range(inner.y, inner.bottom()));
        let (z, n) = if k % 2 == 0 { (1, 1) } else { (1, 2) };
        let v = thermal(sim, 0.012, (z + n) as f32);
        put(sim, at, z, n, v, 0);
    }
}

/// A deuteron beam onto a tritium target (a neutron generator: real ones
/// fire ~100 keV deuterons into tritium-loaded titanium).
fn generator_preset(sim: &mut Sim) {
    use crate::detector::{DetShape, DetectFilter, Detector};
    let c = view_center(sim);
    // A solid target (matter: the tritium stays put and the beam slows in
    // it); tunnelling boosted so the thick-target yield is watchable.
    sim.cfg.tunneling = 300.0;
    // Slowed down less than real (a real deuteron stops within microns),
    // so it meets enough tritons on its way in.
    sim.cfg.stopping = 2.0;
    sim.cfg.cooling = 2.0;
    let target = Rect::new(c.x + 40.0, c.y - 110.0, 110.0, 220.0);
    fill_rect(sim, target, 1, 2, 1.0);
    let mut beam = Spawner::beam(&sim.cfg, vec2(c.x - 300.0, c.y), 1, 1, Vec2::X, 0.5, 15.0);
    beam.width = 160.0;
    beam.energy_spread = 0.03;
    sim.reactor.spawners.push(beam);
    sim.reactor.drains.push(Drain::new(Rect::new(c.x - 370.0, c.y - 200.0, 20.0, 400.0), DrainFilter::All));
    // Neutron counters: one over the target (every neutron made there), one
    // beyond it (the ones that get out).
    sim.reactor.detectors.push(Detector::new(DetShape::Box(Rect::new(target.x - 10.0, target.y - 10.0, target.w + 20.0, target.h + 20.0)), DetectFilter::Element(0), false));
    sim.reactor.detectors.push(Detector::new(DetShape::Line(vec2(c.x + 260.0, c.y - 250.0), vec2(c.x + 260.0, c.y + 250.0)), DetectFilter::Element(0), false));
}

/// A proton beam into a heavy target: spallation.
fn spallation_preset(sim: &mut Sim) {
    use crate::detector::{DetShape, DetectFilter, Detector};
    let c = view_center(sim);
    // Solid lead (matter): the target holds together.
    let target = Rect::new(c.x - 40.0, c.y - 120.0, 150.0, 240.0);
    fill_rect(sim, target, 82, 126, 10.0);
    let e = 1000.0f64.min(crate::reactor::max_beam_energy(&sim.cfg));
    let mut beam = Spawner::beam(&sim.cfg, vec2(c.x - 380.0, c.y), 1, 0, Vec2::X, e, 3.0);
    beam.width = 120.0;
    sim.reactor.spawners.push(beam);
    // Neutrons made in (or crossing) the region round the target, and the
    // charged debris coming out of the back.
    let ring = Rect::new(c.x - 120.0, c.y - 220.0, 340.0, 440.0);
    sim.reactor.detectors.push(Detector::new(DetShape::Box(ring), DetectFilter::Neutral, false));
    sim.reactor.detectors.push(Detector::new(DetShape::Line(vec2(ring.right() + 20.0, ring.y), vec2(ring.right() + 20.0, ring.bottom())), DetectFilter::Charged, false));
    // What flies out is taken away at the sides.
    let (w, h) = (sim.width, sim.height);
    sim.reactor.drains.push(Drain::new(Rect::new(c.x - 420.0, 0.0, 15.0, h), DrainFilter::All));
    sim.reactor.drains.push(Drain::new(Rect::new(w - 15.0, 0.0, 15.0, h), DrainFilter::All));
}

/// Two heavy-ion beams head-on.
fn collider_preset(sim: &mut Sim) {
    use crate::detector::{DetShape, DetectFilter, Detector};
    let c = view_center(sim);
    sim.cfg = sim.cfg.clone().with_medium(false);
    let e = 30.0f64.min(crate::reactor::max_beam_energy(&sim.cfg));
    let mut a = Spawner::beam(&sim.cfg, vec2(c.x - 380.0, c.y), 20, 20, Vec2::X, e, 4.0);
    a.width = 30.0;
    let b = a.mirrored(c);
    sim.reactor.spawners.push(a);
    sim.reactor.spawners.push(b);
    let r = 170.0;
    let up = (vec2(c.x - r, c.y - r), vec2(c.x + r, c.y - r));
    let down = (vec2(c.x - r, c.y + r), vec2(c.x + r, c.y + r));
    sim.reactor.detectors.push(Detector::new(DetShape::Line(up.0, up.1), DetectFilter::Charged, false));
    sim.reactor.detectors.push(Detector::new(DetShape::Line(down.0, down.1), DetectFilter::Neutral, false));
    for (x0, x1) in [(c.x - 420.0, c.x - 400.0), (c.x + 400.0, c.x + 420.0)] {
        sim.reactor.drains.push(Drain::new(Rect::new(x0, c.y - 300.0, x1 - x0, 600.0), DrainFilter::All));
    }
}

/// Heavy-water reactor: a lattice of U-235 fuel channels in a D2O moderator
/// with boron-carbide control rods between the fuel columns, inside a
/// reflecting vessel. The fuelling machine reloads spent channels on power.
///
/// The fuel pitch is near the thermal diffusion length, as in a real core:
/// a fission neutron is slowed by the deuterons in the gap and reaches the
/// next fuel nucleus. Prompt neutrons alone make the core critical only with
/// the rods well out; the delayed neutrons from fission products (Br-87,
/// I-137...) are what make it controllable, as in a real reactor.
fn reactor_preset(sim: &mut Sim) {
    let (w, h) = (sim.width, sim.height);
    let s = sim.types[ids::NEUTRON as usize].radius / 4.0;
    let rf = sim.types[ids::NEUTRON as usize].radius * (235.0f32 / 0.9069).sqrt() + 4.0 * s;
    let pitch = 2.0 * rf + REACTOR_GAP * s;
    let (cols, rows) = (7, 4);
    let x0 = w * 0.5 - pitch * (cols as f32 - 1.0) * 0.5;
    let y0 = h * 0.5 - pitch * (rows as f32 - 1.0) * 0.5;
    let margin = 50.0 * s;
    let (lx0, ly0) = ((x0 - rf - margin).max(12.0), (y0 - rf - margin).max(12.0));
    let (lx1, ly1) = ((x0 + pitch * (cols as f32 - 1.0) + rf + margin).min(w - 12.0), (y0 + pitch * (rows as f32 - 1.0) + rf + margin).min(h - 12.0));
    // Reactor vessel: reflecting walls around the heavy-water pool (added
    // before anything is placed inside).
    sim.walls.half = 4.0 * s;
    let (vx0, vy0, vx1, vy1) = (lx0 - 8.0 * s, ly0 - 8.0 * s, lx1 + 8.0 * s, ly1 + 8.0 * s);
    sim.walls.add(vec2(vx0, vy0), vec2(vx1, vy0));
    sim.walls.add(vec2(vx1, vy0), vec2(vx1, vy1));
    sim.walls.add(vec2(vx1, vy1), vec2(vx0, vy1));
    sim.walls.add(vec2(vx0, vy1), vec2(vx0, vy0));
    // Control rods in the gaps between fuel columns, entering from the top,
    // fully inserted to start with (a reactor starts shut down).
    let re = &mut sim.reactor;
    for gap in REACTOR_RODS {
        let x = x0 + pitch * (gap as f32 + 0.5);
        re.rods.push(Rod { x, top: vy0 + 4.0 * s, len: vy1 - vy0 - 8.0 * s, half_w: 5.0 * s });
    }
    re.insertion = 1.0;
    re.target = 1.0;
    let mut fuel = Vec::new();
    for i in 0..cols {
        for j in 0..rows {
            let c = vec2(x0 + pitch * i as f32, y0 + pitch * j as f32);
            place_nucleus(sim, c, 92, 143, Vec2::ZERO);
            sim.reactor.sites.push(Site { pos: c, z: 92, n: 143, empty_for: 0.0 });
            fuel.push(c);
        }
    }
    // Heavy-water moderator: deuterons on a loose lattice through the core
    // and a reflector margin around it, at room temperature.
    let rd = 2.0 * sim.types[ids::NEUTRON as usize].radius + 11.0 * s;
    let rods = sim.reactor.rects();
    let mut y = ly0.max(12.0);
    let mut row = 0;
    while y < ly1.min(h - 12.0) {
        let mut x = lx0.max(12.0) + if row % 2 == 1 { rd * 0.5 } else { 0.0 };
        while x < lx1.min(w - 12.0) {
            let at = vec2(x, y);
            let in_rod = rods.iter().any(|r| at.x > r.x - 4.0 * s && at.x < r.x + r.w + 4.0 * s);
            if !in_rod && fuel.iter().all(|c| c.distance(at) > rf + 10.0 * s) {
                let v = thermal(sim, 2.5e-8, 2.0);
                place_nucleus(sim, at, 1, 1, v);
            }
            x += rd;
        }
        y += rd * 0.866;
        row += 1;
    }
    // Startup source in the middle of the core, beside the centre rod
    // channel, between two fuel rows.
    let src = vec2(x0 + pitch * 3.5 - 12.0 * s, y0 + pitch * 1.5);
    sim.reactor.sources.push(src);
}

/// Moderator gap between fuel surfaces in the reactor preset (px).
pub const REACTOR_GAP: f32 = 30.0;
/// Fuel-column gaps that hold a control rod.
pub const REACTOR_RODS: [usize; 6] = [0, 1, 2, 3, 4, 5];

/// Centre of the part of the screen the panel doesn't cover.
fn view_center(sim: &Sim) -> Vec2 {
    let left = if sim.width > 1100.0 { PANEL_W } else { 0.0 };
    vec2((left + sim.width) * 0.5, sim.height * 0.5)
}

fn box_walls(sim: &mut Sim, r: Rect, kinds: [WallKind; 4]) {
    let (a, b, c, d) = (vec2(r.x, r.y), vec2(r.x + r.w, r.y), vec2(r.x + r.w, r.y + r.h), vec2(r.x, r.y + r.h));
    sim.walls.add_kind(a, b, kinds[0]); // top
    sim.walls.add_kind(b, c, kinds[1]); // right
    sim.walls.add_kind(c, d, kinds[2]); // bottom
    sim.walls.add_kind(d, a, kinds[3]); // left
}

/// Fill a rectangle with nuclide (z, n) on a hex lattice (`gap` px between
/// surfaces), at room temperature.
fn fill_rect(sim: &mut Sim, r: Rect, z: u32, n: u32, gap: f32) -> Vec<Vec2> {
    let rn = sim.types[ids::NEUTRON as usize].radius;
    let radius = rn * ((z + n) as f32 / 0.9069).sqrt() + rn;
    let d = 2.0 * radius + gap;
    let mut out = Vec::new();
    let mut y = r.y + radius;
    let mut row = 0;
    while y <= r.y + r.h - radius {
        let mut x = r.x + radius + if row % 2 == 1 { d * 0.5 } else { 0.0 };
        while x <= r.x + r.w - radius {
            if place_nucleus(sim, vec2(x, y), z, n, Vec2::ZERO) {
                out.push(vec2(x, y));
            }
            x += d;
        }
        y += d * 0.866;
        row += 1;
    }
    out
}

/// Molten-salt reactor. The fuel is dissolved in the coolant: a salt of
/// U-235 in FLiBe (Li-7, Be-9, F-19) circulates clockwise round a central
/// graphite moderator column inside the vessel, driven by pumps. The
/// vessel's right wall is a heat exchanger (conductor) to a helium stream.
/// On-line processing, as MSRs are designed for: a feed line adds fresh
/// U-235, a reprocessing drain takes fission products out of the flow, and
/// the off-gas drain strips xenon. A freeze plug in the floor holds the salt
/// in: overheat it and the salt drains into the tank below, which stops the
/// chain reaction.
fn msr_preset(sim: &mut Sim) {
    use crate::reactor::{Drain, DrainFilter, Pump, Spawner};
    use crate::walls::WallKind::{Conductor, FreezePlug, Insulator};
    let c = view_center(sim);
    sim.cfg.cooling = 0.0; // heat leaves only through the exchanger
    sim.cfg.heat_fraction = MSR_HEAT;
    sim.cfg.melt_temp = MSR_MELT;
    sim.walls.half = 2.0;
    let rn = sim.types[ids::NEUTRON as usize].radius;
    let (vw, vh) = (640.0, 460.0);
    let v = Rect::new(c.x - vw * 0.5 - 30.0, c.y - vh * 0.5, vw, vh);
    let col = Rect::new(v.center().x - 90.0, v.center().y - 80.0, 180.0, 160.0);
    // Vessel: reflector walls, heat exchanger on the right, freeze plug
    // in the floor.
    let (tl, tr, br, bl) = (v.point(), vec2(v.right(), v.y), vec2(v.right(), v.bottom()), vec2(v.x, v.bottom()));
    sim.walls.add_kind(tl, tr, Insulator);
    sim.walls.add_kind(tr, br, Conductor);
    // Wide enough for the biggest nuclei (U is ~68 px across) to drain.
    let plug = (v.center().x - 50.0, v.center().x + 50.0);
    sim.walls.add_kind(br, vec2(plug.1, v.bottom()), Insulator);
    sim.walls.add_kind(vec2(plug.1, v.bottom()), vec2(plug.0, v.bottom()), FreezePlug);
    sim.walls.add_kind(vec2(plug.0, v.bottom()), bl, Insulator);
    sim.walls.add_kind(bl, tl, Insulator);
    // Graphite column in the middle, thin skin (neutrons pass).
    box_walls(sim, col, [Conductor, Conductor, Conductor, Conductor]);
    fill_rect(sim, Rect::new(col.x + 3.0, col.y + 3.0, col.w - 6.0, col.h - 6.0), 6, 6, 4.0);
    // Drain tank under the freeze plug.
    let tank = Rect::new(plug.0 - 50.0, v.bottom() + 4.0, 200.0, 70.0);
    sim.walls.add_kind(vec2(plug.0, v.bottom()), vec2(tank.x, tank.y + 12.0), Insulator);
    sim.walls.add_kind(vec2(plug.1, v.bottom()), vec2(tank.right(), tank.y + 12.0), Insulator);
    sim.reactor.drains.push(Drain::new(Rect::new(tank.x, tank.y + 18.0, tank.w, tank.h - 18.0), DrainFilter::All));
    // Helium coolant outside the exchanger: fed at the top, pumped down,
    // out at the bottom.
    let he = Rect::new(v.right() + 4.0, v.y, 56.0, vh);
    sim.walls.add_kind(vec2(he.right(), he.y - 40.0), vec2(he.right(), he.bottom() + 40.0), Insulator);
    sim.reactor.spawners.push(Spawner::new(vec2(he.center().x, he.y - 10.0), 2, 2, vec2(0.0, 30.0), 30.0));
    sim.reactor.pumps.push(Pump { rect: Rect::new(he.x, he.y - 30.0, he.w, vh * 0.8), dir: vec2(0.0, 1.0), speed: 40.0 });
    sim.reactor.drains.push(Drain::new(Rect::new(he.x - 2.0, he.bottom() + 10.0, he.w + 4.0, 26.0), DrainFilter::All));
    // On-line processing: a side stream of the salt is cleaned of fission
    // products, here as a removal chance per second for each one (a fission
    // product stays ~3 s, long enough for some delayed neutrons); the
    // off-gas system strips xenon and krypton at the top.
    let mut processing = Drain::new(Rect::new(v.x + 2.0, v.y + 2.0, vw - 4.0, vh - 4.0), DrainFilter::FissionProducts);
    processing.per_second = MSR_PROCESSING;
    sim.reactor.drains.push(processing);
    sim.reactor.drains.push(Drain::new(Rect::new(v.x + 2.0, v.y + 2.0, vw - 4.0, 50.0), DrainFilter::OffGas));
    // Fuel feed: keeps the uranium concentration up either side of the
    // graphite, adding fresh U-235 (displacing carrier salt) as it burns.
    for area in [Rect::new(v.x + 15.0, v.y + 15.0, col.x - v.x - 30.0, vh - 30.0), Rect::new(col.right() + 15.0, v.y + 15.0, v.right() - col.right() - 30.0, vh - 30.0)] {
        let mut feed = Spawner::new(area.center(), 92, 143, Vec2::ZERO, 2.0);
        feed.keep = MSR_KEEP / 2;
        feed.area = Some(area);
        sim.reactor.spawners.push(feed);
    }
    // Control rods: down through the column, and in the salt on either side.
    for x in [col.center().x, v.x + 50.0, v.right() - 50.0, col.x - 45.0, col.right() + 45.0] {
        sim.reactor.rods.push(Rod { x, top: v.y + 3.0, len: vh - 6.0, half_w: 2.0 * rn });
    }
    sim.reactor.insertion = 1.0;
    sim.reactor.target = 1.0;
    sim.reactor.sources.push(vec2(v.x + 100.0, c.y));
    sim.reactor.sources.push(vec2(v.right() - 100.0, c.y));
    // Fuel salt: uranium on a lattice through the ring, then the carrier.
    let pitch = 2.0 * (rn * (235.0f32 / 0.9069).sqrt() + 2.0) + MSR_GAP;
    let mut y = v.y + pitch * 0.5 + 4.0;
    while y < v.bottom() - pitch * 0.5 {
        let mut x = v.x + pitch * 0.5 + 4.0;
        while x < v.right() - pitch * 0.5 {
            let p = vec2(x, y);
            let clear = col.x - pitch * 0.5 + 2.0;
            let in_col = p.x > clear && p.x < col.right() + pitch * 0.5 - 2.0 && p.y > col.y - pitch * 0.5 + 2.0 && p.y < col.bottom() + pitch * 0.5 - 2.0;
            let on_rod = sim.reactor.rods.iter().any(|r| (r.x - p.x).abs() < pitch * 0.5);
            if !in_col && !on_rod {
                place_nucleus(sim, p, 92, 143, Vec2::ZERO);
            }
            x += pitch;
        }
        y += pitch;
    }
    let ring = [
        Rect::new(v.x + 3.0, v.y + 3.0, vw - 6.0, col.y - v.y - 6.0),
        Rect::new(v.x + 3.0, col.bottom() + 3.0, vw - 6.0, v.bottom() - col.bottom() - 6.0),
        Rect::new(v.x + 3.0, col.y, col.x - v.x - 6.0, col.h),
        Rect::new(col.right() + 3.0, col.y, v.right() - col.right() - 6.0, col.h),
    ];
    for (z, n, gap) in [(4u32, 5u32, 16.0f32), (3, 4, 20.0), (9, 10, 36.0)] {
        for r in &ring {
            fill_rect(sim, *r, z, n, gap);
        }
    }
    // Molten salt runs hot: ~900 K.
    crate::thermal::thermalize(sim, 900.0);
}

/// Moderator gap between uranium nuclei in the molten-salt reactor (px).
pub const MSR_GAP: f32 = 14.0;
/// Molten-salt reactor: fission heat kept in the salt, freeze plug melt point.
pub const MSR_HEAT: f32 = 0.001;
pub const MSR_MELT: f32 = 1.0e6;
/// Fission products removed per second by the MSR's processing.
pub const MSR_PROCESSING: f32 = 0.3;
/// Uranium nuclei the molten-salt reactor's fuel feed keeps in the salt.
pub const MSR_KEEP: u32 = 12;


/// Nuclear thermal rocket: a heavy-water moderated U-235 core whose floor
/// is a conductor plate; hydrogen is pumped through the channel beneath it,
/// heats on the plate, and leaves through a converging-diverging nozzle
/// into the exhaust drain, which reads out thrust and exhaust speed.
fn rocket_preset(sim: &mut Sim) {
    use crate::walls::WallKind::{Conductor, Insulator};
    let c = view_center(sim);
    sim.cfg.cooling = 0.0; // heat leaves only in the exhaust
    sim.walls.half = 2.0;
    let rn = sim.types[ids::NEUTRON as usize].radius;
    let rf = rn * (235.0f32 / 0.9069).sqrt() + 2.0;
    let pitch = 2.0 * rf + REACTOR_GAP * 0.5;
    let (cols, rows) = (7, 3);
    let core = Rect::new(c.x - 380.0, c.y - 280.0, pitch * cols as f32 + 30.0, pitch * rows as f32 + 30.0);
    let chan_h = 80.0;
    let chan = Rect::new(core.x, core.bottom(), core.w, chan_h);
    // Core box: insulator reflector, conductor floor.
    box_walls(sim, core, [Insulator, Insulator, Conductor, Insulator]);
    // Propellant channel under it, closed at the left.
    sim.walls.add_kind(vec2(chan.x, chan.y), vec2(chan.x, chan.bottom()), Insulator);
    sim.walls.add_kind(vec2(chan.x, chan.bottom()), vec2(chan.right(), chan.bottom()), Insulator);
    // Nozzle: converge to a throat, then diverge.
    let throat = 24.0;
    let mid = chan.y + chan_h * 0.5;
    let t_x = chan.right() + 70.0;
    let e_x = t_x + 160.0;
    sim.walls.add_kind(vec2(chan.right(), chan.y), vec2(t_x, mid - throat * 0.5), Insulator);
    sim.walls.add_kind(vec2(chan.right(), chan.bottom()), vec2(t_x, mid + throat * 0.5), Insulator);
    sim.walls.add_kind(vec2(t_x, mid - throat * 0.5), vec2(e_x, mid - 70.0), Insulator);
    sim.walls.add_kind(vec2(t_x, mid + throat * 0.5), vec2(e_x, mid + 70.0), Insulator);
    sim.reactor.drains.push(crate::reactor::Drain::new(Rect::new(e_x - 10.0, mid - 80.0, 40.0, 160.0), crate::reactor::DrainFilter::All));
    // Hydrogen feed and turbopump.
    sim.reactor.spawners.push(Spawner::new(vec2(chan.x + 14.0, mid), 1, 0, vec2(15.0, 0.0), 12.0));
    sim.reactor.pumps.push(crate::reactor::Pump { rect: Rect::new(chan.x + 4.0, chan.y + 4.0, 90.0, chan_h - 8.0), dir: vec2(1.0, 0.0), speed: 20.0 });
    // Fuel lattice, moderator, rods, source.
    let x0 = core.x + 15.0 + pitch * 0.5;
    let y0 = core.y + 15.0 + pitch * 0.5;
    let mut fuel = Vec::new();
    for j in 0..rows {
        for i in 0..cols {
            let p = vec2(x0 + pitch * i as f32, y0 + pitch * j as f32);
            place_nucleus(sim, p, 92, 143, Vec2::ZERO);
            sim.reactor.sites.push(Site { pos: p, z: 92, n: 143, empty_for: 0.0 });
            fuel.push(p);
        }
    }
    for gap in 1..cols {
        let x = x0 + pitch * (gap as f32 - 0.5);
        // Thick rods: the hot moderator leaves the spectrum harder.
        sim.reactor.rods.push(Rod { x, top: core.y + 2.0, len: core.h - 4.0, half_w: 2.0 * rn });
    }
    sim.reactor.insertion = 1.0;
    sim.reactor.target = 1.0;
    let rods = sim.reactor.rects();
    let rd = 2.0 * rn + 5.5;
    let mut y = core.y + 6.0;
    let mut row = 0;
    while y < core.bottom() - 6.0 {
        let mut x = core.x + 6.0 + if row % 2 == 1 { rd * 0.5 } else { 0.0 };
        while x < core.right() - 6.0 {
            let at = vec2(x, y);
            let in_rod = rods.iter().any(|r| at.x > r.x - 2.0 && at.x < r.x + r.w + 2.0);
            if !in_rod && fuel.iter().all(|f| f.distance(at) > rf + 5.0) {
                place_nucleus(sim, at, 1, 1, Vec2::ZERO);
            }
            x += rd;
        }
        y += rd * 0.866;
        row += 1;
    }
    sim.reactor.sources.push(vec2(x0 + pitch * 3.0, y0 + pitch * 0.5));
    sim.reactor.sources.push(vec2(x0 + pitch * 1.0, y0 + pitch * 1.5));
    crate::thermal::thermalize(sim, sim.cfg.temperature);
}

/// After a cluster-built scene is turned into point nuclei, each heavy
/// nucleus sits in an empty hole the size its cluster was. Fill the holes
/// with what surrounds them (moderator, salt, propellant: light nuclei),
/// at the surrounding density and temperature.
fn fill_holes(sim: &mut Sim) {
    let rn = sim.types[ids::NEUTRON as usize].radius;
    let bodies: Vec<crate::sim::BodyInfo> = sim.bodies().into_values().collect();
    let light: Vec<&crate::sim::BodyInfo> = bodies.iter().filter(|b| b.a() >= 2 && b.a() <= 20 && b.z > 0).collect();
    // Never inside a control-rod channel (the rods would shove it out).
    let m = crate::nuclear::POINT_RADIUS + 2.0;
    let rods: Vec<Rect> = sim.reactor.rods.iter().map(|r| r.inserted(1.0)).map(|r| Rect::new(r.x - m, r.y - m, r.w + 2.0 * m, r.h + 2.0 * m)).collect();
    let mut add: Vec<(Vec2, Vec2, u32, u32)> = Vec::new();
    for h in bodies.iter().filter(|b| b.a() >= 100) {
        let rf = rn * (h.a() as f32 / 0.9069).sqrt() + rn;
        let ring = rf + 30.0;
        let near: Vec<&&crate::sim::BodyInfo> = light.iter().filter(|b| (rf..ring).contains(&b.pos.distance(h.pos))).collect();
        if near.len() < 4 {
            continue;
        }
        let area = std::f32::consts::PI * (ring * ring - rf * rf);
        let d = (area / near.len() as f32 / 0.866).sqrt();
        let mut y = -rf;
        let mut row = 0;
        while y <= rf {
            let mut x = -rf + if row % 2 == 1 { d * 0.5 } else { 0.0 };
            while x <= rf {
                let p = vec2(x, y);
                if p.length() < rf && p.length() > 3.0 * crate::nuclear::POINT_RADIUS && !rods.iter().any(|r| r.contains(h.pos + p)) {
                    let src = near[gen_range(0, near.len())];
                    add.push((h.pos + p, src.vel, src.z, src.n));
                }
                x += d;
            }
            y += d * 0.866;
            row += 1;
        }
    }
    for (p, v, z, n) in add {
        let jitter = vec2(gen_range(-1.0, 1.0), gen_range(-1.0, 1.0));
        place_nucleus(sim, p + jitter, z, n, v);
    }
    sim.index_nuclei();
}
