//! Neutron economy of the reactor preset: where do neutrons end up, and what
//! is the multiplication factor at a given rod insertion?
//! cargo run --release --example reactor_probe [insertion] [seconds]
//! env: PRESET=n (default 3), POINTS (point nuclei), NOREFUEL, NOSOURCE, NOHELIUM, MELT=<K>,
//! SEED, SPLIT_X=<x> (+DUMP) for where free neutrons are, UEVENTS
use sim_core::game_scenes as scenes;
use sim_core::nuclear::EventKind;
use sim_core::types::ids;
use std::collections::BTreeMap;

fn main() {
    macroquad::rand::srand(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(7));
    let rod: f32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let secs: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let preset: usize = std::env::var("PRESET").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut sim = sim_core::nuclear::game_sim(1600.0, 900.0);
    sim.point_nuclei = std::env::var("POINTS").is_ok();
    scenes::load_preset(&mut sim, preset);
    if let Some(v) = std::env::var("XS").ok().and_then(|s| s.parse().ok()) {
        sim.cfg.xs_scale = v;
    }
    if let Some(v) = std::env::var("XSE").ok().and_then(|s| s.parse().ok()) {
        sim.cfg.xs_exponent = v;
    }
    sim.reactor.insertion = rod;
    sim.reactor.target = rod;
    if std::env::var("NORODS").is_ok() {
        sim.reactor.rods.clear();
    }
    if std::env::var("NOSOURCE").is_ok() {
        sim.reactor.sources.clear();
    }
    if let Some(m) = std::env::var("MELT").ok().and_then(|s| s.parse().ok()) {
        sim.cfg.melt_temp = m;
    }
    if std::env::var("NOHELIUM").is_ok() {
        for sp in sim.reactor.spawners.iter_mut().filter(|s| (s.z, s.n) == (2, 2)) {
            sp.rate = 0.0;
        }
    }
    if std::env::var("NOREFUEL").is_ok() {
        sim.reactor.auto_refuel = false;
    }
    let t0 = std::time::Instant::now();
    for f in 0..secs * 60 {
        sim.step(1.0 / 60.0, None);
        if f % 300 == 299 {
            let s = sim.reactor.samples.back().copied().unwrap_or_default();
            for (k, d) in sim.reactor.drains.iter().enumerate() {
                println!("    drain {k} {:?}: removed {} rate {:.1}/s speed {:.0} px/s E/u {}", d.filter, d.removed, d.rate, d.speed, sim_core::nuclear::format_energy(d.energy));
            }
            if let Ok(xs) = std::env::var("SPLIT_X") {
                let x: f32 = xs.parse().unwrap_or(1000.0);
                let fr: Vec<(f32, f32, f64)> = (0..sim.len())
                    .filter(|&i| sim.kind[i] == ids::NEUTRON && !(0..sim.len()).any(|j| j != i && sim.tag[j] == sim.tag[i] && sim.is_nuclear(j)))
                    .map(|i| (sim.pos[i].x, sim.pos[i].y, sim_core::nuclear::energy_of(&sim.cfg, sim.vel[i].length(), 1.0)))
                    .collect();
                let inside = fr.iter().filter(|p| p.0 < x).count();
                let mut e: Vec<f64> = fr.iter().map(|p| p.2).collect();
                e.sort_by(f64::total_cmp);
                if std::env::var("DUMP").is_ok() {
                    let mut us: std::collections::HashMap<u32, (macroquad::math::Vec2, u32, u32)> = Default::default();
                    for i in 0..sim.len() {
                        if sim.kind[i] <= 1 {
                            let e = us.entry(sim.tag[i]).or_insert((macroquad::math::Vec2::ZERO, 0, 0));
                            e.0 += sim.pos[i];
                            e.1 += 1;
                            e.2 += (sim.kind[i] == ids::PROTON) as u32;
                        }
                    }
                    let u: Vec<macroquad::math::Vec2> = us.values().filter(|e| e.2 == 92).map(|e| e.0 / e.1 as f32).collect();
                    println!("      U nuclei {} ({} in core leg)", u.len(), u.iter().filter(|p| p.x < x).count());
                    let all: Vec<(macroquad::math::Vec2, u32)> = us.values().filter(|e| e.1 >= 20).map(|e| (e.0 / e.1 as f32, e.1)).collect();
                    for (p, a) in all.iter().filter(|(p, _)| p.x > 1300.0 || p.y > 680.0 || p.x < 660.0 || p.y < 220.0) {
                        println!("      OUTSIDE: A={a} at ({:.0},{:.0})", p.x, p.y);
                    }
                    for p in fr.iter().take(12) {
                        let d = u.iter().map(|q| q.distance(macroquad::math::vec2(p.0, p.1))).fold(f32::MAX, f32::min);
                        println!("      n ({:.0},{:.0}) nearest U centre {:.0} px", p.0, p.1, d);
                    }
                }
                if false {
                    for p in fr.iter().take(12) {
                        println!("      n at ({:.0},{:.0}) E {}", p.0, p.1, sim_core::nuclear::format_energy(p.2));
                    }
                }
                println!("    free n: {} in core leg, {} elsewhere; median E {}", inside, fr.len() - inside, e.get(e.len() / 2).map(|&v| sim_core::nuclear::format_energy(v)).unwrap_or_default());
            }
            for sp in &sim.reactor.spawners {
                println!("    spawner {}-{} rate {} spawned {} acc {:.2}", sp.z, sp.n, sp.rate, sp.spawned, sp.acc);
            }
            for w in sim.walls.segs.iter().filter(|w| w.kind == sim_core::walls::WallKind::FreezePlug) {
                println!("    plug theta {:.0}", w.theta);
            }
            let hot = sim.walls.segs.iter().filter(|s| s.kind.conducts()).map(|s| s.theta).fold(0.0f32, f32::max);
            let mat = sim_core::thermal::region_theta(&sim, |_| true).unwrap_or(0.0);
            println!("    hottest wall {}  matter {}  walls {}", sim_core::thermal::format_kelvin(sim_core::thermal::kelvin_of(&sim.cfg, hot)), sim_core::thermal::format_kelvin(sim_core::thermal::kelvin_of(&sim.cfg, mat)), sim.walls.segs.len());
            println!(
                "t={:>3}s free-n {:>4}  fissions {:>4}  power {:>5.1}/s  particles {}",
                (f + 1) / 60,
                s.neutrons,
                sim.nuclear.fissions,
                sim.reactor.power(),
                sim.len()
            );
        }
    }
    {
        let mut es: Vec<f64> = (0..sim.len())
            .filter(|&i| sim.kind[i] == ids::NEUTRON && !(0..sim.len()).any(|j| j != i && sim.tag[j] == sim.tag[i] && sim.kind[j] <= 1))
            .map(|i| sim_core::nuclear::energy_of(&sim.cfg, sim.vel[i].length(), 1.0))
            .collect();
        es.sort_by(f64::total_cmp);
        if !es.is_empty() {
            let q = |f: f64| sim_core::nuclear::format_energy(es[((es.len() - 1) as f64 * f) as usize]);
            println!("free-n energy: 10% {}  25% {}  median {}  75% {}  90% {}  (n={})", q(0.1), q(0.25), q(0.5), q(0.75), q(0.9), es.len());
        }
    }
    let ms = t0.elapsed().as_secs_f32() * 1000.0 / (secs * 60) as f32;
    if std::env::var("UEVENTS").is_ok() {
        let mut c: BTreeMap<String, u32> = BTreeMap::new();
        for ev in &sim.nuclear.history {
            let eq = ev.equation();
            if eq.contains("U-23") && !eq.starts_with("U-235 + n →") {
                *c.entry(format!("{:?}: {}", ev.kind, eq.split('→').next().unwrap_or(""))).or_default() += 1;
            }
        }
        for (k, v) in c.iter().take(20) {
            println!("  U-event {v:>4}  {k}");
        }
    }
    let mut fate: BTreeMap<String, u32> = BTreeMap::new();
    let mut born = 0u32;
    for ev in &sim.nuclear.history {
        let eq = ev.equation();
        let lhs_n = eq.split('→').next().is_some_and(|l| l.split(" + ").any(|x| x.trim() == "n"));
        let rhs = eq.split('→').nth(1).unwrap_or("");
        let out_n: u32 = rhs
            .split(" + ")
            .map(|x| x.trim())
            .map(|x| if x == "n" { 1 } else if let Some(k) = x.strip_suffix('n').and_then(|k| k.parse().ok()) { k } else { 0 })
            .sum();
        born += out_n;
        if !lhs_n && !eq.starts_with("n →") {
            continue;
        }
        let band = |d: &str| -> &'static str {
            let Some(k) = d.find("at ") else { return "" };
            let s = &d[k + 3..];
            if s.ends_with("meV") { "thermal" } else if s.ends_with(" eV") { "epithermal" } else if s.ends_with("keV") { "keV" } else { "MeV" }
        };
        let key = match ev.kind {
            EventKind::Fission => format!("fission ({})", band(ev.details())),
            EventKind::Capture if eq.contains("B-10") => "rod capture".to_string(),
            EventKind::Capture => format!("capture on {}", eq.split(" + ").next().unwrap_or("?")),
            EventKind::Decay => "free decay".into(),
            EventKind::Scatter => continue,
            _ => format!("{:?}", ev.kind),
        };
        *fate.entry(key).or_default() += 1;
    }
    let used: u32 = fate.values().sum();
    println!("== rods {:.0}%  {:.1} ms/frame  neutrons born {born}, absorbed/decayed {used}", rod * 100.0, ms);
    let mut v: Vec<_> = fate.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, c) in v.iter().take(14) {
        println!("  {:>5}  {:>5.1}%  {k}", c, 100.0 * *c as f32 / used.max(1) as f32);
    }
    let fis: u32 = v.iter().filter(|(k, _)| k.starts_with("fission")).map(|x| x.1).sum();
    println!("  k ~ {:.2} (neutrons born per neutron used)", born as f32 / used.max(1) as f32);
    let _ = fis;
}
