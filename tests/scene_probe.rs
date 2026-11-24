//! Headless look at the scenarios (cargo test --release --test scene_probe -- --ignored --nocapture).
use sim_core::game_scenes::{PRESETS, load_preset};
use sim_core::nuclear::game_sim;

#[test]
#[ignore]
fn probe() {
    let which: Vec<usize> = std::env::var("SCENES").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or((8..PRESETS.len()).collect());
    let secs: usize = std::env::var("SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    for k in which {
        let mut sim = game_sim(1600.0, 900.0);
        sim.seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
        load_preset(&mut sim, k);
        let t0 = std::time::Instant::now();
        for _ in 0..secs * 60 {
            sim.step(1.0 / 60.0, None);
        }
        let st = &sim.nuclear;
        println!("== {} {}: {} particles, {:.1}s wall", k, PRESETS[k].0, sim.len(), t0.elapsed().as_secs_f32());
        println!("   fission {} fusion {} breakup {} capture {} q+ {:.1} q- {:.1}", st.fissions, st.fusions, st.breakups, st.captures, st.q_released, st.q_absorbed);
        let mut by: Vec<(String, u32)> = Vec::new();
        for e in st.history.iter() {
            let eq = e.equation().to_string();
            match by.iter_mut().find(|x| x.0 == eq) { Some(x) => x.1 += 1, None => by.push((eq, 1)) }
        }
        by.sort_by(|a, b| b.1.cmp(&a.1));
        for (eq, n) in by.iter().take(6) {
            println!("   {n:5}  {eq}");
        }
        for (i, d) in sim.reactor.detectors.iter().enumerate() {
            let top: Vec<String> = d.top_species().iter().take(5).map(|(s, c)| format!("{} {c}", s.name())).collect();
            println!("   det{} {}: {} ({})", i + 1, d.filter.describe(), d.count, top.join(", "));
        }
        if let Some(e) = st.history.iter().rev().find(|e| e.text.contains("fragmentation")) {
            println!("   e.g. {}", e.text);
        }
    }
}

/// A saved scene (SCENE=<name>, SECS): reactions, fragmentation, and what the
/// light charged products do in the fields.
#[test]
#[ignore]
fn saved() {
    use sim_core::nuclear::energy_of;
    let name = std::env::var("SCENE").unwrap_or("Fusion accelerator".into());
    let secs: usize = std::env::var("SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let clean = std::env::var("CLEAN").is_ok();
    let mut sim = game_sim(1920.0, 1080.0);
    let path = sim_core::game_scenes::scene_dir().join(format!("{name}.scene"));
    let mut text = std::fs::read_to_string(path).unwrap();
    if clean {
        text = text.lines().filter(|l| !l.starts_with("p ")).collect::<Vec<_>>().join("\n");
    }
    sim_core::game_scenes::from_text(&mut sim, &text).unwrap();
    let mut ev_seen = 0usize;
    for s in 0..secs {
        for _ in 0..60 {
            sim.step(1.0 / 60.0, None);
        }
        let b = sim.bodies();
        let mut line: Vec<String> = Vec::new();
        for (zz, nn, nm) in [(1u32, 0u32, "p"), (1, 1, "D"), (1, 2, "T"), (2, 1, "He3"), (2, 2, "He4")] {
            let mut es: Vec<f64> = b.values().filter(|x| x.z == zz && x.n == nn).map(|x| energy_of(&sim.cfg, x.vel.length(), x.a() as f32)).collect();
            es.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !es.is_empty() {
                line.push(format!("{nm} {} [{:.2} {:.2} {:.1}]", es.len(), es[0], es[es.len() / 2], es[es.len() - 1]));
            }
        }
        println!("t {:3} n {:4}  {}", s + 1, sim.len(), line.join("  "));
        let h = &sim.nuclear.history;
        let total = h.len();
        for e in h.iter().skip(ev_seen.min(total)) {
            if s % 5 == 0 || e.text.contains("frag") {
                println!("     {}", e.text);
            }
        }
        ev_seen = total;
    }
    let st = &sim.nuclear;
    println!("fusion {} breakup {} capture {}", st.fusions, st.breakups, st.captures);
    for d in sim.reactor.detectors.iter() {
        println!("det {} mean {:?}", d.count, d.mean_energy());
    }
}

/// Path of the first few beam nuclei and the fate of light products.
#[test]
#[ignore]
fn trace() {
    let name = std::env::var("SCENE").unwrap_or("Fusion accelerator".into());
    let mut sim = game_sim(1920.0, 1080.0);
    let path = sim_core::game_scenes::scene_dir().join(format!("{name}.scene"));
    let text: String = std::fs::read_to_string(path).unwrap().lines().filter(|l| !l.starts_with("p ")).collect::<Vec<_>>().join("\n");
    sim_core::game_scenes::from_text(&mut sim, &text).unwrap();
    let mut first: Option<u32> = None;
    for f in 0..60 * 12 {
        sim.step(1.0 / 60.0, None);
        let b = sim.bodies();
        if first.is_none() {
            first = b.keys().copied().min();
        }
        if f % 15 == 0 {
            if let Some(x) = first.and_then(|t| b.get(&t)) {
                println!("t {:.2} D at ({:.0},{:.0}) v ({:.0},{:.0})", f as f32 / 60.0, x.pos.x, x.pos.y, x.vel.x, x.vel.y);
            }
        }
    }
}

/// Frames where a light nucleus suddenly gains energy, and what was near it.
#[test]
#[ignore]
fn jumps() {
    use sim_core::nuclear::energy_of;
    let name = std::env::var("SCENE").unwrap_or("Fusion accelerator".into());
    let secs: usize = std::env::var("SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(90);
    let mut sim = game_sim(1920.0, 1080.0);
    let path = sim_core::game_scenes::scene_dir().join(format!("{name}.scene"));
    let text: String = std::fs::read_to_string(path).unwrap().lines().filter(|l| !l.starts_with("p ")).collect::<Vec<_>>().join("\n");
    sim_core::game_scenes::from_text(&mut sim, &text).unwrap();
    let mut prev: std::collections::HashMap<u32, f64> = Default::default();
    for f in 0..secs * 60 {
        let before: Vec<(macroquad::math::Vec2, macroquad::math::Vec2, u8)> = (0..sim.len()).map(|i| (sim.pos[i], sim.vel[i], sim.kind[i])).collect();
        sim.step(1.0 / 60.0, None);
        let b = sim.bodies();
        let mut now: std::collections::HashMap<u32, f64> = Default::default();
        for (&t, x) in b.iter() {
            let e = energy_of(&sim.cfg, x.vel.length(), x.a() as f32);
            now.insert(t, e);
            if let Some(&e0) = prev.get(&t) {
                if e > 20.0 && e > e0 * 1.5 {
                    let near: Vec<String> = before.iter().filter(|(p, _, _)| p.distance(x.pos) < 30.0).map(|(p, v, k)| format!("k{} d{:.0} v{:.0}", k, p.distance(x.pos), v.length())).collect();
                    println!("f {} {}-{} {:.1} -> {:.1} at ({:.0},{:.0}) near [{}]", f, x.z, x.n, e0, e, x.pos.x, x.pos.y, near.join(", "));
                }
            }
        }
        prev = now;
    }
}
