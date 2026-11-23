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
        sim.seed = 7;
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
