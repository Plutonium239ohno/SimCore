//! Headless run of the Nuclear Game presets: what happens in each?
//! cargo run --release --example game_probe [preset] [seconds]   (TRACE=1 prints every event)
use sim_core::game_scenes as scenes;
use sim_core::nuclear::nuclide_name;
use std::time::Instant;
fn main() {
    macroquad::rand::srand(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(7));
    let only: Option<usize> = std::env::args().nth(1).and_then(|s| s.parse().ok());
    let secs: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    for k in 0..scenes::PRESETS.len() {
        if only.is_some_and(|o| o != k) { continue; }
        let mut sim = sim_core::nuclear::game_sim(1600.0, 900.0);
        scenes::load_preset(&mut sim, k);
        if let Some(f) = std::env::var("ROD").ok().and_then(|s| s.parse::<f32>().ok()) {
            sim.reactor.insertion = f;
            sim.reactor.target = f;
        }
        if std::env::var("NOREFUEL").is_ok() {
            sim.reactor.auto_refuel = false;
        }
        let n0 = sim.len();
        let t0 = Instant::now();
        let frames = secs * 60;
        let trace = std::env::var("TRACE").is_ok();
        let mut seen = 0usize;
        for f in 0..frames {
            sim.step(1.0 / 60.0, None);
            if trace {
                let total = sim.nuclear.fissions + sim.nuclear.fusions + sim.nuclear.captures + sim.nuclear.counts.values().sum::<u64>();
                let new = (total as usize).saturating_sub(seen).min(sim.nuclear.log.len());
                for l in sim.nuclear.log.iter().take(new).collect::<Vec<_>>().iter().rev() { println!("  f{f} {l}"); }
                seen = total as usize;
            }
            if f % (5 * 60) == 5 * 60 - 1 {
                let ns = &sim.nuclear;
                let ke: f32 = (0..sim.len()).filter(|&i| sim.kind[i] <= 1).map(|i| 0.5 * sim.vel[i].length_squared()).sum::<f32>() / sim.len().max(1) as f32;
                let free_n = sim.reactor.samples.back().map(|s| s.neutrons).unwrap_or(0);
                println!("  t={:>3}s n={} fis {} fus {} cap {} E {:.0} MeV  nucleonKE/n {:.0}  free-n {} rodcap {} refuels {}", (f + 1) / 60, sim.len(), ns.fissions, ns.fusions, ns.captures, ns.energy_released, ke, free_n, sim.reactor.rod_captures, sim.reactor.refuels);
            }
        }
        let ms = t0.elapsed().as_secs_f32() * 1000.0 / frames as f32;
        let ns = &sim.nuclear;
        println!("== {} ({} -> {} particles, {:.1} ms/frame)", scenes::PRESETS[k].0, n0, sim.len(), ms);
        let census: Vec<String> = ns.census.iter().take(14).map(|((z, n), c)| format!("{} x{c}", nuclide_name(*z, *n))).collect();
        println!("  census: {}", census.join(", "));
        let mut counts: Vec<String> = ns.counts.iter().map(|(m, c)| format!("{} {c}", m.short())).collect();
        counts.sort();
        println!("  decays: {}", counts.join(", "));
        for l in ns.log.iter().take(8) { println!("  | {l}"); }
    }
}
