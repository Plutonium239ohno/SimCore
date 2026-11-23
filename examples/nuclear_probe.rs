//! Headless check of nuclear mode.
use macroquad::math::vec2;
use sim_core::nuclear::nuclide_name;
use sim_core::sim::Sim;
use sim_core::types::ids;

fn report(sim: &Sim, label: &str) {
    let mut by_kind = vec![0usize; sim.types.len()];
    for &k in &sim.kind { by_kind[k as usize] += 1; }
    let kinds: Vec<String> = sim.types.iter().zip(&by_kind).filter(|(_, c)| **c > 0).map(|(t, c)| format!("{}:{c}", t.name)).collect();
    println!("--- {label}: {}", kinds.join("  "));
    let mut counts: Vec<_> = sim.nuclear.counts.iter().map(|(m, c)| format!("{}={c}", m.name())).collect();
    counts.sort();
    println!("    decays: {}  annihilations={}", counts.join(", "), sim.nuclear.annihilations);
    let census: Vec<String> = sim.nuclear.census.iter().take(10).map(|((z, n), c)| format!("{}x{c}", nuclide_name(*z, *n))).collect();
    println!("    census: {}", census.join("  "));
    for l in sim.nuclear.log.iter().take(5) { println!("    log: {l}"); }
}

fn main() {
    // 1) Free neutrons: should beta-decay with t_half = 3 * 609^0.15 ~ 7.9 s.
    let mut sim = Sim::new(1600.0, 900.0);
    sim.cfg.nuclear = true;
    sim.cfg.coulomb = false; // isolate decay statistics
    for i in 0..400 { sim.add(vec2(100.0 + (i % 40) as f32 * 35.0, 100.0 + (i / 40) as f32 * 70.0), vec2(0.0, 0.0), ids::NEUTRON); }
    let mut t = 0.0;
    for f in 1..=480 {
        sim.step(1.0 / 60.0, None); t += 1.0 / 60.0;
        if f % 240 == 0 {
            let left = sim.kind.iter().filter(|&&k| k == ids::NEUTRON).count();
            println!("t={t:4.1}s neutrons left {left}/400  (expected {:.0})", 400.0 * 0.5f32.powf(t / (3.0 * 609f32.powf(0.15))));
        }
    }
    report(&sim, "free neutrons after 8 s");

    // 2) Dense p/n/e soup with Coulomb: nuclei should form and decay.
    let mut sim = Sim::new(1600.0, 900.0);
    sim.cfg.nuclear = true;
    for t in sim.types.iter_mut() { t.in_random_spawn = false; }
    for k in [ids::NEUTRON, ids::PROTON, ids::ELECTRON] { sim.types[k as usize].in_random_spawn = true; }
    sim.spawn_random(6000);
    for f in 1..=1200 {
        sim.step(1.0 / 60.0, None);
        if f % 600 == 0 {
            let sane = sim.pos.iter().all(|p| p.is_finite()) && sim.vel.iter().all(|v| v.is_finite());
            report(&sim, &format!("soup t={}s sane={sane}", f / 60));
        }
    }
    let t0 = std::time::Instant::now();
    for _ in 0..60 { sim.step(1.0 / 60.0, None); }
    println!("soup step cost: {:.2} ms/frame", t0.elapsed().as_secs_f32() * 1000.0 / 60.0);
}
