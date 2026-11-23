//! Headless benchmark: cargo run --release --example bench -- [particles] [frames]
use sim_core::sim::Sim;
use std::time::Instant;

fn run(label: &str, n: usize, frames: usize, setup: impl Fn(&mut Sim)) {
    // Scale the world with n so density stays ~constant (a 1600x900 box jams near 13k).
    let k = (n as f32 / 8000.0).sqrt().max(1.0);
    let (w, h) = (1600.0 * k, 900.0 * k);
    let mut sim = Sim::new(w, h);
    sim.rules.randomize();
    sim.cfg.coulomb = false;
    sim.cfg.uniform_gravity = true;
    for t in sim.types.iter_mut() { t.in_random_spawn = !t.ghost && t.charge == 0.0; }
    setup(&mut sim);
    let placed = sim.spawn_random(n);
    for _ in 0..30 { sim.step(1.0 / 60.0, None); } // warm-up
    let t0 = Instant::now();
    let mut far = 0.0;
    for _ in 0..frames { sim.step(1.0 / 60.0, None); far += sim.timings.far; }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    let ok = sim.pos.iter().all(|p| p.is_finite() && p.x >= 0.0 && p.x <= w && p.y >= 0.0 && p.y <= h);
    println!("{label:<34} n={placed:>6}  {ms:7.2} ms/frame  (long-range {:6.2})  sane={ok}", far / frames as f32);
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let n = *args.first().unwrap_or(&10000);
    let frames = *args.get(1).unwrap_or(&200);
    run("collisions + gravity", n, frames, |_| {});
    run("+ type rules", n, frames, |s| s.cfg.rules = true);
    run("+ coulomb (Barnes-Hut)", n, frames, |s| s.cfg.coulomb = true);
    run("coulomb + mutual grav (Barnes-Hut)", n, frames, |s| { s.cfg.coulomb = true; s.cfg.mutual_gravity = true; s.cfg.uniform_gravity = false; });
    run("coulomb + mutual grav (exact)", n, frames.min(30), |s| { s.cfg.coulomb = true; s.cfg.mutual_gravity = true; s.cfg.uniform_gravity = false; s.cfg.barnes_hut = false; });
}
