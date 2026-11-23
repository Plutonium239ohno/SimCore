//! Physics sanity checks: settling, overlap, Barnes–Hut accuracy, gas energy.
use sim_core::sim::Sim;

fn stats(sim: &Sim) -> (f32, f32, f32) {
    let n = sim.len();
    let mut sp: Vec<f32> = sim.vel.iter().map(|v| v.length()).collect();
    let mean = sp.iter().sum::<f32>() / n as f32;
    sp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // worst overlap as a fraction of contact distance (brute force sample)
    let mut worst = 0.0f32;
    for i in (0..n).step_by((n / 500).max(1)) {
        for j in 0..n {
            if i == j { continue; }
            let rs = sim.types[sim.kind[i] as usize].radius + sim.types[sim.kind[j] as usize].radius;
            let d = sim.pos[i].distance(sim.pos[j]);
            if d < rs { worst = worst.max(1.0 - d / rs); }
        }
    }
    (mean, sp[n * 99 / 100], worst)
}

fn main() {
    for &n in &[3000usize, 10000] {
        println!("== settling pile, n={n} ==");
        let mut sim = Sim::new(1600.0, 900.0);
        sim.cfg.coulomb = false;
        sim.cfg.uniform_gravity = true;
        sim.spawn_random(n);
        for f in 0..=1200 {
            sim.step(1.0 / 60.0, None);
            if f % 300 == 0 {
                let (m, p99, ov) = stats(&sim);
                println!("frame {f:4}  mean|v| {m:6.1}  p99 {p99:6.1}  worst overlap {:4.1}%", ov * 100.0);
            }
        }
    }

    println!("== elastic gas (no gravity, e=1, no drag): kinetic energy should stay ~constant ==");
    let mut sim = Sim::new(1600.0, 900.0);
    sim.cfg.uniform_gravity = false;
    sim.cfg.coulomb = false;
    sim.cfg.restitution = 1.0;
    sim.cfg.drag = 0.0;
    let with_electrons = std::env::var("GAS_E").is_ok();
    for (k, t) in sim.types.iter_mut().enumerate() {
        t.in_random_spawn = !t.ghost && t.charge == 0.0 || (with_electrons && k == 2);
    }
    sim.spawn_random(3000);
    for v in sim.vel.iter_mut() { *v *= 4.0; }
    let ke = |s: &Sim| s.vel.iter().zip(&s.kind).map(|(v, k)| 0.5 * s.types[*k as usize].mass * v.length_squared()).sum::<f32>();
    let ke0 = ke(&sim);
    for f in 1..=600 {
        sim.step(1.0 / 60.0, None);
        if f % 150 == 0 { println!("frame {f:4}  KE/KE0 = {:.3}", ke(&sim) / ke0); }
    }

    println!("== Barnes-Hut vs exact (gravity + Coulomb), n=10000 ==");
    let mut sim = Sim::new(1600.0, 900.0);
    sim.spawn_random(10000);
    sim.cfg.mutual_gravity = true;
    sim.cfg.coulomb = true;
    sim.cfg.substeps = 1;
    sim.cfg.barnes_hut = false;
    let snapshot = (sim.pos.clone(), sim.vel.clone(), sim.kind.clone(), sim.tag.clone());
    sim.step(1e-6, None);
    let exact: Vec<_> = sim.pos.iter().cloned().zip(sim.far_accel().iter().cloned()).collect();
    for &theta in &[0.5f32, 0.8, 1.2] {
        let mut b = Sim::new(1600.0, 900.0);
        b.pos = snapshot.0.clone(); b.vel = snapshot.1.clone(); b.kind = snapshot.2.clone(); b.tag = snapshot.3.clone();
        for _ in 0..b.pos.len() { } 
        b.cfg = sim.cfg.clone(); b.cfg.barnes_hut = true; b.cfg.theta = theta;
        // far is sized lazily; pad by re-adding through step
        b.step(1e-6, None);
        // match particles by position (both sorted identically from same start)
        let mut errs: Vec<f32> = b.pos.iter().zip(b.far_accel()).zip(&exact).map(|((p, a), (pe, ae))| {
            assert!((*p - *pe).length() < 1e-2);
            (*a - *ae).length() / ae.length().max(1e-3)
        }).collect();
        errs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!("theta {theta}: median rel err {:.4}  p99 {:.4}", errs[errs.len() / 2], errs[errs.len() * 99 / 100]);
    }
}
