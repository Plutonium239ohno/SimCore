//! Same scene + same seed + same steps = the same run, bit for bit; and
//! a saved scene replays exactly.
use macroquad::math::vec2;
use sim_core::game_scenes as scenes;
use sim_core::nuclear::game_sim;
use sim_core::sim::Sim;

fn fingerprint(sim: &Sim) -> Vec<(u32, u32, u32, u32, u8)> {
    (0..sim.len()).map(|i| (sim.pos[i].x.to_bits(), sim.pos[i].y.to_bits(), sim.vel[i].x.to_bits(), sim.vel[i].y.to_bits(), sim.kind[i])).collect()
}

fn reactor(seed: u64) -> Sim {
    let mut sim = game_sim(1600.0, 900.0);
    sim.point_nuclei = true;
    sim.seed = seed;
    scenes::load_preset(&mut sim, 3); // heavy-water reactor: fission, decays, sources
    sim.reactor.target = 0.0; // rods out: plenty of random events
    sim
}

fn run(sim: &mut Sim, steps: usize) {
    for _ in 0..steps {
        sim.step(1.0 / 60.0, None);
    }
}

#[test]
fn same_seed_same_run() {
    let mut a = reactor(7);
    let mut b = reactor(7);
    run(&mut a, 900);
    run(&mut b, 900);
    assert!(a.nuclear.fissions + a.nuclear.captures > 0, "nothing random happened");
    assert_eq!(fingerprint(&a), fingerprint(&b));
    assert_eq!(a.nuclear.fissions, b.nuclear.fissions);

    let mut c = reactor(8);
    run(&mut c, 900);
    assert_ne!(fingerprint(&a), fingerprint(&c), "a different seed should give a different run");
}

#[test]
fn saved_scene_replays_exactly() {
    let mut a = reactor(11);
    run(&mut a, 60);
    let text = scenes::to_text(&a);
    run(&mut a, 840);
    let mut b = game_sim(1600.0, 900.0);
    scenes::from_text(&mut b, &text).unwrap();
    run(&mut b, 840);
    assert!(a.nuclear.fissions + a.nuclear.captures > 0, "nothing random happened");
    assert_eq!(fingerprint(&a), fingerprint(&b));
}

#[test]
fn sandbox_rules_are_repeatable() {
    let make = || {
        let mut sim = Sim::new(800.0, 600.0);
        sim.seed = 3;
        sim.cfg.rules = true;
        sim.cfg.coulomb = false;
        sim.rules.randomize();
        sim.spawn_random(800);
        sim.add(vec2(10.0, 10.0), vec2(5.0, 0.0), 6);
        sim
    };
    // Spawning draws random numbers outside a step: seed it like the apps do.
    sim_core::rng::seed(99);
    let mut a = make();
    sim_core::rng::seed(99);
    let mut b = make();
    run(&mut a, 120);
    run(&mut b, 120);
    assert_eq!(fingerprint(&a), fingerprint(&b));
}
