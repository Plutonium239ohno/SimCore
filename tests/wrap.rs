//! Wrap-around world: particles leave one edge and come back at the other,
//! and feel each other across the edge.
use macroquad::math::vec2;
use sim_core::sim::Sim;

fn world() -> Sim {
    let mut sim = Sim::new(400.0, 300.0);
    sim.cfg.wrap = true;
    sim.cfg.coulomb = false;
    sim.cfg.drag = 0.0;
    sim
}

#[test]
fn particles_come_back_on_the_other_side() {
    let mut sim = world();
    sim.add(vec2(390.0, 150.0), vec2(300.0, 0.0), 6);
    sim.add(vec2(200.0, 5.0), vec2(0.0, -300.0), 6);
    for _ in 0..6 {
        sim.step(1.0 / 60.0, None);
    }
    let (a, b) = if sim.pos[0].y > 100.0 { (0, 1) } else { (1, 0) };
    assert!(sim.pos[a].x < 100.0, "should have wrapped to the left: {:?}", sim.pos[a]);
    assert!(sim.pos[b].y > 200.0, "should have wrapped to the bottom: {:?}", sim.pos[b]);
    assert!((sim.vel[a].x - 300.0).abs() < 1.0, "speed kept: {:?}", sim.vel[a]);
    for p in &sim.pos {
        assert!((0.0..400.0).contains(&p.x) && (0.0..300.0).contains(&p.y));
    }
}

#[test]
fn collisions_and_rules_cross_the_edge() {
    // Overlapping across the right/left edge: pushed apart.
    let mut sim = world();
    sim.add(vec2(398.0, 150.0), vec2(0.0, 0.0), 6);
    sim.add(vec2(2.0, 150.0), vec2(0.0, 0.0), 6);
    sim.step(1.0 / 60.0, None);
    // Distance the short way round, across the edge: was 4, now apart.
    let across = 400.0 - (sim.pos[0].x - sim.pos[1].x).abs();
    assert!((7.9..100.0).contains(&across), "separated across the edge: {across}");

    // A rule attracting across the edge.
    let mut sim = world();
    sim.cfg.rules = true;
    sim.cfg.collisions = true;
    sim.rules.set(6, 6, 1.0);
    sim.add(vec2(380.0, 150.0), vec2(0.0, 0.0), 6);
    sim.add(vec2(15.0, 150.0), vec2(0.0, 0.0), 6);
    sim.step(1.0 / 60.0, None);
    let right = (0..2).find(|&i| sim.pos[i].x > 200.0).unwrap();
    assert!(sim.vel[right].x > 1.0, "pulled right, towards its partner across the edge: {:?}", sim.vel[right]);
}
