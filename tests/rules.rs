//! Interaction rules: per-cell range and the inner repulsion zone.
use macroquad::math::vec2;
use sim_core::sim::Sim;

/// Velocity B gets after one step at distance d from A, under cell (B, A).
fn pull(d: f32, strength: f32, range: f32, core: f32) -> f32 {
    let mut sim = Sim::new(800.0, 400.0);
    sim.cfg.rules = true;
    sim.cfg.coulomb = false;
    sim.cfg.rule_range = 60.0;
    let (a, b) = (6usize, 7usize);
    sim.rules.set(b, a, strength);
    sim.rules.set_range(b, a, range);
    sim.rules.set_core(b, a, core);
    sim.add(vec2(300.0, 200.0), vec2(0.0, 0.0), a as u8);
    sim.add(vec2(300.0 + d, 200.0), vec2(0.0, 0.0), b as u8);
    sim.step(1.0 / 60.0, None);
    let i = (0..2).find(|&i| sim.kind[i] == b as u8).unwrap();
    // Positive: moved towards A (pulled).
    -sim.vel[i].x
}

#[test]
fn per_cell_range() {
    // 90 px apart, global range 60: out of reach at x1, in reach at x2.
    assert_eq!(pull(90.0, 1.0, 1.0, 0.0), 0.0);
    assert!(pull(90.0, 1.0, 2.0, 0.0) > 1.0);
    // Shorter range: nothing at 40 px with x0.5 (30 px).
    assert_eq!(pull(40.0, 1.0, 0.5, 0.0), 0.0);
}

#[test]
fn core_repels_even_when_attracting() {
    // Close in, inside a 40% core: pushed away despite a positive strength.
    assert!(pull(14.0, 1.0, 1.0, 0.4) < -1.0);
    // Further out, the same cell attracts.
    assert!(pull(45.0, 1.0, 1.0, 0.4) > 1.0);
    // Without a core, close in is attraction too.
    assert!(pull(14.0, 1.0, 1.0, 0.0) > 0.0);
}
