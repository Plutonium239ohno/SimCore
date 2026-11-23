//! Walls stop fast particles (no tunnelling).
use macroquad::math::vec2;
use sim_core::sim::Sim;
use sim_core::types::ids;

#[test]
fn walls_stop_fast_particles() {
    let mut sim = Sim::new(800.0, 400.0);
    sim.cfg.coulomb = false;
    sim.walls.add(vec2(400.0, 0.0), vec2(400.0, 400.0));
    // Includes electrons, which pass through other particles but not walls.
    for k in 0..40 {
        let kind = if k % 2 == 0 { ids::NEUTRON } else { ids::ELECTRON };
        sim.add(vec2(100.0 + k as f32 * 5.0, 10.0 + k as f32 * 9.0), vec2(1500.0, 37.0), kind);
    }
    for _ in 0..120 {
        sim.step(1.0 / 60.0, None);
        assert!(sim.pos.iter().all(|p| p.x < 400.0), "a particle tunnelled through the wall");
    }
}
