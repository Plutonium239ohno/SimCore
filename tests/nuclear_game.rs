use macroquad::math::{vec2, Vec2};
use sim_core::game_scenes as scenes;
use sim_core::nuclear::place_nucleus;
use sim_core::sim::Sim;
use sim_core::types::ids;

fn game_sim() -> Sim {
    sim_core::nuclear::game_sim(1200.0, 800.0)
}

#[test]
fn save_and_load_round_trip() {
    let dir = std::env::temp_dir().join(format!("nuclear-game-test-{}", std::process::id()));
    unsafe { std::env::set_var("NUCLEAR_GAME_SCENES", &dir) };
    let mut sim = game_sim();
    assert!(place_nucleus(&mut sim, vec2(300.0, 300.0), 92, 143, Vec2::ZERO));
    assert!(place_nucleus(&mut sim, vec2(600.0, 300.0), 2, 2, vec2(10.0, 0.0)));
    sim.add(vec2(100.0, 100.0), vec2(5.0, 0.0), ids::NEUTRON);
    sim.walls.add(vec2(50.0, 700.0), vec2(1000.0, 650.0));
    sim.cfg.decay_scale = 1.7;
    sim.cfg.matter = false;
    for _ in 0..10 { sim.step(1.0 / 60.0, None); }
    let census = sim.nuclear.census.clone();
    scenes::save(&sim, "round trip").unwrap();
    assert!(scenes::list_scenes().contains(&"round trip".to_string()));

    let mut other = game_sim();
    scenes::load(&mut other, "round trip").unwrap();
    assert_eq!(other.len(), sim.len());
    assert_eq!(other.walls.segs.len(), 1);
    assert_eq!(other.cfg.decay_scale, 1.7);
    assert!(!other.cfg.matter);
    other.step(1.0 / 60.0, None);
    // Same nuclei (tags were remapped but grouping kept).
    let mut a: Vec<_> = census.iter().map(|c| c.0).collect();
    let mut b: Vec<_> = other.nuclear.census.iter().map(|c| c.0).collect();
    a.sort();
    b.sort();
    assert_eq!(a, b);
    scenes::delete("round trip");
    assert!(!scenes::list_scenes().contains(&"round trip".to_string()));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn presets_load_and_run() {
    for k in 0..scenes::PRESETS.len() {
        let mut sim = game_sim();
        scenes::load_preset(&mut sim, k);
        assert!(sim.len() > 100, "preset {k} is empty");
        for _ in 0..30 { sim.step(1.0 / 60.0, None); }
        assert!(sim.pos.iter().all(|p| p.is_finite()), "preset {k} blew up");
        // Nuclei stay whole: no nucleon wanders far from its nucleus.
        let max_speed = sim.vel.iter().map(|v| v.length()).fold(0.0, f32::max);
        assert!(max_speed <= sim.cfg.max_speed + 1.0);
    }
}

/// Elastic collisions between nuclei: a neutron hitting a nucleus at rest
/// transfers the textbook fraction 4A/(1+A)^2 of its energy.
#[test]
fn neutron_scattering_is_elastic() {
    for (z, n) in [(1u32, 1u32), (6, 6), (92, 146)] {
        let mut sim = game_sim();
        sim.cfg.matter = false;
        sim.cfg.coulomb = false;
        sim.cfg.decay_scale = 1e12;
        place_nucleus(&mut sim, vec2(600.0, 400.0), z, n, Vec2::ZERO);
        let target = sim.tag[0];
        // 150 px/s: ~0.15 MeV at game scale, below any reaction threshold.
        sim.add(vec2(400.0, 400.0), vec2(150.0, 0.0), ids::NEUTRON);
        sim.index_nuclei();
        for _ in 0..120 { sim.step(1.0 / 60.0, None); }
        // Particles are re-sorted every step: find the nucleus by its tag.
        let nucleus_ke: f32 = (0..sim.len()).filter(|&i| sim.tag[i] == target).map(|i| 0.5 * sim.vel[i].length_squared()).sum();
        let a = (z + n) as f32;
        let expect = 4.0 * a / (1.0 + a).powi(2) * 0.5 * 150.0 * 150.0;
        assert!((nucleus_ke - expect).abs() < 0.05 * expect + 1.0, "A={a}: {nucleus_ke} vs {expect}");
    }
}
