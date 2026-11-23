//! Filter walls, and point-nucleus mode (one particle per nucleus,
//! neutron interactions from cross-sections).
use macroquad::math::{Vec2, vec2};
use sim_core::game_scenes as scenes;
use sim_core::nuclear::{energy_of, game_sim, place_nucleus, set_point_mode, speed_for};
use sim_core::reactions::E_THERMAL;
use sim_core::sim::Sim;
use sim_core::types::ids;

fn census(sim: &Sim) -> Vec<(u32, u32, u32)> {
    let mut c: std::collections::BTreeMap<(u32, u32), u32> = Default::default();
    for b in sim.bodies().values() {
        *c.entry((b.z, b.n)).or_default() += 1;
    }
    c.into_iter().map(|((z, n), k)| (z, n, k)).collect()
}

/// A filter passes its element and stops the rest (or the reverse).
#[test]
fn filter_walls_sort_by_element() {
    for points in [false, true] {
        for (only, he_passes) in [(true, true), (false, false)] {
            let mut sim = game_sim(1000.0, 700.0);
            sim.point_nuclei = points;
            sim.cfg.cooling = 0.0;
            sim.walls.half = 2.0;
            sim.walls.add_filter(vec2(500.0, 100.0), vec2(500.0, 600.0), 2, only);
            let v = vec2(60.0, 0.0);
            assert!(place_nucleus(&mut sim, vec2(440.0, 250.0), 2, 2, v)); // He-4
            assert!(place_nucleus(&mut sim, vec2(440.0, 450.0), 6, 6, v)); // C-12
            sim.index_nuclei();
            for _ in 0..90 {
                sim.step(1.0 / 60.0, None);
            }
            for b in sim.bodies().values() {
                let passes = if b.z == 2 { he_passes } else { !he_passes };
                assert_eq!(b.pos.x > 500.0, passes, "points {points} only {only}: Z={} at x={}", b.z, b.pos.x);
            }
        }
    }
}

/// A filter set to neutrons (Z = 0) lets free neutrons through and nothing else.
#[test]
fn neutron_filter() {
    for only in [true, false] {
        let mut sim = game_sim(1000.0, 700.0);
        sim.walls.half = 2.0;
        sim.walls.add_filter(vec2(500.0, 100.0), vec2(500.0, 600.0), 0, only);
        sim.add(vec2(450.0, 350.0), vec2(speed_for(&sim.cfg, 1.0, 1.0), 0.0), ids::NEUTRON);
        sim.index_nuclei();
        for _ in 0..40 {
            sim.step(1.0 / 60.0, None);
        }
        let x = sim.pos[0].x;
        assert_eq!(x > 500.0, only, "only={only}: neutron at {x}");
    }
}

/// Clusters -> points -> clusters keeps every nucleus.
#[test]
fn mode_switch_keeps_nuclei() {
    let mut sim = game_sim(1600.0, 900.0);
    scenes::load_preset(&mut sim, 4);
    let before = census(&sim);
    let n_clusters = sim.len();
    assert_eq!(set_point_mode(&mut sim, true), 0);
    assert!(sim.len() < n_clusters / 5, "points: {} particles", sim.len());
    assert_eq!(census(&sim), before);
    assert_eq!(set_point_mode(&mut sim, false), 0);
    assert_eq!(census(&sim), before);
    assert!(!sim.point_nuclei);
}

/// Thermal neutrons aimed past a point U-235 within its interaction radius
/// are absorbed most of the time (683 b absorption vs 15 b scattering), and
/// absorption mostly means fission.
#[test]
fn point_uranium_fissions_with_thermal_neutrons() {
    let mut fissions = 0;
    let trials = 20;
    for k in 0..trials {
        let mut sim = game_sim(1000.0, 700.0);
        sim.seed = 100 + k;
        sim.point_nuclei = true;
        assert!(place_nucleus(&mut sim, vec2(500.0, 350.0), 92, 143, Vec2::ZERO));
        assert_eq!(sim.len(), 1, "one particle per nucleus");
        let v = speed_for(&sim.cfg, E_THERMAL, 1.0);
        // Passes 5 px from the centre: inside the ~25 px thermal radius,
        // but the particles never touch (no collisions for neutrons here).
        sim.add(vec2(470.0, 355.0), vec2(v, 0.0), ids::NEUTRON);
        sim.index_nuclei();
        for _ in 0..600 {
            sim.step(1.0 / 60.0, None);
            if sim.nuclear.fissions > 0 {
                break;
            }
        }
        fissions += sim.nuclear.fissions;
    }
    assert!(fissions >= 12, "{fissions} fissions in {trials}");
}

/// Fast neutrons in a box of point deuterium slow down to near thermal.
#[test]
fn point_moderation_thermalizes() {
    let mut sim = game_sim(1000.0, 700.0);
    sim.seed = 5;
    sim.point_nuclei = true;
    sim.walls.half = 2.0;
    let (x0, y0, x1, y1) = (200.0, 150.0, 800.0, 550.0);
    for (a, b) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))] {
        sim.walls.add(vec2(a.0, a.1), vec2(b.0, b.1));
    }
    let mut y = y0 + 8.0;
    let mut row = 0;
    while y < y1 - 8.0 {
        let mut x = x0 + 8.0 + if row % 2 == 1 { 6.0 } else { 0.0 };
        while x < x1 - 8.0 {
            place_nucleus(&mut sim, vec2(x, y), 1, 1, Vec2::ZERO);
            x += 12.0;
        }
        y += 10.4;
        row += 1;
    }
    sim_core::thermal::thermalize(&mut sim, 293.0);
    let v = speed_for(&sim.cfg, 2.0, 1.0);
    for j in 0..60 {
        let a = j as f32 * 0.3;
        sim.add(vec2(500.0, 350.0), vec2(a.cos(), a.sin()) * v, ids::NEUTRON);
    }
    sim.index_nuclei();
    for _ in 0..60 * 8 {
        sim.step(1.0 / 60.0, None);
    }
    let mut es: Vec<f64> = (0..sim.len()).filter(|&i| sim.kind[i] == ids::NEUTRON).map(|i| energy_of(&sim.cfg, sim.vel[i].length(), 1.0)).collect();
    es.sort_by(f64::total_cmp);
    let median = es[es.len() / 2];
    assert!(median < 1e-6, "median neutron energy {median} MeV after 8 s");
}

/// Saving and loading keeps point nuclei, the mode and filter walls.
#[test]
fn point_scene_round_trip() {
    let mut sim = game_sim(1600.0, 900.0);
    sim.point_nuclei = true;
    scenes::load_preset(&mut sim, 3);
    sim.walls.add_filter(vec2(10.0, 10.0), vec2(60.0, 10.0), 54, true);
    let before = census(&sim);
    let name = format!("test-points-{}", std::process::id());
    scenes::save(&sim, &name).unwrap();
    let mut other = game_sim(1600.0, 900.0);
    scenes::load(&mut other, &name).unwrap();
    scenes::delete(&name);
    assert!(other.point_nuclei);
    assert_eq!(census(&other), before);
    let f = other.walls.segs.last().unwrap();
    assert_eq!((f.kind, f.filter_z, f.only), (sim_core::walls::WallKind::Filter, 54, true));
}
