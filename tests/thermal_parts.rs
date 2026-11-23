//! Wall kinds, heat transfer, freeze plugs, spawners, drains and pumps.
use macroquad::math::{Rect, Vec2, vec2};
use sim_core::game_scenes as scenes;
use sim_core::nuclear::{game_sim, place_nucleus, speed_for};
use sim_core::reactions::E_THERMAL;
use sim_core::reactor::{Drain, DrainFilter, Pump, Spawner};
use sim_core::sim::Sim;
use sim_core::thermal::{region_theta, thermalize};
use sim_core::types::ids;
use sim_core::walls::WallKind;

fn world() -> Sim {
    let mut sim = game_sim(1000.0, 700.0);
    sim.cfg.cooling = 0.0;
    sim
}

/// A free neutron goes through a conductor wall but bounces off an insulator.
#[test]
fn neutrons_pass_conductors_and_bounce_off_insulators() {
    for (kind, passes) in [(WallKind::Conductor, true), (WallKind::Insulator, false)] {
        let mut sim = world();
        sim.walls.half = 2.0;
        sim.walls.add_kind(vec2(500.0, 100.0), vec2(500.0, 600.0), kind);
        let v = speed_for(&sim.cfg, 1.0, 1.0);
        sim.add(vec2(450.0, 350.0), vec2(v, 0.0), ids::NEUTRON);
        sim.index_nuclei();
        for _ in 0..40 {
            sim.step(1.0 / 60.0, None);
        }
        let x = sim.pos.iter().zip(&sim.kind).find(|p| *p.1 == ids::NEUTRON).map(|p| p.0.x).unwrap();
        assert_eq!(x > 500.0, passes, "{kind:?}: neutron at x = {x}");
    }
}

/// Hot gas on one side of a conductor wall heats cold gas on the other.
#[test]
fn conductor_wall_carries_heat_across() {
    let mut sim = world();
    sim.walls.half = 2.0;
    // Two boxes sharing a conductor wall at x = 500.
    sim.walls.add_kind(vec2(300.0, 200.0), vec2(700.0, 200.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(300.0, 500.0), vec2(700.0, 500.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(300.0, 200.0), vec2(300.0, 500.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(700.0, 200.0), vec2(700.0, 500.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(500.0, 200.0), vec2(500.0, 500.0), WallKind::Conductor);
    for j in 0..8 {
        for i in 0..5 {
            place_nucleus(&mut sim, vec2(325.0 + i as f32 * 35.0, 225.0 + j as f32 * 34.0), 2, 2, Vec2::ZERO);
            place_nucleus(&mut sim, vec2(525.0 + i as f32 * 35.0, 225.0 + j as f32 * 34.0), 2, 2, Vec2::ZERO);
        }
    }
    sim.index_nuclei();
    // Everything cold, then the left side made 10x hotter.
    thermalize(&mut sim, 30.0);
    for i in 0..sim.len() {
        if sim.pos[i].x < 500.0 {
            sim.vel[i] *= 10f32.sqrt();
        }
    }
    let right_before = region_theta(&sim, |p| p.x > 500.0).unwrap();
    let left_before = region_theta(&sim, |p| p.x < 500.0).unwrap();
    assert!(left_before > 5.0 * right_before);
    // 40 atoms a side: one snapshot is noisy, so average the last 5 s.
    let (mut right_after, mut left_after, mut k) = (0.0, 0.0, 0.0);
    for f in 0..2100 {
        sim.step(1.0 / 60.0, None);
        if f % 300 == 0 {
            eprintln!("t={} wall {:.1} left {:.1} right {:.1}", f / 60, sim.walls.segs[4].theta, region_theta(&sim, |p| p.x < 500.0).unwrap(), region_theta(&sim, |p| p.x > 500.0).unwrap());
        }
        if f >= 1800 && f % 10 == 0 {
            right_after += region_theta(&sim, |p| p.x > 500.0).unwrap();
            left_after += region_theta(&sim, |p| p.x < 500.0).unwrap();
            k += 1.0;
        }
    }
    let (right_after, left_after) = (right_after / k, left_after / k);
    assert!(right_after > right_before * 1.25, "cold side warmed: {right_before} -> {right_after}");
    assert!(left_after < left_before * 0.9, "hot side cooled: {left_before} -> {left_after}");
    let wall = sim.walls.segs[4].theta;
    assert!(wall > right_before && wall < left_before, "wall in between: {wall}");
}

/// A freeze plug melts when hot matter heats it, and is logged.
#[test]
fn freeze_plug_melts_when_hot() {
    let mut sim = world();
    sim.cfg.melt_temp = 600.0;
    sim.walls.half = 2.0;
    // A small insulated box whose floor is a freeze plug.
    sim.walls.add_kind(vec2(400.0, 300.0), vec2(500.0, 300.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(400.0, 300.0), vec2(400.0, 400.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(500.0, 300.0), vec2(500.0, 400.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(400.0, 400.0), vec2(500.0, 400.0), WallKind::FreezePlug);
    for j in 0..4 {
        for i in 0..4 {
            place_nucleus(&mut sim, vec2(415.0 + i as f32 * 22.0, 315.0 + j as f32 * 22.0), 2, 2, Vec2::ZERO);
        }
    }
    sim.index_nuclei();
    thermalize(&mut sim, 3e6);
    for _ in 0..1800 {
        sim.step(1.0 / 60.0, None);
        if sim.walls.segs.len() < 4 {
            break;
        }
    }
    assert_eq!(sim.walls.segs.len(), 3, "plug should have melted");
    assert!(sim.nuclear.history.iter().any(|e| e.text.contains("freeze plug melted")));
}

/// Spawners add nuclei; drains take what enters and measure it; pumps push.
#[test]
fn spawner_drain_and_pump() {
    let mut sim = world();
    sim.reactor.spawners.push(Spawner::new(vec2(200.0, 350.0), 1, 0, vec2(40.0, 0.0), 10.0));
    sim.reactor.pumps.push(Pump { rect: Rect::new(150.0, 300.0, 300.0, 100.0), dir: vec2(1.0, 0.0), speed: 60.0 });
    sim.reactor.drains.push(Drain::new(Rect::new(700.0, 250.0, 60.0, 200.0), DrainFilter::All));
    for _ in 0..600 {
        sim.step(1.0 / 60.0, None);
    }
    let sp = &sim.reactor.spawners[0];
    let d = &sim.reactor.drains[0];
    assert!(sp.spawned >= 80, "spawned {}", sp.spawned);
    assert!(d.removed >= 10, "drained {}", d.removed);
    assert!(d.speed > 20.0, "outflow speed {}", d.speed);
}

/// Drain filters: the off-gas drain takes xenon but leaves fuel.
#[test]
fn drain_filters() {
    let f = DrainFilter::OffGas;
    assert!(f.takes(54, 81));
    assert!(!f.takes(92, 143));
    assert!(DrainFilter::FissionProducts.takes(56, 88));
    assert!(!DrainFilter::FissionProducts.takes(92, 143));
    assert!(DrainFilter::NotFuel.takes(9, 10));
    assert!(!DrainFilter::NotFuel.takes(94, 145));
}

/// Several sources, spawners, drains, pumps and wall kinds survive a save.
#[test]
fn parts_survive_save_and_load() {
    let dir = std::env::temp_dir().join(format!("nuclear-game-parts-test-{}", std::process::id()));
    unsafe { std::env::set_var("NUCLEAR_GAME_SCENES", &dir) };
    let mut sim = world();
    scenes::load_preset(&mut sim, scenes::MSR);
    sim.reactor.sources.push(vec2(10.0, 10.0));
    sim.walls.segs[1].theta = 1234.0;
    let re = &sim.reactor;
    let (src, sp, dr, pm) = (re.sources.clone(), re.spawners.clone(), re.drains.len(), re.pumps.clone());
    let kinds: Vec<WallKind> = sim.walls.segs.iter().map(|s| s.kind).collect();
    scenes::save(&sim, "parts").unwrap();
    let mut other = world();
    scenes::load(&mut other, "parts").unwrap();
    assert_eq!(other.reactor.sources, src);
    assert_eq!(other.reactor.spawners.len(), sp.len());
    assert_eq!(other.reactor.spawners.iter().map(|s| (s.keep, s.area)).collect::<Vec<_>>(), sp.iter().map(|s| (s.keep, s.area)).collect::<Vec<_>>());
    assert_eq!(other.reactor.drains.len(), dr);
    assert_eq!(other.reactor.pumps, pm);
    assert_eq!(other.walls.segs.iter().map(|s| s.kind).collect::<Vec<_>>(), kinds);
    assert_eq!(other.walls.segs[1].theta, 1234.0);
    assert_eq!(other.cfg.cooling, 0.0);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = E_THERMAL;
}

/// Gas in an insulated box keeps its heat (the engine doesn't leak energy).
#[test]
fn insulated_gas_keeps_its_heat() {
    let mut sim = world();
    if let Some(d) = std::env::var("NUC_DAMP").ok().and_then(|s| s.parse().ok()) {
        sim.cfg.nuclear_damping = d;
    }
    let (gz, gn) = if std::env::var("GAS_H").is_ok() { (1, 0) } else { (2, 2) };
    sim.walls.half = 2.0;
    sim.walls.add_kind(vec2(300.0, 200.0), vec2(500.0, 200.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(300.0, 500.0), vec2(500.0, 500.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(300.0, 200.0), vec2(300.0, 500.0), WallKind::Insulator);
    sim.walls.add_kind(vec2(500.0, 200.0), vec2(500.0, 500.0), WallKind::Insulator);
    for j in 0..8 {
        for i in 0..5 {
            place_nucleus(&mut sim, vec2(325.0 + i as f32 * 35.0, 225.0 + j as f32 * 34.0), gz, gn, Vec2::ZERO);
        }
    }
    sim.index_nuclei();
    thermalize(&mut sim, 3000.0);
    let before = region_theta(&sim, |_| true).unwrap();
    for f in 0..1800 {
        sim.step(1.0 / 60.0, None);
        if f % 300 == 0 {
            eprintln!("t={} gas {:.1}", f / 60, region_theta(&sim, |_| true).unwrap());
        }
    }
    let after = region_theta(&sim, |_| true).unwrap();
    assert!(after > before * 0.95 && after < before * 1.05, "gas energy drifted: {before} -> {after}");
}
