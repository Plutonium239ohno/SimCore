//! Control rods, fuel channels, the event history and its export.
use macroquad::math::{Vec2, vec2};
use sim_core::game_scenes as scenes;
use sim_core::nuclear::{EventKind, game_config, real_velocity, speed_for};
use sim_core::reactions::E_THERMAL;
use sim_core::reactor::{Rod, Site};
use sim_core::sim::Sim;
use sim_core::types::ids;

fn game_sim() -> Sim {
    sim_core::nuclear::game_sim(1200.0, 800.0)
}

fn free_neutrons(sim: &Sim) -> usize {
    (0..sim.len()).filter(|&i| sim.kind[i] == ids::NEUTRON).count()
}

/// Boron carbide is black to slow neutrons but nearly transparent to MeV ones.
#[test]
fn rods_absorb_slow_neutrons_not_fast_ones() {
    for (mev, expect_absorbed) in [(E_THERMAL, true), (2.0, false)] {
        let mut sim = game_sim();
        sim.seed = 3;
        // A wide slab so every neutron crosses ~40 px of absorber.
        sim.reactor.rods.push(Rod { x: 600.0, top: 50.0, len: 700.0, half_w: 20.0 });
        sim.reactor.insertion = 1.0;
        sim.reactor.target = 1.0;
        let v = speed_for(&sim.cfg, mev, 1.0);
        for k in 0..40 {
            sim.add(vec2(560.0, 100.0 + k as f32 * 15.0), vec2(v, 0.0), ids::NEUTRON);
        }
        sim.index_nuclei();
        // Long enough for the slowest to cross the slab, short of the far wall.
        let secs = (100.0 / v).min(2.0);
        for _ in 0..(secs * 60.0) as usize {
            sim.step(1.0 / 60.0, None);
        }
        let left = free_neutrons(&sim);
        if expect_absorbed {
            assert!(left <= 2, "thermal neutrons got through the rod: {left}/40 left");
            assert!(sim.reactor.rod_captures >= 38);
            assert!(sim.nuclear.history.iter().any(|e| e.text.contains("B-10 + n → Li-7 + α")));
        } else {
            assert!(left >= 30, "fast neutrons should mostly pass: {left}/40 left");
        }
    }
}

/// Withdrawn rods absorb nothing; the inserted part is solid to nuclei.
#[test]
fn withdrawn_rods_are_harmless_and_inserted_rods_push_nuclei() {
    let mut sim = game_sim();
    sim.reactor.rods.push(Rod { x: 600.0, top: 100.0, len: 600.0, half_w: 6.0 });
    sim.reactor.insertion = 0.0;
    sim.reactor.target = 0.0;
    let v = speed_for(&sim.cfg, E_THERMAL, 1.0);
    for k in 0..10 {
        sim.add(vec2(580.0, 200.0 + k as f32 * 20.0), vec2(v, 0.0), ids::NEUTRON);
    }
    assert!(sim_core::nuclear::place_nucleus(&mut sim, vec2(600.0, 650.0), 1, 1, Vec2::ZERO));
    sim.index_nuclei();
    for _ in 0..90 {
        sim.step(1.0 / 60.0, None);
    }
    assert_eq!(sim.reactor.rod_captures, 0);
    // Drive the rods in: the deuteron in the channel is shoved sideways.
    sim.reactor.target = 1.0;
    sim.reactor.drive = 2.0;
    for _ in 0..90 {
        sim.step(1.0 / 60.0, None);
    }
    assert_eq!(sim.reactor.insertion, 1.0);
    for i in 0..sim.len() {
        if sim.kind[i] == ids::PROTON {
            assert!((sim.pos[i].x - 600.0).abs() > 6.0, "deuteron left inside the rod at {:?}", sim.pos[i]);
        }
    }
}

#[test]
fn scram_drops_the_rods_fast() {
    let mut sim = game_sim();
    sim.reactor.rods.push(Rod { x: 600.0, top: 100.0, len: 600.0, half_w: 6.0 });
    sim.reactor.insertion = 0.0;
    sim.reactor.target = 0.0;
    sim.add(vec2(100.0, 100.0), Vec2::ZERO, ids::NEUTRON);
    sim.reactor.scram();
    for _ in 0..60 {
        sim.step(1.0 / 60.0, None);
    }
    assert_eq!(sim.reactor.insertion, 1.0, "rods should be fully in within a second of a scram");
}

#[test]
fn reactor_survives_save_and_load() {
    let dir = std::env::temp_dir().join(format!("nuclear-game-reactor-test-{}", std::process::id()));
    unsafe { std::env::set_var("NUCLEAR_GAME_SCENES", &dir) };
    let mut sim = game_sim();
    scenes::load_preset(&mut sim, 3);
    sim.reactor.target = 0.4;
    sim.reactor.insertion = 0.55;
    sim.reactor.auto_refuel = false;
    let (rods, sites, src) = (sim.reactor.rods.clone(), sim.reactor.sites.clone(), sim.reactor.sources.clone());
    assert!(rods.len() >= 3 && sites.len() >= 20 && !src.is_empty());
    scenes::save(&sim, "reactor").unwrap();
    let mut other = game_sim();
    scenes::load(&mut other, "reactor").unwrap();
    assert_eq!(other.reactor.rods, rods);
    assert_eq!(other.reactor.sites.len(), sites.len());
    assert_eq!(other.reactor.sites[3].pos, sites[3].pos);
    assert_eq!(other.reactor.sources, src);
    assert_eq!(other.reactor.target, 0.4);
    assert_eq!(other.reactor.insertion, 0.55);
    assert!(!other.reactor.auto_refuel);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A spent fuel channel is reloaded by the fuelling machine.
#[test]
fn fuelling_machine_reloads_spent_channels() {
    let mut sim = game_sim();
    let at = vec2(500.0, 400.0);
    sim.reactor.sites.push(Site { pos: at, z: 92, n: 143, empty_for: 0.0 });
    sim.reactor.refuel_delay = 0.5;
    // Spent fuel sits in the channel: U-236 (long-lived, not fissile).
    assert!(sim_core::nuclear::place_nucleus(&mut sim, at, 92, 144, Vec2::ZERO));
    sim.index_nuclei();
    for _ in 0..90 {
        sim.step(1.0 / 60.0, None);
    }
    assert_eq!(sim.reactor.refuels, 1);
    let census = &sim.nuclear.census;
    assert!(census.iter().any(|c| c.0 == (92, 143)), "fresh U-235 loaded: {census:?}");
    assert!(!census.iter().any(|c| c.0 == (92, 144)), "U-236 discharged: {census:?}");
}

/// The history keeps every event with its time; the CSV has them all.
#[test]
fn event_history_and_csv_export() {
    let mut sim = game_sim();
    sim.cfg.decay_scale = 0.05; // fast clock: free neutrons decay quickly
    for k in 0..30 {
        sim.add(vec2(100.0 + k as f32 * 30.0, 300.0), Vec2::ZERO, ids::NEUTRON);
    }
    sim.index_nuclei();
    for _ in 0..600 {
        sim.step(1.0 / 60.0, None);
    }
    let h = &sim.nuclear.history;
    assert!(h.len() >= 20, "expected neutron decays, got {}", h.len());
    assert!(h.iter().all(|e| e.kind == EventKind::Decay));
    assert!(h.iter().zip(h.iter().skip(1)).all(|(a, b)| a.t <= b.t), "events in time order");
    let e = &h[0];
    assert_eq!(e.equation(), "n → p + e- + ν̄");
    assert!((e.q().unwrap() - 0.782).abs() < 1e-3);

    let path = std::env::temp_dir().join(format!("nuclear-game-events-{}.csv", std::process::id()));
    let n = scenes::write_events_csv(&path, h.iter().enumerate().map(|(k, e)| (k as u64 + 1, e))).unwrap();
    assert_eq!(n, h.len());
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next().unwrap(), "index,sim_time_s,type,reaction,Q_MeV,details");
    let row: Vec<&str> = lines.next().unwrap().split(',').collect();
    assert_eq!(row[0], "1");
    assert_eq!(row[2], "decay");
    assert_eq!(row[3], "n → p + e- + ν̄");
    assert_eq!(row[4], "0.782");
    assert_eq!(text.lines().count(), h.len() + 1);
    let _ = std::fs::remove_file(&path);
}

/// Collision energies come from real relative velocities: two bodies moving
/// together have none, whatever the speed compression.
#[test]
fn relative_energy_uses_real_velocities() {
    let cfg = game_config();
    let v = vec2(300.0, 0.0);
    let (a, b) = (real_velocity(&cfg, v), real_velocity(&cfg, v));
    assert_eq!(a, b);
    // Head-on: a 1 MeV neutron into a body at rest carries 1 MeV.
    let u = real_velocity(&cfg, vec2(speed_for(&cfg, 1.0, 1.0), 0.0));
    let e = u.0 * u.0 + u.1 * u.1;
    assert!((e - 1.0).abs() < 1e-3, "{e}");
    // A thermal neutron meeting a fragment flying at 0.7 MeV/u sees about
    // 0.7 MeV, not the GeV that subtracting compressed speeds would give.
    let frag = real_velocity(&cfg, vec2(speed_for(&cfg, 0.7 * 100.0, 100.0), 0.0));
    let n = real_velocity(&cfg, vec2(-speed_for(&cfg, E_THERMAL, 1.0), 0.0));
    let mu = 100.0 / 101.0;
    let e_rel = mu * ((frag.0 - n.0).powi(2) + (frag.1 - n.1).powi(2));
    assert!(e_rel > 0.5 && e_rel < 1.0, "{e_rel}");
}
