//! Beams, fields, detectors, automation, fragmentation, lineage, decay heat.

use macroquad::math::{Rect, Vec2, vec2};
use sim_core::control::{Action, Controller, Measure, Rule, Trigger};
use sim_core::detector::{DetShape, DetectFilter, Detector, Species};
use sim_core::field::Field;
use sim_core::game_scenes as scenes;
use sim_core::nuclear::{EventKind, energy_of, game_sim, place_nucleus, speed_for};
use sim_core::reactor::Spawner;
use sim_core::sim::Sim;
use sim_core::types::ids;

fn vacuum() -> Sim {
    let mut sim = game_sim(1200.0, 800.0);
    sim.seed = 3;
    sim.cfg = sim.cfg.clone().with_medium(false);
    sim.cfg.coulomb = false;
    sim.cfg.decay_scale = 1e12;
    sim
}

fn run(sim: &mut Sim, frames: usize) {
    for _ in 0..frames {
        sim.step(1.0 / 60.0, None);
    }
}

#[test]
fn beam_spawner_sets_energy_and_spread() {
    let mut sim = vacuum();
    let mut b = Spawner::beam(&sim.cfg, vec2(200.0, 400.0), 2, 2, Vec2::X, 5.0, 30.0);
    b.width = 80.0;
    assert!((b.energy_per_nucleon(&sim.cfg) - 5.0).abs() < 0.05);
    sim.reactor.spawners.push(b);
    run(&mut sim, 30);
    let bodies = sim.bodies();
    let alphas: Vec<_> = bodies.values().filter(|b| (b.z, b.n) == (2, 2)).collect();
    assert!(alphas.len() >= 8, "{} alphas", alphas.len());
    for a in &alphas {
        let e = energy_of(&sim.cfg, a.vel.length(), 4.0) / 4.0;
        assert!((e - 5.0).abs() < 0.3, "alpha at {e} MeV/u");
    }
    let ys: Vec<f32> = alphas.iter().map(|a| a.pos.y).collect();
    let spread = ys.iter().cloned().fold(f32::MIN, f32::max) - ys.iter().cloned().fold(f32::MAX, f32::min);
    assert!(spread > 25.0, "beam width {spread}");
    // Mirror meets head on.
    let m = sim.reactor.spawners[0].mirrored(vec2(600.0, 400.0));
    assert!((m.pos - vec2(1000.0, 400.0)).length() < 1e-3 && m.vel.x < 0.0);
}

#[test]
fn pulsed_beam_and_burst() {
    let mut sim = vacuum();
    let mut b = Spawner::beam(&sim.cfg, vec2(200.0, 400.0), 1, 0, Vec2::X, 1.0, 20.0);
    b.pulse_period = 1.0;
    b.pulse_len = 0.25;
    sim.reactor.spawners.push(b);
    run(&mut sim, 120);
    let n = sim.reactor.spawners[0].spawned;
    // 2 s at 20/s for a quarter of the time: ~10.
    assert!((6..=14).contains(&n), "{n} spawned");
    sim.reactor.spawners[0].burst = 7;
    sim.reactor.spawners[0].pulse_len = 0.0;
    run(&mut sim, 2);
    assert_eq!(sim.reactor.spawners[0].spawned, n + 7);
}

#[test]
fn magnetic_field_turns_protons_not_neutrons() {
    let mut sim = vacuum();
    sim.reactor.fields.push(Field::new(Rect::new(0.0, 0.0, 1200.0, 800.0), Vec2::ZERO, 2.0));
    sim.add(vec2(600.0, 300.0), vec2(100.0, 0.0), ids::PROTON);
    sim.add(vec2(600.0, 500.0), vec2(100.0, 0.0), ids::NEUTRON);
    sim.index_nuclei();
    // Half a turn: pi / 2 rad/s.
    run(&mut sim, (std::f32::consts::PI / 2.0 * 60.0) as usize);
    let p = (0..sim.len()).find(|&i| sim.kind[i] == ids::PROTON).unwrap();
    let n = (0..sim.len()).find(|&i| sim.kind[i] == ids::NEUTRON).unwrap();
    assert!(sim.vel[p].x < -90.0, "proton should be heading back: {:?}", sim.vel[p]);
    assert!((sim.vel[p].length() - 100.0).abs() < 2.0, "speed kept: {}", sim.vel[p].length());
    // Positive B turns anticlockwise on screen: up (−y) first, so it ends above.
    assert!(sim.pos[p].y < 300.0 - 80.0, "{:?}", sim.pos[p]);
    assert!((sim.vel[n] - vec2(100.0, 0.0)).length() < 0.05);
}

#[test]
fn detectors_filter_by_charge_and_isotope() {
    let mut sim = vacuum();
    let line = DetShape::Line(vec2(600.0, 0.0), vec2(600.0, 800.0));
    for f in [DetectFilter::All, DetectFilter::Neutral, DetectFilter::Charged, DetectFilter::Positive, DetectFilter::Negative, DetectFilter::Isotope(2, 2)] {
        sim.reactor.detectors.push(Detector::new(line, f, false));
    }
    sim.add(vec2(550.0, 100.0), vec2(300.0, 0.0), ids::NEUTRON);
    sim.add(vec2(550.0, 200.0), vec2(300.0, 0.0), ids::ELECTRON);
    sim.add(vec2(550.0, 300.0), vec2(300.0, 0.0), ids::PROTON);
    assert!(place_nucleus(&mut sim, vec2(550.0, 500.0), 2, 2, vec2(300.0, 0.0)));
    sim.index_nuclei();
    run(&mut sim, 20);
    let c: Vec<u64> = sim.reactor.detectors.iter().map(|d| d.count).collect();
    assert_eq!(c, vec![4, 1, 3, 2, 1, 1], "all, neutral, charged, +, -, He-4");
    assert_eq!(sim.reactor.detectors[5].species.get(&Species::Nuclide(2, 2)), Some(&1));
    assert!(sim.reactor.detectors[0].with_energy == 4);
    // A box counts what is made inside it too.
    let mut sim = vacuum();
    sim.reactor.detectors.push(Detector::new(DetShape::Box(Rect::new(500.0, 300.0, 200.0, 200.0)), DetectFilter::All, false));
    sim.add(vec2(600.0, 400.0), Vec2::ZERO, ids::NEUTRON);
    sim.index_nuclei();
    run(&mut sim, 5);
    assert_eq!(sim.reactor.detectors[0].count, 1);
}

#[test]
fn rules_fire_on_time_and_first_reaction() {
    let mut sim = vacuum();
    sim.reactor.rods.push(sim_core::reactor::Rod { x: 100.0, top: 100.0, len: 300.0, half_w: 3.0 });
    sim.reactor.rules.push(Rule::new(Trigger::At(0.5), Action::Rods(0.25)));
    sim.reactor.rules.push(Rule::new(Trigger::First(EventKind::Decay), Action::Pause));
    sim.add(vec2(800.0, 400.0), Vec2::ZERO, ids::NEUTRON);
    sim.index_nuclei();
    run(&mut sim, 20);
    assert_eq!(sim.reactor.target, 1.0);
    run(&mut sim, 20);
    assert_eq!(sim.reactor.target, 0.25);
    assert!(!sim.reactor.pause_request);
    sim.cfg.decay_scale = 1e-6; // the neutron decays at once
    run(&mut sim, 3);
    assert!(sim.reactor.pause_request);
    assert!(sim.reactor.rules[1].fired == 1);
    // Round trip through text.
    for r in &sim.reactor.rules {
        let line = r.to_line();
        let f: Vec<&str> = line.split_whitespace().collect();
        let back = Rule::from_fields(&f).unwrap();
        assert_eq!(back.trigger, r.trigger);
        assert_eq!(back.action, r.action);
    }
    let c = Controller::new(Measure::FusionRate, 3.0, sim_core::control::Actuator::Rods);
    let f: Vec<String> = c.to_line().split_whitespace().map(String::from).collect();
    let f: Vec<&str> = f.iter().map(|s| s.as_str()).collect();
    assert_eq!(Controller::from_fields(&f).unwrap().measure, Measure::FusionRate);
}

#[test]
fn feed_controller_throttles_beams() {
    // Hold the fusion rate of a D beam on a D target down with the throttle.
    let mut sim = vacuum();
    sim.cfg.tunneling = 1000.0;
    sim.reactor.controller = Controller::new(Measure::FusionRate, 0.01, sim_core::control::Actuator::Feeds);
    let b = Spawner::beam(&sim.cfg, vec2(100.0, 400.0), 1, 1, Vec2::X, 0.2, 20.0);
    sim.reactor.spawners.push(b);
    for y in 0..10 {
        place_nucleus(&mut sim, vec2(600.0, 300.0 + y as f32 * 20.0), 1, 1, Vec2::ZERO);
    }
    sim.index_nuclei();
    run(&mut sim, 60 * 12);
    assert!(sim.reactor.throttle < 0.9, "throttle {}", sim.reactor.throttle);
}

#[test]
fn fragmentation_conserves_charge_and_mass() {
    let mut fragmented = 0;
    for seed in 0..40 {
        let mut sim = vacuum();
        sim.point_nuclei = true;
        sim.seed = seed;
        let v = speed_for(&sim.cfg, 200.0, 1.0).min(sim.cfg.max_speed);
        assert!(place_nucleus(&mut sim, vec2(560.0, 400.0), 20, 20, vec2(v, 0.0)));
        assert!(place_nucleus(&mut sim, vec2(640.0, 400.0), 20, 20, vec2(-v, 0.0)));
        sim.index_nuclei();
        run(&mut sim, 20);
        let st = &sim.nuclear;
        if let Some((g, e)) = st.history.iter().enumerate().find(|(_, e)| e.text.contains("fragmentation")) {
            let g = st.dropped + g as u64;
            fragmented += 1;
            assert_eq!(e.kind, EventKind::Breakup);
            // Every nucleon is still there (decays are off).
            let b = sim.bodies();
            let (z, a): (u32, u32) = b.values().fold((0, 0), |s, b| (s.0 + b.z, s.1 + b.a()));
            assert_eq!((z, a), (40, 80), "{}", e.text);
            assert!(e.q().unwrap() < 0.0, "shattering costs energy: {}", e.text);
            // The products were born from the two calciums.
            let child = st.born.values().find(|bi| bi.event == Some(g)).unwrap();
            assert!(child.parents.iter().all(|p| sim_core::sim::unpack_zn(p.1) == (20, 20)));
        }
    }
    assert!(fragmented >= 10, "only {fragmented} of 40 collisions at 200 MeV/u fragmented");
}

#[test]
fn decay_heat_is_measured() {
    let mut sim = game_sim(1200.0, 800.0);
    sim.seed = 2;
    sim.cfg.decay_scale = 0.05;
    for k in 0..30 {
        place_nucleus(&mut sim, vec2(200.0 + (k % 10) as f32 * 60.0, 200.0 + (k / 10) as f32 * 80.0), 27, 33, Vec2::ZERO);
    }
    sim.index_nuclei();
    run(&mut sim, 600);
    assert!(sim.nuclear.decay_heat > 0.0);
    assert!(sim.reactor.samples.iter().any(|s| s.decay_w > 0.0));
    let inv = sim_core::inventory::take(&sim.cfg, sim.nuclear.census.iter().map(|c| (c.0, c.1 as f64)));
    assert!(inv.heat_w > 0.0);
    assert_eq!(sim_core::inventory::category(92, 143), sim_core::inventory::Category::FissileFuel);
    assert_eq!(sim_core::inventory::category(2, 2), sim_core::inventory::Category::FusionAsh);
    assert_eq!(sim_core::inventory::category(54, 81), sim_core::inventory::Category::Poison);
}

#[test]
fn new_parts_survive_save_and_load() {
    let mut sim = vacuum();
    let mut b = Spawner::beam(&sim.cfg, vec2(100.0, 400.0), 1, 1, Vec2::X, 2.0, 5.0);
    b.width = 40.0;
    b.pulse_period = 2.0;
    sim.reactor.spawners.push(b);
    sim.reactor.fields.push(Field::new(Rect::new(10.0, 20.0, 300.0, 200.0), vec2(5.0, 0.0), -1.5));
    sim.reactor.detectors.push(Detector::new(DetShape::Line(vec2(1.0, 2.0), vec2(3.0, 400.0)), DetectFilter::Isotope(1, 2), true));
    sim.reactor.rules.push(Rule::new(Trigger::Above(Measure::Neutrons, 50.0), Action::Scram));
    sim.reactor.controller = Controller::new(Measure::FissionRate, 2.0, sim_core::control::Actuator::Rods);
    let text = scenes::to_text(&sim);
    let mut other = vacuum();
    scenes::from_text(&mut other, &text).unwrap();
    let re = &other.reactor;
    assert_eq!(re.spawners[0].width, 40.0);
    assert!(re.spawners[0].beam && re.spawners[0].pulse_period == 2.0);
    assert_eq!(re.fields, sim.reactor.fields);
    assert_eq!(re.detectors[0].filter, DetectFilter::Isotope(1, 2));
    assert!(re.detectors[0].light);
    assert_eq!(re.rules[0].trigger, Trigger::Above(Measure::Neutrons, 50.0));
    assert!(re.controller.on);
}

#[test]
fn spawner_mixes_nuclides() {
    // In turn: D, T, T, D, T, T...
    let mut sim = vacuum();
    let mut b = Spawner::beam(&sim.cfg, vec2(200.0, 400.0), 1, 1, Vec2::X, 1.0, 30.0);
    b.width = 200.0;
    b.set_mix(&[(1, 1, 1), (1, 2, 2)]);
    assert!(b.is_mix() && b.label() == "H-2 + 2×H-3", "{}", b.label());
    sim.reactor.spawners.push(b);
    run(&mut sim, 60);
    let bodies = sim.bodies();
    let d = bodies.values().filter(|b| (b.z, b.n) == (1, 1)).count() as f32;
    let t = bodies.values().filter(|b| (b.z, b.n) == (1, 2)).count() as f32;
    assert!(d > 3.0 && (t / d - 2.0).abs() < 0.5, "D {d} T {t}");
    // At random, by weight.
    let mut sim = vacuum();
    let mut b = Spawner::beam(&sim.cfg, vec2(200.0, 400.0), 1, 0, Vec2::X, 1.0, 30.0);
    b.width = 300.0;
    b.set_mix(&[(1, 0, 3), (2, 2, 1), (5, 6, 0)]);
    b.mix_random = true;
    sim.reactor.spawners.push(b);
    run(&mut sim, 240);
    let bodies = sim.bodies();
    let p = bodies.values().filter(|b| (b.z, b.n) == (1, 0)).count();
    let a = bodies.values().filter(|b| (b.z, b.n) == (2, 2)).count();
    assert!(bodies.values().all(|b| (b.z, b.n) != (5, 6)));
    assert!(p > a && a > 3, "p {p} alpha {a}");
    // Saved and loaded.
    let text = scenes::to_text(&sim);
    let mut other = vacuum();
    scenes::from_text(&mut other, &text).unwrap();
    assert_eq!(other.reactor.spawners[0].nuclides(), vec![(1, 0, 3), (2, 2, 1)]);
    assert!(other.reactor.spawners[0].mix_random);
}
