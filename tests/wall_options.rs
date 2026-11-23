//! Per-wall settings: thickness, neutron reflection, filter rules (charge,
//! energy, mass number), and heat not leaking through walls.
use macroquad::math::vec2;
use sim_core::game_scenes as scenes;
use sim_core::nuclear::{energy_of, game_sim, place_nucleus, speed_for};
use sim_core::sim::Sim;
use sim_core::types::ids;
use sim_core::walls::{FilterRule, WallKind, WallSpec};

fn run(sim: &mut Sim, frames: usize) {
    sim.index_nuclei();
    for _ in 0..frames {
        sim.step(1.0 / 60.0, None);
    }
}

fn wall(sim: &mut Sim, spec: WallSpec) {
    sim.walls.add_spec(vec2(500.0, 100.0), vec2(500.0, 600.0), &spec);
}

/// A conductor passes neutrons by default and reflects them when set to.
#[test]
fn conductor_can_reflect_neutrons() {
    for reflect in [false, true] {
        let mut sim = game_sim(1000.0, 700.0);
        wall(&mut sim, WallSpec { reflect_neutrons: reflect, ..WallSpec::new(WallKind::Conductor, 2.0) });
        sim.add(vec2(450.0, 350.0), vec2(speed_for(&sim.cfg, 1.0, 1.0), 0.0), ids::NEUTRON);
        run(&mut sim, 40);
        assert_eq!(sim.pos[0].x > 500.0, !reflect, "reflect={reflect}: neutron at {}", sim.pos[0].x);
    }
    // ...and an insulator can be made a neutron window.
    let mut sim = game_sim(1000.0, 700.0);
    wall(&mut sim, WallSpec { reflect_neutrons: false, ..WallSpec::new(WallKind::Insulator, 2.0) });
    sim.add(vec2(450.0, 350.0), vec2(speed_for(&sim.cfg, 1.0, 1.0), 0.0), ids::NEUTRON);
    run(&mut sim, 40);
    assert!(sim.pos[0].x > 500.0);
}

fn filter(rule: FilterRule, z: u32, threshold: f32, only: bool) -> WallSpec {
    WallSpec { rule, filter_z: z, threshold, only, ..WallSpec::new(WallKind::Filter, 2.0) }
}

/// Charge filters: charged-only passes the helium and stops the neutron;
/// neutral-only the reverse.
#[test]
fn charge_filter() {
    for points in [false, true] {
        for only in [true, false] {
            let mut sim = game_sim(1000.0, 700.0);
            sim.point_nuclei = points;
            sim.cfg.cooling = 0.0;
            wall(&mut sim, filter(FilterRule::Charged, 0, 0.0, only));
            assert!(place_nucleus(&mut sim, vec2(440.0, 250.0), 2, 2, vec2(60.0, 0.0)));
            sim.add(vec2(450.0, 450.0), vec2(speed_for(&sim.cfg, 1.0, 1.0), 0.0), ids::NEUTRON);
            run(&mut sim, 90);
            for b in sim.bodies().values().filter(|b| b.z == 2) {
                assert_eq!(b.pos.x > 500.0, only, "points {points} only {only}: He at {}", b.pos.x);
            }
            let n = sim.bodies().values().find(|b| (b.z, b.n) == (0, 1)).unwrap().pos.x;
            assert_eq!(n > 500.0, !only, "points {points} only {only}: neutron at {n}");
        }
    }
}

/// Energy filters: a fast and a slow helium; only the one on the passing
/// side of the threshold gets through.
#[test]
fn energy_filter() {
    for points in [false, true] {
        for only in [true, false] {
            let mut sim = game_sim(1000.0, 700.0);
            sim.point_nuclei = points;
            sim.cfg.cooling = 0.0;
            let (slow, fast) = (25.0, 90.0);
            let thr = (energy_of(&sim.cfg, slow, 4.0) * energy_of(&sim.cfg, fast, 4.0)).sqrt() as f32;
            wall(&mut sim, filter(FilterRule::Energy, 0, thr, only));
            assert!(place_nucleus(&mut sim, vec2(470.0, 250.0), 2, 2, vec2(slow, 0.0)));
            assert!(place_nucleus(&mut sim, vec2(440.0, 450.0), 2, 2, vec2(fast, 0.0)));
            run(&mut sim, 120);
            for b in sim.bodies().values() {
                let was_fast = b.pos.y > 350.0;
                assert_eq!(b.pos.x > 500.0, was_fast == only, "points {points} only {only}: fast={was_fast} at {}", b.pos.x);
            }
        }
    }
}

/// Mass-number filters: A >= 10 passes carbon, not helium.
#[test]
fn mass_filter() {
    for points in [false, true] {
        let mut sim = game_sim(1000.0, 700.0);
        sim.point_nuclei = points;
        sim.cfg.cooling = 0.0;
        wall(&mut sim, filter(FilterRule::Mass, 10, 0.0, true));
        let v = vec2(60.0, 0.0);
        assert!(place_nucleus(&mut sim, vec2(440.0, 250.0), 2, 2, v));
        assert!(place_nucleus(&mut sim, vec2(440.0, 450.0), 6, 6, v));
        run(&mut sim, 90);
        for b in sim.bodies().values() {
            assert_eq!(b.pos.x > 500.0, b.z == 6, "points {points}: Z={} at {}", b.z, b.pos.x);
        }
    }
}

/// Changing the thickness only affects walls drawn afterwards, and it
/// survives saving.
#[test]
fn thickness_is_per_wall_and_saved() {
    let mut sim = game_sim(1000.0, 700.0);
    sim.walls.half = 2.0;
    sim.walls.add_kind(vec2(100.0, 100.0), vec2(300.0, 100.0), WallKind::Conductor);
    sim.walls.half = 6.0;
    sim.walls.add_kind(vec2(100.0, 300.0), vec2(300.0, 300.0), WallKind::Insulator);
    let spec = WallSpec { reflect_neutrons: true, ..filter(FilterRule::Energy, 0, 0.5, true) };
    sim.walls.add_spec(vec2(100.0, 500.0), vec2(300.0, 500.0), &WallSpec { half: 3.5, ..spec });
    sim.walls.add_spec(vec2(600.0, 100.0), vec2(600.0, 500.0), &WallSpec { reflect_neutrons: true, ..WallSpec::new(WallKind::Conductor, 1.0) });
    assert_eq!(sim.walls.segs[0].half, 2.0);
    // A particle 4 px from the thin wall is clear of it, not of the thick one.
    assert!(!sim.walls.blocked(vec2(200.0, 104.5), 2.0));
    assert!(sim.walls.blocked(vec2(200.0, 304.5), 2.0));

    let name = format!("test-walls-{}", std::process::id());
    scenes::save(&sim, &name).unwrap();
    let mut other = game_sim(1000.0, 700.0);
    scenes::load(&mut other, &name).unwrap();
    scenes::delete(&name);
    let got: Vec<WallSpec> = other.walls.segs.iter().map(|s| s.spec()).collect();
    let want: Vec<WallSpec> = sim.walls.segs.iter().map(|s| s.spec()).collect();
    assert_eq!(got, want);
    assert!(other.walls.needs_energy);
}

/// A fast ion slowing down next to a wall heats what is on its side, not
/// what is on the other side (it used to kick atoms through the wall).
#[test]
fn stopping_heat_stays_behind_walls() {
    let kick = |with_wall: bool| {
        let mut sim = game_sim(1000.0, 700.0);
        sim.point_nuclei = true;
        sim.cfg.cooling = 0.0;
        sim.cfg.heat_fraction = 1.0;
        if with_wall {
            sim.walls.add_kind(vec2(500.0, 100.0), vec2(500.0, 600.0), WallKind::Insulator);
        }
        // A 20 MeV carbon heading away from the wall, a helium at rest 23 px
        // away on the far side.
        let v = vec2(-speed_for(&sim.cfg, 20.0, 12.0), 0.0);
        assert!(place_nucleus(&mut sim, vec2(490.0, 350.0), 6, 6, v));
        assert!(place_nucleus(&mut sim, vec2(508.0, 350.0), 2, 2, vec2(0.0, 0.0)));
        // Matter on the carbon's own side for it to slow down in.
        assert!(place_nucleus(&mut sim, vec2(490.0, 357.0), 1, 1, vec2(0.0, 0.0)));
        assert!(place_nucleus(&mut sim, vec2(490.0, 343.0), 1, 1, vec2(0.0, 0.0)));
        run(&mut sim, 1);
        sim.bodies().values().find(|b| b.z == 2).unwrap().vel.length()
    };
    let open = kick(false);
    let walled = kick(true);
    assert!(open > 1.0, "without a wall the helium should be heated ({open} px/s)");
    assert!(walled < 1e-3, "through a wall the helium was kicked to {walled} px/s");
}
