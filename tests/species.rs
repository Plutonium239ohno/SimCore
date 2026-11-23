use macroquad::math::vec2;
use sim_core::sim::Sim;
use sim_core::types::ids;

#[test]
fn add_and_remove_species_while_running() {
    let mut sim = Sim::new(800.0, 600.0);
    sim.cfg.rules = true;
    sim.spawn_random(500);
    let k = sim.add_species().unwrap();
    assert_eq!(sim.rules.n, sim.types.len());
    sim.rules.set(k as usize, ids::PROTON as usize, 0.8);
    sim.types[k as usize].radius = 9.0; // bigger than anything else -> grid cell grows
    for i in 0..50 { sim.add(vec2(100.0 + i as f32 * 12.0, 300.0), vec2(0.0, 0.0), k); }
    for _ in 0..30 { sim.step(1.0 / 60.0, None); }
    sim.remove_last_species();
    assert!(sim.kind.iter().all(|&kk| (kk as usize) < sim.types.len()));
    assert_eq!(sim.rules.n, sim.types.len());
    for _ in 0..30 { sim.step(1.0 / 60.0, None); }
    assert!(sim.pos.iter().all(|p| p.is_finite()));
}

#[test]
fn nuclear_off_means_no_decays() {
    let mut sim = Sim::new(800.0, 600.0);
    for i in 0..200 { sim.add(vec2(20.0 + (i % 20) as f32 * 30.0, 50.0 + (i / 20) as f32 * 40.0), vec2(0.0, 0.0), ids::NEUTRON); }
    sim.cfg.decay_scale = 0.1; // would decay almost instantly if nuclear were on
    for _ in 0..120 { sim.step(1.0 / 60.0, None); }
    assert!(sim.kind.iter().all(|&k| k == ids::NEUTRON));
    sim.cfg.nuclear = true;
    for _ in 0..120 { sim.step(1.0 / 60.0, None); }
    assert!(sim.kind.iter().any(|&k| k == ids::PROTON), "neutrons should decay in nuclear mode");
}

#[test]
fn photons_leave_the_box() {
    let mut sim = Sim::new(400.0, 400.0);
    sim.add(vec2(200.0, 200.0), vec2(1200.0, 0.0), ids::PHOTON);
    sim.add(vec2(200.0, 200.0), vec2(0.0, 0.0), ids::PROTON);
    for _ in 0..30 { sim.step(1.0 / 60.0, None); }
    assert_eq!(sim.len(), 1);
    assert_eq!(sim.kind[0], ids::PROTON);
}

/// Bound nuclei must not heat up or spin up on their own (regression test:
/// stale strong force + bouncing on force-held contacts used to blow them apart).
#[test]
fn nuclei_do_not_self_heat() {
    use macroquad::math::Vec2;
    let d = 8.0;
    // Compact (rhombus) packing: the shape the roundness pull settles into.
    let h = d * 0.866;
    let he4 = [
        (vec2(-d / 2.0, 0.0), ids::PROTON), (vec2(d / 2.0, 0.0), ids::PROTON),
        (vec2(0.0, -h), ids::NEUTRON), (vec2(0.0, h), ids::NEUTRON),
    ];
    for spin in [0.0f32, 2.0] {
        let mut sim = Sim::new(800.0, 800.0);
        sim.cfg.nuclear = true;
        sim.cfg.drag = 0.0;
        sim.cfg.decay_scale = 1e9;
        let c = vec2(400.0, 400.0);
        for (p, k) in he4 {
            sim.add(c + p, vec2(-p.y, p.x) * spin, k);
        }
        // Start as one nucleus.
        let t0 = sim.tag[0];
        for t in sim.tag.iter_mut() { *t = t0; }
        let ke = |s: &Sim| {
            let vc = s.vel.iter().copied().sum::<Vec2>() / s.len() as f32;
            s.vel.iter().map(|v| 0.5 * (*v - vc).length_squared()).sum::<f32>()
        };
        let ke0 = ke(&sim);
        for _ in 0..300 { sim.step(1.0 / 60.0, None); }
        assert_eq!(sim.len(), 4, "nucleus lost particles");
        assert!(ke(&sim) <= ke0 * 1.05 + 1.0, "spin {spin}: KE {} -> {}", ke0, ke(&sim));
    }
}

/// Real reaction channels in the engine: He-3 + slow n -> T + p (99.99% of
/// thermal absorptions), p + p never binds, slow D + D doesn't tunnel,
/// U-235 + slow n fissions.
#[test]
fn reactions_in_the_engine() {
    use sim_core::nuclear::place_nucleus;
    let setup = || {
        let mut sim = Sim::new(600.0, 400.0);
        sim.cfg.nuclear = true;
        sim.cfg.coulomb = false;
        sim.cfg.decay_scale = 1e9;
        sim
    };
    let has = |sim: &Sim, zn: (u32, u32)| sim.nuclear.census.iter().any(|c| c.0 == zn);

    let mut sim = setup();
    assert!(place_nucleus(&mut sim, vec2(300.0, 200.0), 2, 1, vec2(0.0, 0.0)));
    sim.add(vec2(250.0, 200.0), vec2(20.0, 0.0), ids::NEUTRON);
    for _ in 0..240 { sim.step(1.0 / 60.0, None); }
    assert!(has(&sim, (1, 2)) && has(&sim, (1, 0)), "He-3 + n -> T + p: {:?}", sim.nuclear.census);

    let mut sim = setup();
    sim.add(vec2(290.0, 200.0), vec2(30.0, 0.0), ids::PROTON);
    sim.add(vec2(310.0, 200.0), vec2(-30.0, 0.0), ids::PROTON);
    for _ in 0..60 { sim.step(1.0 / 60.0, None); }
    assert!(sim.nuclear.census.iter().all(|((z, n), _)| z + n == 1), "p + p must not bind");

    let mut sim = setup();
    place_nucleus(&mut sim, vec2(270.0, 200.0), 1, 1, vec2(20.0, 0.0));
    place_nucleus(&mut sim, vec2(330.0, 200.0), 1, 1, vec2(-20.0, 0.0));
    for _ in 0..120 { sim.step(1.0 / 60.0, None); }
    assert_eq!(sim.nuclear.fusions, 0, "slow D + D can't tunnel");

    let mut fissions = 0;
    for k in 0..5 {
        let mut sim = setup();
        sim.seed = 20 + k;
        place_nucleus(&mut sim, vec2(330.0, 200.0), 92, 143, vec2(0.0, 0.0));
        sim.add(vec2(220.0, 200.0), vec2(15.0, 0.0), ids::NEUTRON);
        for _ in 0..600 { sim.step(1.0 / 60.0, None); }
        fissions += sim.nuclear.fissions;
    }
    assert!(fissions >= 3, "U-235 + slow n should usually fission ({fissions}/5)");
}
