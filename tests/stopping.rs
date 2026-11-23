//! Fast ions slow down in matter, not in vacuum, and not because of matter
//! on the other side of a wall.
use macroquad::math::{Vec2, vec2};
use sim_core::nuclear::{game_sim, place_nucleus, speed_for};
use sim_core::sim::Sim;
use sim_core::walls::WallKind;

/// A 5 MeV alpha's speed after 0.25 s, starting at (x, 350) heading up.
fn alpha_after(setup: impl Fn(&mut Sim)) -> (f32, f32) {
    let mut sim = game_sim(1200.0, 700.0);
    sim.point_nuclei = true;
    sim.cfg.cooling = 0.0;
    let v0 = speed_for(&sim.cfg, 5.0, 4.0);
    assert!(place_nucleus(&mut sim, vec2(600.0, 600.0), 2, 2, vec2(0.0, -v0)));
    setup(&mut sim);
    sim.index_nuclei();
    for _ in 0..15 {
        sim.step(1.0 / 60.0, None);
    }
    let he = sim.bodies().values().find(|b| (b.z, b.n) == (2, 2)).unwrap().vel.length();
    (v0, he)
}

/// Deuterium filling x in [lo, hi], everywhere vertically.
fn deuterium(sim: &mut Sim, lo: f32, hi: f32) {
    let mut y = 20.0;
    let mut row = 0;
    while y < 680.0 {
        let mut x = lo + if row % 2 == 1 { 6.0 } else { 0.0 };
        while x < hi {
            place_nucleus(sim, vec2(x, y), 1, 1, Vec2::ZERO);
            x += 12.0;
        }
        y += 10.4;
        row += 1;
    }
}

#[test]
fn stopping_needs_matter() {
    let (v0, vacuum) = alpha_after(|_| {});
    let (_, matter) = alpha_after(|s| deuterium(s, 560.0, 640.0));
    let (_, walled) = alpha_after(|s| {
        // Matter right next to the alpha's path, behind a wall.
        deuterium(s, 620.0, 700.0);
        s.walls.add_kind(vec2(612.0, 0.0), vec2(612.0, 700.0), WallKind::Insulator);
    });
    assert!(vacuum > 0.98 * v0, "vacuum: {vacuum} of {v0}");
    assert!(matter < 0.5 * v0, "matter: {matter} of {v0}");
    assert!(walled > 0.98 * v0, "behind a wall: {walled} of {v0}");
}
