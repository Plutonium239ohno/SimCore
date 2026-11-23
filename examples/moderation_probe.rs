//! Moderation check: fast neutrons in a box of moderator (point mode or
//! clusters). Prints the free-neutron energy quantiles over time.
//! env: POINTS, Z, N (moderator, default D), GAP
use sim_core::nuclear::{energy_of, format_energy, speed_for};
use sim_core::types::ids;
use macroquad::math::{vec2, Vec2};

fn main() {
    macroquad::rand::srand(3);
    let mut sim = sim_core::nuclear::game_sim(1600.0, 900.0);
    sim.point_nuclei = std::env::var("POINTS").is_ok();
    let z: u32 = std::env::var("Z").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let n: u32 = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let gap: f32 = std::env::var("GAP").ok().and_then(|s| s.parse().ok()).unwrap_or(5.5);
    sim.walls.half = 2.0;
    let (x0, y0, x1, y1) = (300.0, 150.0, 1300.0, 750.0);
    for (a, b) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))] {
        sim.walls.add(vec2(a.0, a.1), vec2(b.0, b.1));
    }
    let r = if sim.point_nuclei { sim_core::nuclear::POINT_RADIUS } else { 2.0 * ((z + n) as f32 / 0.9069).sqrt() + 2.0 };
    let d = 2.0 * r + gap;
    let mut y = y0 + 8.0;
    let mut row = 0;
    let mut k = 0;
    while y < y1 - 8.0 {
        let mut x = x0 + 8.0 + if row % 2 == 1 { d * 0.5 } else { 0.0 };
        while x < x1 - 8.0 {
            if sim_core::nuclear::place_nucleus(&mut sim, vec2(x, y), z, n, Vec2::ZERO) {
                k += 1;
            }
            x += d;
        }
        y += d * 0.866;
        row += 1;
    }
    sim_core::thermal::thermalize(&mut sim, 293.0);
    let v = speed_for(&sim.cfg, 2.0, 1.0);
    for j in 0..200 {
        let a = j as f32 * 0.1;
        sim.add(vec2(800.0, 450.0) + vec2(a.cos(), a.sin()) * 3.0, vec2(a.cos(), a.sin()) * v, ids::NEUTRON);
    }
    sim.index_nuclei();
    println!("{k} moderator nuclei, spacing {d:.1}");
    for f in 0..60 * 20 {
        sim.step(1.0 / 60.0, None);
        if f % 120 == 119 {
            let mut es: Vec<f64> = (0..sim.len())
                .filter(|&i| sim.kind[i] == ids::NEUTRON && !(0..sim.len()).any(|j| j != i && sim.tag[j] == sim.tag[i]))
                .map(|i| energy_of(&sim.cfg, sim.vel[i].length(), 1.0))
                .collect();
            es.sort_by(f64::total_cmp);
            if es.is_empty() {
                println!("t={} none", (f + 1) / 60);
                continue;
            }
            let q = |p: f64| format_energy(es[((es.len() - 1) as f64 * p) as usize]);
            println!("t={:>2}s n={} 10% {} 50% {} 90% {}  matter {}", (f + 1) / 60, es.len(), q(0.1), q(0.5), q(0.9),
                sim_core::thermal::format_kelvin(sim_core::thermal::kelvin_of(&sim.cfg, sim_core::thermal::region_theta(&sim, |_| true).unwrap_or(0.0))));
        }
    }
}
