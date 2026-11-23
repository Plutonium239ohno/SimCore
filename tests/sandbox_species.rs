//! Sandbox species editing: removing any species re-numbers the particles
//! and the interaction matrix; duplicating copies its rules.
use macroquad::math::vec2;
use sim_core::sim::Sim;
use sim_core::types::ids;

fn customs(sim: &Sim) -> Vec<u8> {
    (0..sim.types.len() as u8).filter(|&k| sim.types[k as usize].custom).collect()
}

#[test]
fn remove_middle_species_keeps_the_rest_consistent() {
    let mut sim = Sim::new(800.0, 600.0);
    let c = customs(&sim);
    let (a, b, d) = (c[0], c[1], c[2]);
    let n = sim.types.len();
    for x in 0..n {
        for y in 0..n {
            sim.rules.set(x, y, (x * 100 + y) as f32);
        }
    }
    sim.add(vec2(100.0, 100.0), vec2(0.0, 0.0), a);
    sim.add(vec2(200.0, 100.0), vec2(0.0, 0.0), b);
    sim.add(vec2(300.0, 100.0), vec2(0.0, 0.0), d);
    let d_name = sim.types[d as usize].name.clone();
    sim.remove_species(b);
    assert_eq!(sim.types.len(), n - 1);
    assert_eq!(sim.len(), 2, "b's particle removed");
    // d moved down into b's id; its particle followed.
    assert_eq!(sim.types[b as usize].name, d_name);
    assert!(sim.kind.contains(&a) && sim.kind.contains(&b));
    // Rules: old (d, a) is now (b, a); (a, d) now (a, b); (d, d) now (b, b).
    let old = |x: u8, y: u8| (x as usize * 100 + y as usize) as f32;
    assert_eq!(sim.rules.get(b as usize, a as usize), old(d, a));
    assert_eq!(sim.rules.get(a as usize, b as usize), old(a, d));
    assert_eq!(sim.rules.get(b as usize, b as usize), old(d, d));
    assert_eq!(sim.rules.n, sim.types.len());
    // Built-ins can't be removed.
    sim.remove_species(ids::PROTON);
    assert_eq!(sim.types.len(), n - 1);
}

#[test]
fn duplicate_copies_rules_both_ways() {
    let mut sim = Sim::new(800.0, 600.0);
    let c = customs(&sim);
    let (a, b) = (c[0] as usize, c[1] as usize);
    sim.rules.set(a, b, 0.7);
    sim.rules.set(b, a, -0.4);
    sim.rules.set(a, a, 0.3);
    let k = sim.duplicate_species(a as u8).unwrap() as usize;
    assert!(sim.types[k].custom);
    assert_eq!(sim.rules.get(k, b), 0.7);
    assert_eq!(sim.rules.get(b, k), -0.4);
    assert_eq!(sim.rules.get(k, k), 0.3);
    assert_eq!(sim.rules.get(k, a), sim.rules.get(a, a));
}
