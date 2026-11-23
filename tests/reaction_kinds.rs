//! Nucleus-nucleus reactions are labelled by what comes out: fusion makes
//! something heavier than both reactants, breakup doesn't. Products are
//! listed heaviest first, and the energy books balance.
use macroquad::math::vec2;
use sim_core::nuclear::{EventKind, game_sim, place_nucleus, speed_for};

fn mass(name: &str) -> u32 {
    if let Some((_, a)) = name.rsplit_once('-') {
        return a.parse().unwrap_or(0);
    }
    let digits: String = name.chars().take_while(|c| c.is_ascii_digit()).collect();
    let k: u32 = digits.parse().unwrap_or(1);
    if matches!(&name[digits.len()..], "n" | "p") { k } else { 0 }
}

#[test]
fn fusion_and_breakup_labels_match_products() {
    let (mut fusions, mut breakups) = (0, 0);
    for seed in 0..120u64 {
        let mut sim = game_sim(1000.0, 700.0);
        sim.seed = seed;
        sim.point_nuclei = true;
        sim.cfg.matter = false;
        // He-4 head-on into p or He-4 at 8-60 MeV per nucleon.
        let per_nucleon = [8.0, 15.0, 30.0, 60.0][(seed / 2 % 4) as usize];
        let v = speed_for(&sim.cfg, per_nucleon * 4.0, 4.0);
        assert!(place_nucleus(&mut sim, vec2(400.0, 350.0), 2, 2, vec2(v, 0.0)));
        let (z, n) = if seed % 2 == 0 { (1, 0) } else { (2, 2) };
        assert!(place_nucleus(&mut sim, vec2(600.0, 350.0), z, n, vec2(-v * 0.5, 0.0)));
        sim.index_nuclei();
        for _ in 0..120 {
            sim.step(1.0 / 60.0, None);
        }
        for ev in &sim.nuclear.history {
            if !matches!(ev.kind, EventKind::Fusion | EventKind::Breakup) {
                continue;
            }
            let (lhs, rhs) = ev.equation().split_once(" → ").unwrap();
            let biggest_in = lhs.split(" + ").map(mass).max().unwrap();
            let out: Vec<u32> = rhs.split(" + ").map(mass).collect();
            assert!(out.windows(2).all(|w| w[0] >= w[1]), "heaviest first: {}", ev.text);
            let biggest_out = out[0];
            match ev.kind {
                EventKind::Fusion => {
                    fusions += 1;
                    assert!(biggest_out > biggest_in, "fusion must make something heavier: {}", ev.text);
                }
                _ => {
                    breakups += 1;
                    assert!(biggest_out <= biggest_in, "breakup can't: {}", ev.text);
                }
            }
            assert!(ev.energy().is_some_and(|e| e > 1.0), "collision energy parsed: {}", ev.text);
        }
        let st = &sim.nuclear;
        assert_eq!(st.fusions + st.breakups, sim.nuclear.history.iter().filter(|e| matches!(e.kind, EventKind::Fusion | EventKind::Breakup)).count() as u64);
        let q_net: f64 = st.history.iter().filter(|e| !matches!(e.kind, EventKind::Decay)).filter_map(|e| e.q()).sum();
        assert!((st.q_released - st.q_absorbed - q_net).abs() < 0.05 * (1.0 + st.q_released + st.q_absorbed), "books: {} - {} vs {q_net}", st.q_released, st.q_absorbed);
    }
    eprintln!("fusions {fusions} breakups {breakups}");
    assert!(fusions > 0 && breakups > 0, "both kinds seen: fusions {fusions} breakups {breakups}");
}
