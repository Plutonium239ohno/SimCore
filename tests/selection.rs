//! Copy, cut, paste, rotate and mirror parts of a scene.

use macroquad::math::{Rect, Vec2, vec2};
use sim_core::detector::{DetShape, DetectFilter, Detector};
use sim_core::field::Field;
use sim_core::nuclear::{game_sim, place_nucleus};
use sim_core::reactor::Spawner;
use sim_core::selection::{self, Kinds};
use sim_core::sim::Sim;

fn scene() -> Sim {
    let mut sim = game_sim(1200.0, 800.0);
    sim.walls.add(vec2(100.0, 100.0), vec2(300.0, 100.0));
    sim.walls.add(vec2(300.0, 100.0), vec2(300.0, 200.0));
    sim.walls.add(vec2(900.0, 100.0), vec2(1000.0, 100.0)); // outside the region
    sim.reactor.fields.push(Field::new(Rect::new(120.0, 120.0, 100.0, 40.0), vec2(0.0, -50.0), 2.0));
    sim.reactor.spawners.push(Spawner::new(vec2(150.0, 180.0), 1, 1, vec2(100.0, 0.0), 1.0));
    sim.reactor.detectors.push(Detector::new(DetShape::Line(vec2(250.0, 110.0), vec2(250.0, 190.0)), DetectFilter::All, false));
    assert!(place_nucleus(&mut sim, vec2(200.0, 150.0), 2, 2, vec2(10.0, 0.0)));
    sim.index_nuclei();
    sim
}

const REGION: Rect = Rect { x: 90.0, y: 90.0, w: 220.0, h: 120.0 };

#[test]
fn copy_and_paste_everything() {
    let mut sim = scene();
    let (clip, centre) = selection::copy(&sim, REGION, Kinds::default());
    assert_eq!(clip.walls.len(), 2);
    assert_eq!((clip.fields.len(), clip.spawners.len(), clip.detectors.len()), (1, 1, 1));
    assert_eq!(clip.bodies().0, 1, "the He-4 nucleus goes whole");
    let n0 = sim.len();
    let to = centre + vec2(400.0, 300.0);
    let r = selection::place(&mut sim, &clip, to);
    assert_eq!(sim.walls.segs.len(), 5);
    assert_eq!(sim.reactor.fields.len(), 2);
    assert_eq!(sim.reactor.fields[1].rect.x, 520.0);
    assert_eq!(sim.len(), n0 + clip.particles.len());
    assert!(r.contains(vec2(600.0, 450.0)));
    // The copy is a new nucleus, not the old one again.
    let tags: std::collections::HashSet<u32> = sim.tag.iter().copied().filter(|&t| t != u32::MAX).collect();
    assert_eq!(tags.len(), 2);
}

#[test]
fn cut_takes_only_whats_inside_and_kinds_filter() {
    let mut sim = scene();
    let walls_only = Kinds { walls: true, matter: false, parts: false };
    assert!(selection::delete(&mut sim, REGION, walls_only));
    assert_eq!(sim.walls.segs.len(), 1, "the far wall stays");
    assert_eq!(sim.reactor.fields.len(), 1);
    assert!(sim.len() > 0);
    assert!(selection::delete(&mut sim, REGION, Kinds::default()));
    assert!(sim.reactor.fields.is_empty() && sim.reactor.spawners.is_empty() && sim.reactor.detectors.is_empty());
    assert_eq!(sim.len(), 0);
    assert!(!selection::delete(&mut sim, REGION, Kinds::default()));
}

#[test]
fn rotate_and_mirror() {
    let sim = scene();
    let (clip, _) = selection::copy(&sim, REGION, Kinds::default());
    // Four quarter turns: back where it started.
    let mut c = clip.clone();
    for _ in 0..4 {
        c.rotate(std::f32::consts::FRAC_PI_2);
    }
    for (a, b) in c.walls.iter().zip(&clip.walls) {
        assert!(a.a.distance(b.a) < 1e-3 && a.b.distance(b.b) < 1e-3);
    }
    // A quarter turn (anticlockwise on screen): the spawner fired right, now up.
    let mut q = clip.clone();
    q.rotate(std::f32::consts::FRAC_PI_2);
    let v = q.spawners[0].vel;
    assert!(v.x.abs() < 1e-3 && v.y < -99.0, "{v:?}");
    assert_eq!((q.fields[0].rect.w, q.fields[0].rect.h), (40.0, 100.0), "the box turns on its side");
    assert_eq!(q.fields[0].b, 2.0, "turning doesn't change B");
    assert!(q.fields[0].e.x < -49.0, "E (up) now points left");
    // A mirror image: B turns the other way, the spawner fires left.
    let mut m = clip.clone();
    m.flip_x();
    assert_eq!(m.fields[0].b, -2.0);
    assert!(m.spawners[0].vel.x < -99.0);
    let (b0, b1) = (clip.bounds(), m.bounds());
    assert!((b0.w - b1.w).abs() < 1e-3 && (b0.h - b1.h).abs() < 1e-3);
    let mut f = clip.clone();
    f.flip_y();
    assert_eq!(f.fields[0].b, -2.0);
    assert!(f.fields[0].e.y > 49.0);
    if let DetShape::Line(a, b) = f.detectors[0].shape {
        assert!((a - b).length() > 79.0);
    }
}

#[test]
fn align_to_grid() {
    let sim = scene();
    let (mut clip, _) = selection::copy(&sim, REGION, Kinds::default());
    let at = clip.grid_anchor(vec2(503.0, 287.0), 20.0);
    let b = clip.bounds();
    let corner = at + vec2(b.x, b.y);
    assert!((corner.x / 20.0).fract().abs() < 1e-3 && (corner.y / 20.0).fract().abs() < 1e-3, "{corner:?}");
    assert!(at.distance(vec2(503.0, 287.0)) <= 15.0);
    // Snapping every point: wall ends land on the grid.
    clip.snap_points(vec2(7.0, 3.0), 25.0);
    for s in &clip.walls {
        for p in [s.a, s.b] {
            let q = p + vec2(7.0, 3.0);
            assert!((q.x / 25.0).fract().abs() < 1e-3 && (q.y / 25.0).fract().abs() < 1e-3);
        }
    }
    let _ = Vec2::ZERO;
}
