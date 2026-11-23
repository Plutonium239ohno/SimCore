//! Electric and magnetic field regions: rectangles with a uniform field
//! that steer charged particles (beams, plasma, fission fragments). Free
//! neutrons, photons and neutrinos don't notice them.
//!
//! The plane is the x-y plane and B points out of it (a scalar), so a
//! charge q of mass m feels F = q (E + v x B): E pushes along its direction,
//! B turns the velocity at the cyclotron rate w = q B / m without changing
//! the speed.
//!
//! Units: E is the acceleration (px/s^2) it gives a proton; B is the angular
//! speed (rad/s) it turns a proton at. Anything else scales with its
//! charge-to-mass ratio (an alpha feels half, an electron 20x more and the
//! opposite way: the electrons here are 20x lighter than a nucleon).

use macroquad::math::{Rect, Vec2};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Field {
    pub rect: Rect,
    /// Electric field: acceleration of a proton (px/s^2).
    pub e: Vec2,
    /// Magnetic field out of the plane: a proton's turn rate (rad/s,
    /// positive = counter-clockwise on screen).
    pub b: f32,
}

impl Field {
    pub fn new(rect: Rect, e: Vec2, b: f32) -> Self {
        Self { rect, e, b }
    }

    /// One-line description.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.e.length() > 0.0 {
            parts.push(format!("E {:.0} px/s² (for a proton)", self.e.length()));
        }
        if self.b != 0.0 {
            parts.push(format!(
                "B: a proton turns {:.2} rad/s {} (one turn in {:.1} s)",
                self.b.abs(),
                if self.b > 0.0 { "anticlockwise" } else { "clockwise" },
                std::f32::consts::TAU / self.b.abs()
            ));
        }
        if parts.is_empty() { "no field".into() } else { parts.join(";  ") }
    }

    /// Radius (px) a proton at `speed` px/s circles at in this field.
    pub fn proton_radius(&self, speed: f32) -> Option<f32> {
        (self.b.abs() > 1e-6).then(|| speed / self.b.abs())
    }
}

/// Velocity after `h` seconds in the fields at `p`, for charge-to-mass ratio
/// `qm` (proton = 1). The magnetic turn is exact (no spiralling out).
#[inline]
pub fn apply(fields: &[Field], p: Vec2, v: Vec2, qm: f32, h: f32) -> Vec2 {
    if qm == 0.0 {
        return v;
    }
    let mut v = v;
    for f in fields {
        if !f.rect.contains(p) {
            continue;
        }
        v += f.e * (qm * h);
        if f.b != 0.0 {
            // Positive B turns positive charges anticlockwise on screen
            // (y points down, so that is a negative angle in maths terms).
            let a = -f.b * qm * h;
            let (s, c) = a.sin_cos();
            v = Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c);
        }
    }
    v
}
