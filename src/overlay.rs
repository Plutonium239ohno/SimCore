//! Field overlays drawn over the scene: matter temperature and neutron flux
//! on a coarse grid, averaged over sim time and drawn as a smooth,
//! semi-transparent heatmap.

use crate::nuclear::energy_of;
use crate::sim::Sim;
use crate::thermal::{format_kelvin, kelvin_of};
use crate::types::ids;
use macroquad::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OverlayKind {
    #[default]
    None,
    /// Temperature of the matter (motion relative to the local flow).
    Temperature,
    /// Neutron flux: free-neutron track length per area per time.
    Flux,
    /// The same for slow (thermal, < 1 eV) neutrons only.
    ThermalFlux,
}

impl OverlayKind {
    pub const ALL: [OverlayKind; 4] = [OverlayKind::None, OverlayKind::Temperature, OverlayKind::Flux, OverlayKind::ThermalFlux];

    pub fn name(self) -> &'static str {
        match self {
            OverlayKind::None => "None",
            OverlayKind::Temperature => "Temperature",
            OverlayKind::Flux => "Neutron flux",
            OverlayKind::ThermalFlux => "Thermal flux",
        }
    }
}

/// Grid cell (px). Fine enough to show a fuel channel, coarse enough to
/// average over a few nuclei.
const CELL: f32 = 20.0;
/// Averaging time (sim seconds).
const TAU_FLUX: f64 = 2.0;
const TAU_TEMP: f64 = 0.6;
/// Thermal neutrons: below 1 eV.
const THERMAL_MEV: f64 = 1e-6;

pub struct Overlay {
    pub kind: OverlayKind,
    /// Overlay opacity (0..1).
    pub opacity: f32,
    cols: usize,
    rows: usize,
    /// Averaged field per cell, and whether the cell has ever had matter
    /// (temperature) or neutrons (flux).
    field: Vec<f32>,
    seen: Vec<f32>,
    last_t: f64,
    /// Smoothed top of the colour scale.
    top: f32,
    image: Option<Image>,
    texture: Option<Texture2D>,
}

impl Default for Overlay {
    fn default() -> Self {
        Self { kind: OverlayKind::None, opacity: 0.55, cols: 0, rows: 0, field: Vec::new(), seen: Vec::new(), last_t: 0.0, top: 0.0, image: None, texture: None }
    }
}

impl Overlay {
    /// Fold in what happened since the last call (by sim time, so a
    /// paused sim keeps its picture).
    pub fn update(&mut self, sim: &Sim) {
        if self.kind == OverlayKind::None {
            self.field.clear();
            return;
        }
        let (cols, rows) = (((sim.width / CELL).ceil() as usize).max(1), ((sim.height / CELL).ceil() as usize).max(1));
        let t = sim.nuclear.time;
        if cols != self.cols || rows != self.rows || self.field.is_empty() || t < self.last_t {
            self.cols = cols;
            self.rows = rows;
            self.field = vec![0.0; cols * rows];
            self.seen = vec![0.0; cols * rows];
            self.last_t = t;
            self.top = 0.0;
        }
        let dt = t - self.last_t;
        if dt <= 0.0 {
            return;
        }
        self.last_t = t;
        let cell_of = |p: Vec2| {
            let cx = ((p.x / CELL) as i32).clamp(0, cols as i32 - 1) as usize;
            let cy = ((p.y / CELL) as i32).clamp(0, rows as i32 - 1) as usize;
            cy * cols + cx
        };
        let mut now = vec![0.0f32; cols * rows];
        let mut have = vec![0.0f32; cols * rows];
        let tau = match self.kind {
            OverlayKind::Temperature => TAU_TEMP,
            _ => TAU_FLUX,
        };
        match self.kind {
            OverlayKind::Temperature => {
                // Mean kinetic energy per body relative to the cell's flow.
                let bodies: Vec<_> = sim.bodies().into_values().filter(|b| b.a() > 1 || b.z > 0).collect();
                let mut flow = vec![(Vec2::ZERO, 0.0f32); cols * rows];
                for b in &bodies {
                    let c = cell_of(b.pos);
                    flow[c].0 += b.vel * b.a() as f32;
                    flow[c].1 += b.a() as f32;
                }
                for b in &bodies {
                    let c = cell_of(b.pos);
                    let u = flow[c].0 / flow[c].1.max(1e-6);
                    now[c] += 0.5 * b.a() as f32 * (b.vel - u).length_squared();
                    have[c] += 1.0;
                }
                for (x, &k) in now.iter_mut().zip(&have) {
                    if k > 0.0 {
                        *x /= k;
                    }
                }
            }
            OverlayKind::Flux | OverlayKind::ThermalFlux => {
                let thermal = self.kind == OverlayKind::ThermalFlux;
                let sizes = body_sizes(sim);
                for i in 0..sim.len() {
                    if sim.kind[i] != ids::NEUTRON || sizes.get(sim.nuc[i] as usize).is_some_and(|&n| n > 1) {
                        continue;
                    }
                    let v = sim.vel[i].length();
                    if thermal && energy_of(&sim.cfg, v, 1.0) > THERMAL_MEV {
                        continue;
                    }
                    let c = cell_of(sim.pos[i]);
                    // Track length per unit time and area.
                    now[c] += v / (CELL * CELL);
                    have[c] = 1.0;
                }
            }
            OverlayKind::None => {}
        }
        let a = (1.0 - (-dt / tau).exp()) as f32;
        let temperature = self.kind == OverlayKind::Temperature;
        for ((f, s), (&x, &h)) in self.field.iter_mut().zip(self.seen.iter_mut()).zip(now.iter().zip(&have)) {
            if temperature {
                // Empty cells keep their last value but fade out.
                if h > 0.0 {
                    *f += (x - *f) * a;
                }
                *s += ((h > 0.0) as u8 as f32 - *s) * a;
            } else {
                *f += (x - *f) * a;
                *s = s.max(h);
            }
        }
        let max = self.field.iter().zip(&self.seen).filter(|(_, s)| **s > 0.05).map(|(f, _)| *f).fold(0.0f32, f32::max);
        self.top = if self.top <= 0.0 { max } else { self.top + (max - self.top) * a.max(0.05) };
    }

    /// (title, low end, high end) of the colour scale.
    pub fn legend(&self, sim: &Sim) -> Option<(String, String, String)> {
        let amb = crate::thermal::theta_of(&sim.cfg, sim.cfg.temperature);
        match self.kind {
            OverlayKind::None => None,
            OverlayKind::Temperature => Some((
                "Temperature".into(),
                format_kelvin(sim.cfg.temperature as f64),
                format_kelvin(kelvin_of(&sim.cfg, self.top.max(amb * 1.5))),
            )),
            OverlayKind::Flux => Some(("Neutron flux (relative)".into(), "0".into(), "max".into())),
            OverlayKind::ThermalFlux => Some(("Thermal (< 1 eV) neutron flux".into(), "0".into(), "max".into())),
        }
    }

    /// Turn the field into this frame's texture (call after `update`).
    pub fn prepare(&mut self, sim: &Sim) {
        if self.kind == OverlayKind::None || self.field.is_empty() {
            self.texture = None;
            return;
        }
        let (cols, rows) = (self.cols, self.rows);
        let img = self.image.get_or_insert_with(|| Image::gen_image_color(cols as u16, rows as u16, BLANK));
        if img.width() != cols || img.height() != rows {
            *img = Image::gen_image_color(cols as u16, rows as u16, BLANK);
            self.texture = None;
        }
        let amb = crate::thermal::theta_of(&sim.cfg, sim.cfg.temperature).max(1e-6);
        let top = self.top.max(1e-9);
        for (k, (&f, &s)) in self.field.iter().zip(&self.seen).enumerate() {
            let (x, y) = ((k % cols) as u32, (k / cols) as u32);
            let mut c = match self.kind {
                // Log scale from the ambient temperature up.
                OverlayKind::Temperature => {
                    let hi = (top.max(amb * 1.5) / amb).ln();
                    crate::render::heat(((f.max(amb) / amb).ln() / hi).clamp(0.0, 1.0))
                }
                // Square-root scale: shows the edges of the flux too.
                _ => crate::render::heat((f / top).clamp(0.0, 1.0).sqrt()),
            };
            let shown = match self.kind {
                OverlayKind::Temperature => s,
                _ => (f / top * 20.0).min(1.0) * s,
            };
            c.a = self.opacity * shown.clamp(0.0, 1.0);
            img.set_pixel(x, y, c);
        }
        match &self.texture {
            Some(t) if t.width() as usize == cols && t.height() as usize == rows => t.update(img),
            _ => {
                let t = Texture2D::from_image(img);
                t.set_filter(FilterMode::Linear);
                self.texture = Some(t);
            }
        }
    }

    /// Draw over the world (world coordinates, under the current camera).
    pub fn draw(&self) {
        if self.kind == OverlayKind::None {
            return;
        }
        if let Some(t) = &self.texture {
            draw_texture_ex(t, 0.0, 0.0, WHITE, DrawTextureParams { dest_size: Some(vec2(self.cols as f32 * CELL, self.rows as f32 * CELL)), ..Default::default() });
        }
    }
}

/// Members per nucleus index (free neutrons are 1).
fn body_sizes(sim: &Sim) -> Vec<u32> {
    let mut sizes = vec![0u32; sim.n_nuclei];
    for &k in &sim.nuc {
        if let Some(s) = sizes.get_mut(k as usize) {
            *s += 1;
        }
    }
    sizes
}
