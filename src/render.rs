//! Drawing: particles (one textured quad each),
//! view modes (normal / speed heatmap / element), trails and walls.

use macroquad::miniquad::{BlendState, PipelineParams};
use macroquad::prelude::*;
use crate::nucdata::nuclide_name;
use crate::sim::Sim;
use crate::types::{hsv, ids};
use crate::rng::HashMap;

pub const BACKGROUND: Color = Color::new(8.0 / 255.0, 8.0 / 255.0, 13.0 / 255.0, 1.0);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewMode {
    /// Species colours.
    Normal,
    /// Heatmap of speed.
    Speed,
    /// Nucleons coloured by the element of the nucleus they belong to.
    Element,
}

impl ViewMode {
    pub const ALL: [ViewMode; 3] = [ViewMode::Normal, ViewMode::Speed, ViewMode::Element];
    pub fn name(self) -> &'static str {
        match self {
            ViewMode::Normal => "Species",
            ViewMode::Speed => "Speed heatmap",
            ViewMode::Element => "Elements",
        }
    }
}

pub struct View {
    pub mode: ViewMode,
    /// Motion trails (particles leave fading streaks).
    pub trails: bool,
    /// Fraction of the trail image cleared per frame (smaller = longer trails).
    pub trail_fade: f32,
    /// Speed (px/s) that maps to the hot end of the heatmap.
    pub speed_scale: f32,
    /// Element view: write nuclide names next to nuclei.
    pub labels: bool,
    /// Field drawn over the scene (temperature, neutron flux).
    pub overlay: crate::overlay::OverlayKind,
}

impl Default for View {
    fn default() -> Self {
        Self { mode: ViewMode::Normal, trails: false, trail_fade: 0.12, speed_scale: 500.0, labels: true, overlay: crate::overlay::OverlayKind::None }
    }
}

const VERTEX: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;
varying lowp vec2 uv;
varying lowp vec4 color;
uniform mat4 Model;
uniform mat4 Projection;
void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    color = color0 / 255.0;
    uv = texcoord;
}
"#;

/// Plain copy with alpha forced to 1 (the trail image already contains the
/// background, and its alpha channel is meaningless after fading).
const OPAQUE_FRAG: &str = r#"#version 100
precision mediump float;
varying lowp vec2 uv;
varying lowp vec4 color;
uniform sampler2D Texture;
void main() {
    gl_FragColor = vec4(texture2D(Texture, uv).rgb, 1.0);
}
"#;

struct Target {
    rt: RenderTarget,
    w: u32,
    h: u32,
}

impl Target {
    fn ensure(slot: &mut Option<Target>, w: u32, h: u32) -> bool {
        if slot.as_ref().is_some_and(|t| t.w == w && t.h == h) {
            return false;
        }
        let rt = render_target(w, h);
        rt.texture.set_filter(FilterMode::Linear);
        *slot = Some(Target { rt, w, h });
        true
    }

    /// Draw the target over the whole screen.
    fn blit(&self) {
        draw_texture_ex(&self.rt.texture, 0.0, 0.0, WHITE, DrawTextureParams {
            dest_size: Some(vec2(self.w as f32, self.h as f32)),
            // Render targets come back upside down relative to the screen.
            flip_y: true,
            ..Default::default()
        });
    }
}

pub struct Renderer {
    disc: Texture2D,
    opaque_mat: Material,
    /// World rectangle the trail image was drawn with.
    trail_src: Option<Rect>,
    /// Temperature / flux overlay (see `View::overlay`).
    pub overlay: crate::overlay::Overlay,
    trail: Option<Target>,
    colors: Vec<Color>,
    nuclei: HashMap<u32, (u32, u32, Vec2)>,
}

/// Soft-edged white disc; tinted per particle.
fn disc_texture() -> Texture2D {
    radial_texture(32, |d| d)
}

/// Texture whose alpha is `f(distance from centre in pixels-to-edge units)`.
fn radial_texture(s: u16, f: impl Fn(f32) -> f32) -> Texture2D {
    let mut img = Image::gen_image_color(s, s, Color::new(1.0, 1.0, 1.0, 0.0));
    let c = s as f32 / 2.0;
    for y in 0..s as u32 {
        for x in 0..s as u32 {
            let d = ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)).sqrt();
            img.set_pixel(x, y, Color::new(1.0, 1.0, 1.0, f(c - d).clamp(0.0, 1.0)));
        }
    }
    let tex = Texture2D::from_image(&img);
    tex.set_filter(FilterMode::Linear);
    tex
}

fn material(frag: &str, blend: Option<BlendState>, uniforms: Vec<UniformDesc>) -> Material {
    load_material(
        ShaderSource::Glsl { vertex: VERTEX, fragment: frag },
        MaterialParams {
            pipeline_params: PipelineParams {
                color_blend: blend,
                alpha_blend: blend,
                ..Default::default()
            },
            uniforms,
            ..Default::default()
        },
    )
    .expect("shader")
}

/// Perceptual-ish heatmap: deep blue -> cyan -> green -> yellow -> red -> white.
pub fn heat(x: f32) -> Color {
    const STOPS: [(f32, f32, f32); 6] = [
        (0.10, 0.12, 0.45),
        (0.10, 0.55, 0.90),
        (0.20, 0.85, 0.45),
        (0.98, 0.85, 0.20),
        (0.95, 0.30, 0.15),
        (1.00, 0.95, 0.90),
    ];
    let x = x.clamp(0.0, 1.0) * (STOPS.len() - 1) as f32;
    let i = (x as usize).min(STOPS.len() - 2);
    let f = x - i as f32;
    let (a, b) = (STOPS[i], STOPS[i + 1]);
    Color::new(a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f, a.2 + (b.2 - a.2) * f, 1.0)
}

/// Colour for a nucleus with `z` protons and `n` neutrons.
/// Drawn radius of particle i: point nuclei grow a little with A (A^1/3,
/// like real nuclear radii) so heavy ones stand out; their collision
/// radius stays `POINT_RADIUS`.
fn dot_radius(sim: &Sim, i: usize) -> f32 {
    if sim.zn[i] != 0 {
        let (z, n) = sim.zn_of(i);
        2.4 + 0.4 * ((z + n) as f32).cbrt()
    } else {
        sim.types[sim.kind[i] as usize].radius
    }
}

fn element_color(z: u32, n: u32) -> Color {
    match (z, n) {
        (0, _) => Color::from_rgba(150, 152, 165, 140),
        (1, 0) => Color::from_rgba(200, 80, 80, 150),
        _ => hsv((z as f32 * 67.0 + 190.0) % 360.0, 0.7, 1.0),
    }
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            disc: disc_texture(),
            opaque_mat: material(OPAQUE_FRAG, None, vec![]),
            overlay: Default::default(),
            trail_src: None,
            trail: None,
            colors: Vec::new(),
            nuclei: HashMap::default(),
        }
    }

    fn compute_colors(&mut self, sim: &Sim, view: &View) {
        let n = sim.len();
        self.colors.clear();
        match view.mode {
            ViewMode::Normal => {
                self.colors.extend(sim.kind.iter().map(|&k| sim.types[k as usize].color));
            }
            ViewMode::Speed => {
                let inv = 1.0 / view.speed_scale.max(1.0);
                self.colors.extend((0..n).map(|i| {
                    if sim.types[sim.kind[i] as usize].ghost {
                        sim.types[sim.kind[i] as usize].color
                    } else {
                        heat(sim.vel[i].length() * inv)
                    }
                }));
            }
            ViewMode::Element => {
                // Group nucleons by nucleus tag: (Z, N, position sum).
                self.nuclei.clear();
                for i in 0..n {
                    if sim.is_nuclear(i) {
                        let (z, nn) = sim.zn_of(i);
                        let e = self.nuclei.entry(sim.tag[i]).or_insert((0, 0, Vec2::ZERO));
                        e.0 += z;
                        e.1 += nn;
                        // Weighted by nucleons, so sum / A is the centre.
                        e.2 += sim.pos[i] * (z + nn) as f32;
                    }
                }
                for i in 0..n {
                    let k = sim.kind[i];
                    let t = &sim.types[k as usize];
                    self.colors.push(if sim.is_nuclear(i) {
                        let (z, nn, _) = self.nuclei[&sim.tag[i]];
                        let c = element_color(z, nn);
                        // Neutrons: a darker shade of their nucleus' colour.
                        if k == ids::NEUTRON && z > 0 {
                            Color::new(c.r * 0.55, c.g * 0.55, c.b * 0.55, c.a)
                        } else {
                            c
                        }
                    } else if t.custom {
                        t.color
                    } else {
                        Color::new(t.color.r, t.color.g, t.color.b, 0.35)
                    });
                }
            }
        }
    }

    fn draw_dots(&self, sim: &Sim) {
        for i in 0..sim.len() {
            let r = dot_radius(sim, i);
            let p = sim.pos[i];
            draw_texture_ex(&self.disc, p.x - r, p.y - r, self.colors[i], DrawTextureParams {
                dest_size: Some(vec2(2.0 * r, 2.0 * r)),
                ..Default::default()
            });
        }
    }

    pub fn draw(&mut self, sim: &Sim, view: &View) {
        let screen = Rect::new(0.0, 0.0, screen_width().max(1.0), screen_height().max(1.0));
        self.draw_view(sim, view, screen);
    }

    /// Draw the world rectangle `src` over the whole screen (pan and zoom).
    /// World areas outside the box are left dark and the box is outlined.
    pub fn draw_view(&mut self, sim: &Sim, view: &View, src: Rect) {
        let (w, h) = (screen_width().max(1.0) as u32, screen_height().max(1.0) as u32);
        clear_background(BACKGROUND);
        self.compute_colors(sim, view);
        let screen_cam = {
            let mut cam = Camera2D::from_display_rect(src);
            // from_display_rect is y-up (meant for render targets).
            cam.zoom.y = -cam.zoom.y;
            cam
        };

        // ---- Particles (optionally into the persistent trail image) ----
        if view.trails {
            // A moved camera would smear the old image: start afresh.
            let fresh = Target::ensure(&mut self.trail, w, h) || self.trail_src != Some(src);
            self.trail_src = Some(src);
            let t = self.trail.as_ref().unwrap();
            let mut cam = Camera2D::from_display_rect(src);
            cam.render_target = Some(t.rt.clone());
            set_camera(&cam);
            if fresh {
                clear_background(BACKGROUND);
            }
            let fade = Color::new(BACKGROUND.r, BACKGROUND.g, BACKGROUND.b, view.trail_fade.clamp(0.01, 1.0));
            draw_rectangle(src.x, src.y, src.w, src.h, fade);
            self.draw_dots(sim);
            set_default_camera();
            gl_use_material(&self.opaque_mat);
            self.trail.as_ref().unwrap().blit();
            gl_use_default_material();
        } else {
            self.trail = None;
            self.trail_src = None;
            set_camera(&screen_cam);
            self.draw_dots(sim);
        }
        set_camera(&screen_cam);

        // ---- Overlay ----
        self.overlay.kind = view.overlay;
        self.overlay.update(sim);
        self.overlay.prepare(sim);
        self.overlay.draw();

        // ---- Walls ----
        draw_walls(sim);
        draw_reactor(sim);
        let world = Rect::new(0.0, 0.0, sim.width, sim.height);
        if src != world {
            let k = src.w / w as f32;
            draw_rectangle_lines(0.0, 0.0, sim.width, sim.height, 1.5 * k, Color::new(0.45, 0.5, 0.65, 0.6));
        }
        set_default_camera();

        // ---- Element labels ----
        if view.mode == ViewMode::Element && view.labels {
            let k = w as f32 / src.w;
            self.draw_labels(sim, |p| (p - src.point()) * k, k, None);
        }
    }

    /// Nuclide names next to nuclei; `map` takes world to screen coordinates.
    fn draw_labels(&self, sim: &Sim, map: impl Fn(Vec2) -> Vec2, scale: f32, clip: Option<Rect>) {
        let r = sim.types[ids::NEUTRON as usize].radius;
        let mut shown = 0;
        // Species making up a big share of the nuclei (moderator, carrier
        // salt, propellant) aren't labelled when zoomed out: the labels would
        // bury the fuel and the fission products. Hover still names them.
        let total = self.nuclei.len().max(1);
        // Point nuclei are tiny: label them only when zoomed in or few.
        if sim.point_nuclei && scale < 2.0 && total > 150 {
            return;
        }
        let common: Vec<(u32, u32)> = if scale > 2.5 || total < 60 {
            Vec::new()
        } else {
            sim.nuclear.census.iter().filter(|c| c.1 as usize * 25 > total).map(|c| c.0).collect()
        };
        for &(z, n, sum) in self.nuclei.values() {
            let a = z + n;
            if a < 2 || shown > 400 || common.contains(&(z, n)) {
                continue;
            }
            let c = map(sum / a as f32);
            // Offset past the nucleus' edge (radius of a packed disc).
            let edge = if sim.point_nuclei { crate::nuclear::POINT_RADIUS } else { r * (a as f32 / 0.9069).sqrt() } * scale;
            let p = c + vec2(edge * 0.7 + 2.0, -edge * 0.7 - 2.0);
            if clip.is_some_and(|cl| !cl.contains(p)) {
                continue;
            }
            shown += 1;
            let size = (13.0 + 2.0 * scale).min(22.0);
            draw_text(&nuclide_name(z, n), p.x, p.y, size, Color::new(1.0, 1.0, 1.0, 0.85));
        }
    }

    /// Magnified view of world rectangle `src` drawn into screen rectangle
    /// `dst` (the zoom window). Uses the colours from this frame's `draw`.
    pub fn draw_region(&self, sim: &Sim, view: &View, src: Rect, dst: Rect) {
        let mut cam = Camera2D::from_display_rect(src);
        // from_display_rect is y-up (meant for render targets); the screen is y-down.
        cam.zoom.y = -cam.zoom.y;
        cam.viewport = Some((dst.x as i32, (screen_height() - dst.y - dst.h) as i32, dst.w as i32, dst.h as i32));
        set_camera(&cam);
        draw_rectangle(src.x, src.y, src.w, src.h, BACKGROUND);
        let grow = src.w * 0.05 + 8.0;
        let area = Rect::new(src.x - grow, src.y - grow, src.w + 2.0 * grow, src.h + 2.0 * grow);
        for i in 0..sim.len() {
            let p = sim.pos[i];
            if !area.contains(p) {
                continue;
            }
            let r = dot_radius(sim, i);
            draw_texture_ex(&self.disc, p.x - r, p.y - r, self.colors[i], DrawTextureParams {
                dest_size: Some(vec2(2.0 * r, 2.0 * r)),
                ..Default::default()
            });
        }
        self.overlay.draw();
        draw_walls(sim);
        draw_reactor(sim);
        set_default_camera();
        if view.mode == ViewMode::Element && view.labels {
            let k = dst.w / src.w;
            self.draw_labels(sim, |p| dst.point() + (p - src.point()) * k, k, Some(dst));
        }
    }
}

/// Reactor hardware in world coordinates: rod channels and the inserted
/// boron-carbide, empty fuel channels, the startup source.
pub fn draw_reactor(sim: &Sim) {
    let re = &sim.reactor;
    for rod in &re.rods {
        let ch = rod.inserted(1.0);
        draw_rectangle_lines(ch.x - 1.0, ch.y, ch.w + 2.0, ch.h, 1.0, Color::new(0.45, 0.48, 0.58, 0.35));
        let r = rod.inserted(re.insertion);
        if r.h > 0.5 {
            draw_rectangle(r.x, r.y, r.w, r.h, Color::from_rgba(64, 66, 78, 235));
            // Hatching reads as "solid absorber" at any zoom.
            let mut y = r.y + 4.0;
            while y < r.y + r.h {
                let y1 = (y + r.w * 0.8).min(r.y + r.h);
                draw_line(r.x, y, r.x + r.w * (y1 - y) / (r.w * 0.8), y1, 1.0, Color::from_rgba(96, 100, 116, 255));
                y += 7.0;
            }
            draw_rectangle(r.x, r.y + r.h - 2.0, r.w, 2.0, if re.scram { Color::from_rgba(255, 110, 90, 255) } else { Color::from_rgba(150, 156, 176, 255) });
        }
    }
    let rn = sim.types[crate::types::ids::NEUTRON as usize].radius;
    for s in &re.sites {
        let r = if sim.point_nuclei { crate::nuclear::POINT_RADIUS + 3.0 } else { rn * ((s.z + s.n) as f32 / 0.9069).sqrt() + 2.0 };
        if s.empty_for > 0.0 {
            let pulse = 0.35 + 0.25 * (get_time() as f32 * 4.0).sin();
            draw_circle_lines(s.pos.x, s.pos.y, r, 1.0, Color::new(1.0, 0.75, 0.3, pulse));
        } else {
            draw_circle_lines(s.pos.x, s.pos.y, r + 1.5, 1.0, Color::new(0.5, 0.55, 0.7, 0.18));
        }
    }
    for p in &re.sources {
        let c = Color::from_rgba(120, 255, 170, if re.source_rate > 0.0 { 220 } else { 70 });
        for k in 0..4 {
            let a = k as f32 * std::f32::consts::FRAC_PI_4 + get_time() as f32 * 0.5;
            let d = vec2(a.cos(), a.sin()) * 4.0;
            draw_line(p.x - d.x, p.y - d.y, p.x + d.x, p.y + d.y, 1.0, c);
        }
        draw_circle(p.x, p.y, 1.2, c);
    }
    for f in &re.fields {
        let r = f.rect;
        let (fill, edge) = if f.b != 0.0 { (Color::new(0.55, 0.35, 1.0, 0.06), Color::new(0.7, 0.55, 1.0, 0.5)) } else { (Color::new(0.3, 0.9, 1.0, 0.05), Color::new(0.4, 0.9, 1.0, 0.5)) };
        draw_rectangle(r.x, r.y, r.w, r.h, fill);
        draw_rectangle_lines(r.x, r.y, r.w, r.h, 1.0, edge);
        // B: dots (out of the screen) or crosses (into it) on a grid; E: arrows.
        let step = 36.0;
        let mut y = r.y + step * 0.5;
        while y < r.bottom() {
            let mut x = r.x + step * 0.5;
            while x < r.right() {
                if f.b > 0.0 {
                    draw_circle_lines(x, y, 3.0, 1.0, Color::new(0.7, 0.55, 1.0, 0.35));
                    draw_circle(x, y, 1.0, Color::new(0.7, 0.55, 1.0, 0.5));
                } else if f.b < 0.0 {
                    draw_line(x - 2.5, y - 2.5, x + 2.5, y + 2.5, 1.0, Color::new(0.7, 0.55, 1.0, 0.4));
                    draw_line(x - 2.5, y + 2.5, x + 2.5, y - 2.5, 1.0, Color::new(0.7, 0.55, 1.0, 0.4));
                }
                if f.e.length() > 0.0 {
                    let d = f.e.normalize() * 9.0;
                    let (a, b) = (vec2(x, y) - d, vec2(x, y) + d);
                    let c = Color::new(0.4, 0.9, 1.0, 0.4);
                    draw_line(a.x, a.y, b.x, b.y, 1.0, c);
                    let back = -d.normalize() * 3.5;
                    let side = vec2(-back.y, back.x);
                    draw_line(b.x, b.y, b.x + back.x + side.x, b.y + back.y + side.y, 1.0, c);
                    draw_line(b.x, b.y, b.x + back.x - side.x, b.y + back.y - side.y, 1.0, c);
                }
                x += step;
            }
            y += step;
        }
        let label = match (f.e.length() > 0.0, f.b != 0.0) {
            (true, true) => "E + B field",
            (true, false) => "E field",
            _ => "B field",
        };
        draw_text(label, r.x + 3.0, r.y + 11.0, 13.0, edge);
    }
    for (k, d) in re.detectors.iter().enumerate() {
        let c = Color::new(0.55, 1.0, 0.6, 0.75);
        let label = format!("D{}  {}", k + 1, d.count);
        match d.shape {
            crate::detector::DetShape::Box(r) => {
                dashed_rect(r, c);
                draw_text(&label, r.x + 3.0, r.y + r.h - 4.0, 13.0, c);
            }
            crate::detector::DetShape::Line(a, b) => {
                dashed(a, b, Color::new(c.r, c.g, c.b, 0.9), 1.5);
                let m = (a + b) * 0.5;
                draw_text(&label, m.x + 5.0, m.y - 5.0, 13.0, c);
            }
        }
    }
    for sp in &re.spawners {
        let c = Color::from_rgba(255, 210, 90, 230);
        if sp.beam && sp.width > 0.5 && sp.vel.length() > 0.5 {
            let d = sp.vel.normalize();
            let side = vec2(-d.y, d.x) * sp.width * 0.5;
            let (a, b) = (sp.pos - side, sp.pos + side);
            let on = if sp.firing() { 0.85 } else { 0.3 };
            draw_line(a.x, a.y, b.x, b.y, 2.0, Color::new(1.0, 0.82, 0.35, on));
        }
        draw_circle_lines(sp.pos.x, sp.pos.y, 5.0, 1.0, c);
        if sp.vel.length() > 0.5 {
            let tip = sp.pos + sp.vel.normalize() * (8.0 + sp.vel.length().sqrt() * 1.5);
            draw_line(sp.pos.x, sp.pos.y, tip.x, tip.y, 1.0, c);
            let back = (sp.pos - tip).normalize() * 4.0;
            let side = vec2(-back.y, back.x) * 0.6;
            draw_line(tip.x, tip.y, tip.x + back.x + side.x, tip.y + back.y + side.y, 1.0, c);
            draw_line(tip.x, tip.y, tip.x + back.x - side.x, tip.y + back.y - side.y, 1.0, c);
        }
        draw_text(&nuclide_name(sp.z, sp.n), sp.pos.x + 6.0, sp.pos.y - 6.0, 13.0, c);
    }
    for d in &re.drains {
        let r = d.rect;
        draw_rectangle(r.x, r.y, r.w, r.h, Color::new(0.9, 0.3, 0.3, 0.08));
        draw_rectangle_lines(r.x, r.y, r.w, r.h, 1.0, Color::new(1.0, 0.45, 0.4, 0.6));
        let label = match d.filter {
            crate::reactor::DrainFilter::All => "drain",
            crate::reactor::DrainFilter::FissionProducts => "reprocess",
            crate::reactor::DrainFilter::OffGas => "off-gas",
            crate::reactor::DrainFilter::NotFuel => "all but fuel & n",
            crate::reactor::DrainFilter::SpentProducts => "reprocessing",
        };
        draw_text(label, r.x + 3.0, r.y + 11.0, 13.0, Color::new(1.0, 0.55, 0.5, 0.8));
    }
    for pump in &re.pumps {
        let r = pump.rect;
        draw_rectangle(r.x, r.y, r.w, r.h, Color::new(0.3, 0.6, 1.0, 0.06));
        draw_rectangle_lines(r.x, r.y, r.w, r.h, 1.0, Color::new(0.4, 0.65, 1.0, 0.45));
        let dir = pump.dir.normalize_or_zero();
        if dir != Vec2::ZERO {
            // Chevrons that march along the flow.
            let side = vec2(-dir.y, dir.x);
            let len = (r.w.abs() * dir.x.abs() + r.h.abs() * dir.y.abs()).max(1.0);
            let c = r.center();
            let phase = (get_time() as f32 * pump.speed.max(5.0) * 0.5) % 14.0;
            let mut t = -len * 0.5 + phase;
            let half = (r.w.abs() * dir.y.abs() + r.h.abs() * dir.x.abs()) * 0.3;
            while t < len * 0.5 {
                let p = c + dir * t;
                let a = p - dir * 4.0 + side * half.min(8.0);
                let b = p - dir * 4.0 - side * half.min(8.0);
                draw_line(a.x, a.y, p.x, p.y, 1.0, Color::new(0.5, 0.75, 1.0, 0.55));
                draw_line(b.x, b.y, p.x, p.y, 1.0, Color::new(0.5, 0.75, 1.0, 0.55));
                t += 14.0;
            }
        }
    }
}

/// A dashed line.
fn dashed(a: Vec2, b: Vec2, c: Color, w: f32) {
    let len = a.distance(b);
    if len < 1e-3 {
        return;
    }
    let d = (b - a) / len;
    let mut t = 0.0;
    while t < len {
        let (p, q) = (a + d * t, a + d * (t + 6.0).min(len));
        draw_line(p.x, p.y, q.x, q.y, w, c);
        t += 10.0;
    }
}

fn dashed_rect(r: Rect, c: Color) {
    let (a, b, cc, d) = (vec2(r.x, r.y), vec2(r.right(), r.y), vec2(r.right(), r.bottom()), vec2(r.x, r.bottom()));
    dashed(a, b, c, 1.0);
    dashed(b, cc, c, 1.0);
    dashed(cc, d, c, 1.0);
    dashed(d, a, c, 1.0);
}

/// Colour for a temperature ratio to ambient: blue-grey cold, through
/// orange, to white hot.
pub fn heat_color(ratio: f32) -> Color {
    let t = (ratio.max(1e-3).ln() / 6.0).clamp(-0.3, 1.0);
    if t < 0.0 {
        Color::new(0.45 + t, 0.6 + t * 0.5, 0.95, 1.0)
    } else if t < 0.5 {
        let u = t / 0.5;
        Color::new(0.55 + 0.45 * u, 0.6 - 0.15 * u, 0.7 - 0.6 * u, 1.0)
    } else {
        let u = (t - 0.5) / 0.5;
        Color::new(1.0, 0.45 + 0.55 * u, 0.1 + 0.85 * u, 1.0)
    }
}

/// Walls: insulators grey, conductors by temperature, freeze plugs icy.
pub fn draw_walls(sim: &Sim) {
    use crate::walls::WallKind;
    let amb = crate::thermal::theta_of(&sim.cfg, sim.cfg.temperature).max(1e-3);
    for s in &sim.walls.segs {
        let half = s.half;
        let c = match s.kind {
            WallKind::Insulator => Color::from_rgba(120, 128, 150, 255),
            WallKind::Conductor => heat_color(s.theta / amb),
            WallKind::FreezePlug => {
                let melt = crate::thermal::theta_of(&sim.cfg, sim.cfg.melt_temp).max(amb * 1.01);
                let f = ((s.theta - amb) / (melt - amb)).clamp(0.0, 1.0);
                Color::new(0.55 + 0.45 * f, 0.85 - 0.4 * f, 1.0 - 0.7 * f, 1.0)
            }
            WallKind::Filter => {
                // Dashed in the colour of what it sorts by: bright dashes on
                // a dark bed pass only that, dark dashes on a bright bed
                // pass everything else.
                use crate::walls::FilterRule;
                let ec = match s.rule {
                    FilterRule::Element if s.filter_z == 0 => sim.types[crate::types::ids::NEUTRON as usize].color,
                    FilterRule::Element => element_color(s.filter_z, s.filter_z),
                    FilterRule::Charged => Color::from_rgba(255, 120, 200, 255),
                    FilterRule::Energy => Color::from_rgba(255, 200, 60, 255),
                    FilterRule::Mass => Color::from_rgba(170, 140, 255, 255),
                };
                let dim = Color::new(ec.r * 0.3, ec.g * 0.3, ec.b * 0.3, 1.0);
                let (bed, dash) = if s.only { (dim, ec) } else { (ec, Color::new(0.08, 0.08, 0.1, 1.0)) };
                draw_line(s.a.x, s.a.y, s.b.x, s.b.y, half * 2.0, bed);
                draw_circle(s.a.x, s.a.y, half, bed);
                draw_circle(s.b.x, s.b.y, half, bed);
                let len = s.a.distance(s.b);
                let u = (s.b - s.a) / len.max(1e-3);
                let mut t = 2.0;
                while t < len {
                    let (p, q) = (s.a + u * t, s.a + u * (t + 4.0).min(len));
                    draw_line(p.x, p.y, q.x, q.y, (half * 1.2).max(1.0), dash);
                    t += 8.0;
                }
                continue;
            }
        };
        draw_line(s.a.x, s.a.y, s.b.x, s.b.y, half * 2.0, c);
        draw_circle(s.a.x, s.a.y, half, c);
        draw_circle(s.b.x, s.b.y, half, c);
        if s.kind == WallKind::Conductor {
            // A thin dark core line tells conductors from insulators.
            draw_line(s.a.x, s.a.y, s.b.x, s.b.y, (half * 0.5).max(1.0), Color::new(0.1, 0.1, 0.12, 0.5));
        }
        if s.reflect_neutrons != s.kind.reflects_neutrons() {
            // Non-default neutron behaviour: a neutron reflector gets bright
            // cyan edges, a neutron window a dashed cyan centre line.
            let cyan = Color::new(0.35, 0.95, 1.0, 1.0);
            let len = s.a.distance(s.b).max(1e-3);
            let u = (s.b - s.a) / len;
            if s.reflect_neutrons {
                let off = macroquad::math::vec2(-u.y, u.x) * (half + 0.5);
                for o in [off, -off] {
                    draw_line(s.a.x + o.x, s.a.y + o.y, s.b.x + o.x, s.b.y + o.y, 1.5, cyan);
                }
            } else {
                let mut t = 0.0;
                while t < len {
                    let (p, q) = (s.a + u * t, s.a + u * (t + 5.0).min(len));
                    draw_line(p.x, p.y, q.x, q.y, (half * 0.5).max(1.0), cyan);
                    t += 10.0;
                }
            }
        }
    }
}
