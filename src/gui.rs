//! A small immediate-mode GUI for the control panels (sandbox and Nuclear Game).
//!
//! Why not megaui: its default font has no arrows, Greek or superscripts
//! (the event log needs →, γ, ν, α), its sliders are tiny drag handles, and
//! its windows scroll awkwardly. This one draws with a bundled Unicode font,
//! makes whole rows clickable, scrolls with the wheel, and clips properly.

use macroquad::prelude::*;
use crate::rng::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

pub use crate::skin::FONT_BYTES;

pub mod col {
    use macroquad::prelude::Color;
    const fn c(r: u8, g: u8, b: u8, a: u8) -> Color {
        Color { r: r as f32 / 255.0, g: g as f32 / 255.0, b: b as f32 / 255.0, a: a as f32 / 255.0 }
    }
    pub const PANEL: Color = c(17, 18, 25, 246);
    pub const EDGE: Color = c(52, 56, 74, 255);
    pub const TEXT: Color = c(220, 224, 236, 255);
    pub const DIM: Color = c(140, 146, 166, 255);
    pub const FAINT: Color = c(96, 101, 120, 255);
    pub const ACCENT: Color = c(118, 162, 255, 255);
    pub const WIDGET: Color = c(36, 39, 52, 255);
    pub const HOVER: Color = c(48, 52, 70, 255);
    pub const PRESS: Color = c(62, 68, 92, 255);
    pub const ON: Color = c(58, 86, 150, 255);
    pub const TRACK: Color = c(30, 32, 43, 255);
    pub const WARN: Color = c(255, 196, 92, 255);
    pub const BAD: Color = c(255, 104, 96, 255);
    pub const GOOD: Color = c(110, 214, 140, 255);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    Lin,
    Log,
}

pub struct TextResp {
    pub changed: bool,
    pub submitted: bool,
}

#[derive(Default, Clone, Copy)]
struct Scroll {
    offset: f32,
    content: f32,
}

pub struct Gui {
    pub font: Font,
    pub size: f32,
    // Input, sampled once per frame.
    pub mouse: Vec2,
    last_mouse: Vec2,
    pressed: bool,
    down: bool,
    released: bool,
    right_pressed: bool,
    wheel: f32,
    shift: bool,
    /// Widget being pressed or dragged.
    active: Option<u64>,
    drag_t: f32,
    /// Text field with the keyboard.
    pub focus: Option<u64>,
    /// The focused field's text is selected: typing replaces it.
    fresh_focus: bool,
    focus_used: bool,
    focus_claimed: bool,
    repeat_at: f64,
    // Layout.
    x: f32,
    pub y: f32,
    pub w: f32,
    clip: Rect,
    scope: u64,
    scrolls: HashMap<u64, Scroll>,
    scroll_stack: Vec<(u64, Rect, f32, Rect)>,
    pub tooltip: Option<Vec<String>>,
    char_buf: Vec<char>,
}

fn hash_of(a: u64, s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    a.hash(&mut h);
    s.hash(&mut h);
    h.finish()
}

impl Gui {
    pub fn new() -> Self {
        let font = load_ttf_font_from_bytes(FONT_BYTES).expect("bundled font");
        Self {
            font,
            size: 16.0,
            mouse: Vec2::ZERO,
            last_mouse: Vec2::ZERO,
            pressed: false,
            down: false,
            released: false,
            right_pressed: false,
            wheel: 0.0,
            shift: false,
            active: None,
            drag_t: 0.0,
            focus: None,
            fresh_focus: false,
            focus_used: false,
            focus_claimed: false,
            repeat_at: 0.0,
            x: 0.0,
            y: 0.0,
            w: 0.0,
            clip: Rect::new(0.0, 0.0, 0.0, 0.0),
            scope: 0,
            scrolls: HashMap::default(),
            scroll_stack: Vec::new(),
            tooltip: None,
            char_buf: Vec::new(),
        }
    }

    /// Start a frame. `owns` = the mouse is over the GUI (its clicks are ours).
    pub fn begin(&mut self) {
        self.last_mouse = self.mouse;
        self.mouse = mouse_position().into();
        self.pressed = is_mouse_button_pressed(MouseButton::Left);
        self.down = is_mouse_button_down(MouseButton::Left);
        self.released = is_mouse_button_released(MouseButton::Left);
        self.right_pressed = is_mouse_button_pressed(MouseButton::Right);
        self.wheel = mouse_wheel().1;
        self.shift = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
        self.tooltip = None;
        self.focus_used = false;
        self.focus_claimed = false;
        self.char_buf.clear();
        while let Some(c) = get_char_pressed() {
            self.char_buf.push(c);
        }
        if !self.down && !self.released {
            self.active = None;
        }
        self.clip = Rect::new(0.0, 0.0, screen_width(), screen_height());
    }

    /// End a frame: a click that hit no text field, or a focused field that
    /// wasn't drawn (its tab was closed), drops the keyboard focus.
    pub fn end(&mut self) {
        if (self.pressed && !self.focus_claimed) || !self.focus_used {
            self.focus = None;
        }
        if self.released {
            self.active = None;
        }
    }

    /// Is a drag in progress (the GUI keeps the mouse until release)?
    pub fn dragging(&self) -> bool {
        self.active.is_some() && self.down
    }

    pub fn typing(&self) -> bool {
        self.focus.is_some()
    }

    // ------------------------------------------------------------- layout

    pub fn layout(&mut self, x: f32, y: f32, w: f32) {
        self.x = x;
        self.y = y;
        self.w = w;
    }

    pub fn scope(&mut self, s: &str) {
        self.scope = hash_of(0, s);
    }

    fn id(&self, label: &str) -> u64 {
        hash_of(self.scope, label)
    }

    pub fn space(&mut self, h: f32) {
        self.y += h;
    }

    /// Take a full-width rect of height h from the layout.
    pub fn alloc(&mut self, h: f32) -> Rect {
        let r = Rect::new(self.x, self.y, self.w, h);
        self.y += h;
        r
    }

    pub fn x(&self) -> f32 {
        self.x
    }

    fn visible(&self, r: &Rect) -> bool {
        r.y + r.h >= self.clip.y && r.y <= self.clip.y + self.clip.h
    }

    fn hovered(&self, r: &Rect) -> bool {
        r.contains(self.mouse) && self.clip.contains(self.mouse)
    }

    // ------------------------------------------------------------- text

    pub fn text_width(&self, s: &str, size: f32) -> f32 {
        if s.contains('\u{304}') {
            let plain: String = s.chars().filter(|&c| c != '\u{304}').collect();
            return measure_text(&plain, Some(&self.font), size as u16, 1.0).width;
        }
        measure_text(s, Some(&self.font), size as u16, 1.0).width
    }

    /// Draw text with its top-left at (x, y). A combining macron (U+0304,
    /// as in ν̄) is drawn as a bar over its letter: the glyph renderer would
    /// put it beside the letter.
    pub fn text(&self, s: &str, x: f32, y: f32, size: f32, color: Color) {
        let params = TextParams { font: Some(&self.font), font_size: size as u16, color, ..Default::default() };
        let baseline = (y + size * 0.78).round();
        if !s.contains('\u{304}') {
            draw_text_ex(s, x.round(), baseline, params);
            return;
        }
        let mut plain = String::with_capacity(s.len());
        let mut bars = Vec::new();
        for c in s.chars() {
            if c == '\u{304}' {
                if let Some(base) = plain.chars().last() {
                    let x0 = self.text_width(&plain[..plain.len() - base.len_utf8()], size);
                    let wb = self.text_width(&base.to_string(), size);
                    bars.push((x0, wb));
                }
            } else {
                plain.push(c);
            }
        }
        draw_text_ex(&plain, x.round(), baseline, params);
        let yb = baseline - size * 0.66;
        for (x0, wb) in bars {
            draw_line(x.round() + x0 + wb * 0.12, yb, x.round() + x0 + wb * 0.88, yb, (size / 14.0).max(1.0), color);
        }
    }

    /// Shorten s with an ellipsis to fit width w.
    pub fn fit(&self, s: &str, w: f32, size: f32) -> (String, bool) {
        if self.text_width(s, size) <= w {
            return (s.to_string(), false);
        }
        let chars: Vec<char> = s.chars().collect();
        let (mut lo, mut hi) = (0usize, chars.len());
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            let t: String = chars[..mid].iter().collect::<String>() + "…";
            if self.text_width(&t, size) <= w { lo = mid } else { hi = mid - 1 }
        }
        (chars[..lo].iter().collect::<String>() + "…", true)
    }

    fn wrap(&self, s: &str, w: f32, size: f32) -> Vec<String> {
        let mut lines = Vec::new();
        for para in s.split('\n') {
            let mut line = String::new();
            for word in para.split(' ') {
                let cand = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
                if self.text_width(&cand, size) > w && !line.is_empty() {
                    lines.push(std::mem::take(&mut line));
                    line = word.to_string();
                } else {
                    line = cand;
                }
            }
            lines.push(line);
        }
        lines
    }

    pub fn label_sized(&mut self, s: &str, size: f32, color: Color) {
        let lh = size * 1.3;
        for l in self.wrap(s, self.w, size) {
            let r = Rect::new(self.x, self.y, self.w, lh);
            if self.visible(&r) {
                self.text(&l, self.x, self.y + (lh - size) * 0.5, size, color);
            }
            self.y += lh;
        }
    }

    pub fn label(&mut self, s: &str) {
        self.label_sized(s, self.size, col::TEXT);
    }

    pub fn label_col(&mut self, s: &str, c: Color) {
        self.label_sized(s, self.size, c);
    }

    pub fn note(&mut self, s: &str) {
        self.label_sized(s, self.size - 2.5, col::DIM);
    }

    pub fn heading(&mut self, s: &str) {
        self.y += 6.0;
        let r = Rect::new(self.x, self.y, self.w, 22.0);
        if self.visible(&r) {
            self.text(&s.to_uppercase(), self.x, self.y + 3.0, self.size - 3.0, col::ACCENT);
            let tw = self.text_width(&s.to_uppercase(), self.size - 3.0);
            draw_line(self.x + tw + 8.0, self.y + 11.0, self.x + self.w, self.y + 11.0, 1.0, col::EDGE);
        }
        self.y += 24.0;
    }

    /// Collapsible section header: click to open or close it. Returns
    /// whether it's open (draw its contents only then). `badge` is shown
    /// dimmed on the right (a count, a state).
    pub fn section(&mut self, title: &str, badge: &str, open: &mut bool) -> bool {
        self.collapsible(title, badge, open, true)
    }

    /// A smaller collapsible header inside a section (one item of a list).
    pub fn subsection(&mut self, title: &str, badge: &str, open: &mut bool) -> bool {
        self.collapsible(title, badge, open, false)
    }

    fn collapsible(&mut self, title: &str, badge: &str, open: &mut bool, top: bool) -> bool {
        self.y += if top { 6.0 } else { 2.0 };
        let h = if top { 26.0 } else { 28.0 };
        let r = Rect::new(self.x, self.y, self.w, h);
        let shown = title.split_once("##").map_or(title, |p| p.0);
        let shown = if top { shown.to_uppercase() } else { shown.to_string() };
        if self.visible(&r) {
            let id = self.id(&format!("section {title}"));
            let (clicked, hov, _) = self.press_box(id, r);
            if top {
                if hov {
                    draw_rectangle(r.x - 4.0, r.y, r.w + 4.0, r.h, col::HOVER);
                }
            } else {
                draw_rectangle(r.x, r.y, r.w, r.h, if hov { col::HOVER } else { col::WIDGET });
                draw_rectangle_lines(r.x, r.y, r.w, r.h, 1.0, if *open { col::ACCENT } else { col::EDGE });
            }
            let size = if top { self.size - 3.0 } else { self.size - 1.0 };
            let c = if top { col::ACCENT } else { col::TEXT };
            // Disclosure triangle (the font has no arrow glyphs).
            let (tx, ty) = (r.x + if top { 4.0 } else { 9.0 }, r.y + h * 0.5);
            let tc = if top { col::ACCENT } else { col::DIM };
            if *open {
                draw_triangle(vec2(tx - 4.0, ty - 2.5), vec2(tx + 4.0, ty - 2.5), vec2(tx, ty + 3.0), tc);
            } else {
                draw_triangle(vec2(tx - 2.5, ty - 4.0), vec2(tx - 2.5, ty + 4.0), vec2(tx + 3.0, ty), tc);
            }
            let x0 = r.x + if top { 14.0 } else { 20.0 };
            let bw = if badge.is_empty() { 0.0 } else { self.text_width(badge, size - 1.0) + 8.0 };
            let (t, _) = self.fit(&shown, r.w - (x0 - r.x) - bw - 8.0, size);
            self.text(&t, x0, r.y + (h - size) * 0.5 - 1.0, size, c);
            if !badge.is_empty() {
                self.text(badge, r.x + r.w - bw + (if top { 8.0 } else { 2.0 }), r.y + (h - size) * 0.5, size - 1.0, col::DIM);
            }
            if top {
                let tw = self.text_width(&t, size) + 14.0;
                draw_line(r.x + tw + 8.0, r.y + 13.0, r.x + r.w - bw - 4.0, r.y + 13.0, 1.0, col::EDGE);
            }
            if clicked {
                *open = !*open;
            }
        }
        self.y += h + 4.0;
        *open
    }

    /// Two-column "key   value" row.
    pub fn kv(&mut self, k: &str, v: &str, vc: Color) {
        let h = self.size * 1.35;
        let r = Rect::new(self.x, self.y, self.w, h);
        if self.visible(&r) {
            self.text(k, self.x, self.y + 2.0, self.size - 1.0, col::DIM);
            let tw = self.text_width(v, self.size - 1.0);
            self.text(v, self.x + self.w - tw, self.y + 2.0, self.size - 1.0, vc);
        }
        self.y += h;
    }

    // ------------------------------------------------------------- widgets

    /// A clickable box; returns (clicked, hovered, held).
    fn press_box(&mut self, id: u64, r: Rect) -> (bool, bool, bool) {
        let hov = self.hovered(&r);
        if hov && self.pressed {
            self.active = Some(id);
        }
        let held = self.active == Some(id) && self.down;
        let clicked = self.active == Some(id) && self.released && hov;
        (clicked, hov, held)
    }

    fn box_bg(&self, r: Rect, hov: bool, held: bool, on: bool) {
        let c = if held {
            col::PRESS
        } else if on {
            col::ON
        } else if hov {
            col::HOVER
        } else {
            col::WIDGET
        };
        draw_rectangle(r.x, r.y, r.w, r.h, c);
        draw_rectangle_lines(r.x, r.y, r.w, r.h, 1.0, if on { col::ACCENT } else { col::EDGE });
    }

    fn button_in(&mut self, label: &str, r: Rect, on: bool) -> bool {
        let id = self.id(label);
        if !self.visible(&r) {
            return false;
        }
        let (clicked, hov, held) = self.press_box(id, r);
        self.box_bg(r, hov, held, on);
        let size = self.size - 1.0;
        // "text##key": only the text is shown, the key keeps ids unique.
        let shown = label.split_once("##").map_or(label, |p| p.0);
        let (t, _) = self.fit(shown, r.w - 8.0, size);
        let tw = self.text_width(&t, size);
        self.text(&t, r.x + (r.w - tw) * 0.5, r.y + (r.h - size) * 0.5 - 1.0, size, col::TEXT);
        clicked
    }

    pub fn button(&mut self, label: &str) -> bool {
        let r = self.alloc(28.0);
        self.y += 4.0;
        self.button_in(label, Rect::new(r.x, r.y, r.w, r.h), false)
    }

    /// A row of equal buttons; returns the clicked one.
    pub fn buttons(&mut self, labels: &[&str]) -> Option<usize> {
        self.button_grid(labels, labels.len().max(1), None)
    }

    /// Buttons in a grid of `cols` columns; `on` highlights one.
    pub fn button_grid(&mut self, labels: &[&str], cols: usize, on: Option<usize>) -> Option<usize> {
        let gap = 4.0;
        let bw = (self.w - gap * (cols as f32 - 1.0)) / cols as f32;
        let bh = 28.0;
        let mut hit = None;
        for (k, l) in labels.iter().enumerate() {
            let (c, r) = (k % cols, k / cols);
            let rect = Rect::new(self.x + c as f32 * (bw + gap), self.y + r as f32 * (bh + gap), bw, bh);
            if self.button_in(l, rect, on == Some(k)) {
                hit = Some(k);
            }
        }
        let rows = labels.len().div_ceil(cols);
        self.y += rows as f32 * (bh + gap);
        hit
    }

    /// Tab bar: widths follow the labels, filling the row.
    pub fn tabs(&mut self, labels: &[&str], sel: &mut usize) -> bool {
        let gap = 3.0;
        let size = self.size - 1.0;
        let natural: Vec<f32> = labels.iter().map(|l| self.text_width(l, size) + 14.0).collect();
        let total: f32 = natural.iter().sum::<f32>() + gap * (labels.len() as f32 - 1.0);
        let extra = ((self.w - total) / labels.len() as f32).max(0.0);
        let squeeze = if total > self.w { self.w / total } else { 1.0 };
        let mut x = self.x;
        let mut changed = false;
        for (k, l) in labels.iter().enumerate() {
            let w = (natural[k] + extra) * squeeze;
            if self.button_in(l, Rect::new(x, self.y, w, 30.0), k == *sel) && k != *sel {
                *sel = k;
                changed = true;
            }
            x += w + gap;
        }
        self.y += 34.0;
        changed
    }

    /// One-of-n selector.
    pub fn segmented(&mut self, labels: &[&str], sel: &mut usize, cols: usize) -> bool {
        match self.button_grid(labels, cols, Some(*sel)) {
            Some(k) if k != *sel => {
                *sel = k;
                true
            }
            _ => false,
        }
    }

    pub fn checkbox(&mut self, label: &str, v: &mut bool) -> bool {
        let r = self.alloc(24.0);
        if !self.visible(&r) {
            return false;
        }
        let id = self.id(label);
        let (clicked, hov, _) = self.press_box(id, r);
        let b = Rect::new(r.x, r.y + 3.0, 18.0, 18.0);
        draw_rectangle(b.x, b.y, b.w, b.h, if *v { col::ON } else if hov { col::HOVER } else { col::WIDGET });
        draw_rectangle_lines(b.x, b.y, b.w, b.h, 1.0, if *v { col::ACCENT } else { col::EDGE });
        if *v {
            draw_line(b.x + 4.0, b.y + 9.5, b.x + 7.5, b.y + 13.0, 2.0, col::TEXT);
            draw_line(b.x + 7.5, b.y + 13.0, b.x + 14.0, b.y + 5.0, 2.0, col::TEXT);
        }
        let shown = label.split_once("##").map_or(label, |p| p.0);
        let (t, _) = self.fit(shown, r.w - 28.0, self.size - 1.0);
        self.text(&t, r.x + 26.0, r.y + 3.0, self.size - 1.0, if hov { col::TEXT } else { Color::new(0.82, 0.84, 0.9, 1.0) });
        if clicked {
            *v = !*v;
        }
        clicked
    }

    /// Full-width slider: label and value on top, a track below. Click
    /// anywhere on the track to jump there, drag to adjust (hold Shift for
    /// fine control), right-click to reset to `default`.
    pub fn slider(&mut self, label: &str, v: &mut f32, lo: f32, hi: f32, scale: Scale, default: f32, fmt: impl Fn(f32) -> String) -> bool {
        let r = self.alloc(40.0);
        if !self.visible(&r) {
            return false;
        }
        let id = self.id(label);
        let to_t = |x: f32| match scale {
            Scale::Lin => (x - lo) / (hi - lo),
            Scale::Log => (x.max(lo).ln() - lo.ln()) / (hi.ln() - lo.ln()),
        };
        let from_t = |t: f32| {
            let t = t.clamp(0.0, 1.0);
            match scale {
                Scale::Lin => lo + (hi - lo) * t,
                Scale::Log => (lo.ln() + (hi.ln() - lo.ln()) * t).exp(),
            }
        };
        let track = Rect::new(r.x + 2.0, r.y + 24.0, r.w - 4.0, 8.0);
        let hit = Rect::new(r.x, r.y, r.w, r.h - 2.0);
        let hov = self.hovered(&hit);
        let old = *v;
        if hov && self.pressed {
            self.active = Some(id);
            if self.mouse.y >= r.y + 17.0 {
                // Clicked on the track: jump there.
                self.drag_t = (self.mouse.x - track.x) / track.w;
                *v = from_t(self.drag_t);
            } else {
                self.drag_t = to_t(*v);
            }
        }
        if hov && self.right_pressed {
            *v = default;
        }
        let held = self.active == Some(id) && self.down;
        if held {
            let dx = self.mouse.x - self.last_mouse.x;
            self.drag_t += dx / track.w * if self.shift { 0.1 } else { 1.0 };
            *v = from_t(self.drag_t);
        }
        let t = to_t(*v).clamp(0.0, 1.0);
        let size = self.size - 1.0;
        let shown = label.split_once("##").map_or(label, |p| p.0);
        let (shown, _) = self.fit(shown, r.w - 90.0, size);
        self.text(&shown, r.x, r.y + 2.0, size, if hov || held { col::TEXT } else { Color::new(0.8, 0.82, 0.88, 1.0) });
        let val = fmt(*v);
        let tw = self.text_width(&val, size);
        self.text(&val, r.x + r.w - tw, r.y + 2.0, size, if held { col::WARN } else { col::ACCENT });
        draw_rectangle(track.x, track.y, track.w, track.h, col::TRACK);
        draw_rectangle(track.x, track.y, track.w * t, track.h, if held { Color::new(0.5, 0.62, 1.0, 1.0) } else { col::ON });
        draw_rectangle_lines(track.x, track.y, track.w, track.h, 1.0, col::EDGE);
        let kx = track.x + track.w * t;
        draw_rectangle(kx - 3.0, track.y - 4.0, 6.0, track.h + 8.0, if held || hov { col::TEXT } else { Color::new(0.75, 0.78, 0.86, 1.0) });
        *v != old
    }

    /// Button that fires on press and then repeats while held.
    fn repeat_button(&mut self, key: &str, label: &str, r: Rect) -> bool {
        let id = self.id(key);
        if !self.visible(&r) {
            return false;
        }
        let (_, hov, held) = self.press_box(id, r);
        self.box_bg(r, hov, held, false);
        let size = self.size + 2.0;
        let tw = self.text_width(label, size);
        self.text(label, r.x + (r.w - tw) * 0.5, r.y + (r.h - size) * 0.5 - 1.0, size, col::TEXT);
        let now = get_time();
        if hov && self.pressed {
            self.repeat_at = now + 0.4;
            return true;
        }
        if held && now >= self.repeat_at {
            self.repeat_at = now + 0.06;
            return true;
        }
        false
    }

    /// "label   [−] value [+]" integer stepper with hold-to-repeat.
    pub fn stepper(&mut self, label: &str, v: &mut u32, lo: u32, hi: u32) -> bool {
        let r = self.alloc(30.0);
        self.y += 3.0;
        if !self.visible(&r) {
            return false;
        }
        let old = *v;
        let shown = label.split_once("##").map_or(label, |p| p.0);
        self.text(shown, r.x, r.y + 6.0, self.size, col::TEXT);
        let bw = 34.0;
        let vw = 56.0;
        let x1 = r.x + r.w - bw;
        let x0 = x1 - vw - bw;
        if self.repeat_button(&format!("{label}#dec"), "−", Rect::new(x0, r.y, bw, r.h)) {
            *v = v.saturating_sub(1).max(lo);
        }
        if self.repeat_button(&format!("{label}#inc"), "+", Rect::new(x1, r.y, bw, r.h)) {
            *v = (*v + 1).min(hi);
        }
        let s = format!("{}", *v);
        draw_rectangle(x0 + bw, r.y, vw, r.h, col::TRACK);
        draw_rectangle_lines(x0 + bw, r.y, vw, r.h, 1.0, col::EDGE);
        let tw = self.text_width(&s, self.size + 1.0);
        self.text(&s, x0 + bw + (vw - tw) * 0.5, r.y + 6.0, self.size + 1.0, col::TEXT);
        *v != old
    }

    /// Single-line text field. Clicking it selects all (typing replaces).
    pub fn text_field(&mut self, label: &str, s: &mut String, hint: &str) -> TextResp {
        let r = self.alloc(30.0);
        self.y += 4.0;
        let id = self.id(label);
        let mut resp = TextResp { changed: false, submitted: false };
        if !self.visible(&r) {
            if self.focus == Some(id) {
                self.focus_used = true;
            }
            return resp;
        }
        let hov = self.hovered(&r);
        if hov && self.pressed {
            self.focus = Some(id);
            self.fresh_focus = true;
            self.focus_claimed = true;
        }
        let focused = self.focus == Some(id);
        if focused {
            self.focus_used = true;
            for &c in &self.char_buf {
                if c.is_control() {
                    continue;
                }
                if self.fresh_focus {
                    s.clear();
                    self.fresh_focus = false;
                }
                s.push(c);
                resp.changed = true;
            }
            let now = get_time();
            let bs = if is_key_pressed(KeyCode::Backspace) {
                self.repeat_at = now + 0.4;
                true
            } else if is_key_down(KeyCode::Backspace) && now >= self.repeat_at {
                self.repeat_at = now + 0.04;
                true
            } else {
                false
            };
            if bs {
                if self.fresh_focus {
                    s.clear();
                    self.fresh_focus = false;
                } else {
                    s.pop();
                }
                resp.changed = true;
            }
            if is_key_pressed(KeyCode::Enter) || is_key_pressed(KeyCode::KpEnter) {
                resp.submitted = true;
                self.focus = None;
            }
            if is_key_pressed(KeyCode::Escape) {
                self.focus = None;
            }
        }
        draw_rectangle(r.x, r.y, r.w, r.h, if focused { Color::new(0.1, 0.11, 0.16, 1.0) } else { col::TRACK });
        draw_rectangle_lines(r.x, r.y, r.w, r.h, if focused { 1.5 } else { 1.0 }, if focused { col::ACCENT } else if hov { col::DIM } else { col::EDGE });
        let size = self.size;
        let label = label.split_once("##").map_or(label, |p| p.0);
        let lw = self.text_width(label, size - 2.0);
        self.text(label, r.x + 8.0, r.y + 7.0, size - 2.0, col::DIM);
        let tx = r.x + 16.0 + lw;
        if s.is_empty() && !focused {
            self.text(hint, tx, r.y + 6.0, size, col::FAINT);
        } else {
            if focused && self.fresh_focus && !s.is_empty() {
                let tw = self.text_width(s, size);
                draw_rectangle(tx - 2.0, r.y + 5.0, tw + 4.0, r.h - 10.0, Color::new(0.25, 0.35, 0.65, 0.8));
            }
            let (t, _) = self.fit(s, r.w - (tx - r.x) - 10.0, size);
            self.text(&t, tx, r.y + 6.0, size, col::TEXT);
            if focused && !self.fresh_focus && (get_time() * 2.0) as i64 % 2 == 0 {
                let cx = tx + self.text_width(&t, size) + 1.0;
                draw_line(cx, r.y + 6.0, cx, r.y + r.h - 6.0, 1.5, col::TEXT);
            }
        }
        resp
    }

    /// Tooltip `lines` when the mouse is over r.
    pub fn tip_if_hover(&mut self, r: Rect, lines: Vec<String>) {
        if self.hovered(&r) {
            self.tooltip = Some(lines);
        }
    }

    /// Tooltip shown at the mouse after the frame (by the caller).
    pub fn tip(&mut self, lines: Vec<String>) {
        self.tooltip = Some(lines);
    }

    /// Is the mouse over r (and inside the current clip)?
    pub fn hover(&self, r: Rect) -> bool {
        self.hovered(&r)
    }

    /// A custom-drawn clickable area: (clicked, hovered, held). `key` must
    /// be unique in the current scope. While held, the GUI keeps the mouse.
    pub fn press(&mut self, key: &str, r: Rect) -> (bool, bool, bool) {
        if !self.visible(&r) {
            return (false, false, false);
        }
        let id = self.id(key);
        self.press_box(id, r)
    }

    /// Was the left button pressed over r this frame?
    pub fn pressed_in(&self, r: Rect) -> bool {
        self.pressed && self.hovered(&r)
    }

    /// Was the right button pressed over r this frame?
    pub fn right_clicked(&self, r: Rect) -> bool {
        self.right_pressed && self.hovered(&r)
    }

    /// Mouse movement since last frame (screen px).
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse - self.last_mouse
    }

    pub fn shift(&self) -> bool {
        self.shift
    }

    // ------------------------------------------------------------- scrolling

    /// Begin a scrolling region filling `r`: everything laid out until
    /// `scroll_end` is clipped to it and moved by its scroll offset.
    pub fn scroll_begin(&mut self, name: &str, r: Rect) {
        let id = hash_of(1, name);
        let st = *self.scrolls.entry(id).or_default();
        // Wheel over the region scrolls it.
        let max = (st.content - r.h).max(0.0);
        let mut off = st.offset.clamp(0.0, max);
        if r.contains(self.mouse) && self.wheel != 0.0 && max > 0.0 {
            off = (off - self.wheel.signum() * 60.0).clamp(0.0, max);
            self.wheel = 0.0;
        }
        self.scrolls.get_mut(&id).unwrap().offset = off;
        self.scroll_stack.push((id, r, self.y, self.clip));
        self.clip = r;
        flush_scissor(Some(r));
        self.x = r.x;
        self.w = r.w - 10.0;
        self.y = r.y - off;
    }

    /// Current scroll offset of a region (to keep a view stable as rows are added).
    pub fn scroll_offset(&mut self, name: &str) -> &mut f32 {
        &mut self.scrolls.entry(hash_of(1, name)).or_default().offset
    }

    pub fn scroll_end(&mut self) {
        let Some((id, r, _y0, clip)) = self.scroll_stack.pop() else { return };
        let off = self.scrolls[&id].offset;
        let content = self.y - (r.y - off);
        self.scrolls.get_mut(&id).unwrap().content = content;
        flush_scissor(if self.scroll_stack.is_empty() { None } else { Some(clip) });
        self.clip = clip;
        // Scrollbar.
        if content > r.h + 1.0 {
            let bar = Rect::new(r.x + r.w - 6.0, r.y, 6.0, r.h);
            let kh = (r.h * r.h / content).max(24.0);
            let max = content - r.h;
            let ky = bar.y + (bar.h - kh) * (off / max);
            let sid = id ^ 0x5c;
            let grab = Rect::new(bar.x - 4.0, bar.y, bar.w + 4.0, bar.h);
            let hov = grab.contains(self.mouse);
            if hov && self.pressed {
                self.active = Some(sid);
            }
            if self.active == Some(sid) && self.down {
                let t = ((self.mouse.y - bar.y - kh * 0.5) / (bar.h - kh)).clamp(0.0, 1.0);
                self.scrolls.get_mut(&id).unwrap().offset = t * max;
            }
            draw_rectangle(bar.x, bar.y, bar.w, bar.h, col::TRACK);
            draw_rectangle(bar.x, ky, bar.w, kh, if hov || self.active == Some(sid) { col::DIM } else { col::FAINT });
        }
        self.y = r.y + r.h;
    }

    /// Draw a tooltip box near the mouse.
    pub fn draw_tooltip(&self, lines: &[String]) {
        if lines.is_empty() {
            return;
        }
        let size = self.size - 1.0;
        let w = lines.iter().map(|l| self.text_width(l, size)).fold(0.0, f32::max) + 16.0;
        let h = lines.len() as f32 * (size + 4.0) + 10.0;
        let x = (self.mouse.x + 16.0).min(screen_width() - w - 4.0).max(4.0);
        let y = (self.mouse.y + 18.0).min(screen_height() - h - 4.0).max(4.0);
        draw_rectangle(x, y, w, h, Color::new(0.06, 0.065, 0.09, 0.96));
        draw_rectangle_lines(x, y, w, h, 1.0, col::EDGE);
        for (k, l) in lines.iter().enumerate() {
            self.text(l, x + 8.0, y + 6.0 + k as f32 * (size + 4.0), size, if k == 0 { col::TEXT } else { Color::new(0.78, 0.8, 0.88, 1.0) });
        }
    }
}

/// Set the clip rectangle. (No flush: the scissor is part of the draw-call
/// batch key, and flushing mid-frame breaks glyphs the font atlas adds later
/// in the frame.)
fn flush_scissor(r: Option<Rect>) {
    let gl = unsafe { get_internal_gl() };
    gl.quad_gl.scissor(r.map(|r| (r.x as i32, r.y as i32, r.w.max(0.0) as i32, r.h.max(0.0) as i32)));
}
