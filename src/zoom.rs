//! Magnifier window shared by the sandbox and the Nuclear Game: a source
//! square that follows the mouse (or is locked in place), drawn enlarged
//! in a window on the other side of the screen.

use crate::render::{Renderer, View};
use crate::sim::Sim;
use macroquad::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ZoomState {
    Off,
    /// Source square follows the mouse; click to lock it.
    Aiming,
    /// Source fixed; the mouse works inside the magnified window.
    Locked,
}

pub struct Zoom {
    pub state: ZoomState,
    /// World point at the centre of the source square.
    pub center: Vec2,
    pub factor: f32,
    /// Side of the magnified window (screen px).
    pub size: f32,
}

impl Default for Zoom {
    fn default() -> Self {
        Self { state: ZoomState::Off, center: Vec2::ZERO, factor: 4.0, size: 380.0 }
    }
}

impl Zoom {
    pub fn src(&self) -> Rect {
        let s = self.size / self.factor;
        Rect::new(self.center.x - s / 2.0, self.center.y - s / 2.0, s, s)
    }

    /// The magnified window, on the side of the screen away from the
    /// source; `left` = screen px covered by a panel on the left; `view` =
    /// the world rectangle the main view shows (the screen, unless the app
    /// pans and zooms).
    pub fn dst(&self, left: f32, view: Rect) -> Rect {
        let on_screen = (self.center.x - view.x) / view.w * screen_width();
        let right = on_screen < screen_width() * 0.5;
        let x = if right { screen_width() - self.size - 10.0 } else { left + 10.0 };
        Rect::new(x, 10.0, self.size, self.size)
    }

    pub fn clamp_center(&mut self, w: f32, h: f32) {
        let half = self.size / self.factor / 2.0;
        self.center.x = self.center.x.clamp(half.min(w / 2.0), (w - half).max(w / 2.0));
        self.center.y = self.center.y.clamp(half.min(h / 2.0), (h - half).max(h / 2.0));
    }

    /// Z toggles aiming, Esc closes.
    pub fn toggle(&mut self) {
        self.state = if self.state == ZoomState::Off { ZoomState::Aiming } else { ZoomState::Off };
    }

    /// Source frame (on the main view, which shows world rect `main`),
    /// magnified view and hint.
    pub fn draw(&self, renderer: &mut Renderer, sim: &Sim, view: &View, left: f32, main: Rect) {
        if self.state == ZoomState::Off {
            return;
        }
        let (src, dst) = (self.src(), self.dst(left, main));
        let frame_col = if self.state == ZoomState::Locked { Color::new(1.0, 0.85, 0.3, 0.9) } else { Color::new(1.0, 1.0, 1.0, 0.7) };
        let k = screen_width() / main.w;
        let f = Rect::new((src.x - main.x) * k, (src.y - main.y) * k, src.w * k, src.h * k);
        draw_rectangle_lines(f.x, f.y, f.w, f.h, 1.0, frame_col);
        renderer.draw_region(sim, view, src, dst);
        draw_rectangle_lines(dst.x - 1.0, dst.y - 1.0, dst.w + 2.0, dst.h + 2.0, 2.0, frame_col);
        let hint = if self.state == ZoomState::Aiming { "click to lock, scroll to zoom" } else { "Z / Esc to close, scroll to zoom" };
        draw_text(&format!("x{:.1}  {hint}", self.factor), dst.x + 6.0, dst.y + dst.h - 8.0, 16.0, frame_col);
    }
}
