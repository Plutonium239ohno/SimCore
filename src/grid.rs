use macroquad::math::Vec2;
use rayon::prelude::*;

/// Uniform spatial grid built with a counting sort.
///
/// After `build`, `order` is a permutation that sorts particles by cell.
/// The simulation physically reorders its particle arrays with it, so every
/// cell's particles are contiguous in memory and a whole row of neighbouring
/// cells is one contiguous index range. This keeps neighbour loops cache-friendly
/// and branch-light.
#[derive(Default)]
pub struct Grid {
    pub cell: f32,
    pub inv: f32,
    pub cols: usize,
    pub rows: usize,
    /// Wrap-around world: neighbour searches continue across the edges.
    pub wrap: bool,
    /// start[c]..start[c+1] = indices (in sorted order) of particles in cell c.
    pub start: Vec<u32>,
    pub order: Vec<u32>,
    cell_of: Vec<u32>,
    cursor: Vec<u32>,
}

impl Grid {
    pub fn build(&mut self, pos: &[Vec2], width: f32, height: f32, cell: f32) {
        self.cell = cell;
        self.inv = 1.0 / cell;
        self.cols = ((width * self.inv).ceil() as usize).max(1);
        self.rows = ((height * self.inv).ceil() as usize).max(1);
        let ncells = self.cols * self.rows;
        let n = pos.len();

        let (inv, cols, rows) = (self.inv, self.cols as i32, self.rows as i32);
        self.cell_of.resize(n, 0);
        self.cell_of.par_iter_mut().zip(pos.par_iter()).for_each(|(c, p)| {
            let cx = ((p.x * inv) as i32).clamp(0, cols - 1);
            let cy = ((p.y * inv) as i32).clamp(0, rows - 1);
            *c = (cy * cols + cx) as u32;
        });

        // Counting sort: histogram -> prefix sum -> scatter.
        self.cursor.clear();
        self.cursor.resize(ncells, 0);
        for &c in &self.cell_of {
            self.cursor[c as usize] += 1;
        }
        self.start.resize(ncells + 1, 0);
        self.start[0] = 0;
        for c in 0..ncells {
            self.start[c + 1] = self.start[c] + self.cursor[c];
        }
        self.cursor.copy_from_slice(&self.start[..ncells]);
        self.order.resize(n, 0);
        for (i, &c) in self.cell_of.iter().enumerate() {
            let slot = &mut self.cursor[c as usize];
            self.order[*slot as usize] = i as u32;
            *slot += 1;
        }
    }

    #[inline(always)]
    fn coords(&self, p: Vec2) -> (i32, i32) {
        let cx = ((p.x * self.inv) as i32).clamp(0, self.cols as i32 - 1);
        let cy = ((p.y * self.inv) as i32).clamp(0, self.rows as i32 - 1);
        (cx, cy)
    }

    /// Calls `f(j)` for every particle in the (2*reach+1)^2 block of cells around `p`.
    /// Only valid after the particle arrays have been reordered with `order`.
    #[inline(always)]
    pub fn for_each_near(&self, p: Vec2, reach: i32, mut f: impl FnMut(usize)) {
        let (cx, cy) = self.coords(p);
        if self.wrap {
            // Each cell at most once, even if the block is wider than the world.
            let (cols, rows) = (self.cols as i32, self.rows as i32);
            let ny = (2 * reach + 1).min(rows);
            let nx = (2 * reach + 1).min(cols);
            let xs = (cx - reach).rem_euclid(cols);
            let xe = xs + nx - 1;
            for k in 0..ny {
                let row = ((cy - reach + k).rem_euclid(rows) * cols) as usize;
                let mut span = |a: i32, b: i32| {
                    for j in self.start[row + a as usize] as usize..self.start[row + b as usize + 1] as usize {
                        f(j);
                    }
                };
                if xe < cols {
                    span(xs, xe);
                } else {
                    span(xs, cols - 1);
                    span(0, xe - cols);
                }
            }
            return;
        }
        let x0 = (cx - reach).max(0) as usize;
        let x1 = (cx + reach).min(self.cols as i32 - 1) as usize;
        let y0 = (cy - reach).max(0) as usize;
        let y1 = (cy + reach).min(self.rows as i32 - 1) as usize;
        for y in y0..=y1 {
            let row = y * self.cols;
            let a = self.start[row + x0] as usize;
            let b = self.start[row + x1 + 1] as usize;
            for j in a..b {
                f(j);
            }
        }
    }
}
