//! Random numbers for the simulation.
//!
//! One generator per thread (so tests running in parallel don't disturb
//! each other), reseeded by `Sim::step` from the scene's seed and step
//! count: the same scene, stepped the same way, plays out identically.

use std::cell::Cell;

thread_local! {
    static STATE: Cell<u64> = const { Cell::new(0x9E37_79B9_7F4A_7C15) };
}

/// Restart the sequence from `seed`.
pub fn seed(seed: u64) {
    STATE.with(|s| s.set(mix(seed)));
}

/// SplitMix64 finaliser: spreads nearby seeds (step counts) far apart.
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Next 64 random bits (wyrand).
pub fn next_u64() -> u64 {
    STATE.with(|s| {
        let x = s.get().wrapping_add(0xA076_1D64_78BD_642F);
        s.set(x);
        let t = (x as u128).wrapping_mul((x ^ 0xE703_7ED1_A0B4_28DB) as u128);
        ((t >> 64) as u64) ^ (t as u64)
    })
}

/// Uniform in [0, 1).
pub fn unit() -> f64 {
    (next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

pub trait RandomRange: Copy {
    fn gen_range(low: Self, high: Self) -> Self;
}

impl RandomRange for f64 {
    fn gen_range(low: f64, high: f64) -> f64 {
        low + (high - low) * unit()
    }
}

impl RandomRange for f32 {
    fn gen_range(low: f32, high: f32) -> f32 {
        let x = low + (high - low) * unit() as f32;
        // Rounding can land exactly on `high`.
        if x >= high && high > low { low } else { x }
    }
}

macro_rules! int_range {
    ($($t:ty),*) => {$(
        impl RandomRange for $t {
            /// [low, high): `high` is excluded.
            fn gen_range(low: $t, high: $t) -> $t {
                if high <= low {
                    return low;
                }
                let span = (high as i128 - low as i128) as u128;
                (low as i128 + (next_u64() as u128 % span) as i128) as $t
            }
        }
    )*};
}
int_range!(i32, u32, i64, u64, usize);

/// Uniform in [low, high) (floats) or low..high (integers), like
/// `macroquad::rand::gen_range`.
pub fn gen_range<T: RandomRange>(low: T, high: T) -> T {
    T::gen_range(low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reseeding_repeats_the_sequence() {
        seed(42);
        let a: Vec<u64> = (0..5).map(|_| next_u64()).collect();
        seed(42);
        let b: Vec<u64> = (0..5).map(|_| next_u64()).collect();
        assert_eq!(a, b);
        seed(43);
        assert_ne!(next_u64(), a[0]);
    }

    #[test]
    fn ranges_stay_in_bounds() {
        seed(1);
        for _ in 0..10_000 {
            let f = gen_range(-2.0f32, 3.0);
            assert!((-2.0..3.0).contains(&f));
            let k = gen_range(0usize, 7);
            assert!(k < 7);
        }
        let mean: f64 = (0..100_000).map(|_| unit()).sum::<f64>() / 100_000.0;
        assert!((mean - 0.5).abs() < 0.01);
    }
}

/// Hash maps and sets with a fixed hasher: iteration order depends only on
/// the contents, never on a per-process random key, so loops over them
/// (and the random draws inside) happen in the same order every run.
pub type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<FixedHasher>>;
pub type HashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<FixedHasher>>;

/// FxHash (rustc's hasher): fast, deterministic, fine for integer keys.
#[derive(Default, Clone, Copy)]
pub struct FixedHasher(u64);

impl std::hash::Hasher for FixedHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

impl FixedHasher {
    #[inline]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}
