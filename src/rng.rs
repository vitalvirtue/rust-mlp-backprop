//! A small, seedable pseudo-random number generator.
//!
//! We implement SplitMix64 ourselves instead of pulling in a crate so that
//! every experiment is bit-for-bit reproducible from a single `u64` seed,
//! on every platform, regardless of dependency versions.

/// SplitMix64 pseudo-random number generator.
///
/// Not cryptographically secure; intended for weight initialization and
/// data shuffling only.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a generator from a seed. The same seed always yields the
    /// same sequence.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a uniformly distributed `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits: exactly the precision of an f64 mantissa.
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// Returns a uniformly distributed `f64` in `[low, high)`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    /// Returns a sample from the standard normal distribution N(0, 1)
    /// using the Box–Muller transform.
    pub fn normal(&mut self) -> f64 {
        // 1 - u maps [0, 1) to (0, 1], so ln() never sees zero.
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Returns a uniformly distributed integer in `[0, n)`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0) is undefined");
        // Lemire's multiply-shift method: unbiased enough for our purposes
        // and avoids the modulo bias of `next_u64() % n`.
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    /// Shuffles a slice in place using the Fisher–Yates algorithm.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn next_f64_stays_in_unit_interval() {
        let mut rng = Rng::new(1);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn normal_has_roughly_zero_mean_and_unit_variance() {
        let mut rng = Rng::new(7);
        let n = 50_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.03, "variance = {var}");
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::new(3);
        let mut v: Vec<usize> = (0..50).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert_ne!(v, sorted, "50 elements should almost surely move");
    }
}
