//! Categorical cross-entropy loss for a softmax output layer.

use crate::matrix::Matrix;

/// Smallest probability fed into `ln`, so a confident wrong prediction gives
/// a large but finite loss instead of `inf`.
const PROB_FLOOR: f64 = 1e-12;

/// Mean categorical cross-entropy over a batch:
///
/// `L = -(1/N) Σ_n Σ_k y_nk · ln(p_nk)`
///
/// # Panics
///
/// Panics if `probs` and `targets` have different shapes.
#[must_use]
pub fn cross_entropy(probs: &Matrix, targets: &Matrix) -> f64 {
    assert_eq!(probs.shape(), targets.shape(), "loss shape mismatch");
    let n = probs.rows() as f64;
    let total: f64 = probs
        .as_slice()
        .iter()
        .zip(targets.as_slice())
        .filter(|(_, y)| **y != 0.0)
        .map(|(&p, &y)| -y * p.max(PROB_FLOOR).ln())
        .sum();
    total / n
}

/// Gradient of the mean cross-entropy with respect to the **logits** `Z`
/// of a softmax output layer:
///
/// `∂L/∂Z = (P - Y) / N`
///
/// Softmax and cross-entropy are differentiated together because their
/// combined derivative collapses to this simple, numerically stable form.
///
/// # Panics
///
/// Panics if `probs` and `targets` have different shapes.
#[must_use]
pub fn softmax_cross_entropy_grad(probs: &Matrix, targets: &Matrix) -> Matrix {
    probs.sub(targets).scale(1.0 / probs.rows() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: usize, cols: usize, data: &[f64]) -> Matrix {
        Matrix::from_vec(rows, cols, data.to_vec()).unwrap()
    }

    #[test]
    fn perfect_prediction_has_zero_loss() {
        let y = mat(1, 3, &[0., 1., 0.]);
        assert!(cross_entropy(&y, &y).abs() < 1e-12);
    }

    #[test]
    fn uniform_prediction_has_ln_k_loss() {
        let p = mat(2, 3, &[1. / 3.; 6]);
        let y = mat(2, 3, &[1., 0., 0., 0., 0., 1.]);
        assert!((cross_entropy(&p, &y) - 3f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn zero_probability_gives_finite_loss() {
        let p = mat(1, 2, &[1.0, 0.0]);
        let y = mat(1, 2, &[0.0, 1.0]);
        assert!(cross_entropy(&p, &y).is_finite());
    }

    #[test]
    fn gradient_is_p_minus_y_over_n() {
        let p = mat(2, 2, &[0.8, 0.2, 0.4, 0.6]);
        let y = mat(2, 2, &[1., 0., 0., 1.]);
        let g = softmax_cross_entropy_grad(&p, &y);
        let expected = [-0.1, 0.1, 0.2, -0.2];
        for (a, b) in g.as_slice().iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
