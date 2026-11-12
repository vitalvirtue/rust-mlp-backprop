//! Activation functions for hidden layers, and softmax for the output layer.

use std::fmt;
use std::str::FromStr;

use crate::matrix::Matrix;

/// Hidden-layer activation function.
///
/// `Copy` because the enum is a single byte: passing it by value is cheaper
/// and simpler than passing a reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    /// σ(z) = 1 / (1 + e^(-z)), output in (0, 1).
    Sigmoid,
    /// tanh(z), output in (-1, 1), zero-centred.
    Tanh,
    /// max(0, z), cheap and does not saturate for z > 0.
    Relu,
}

impl Activation {
    /// All variants, handy for experiment grids.
    pub const ALL: [Activation; 3] = [Activation::Sigmoid, Activation::Tanh, Activation::Relu];

    /// f(z) for a single value.
    #[must_use]
    pub fn apply(self, z: f64) -> f64 {
        match self {
            Activation::Sigmoid => sigmoid(z),
            Activation::Tanh => z.tanh(),
            Activation::Relu => z.max(0.0),
        }
    }

    /// f'(z) for a single value, expressed in terms of the pre-activation `z`.
    #[must_use]
    pub fn derivative(self, z: f64) -> f64 {
        match self {
            Activation::Sigmoid => {
                let s = sigmoid(z);
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - z.tanh().powi(2),
            // The derivative at exactly 0 is undefined; 0 is the usual convention.
            Activation::Relu => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Applies f element-wise to a matrix.
    #[must_use]
    pub fn forward(self, z: &Matrix) -> Matrix {
        z.map(|x| self.apply(x))
    }

    /// Applies f' element-wise to a matrix.
    #[must_use]
    pub fn derivative_matrix(self, z: &Matrix) -> Matrix {
        z.map(|x| self.derivative(x))
    }

    /// Lower-case name used in file names and the CLI.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::Relu => "relu",
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Parses `"sigmoid"`, `"tanh"` or `"relu"` (case-insensitive).
impl FromStr for Activation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "sigmoid" => Ok(Activation::Sigmoid),
            "tanh" => Ok(Activation::Tanh),
            "relu" => Ok(Activation::Relu),
            other => Err(format!(
                "unknown activation `{other}` (expected sigmoid, tanh or relu)"
            )),
        }
    }
}

/// Numerically stable sigmoid: never evaluates `exp` of a large positive number.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Row-wise softmax: turns each row of logits into a probability distribution.
///
/// Subtracting the row maximum before `exp` does not change the result
/// (softmax is shift-invariant) but prevents overflow for large logits.
#[must_use]
pub fn softmax_rows(z: &Matrix) -> Matrix {
    let mut data = Vec::with_capacity(z.rows() * z.cols());
    for row in z.row_iter() {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        data.extend(exps.iter().map(|e| e / sum));
    }
    Matrix::from_vec(z.rows(), z.cols(), data).expect("same shape as input")
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    /// Central finite difference: (f(z+h) - f(z-h)) / 2h.
    fn numerical_derivative(act: Activation, z: f64) -> f64 {
        (act.apply(z + EPS) - act.apply(z - EPS)) / (2.0 * EPS)
    }

    #[test]
    fn known_values() {
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-12);
        assert!(Activation::Tanh.apply(0.0).abs() < 1e-12);
        assert_eq!(Activation::Relu.apply(-3.0), 0.0);
        assert_eq!(Activation::Relu.apply(2.5), 2.5);
    }

    #[test]
    fn derivatives_match_finite_differences() {
        for act in Activation::ALL {
            // Avoid z = 0 where ReLU is not differentiable.
            for z in [-3.0, -0.7, 0.3, 1.9, 4.0] {
                let analytic = act.derivative(z);
                let numeric = numerical_derivative(act, z);
                assert!(
                    (analytic - numeric).abs() < 1e-6,
                    "{act}: f'({z}) analytic={analytic} numeric={numeric}"
                );
            }
        }
    }

    #[test]
    fn sigmoid_is_stable_for_extreme_inputs() {
        assert_eq!(Activation::Sigmoid.apply(1000.0), 1.0);
        assert_eq!(Activation::Sigmoid.apply(-1000.0), 0.0);
        assert!(Activation::Sigmoid.apply(-1000.0).is_finite());
    }

    #[test]
    fn softmax_rows_sum_to_one_and_survive_large_logits() {
        let z = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 1000.0, 1000.0, 1000.0]).unwrap();
        let p = softmax_rows(&z);
        for row in p.row_iter() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(row.iter().all(|x| x.is_finite() && *x > 0.0));
        }
        assert!((p[(1, 0)] - 1.0 / 3.0).abs() < 1e-12);
        assert!(p[(0, 2)] > p[(0, 1)] && p[(0, 1)] > p[(0, 0)]);
    }

    #[test]
    fn parses_names_case_insensitively() {
        assert_eq!("ReLU".parse::<Activation>(), Ok(Activation::Relu));
        assert!("swish".parse::<Activation>().is_err());
    }
}
