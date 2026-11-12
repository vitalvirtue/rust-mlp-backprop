//! The multilayer perceptron: parameters, forward propagation,
//! backpropagation and the gradient descent update.
//!
//! Notation (batch of N samples, layers l = 1..L):
//!
//! ```text
//! A⁰ = X                                   (N × n₀)
//! Zˡ = Aˡ⁻¹ · Wˡ + bˡ                      (N × nₗ)   Wˡ: nₗ₋₁ × nₗ, bˡ: 1 × nₗ
//! Aˡ = f(Zˡ)            for hidden layers
//! P  = softmax(Zᴸ)      for the output layer
//! L  = -(1/N) Σ Σ Y ⊙ ln P
//! ```
//!
//! Backpropagation (chain rule), from the output layer backwards:
//!
//! ```text
//! δᴸ   = ∂L/∂Zᴸ = (P - Y) / N
//! ∂L/∂Wˡ = (Aˡ⁻¹)ᵀ · δˡ
//! ∂L/∂bˡ = Σ_rows δˡ
//! δˡ⁻¹ = (δˡ · (Wˡ)ᵀ) ⊙ f'(Zˡ⁻¹)
//! ```
//!
//! Update (gradient descent): `W ← W - η ∂L/∂W`, `b ← b - η ∂L/∂b`.

use crate::activation::{Activation, softmax_rows};
use crate::loss::{cross_entropy, softmax_cross_entropy_grad};
use crate::matrix::Matrix;
use crate::rng::Rng;

/// A fully connected layer: `Z = A · W + b`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLayer {
    weights: Matrix,
    biases: Matrix,
}

impl DenseLayer {
    /// Creates a layer with weights drawn from N(0, std²) and zero biases.
    fn new(fan_in: usize, fan_out: usize, std: f64, rng: &mut Rng) -> Self {
        let weights = (0..fan_in * fan_out).map(|_| rng.normal() * std).collect();
        Self {
            weights: Matrix::from_vec(fan_in, fan_out, weights).expect("fan_in * fan_out values"),
            biases: Matrix::zeros(1, fan_out),
        }
    }

    /// Weight matrix, shape `fan_in × fan_out`.
    #[must_use]
    pub fn weights(&self) -> &Matrix {
        &self.weights
    }

    /// Bias row vector, shape `1 × fan_out`.
    #[must_use]
    pub fn biases(&self) -> &Matrix {
        &self.biases
    }

    /// `Z = A · W + b`.
    fn linear(&self, input: &Matrix) -> Matrix {
        input.matmul(&self.weights).add_row_vector(&self.biases)
    }
}

/// Everything computed during a forward pass that backpropagation needs.
#[derive(Debug, Clone)]
pub struct ForwardPass {
    /// `inputs[l]` is the input to layer `l` (so `inputs[0] = X`).
    pub inputs: Vec<Matrix>,
    /// `pre_activations[l]` is `Z` of layer `l`.
    pub pre_activations: Vec<Matrix>,
    /// Softmax output probabilities, `N × n_classes`.
    pub output: Matrix,
}

/// Gradients of the loss with respect to every parameter, one entry per layer.
#[derive(Debug, Clone)]
pub struct Gradients {
    /// `∂L/∂Wˡ` for each layer.
    pub weights: Vec<Matrix>,
    /// `∂L/∂bˡ` for each layer.
    pub biases: Vec<Matrix>,
}

/// A multilayer perceptron with one activation function shared by all
/// hidden layers and a softmax output layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    layers: Vec<DenseLayer>,
    hidden_activation: Activation,
}

impl Mlp {
    /// Builds a network from layer sizes, e.g. `[4, 8, 3]` = 4 inputs,
    /// one hidden layer of 8 neurons, 3 output classes.
    ///
    /// Weights are initialized with **He** initialization (`std = √(2/fan_in)`)
    /// for ReLU and **Xavier/Glorot** (`std = √(2/(fan_in+fan_out))`) for
    /// sigmoid and tanh, which keeps activation variance roughly constant from
    /// layer to layer. Biases start at zero.
    ///
    /// # Panics
    ///
    /// Panics if fewer than two sizes are given or any size is zero.
    #[must_use]
    pub fn new(layer_sizes: &[usize], hidden_activation: Activation, rng: &mut Rng) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "need at least input and output sizes"
        );
        assert!(
            layer_sizes.iter().all(|&n| n > 0),
            "layer sizes must be positive"
        );
        let layers = layer_sizes
            .windows(2)
            .map(|pair| {
                let (fan_in, fan_out) = (pair[0], pair[1]);
                let std = match hidden_activation {
                    Activation::Relu => (2.0 / fan_in as f64).sqrt(),
                    Activation::Sigmoid | Activation::Tanh => {
                        (2.0 / (fan_in + fan_out) as f64).sqrt()
                    }
                };
                DenseLayer::new(fan_in, fan_out, std, rng)
            })
            .collect();
        Self {
            layers,
            hidden_activation,
        }
    }

    /// Layer sizes, including input and output, e.g. `[4, 8, 3]`.
    #[must_use]
    pub fn layer_sizes(&self) -> Vec<usize> {
        std::iter::once(self.layers[0].weights.rows())
            .chain(self.layers.iter().map(|l| l.weights.cols()))
            .collect()
    }

    /// Activation used by every hidden layer.
    #[must_use]
    pub fn hidden_activation(&self) -> Activation {
        self.hidden_activation
    }

    /// The layers, input side first.
    #[must_use]
    pub fn layers(&self) -> &[DenseLayer] {
        &self.layers
    }

    /// Total number of trainable parameters (weights + biases).
    #[must_use]
    pub fn n_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.as_slice().len() + l.biases.as_slice().len())
            .sum()
    }

    /// Forward propagation, keeping every intermediate value for backprop.
    #[must_use]
    pub fn forward(&self, x: &Matrix) -> ForwardPass {
        let mut inputs = Vec::with_capacity(self.layers.len());
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        let mut a = x.clone();
        let last = self.layers.len() - 1;
        for (l, layer) in self.layers.iter().enumerate() {
            let z = layer.linear(&a);
            let next = if l == last {
                softmax_rows(&z)
            } else {
                self.hidden_activation.forward(&z)
            };
            inputs.push(a);
            pre_activations.push(z);
            a = next;
        }
        ForwardPass {
            inputs,
            pre_activations,
            output: a,
        }
    }

    /// Class probabilities for each row of `x`.
    #[must_use]
    pub fn predict_proba(&self, x: &Matrix) -> Matrix {
        self.forward(x).output
    }

    /// Most likely class for each row of `x`.
    #[must_use]
    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        self.predict_proba(x).argmax_rows()
    }

    /// Mean cross-entropy loss on `(x, targets)` where `targets` is one-hot.
    #[must_use]
    pub fn loss(&self, x: &Matrix, targets: &Matrix) -> f64 {
        cross_entropy(&self.predict_proba(x), targets)
    }

    /// Backpropagation: gradients of the mean loss w.r.t. every parameter.
    #[must_use]
    pub fn backward(&self, pass: &ForwardPass, targets: &Matrix) -> Gradients {
        let n_layers = self.layers.len();
        let mut grad_w = Vec::with_capacity(n_layers);
        let mut grad_b = Vec::with_capacity(n_layers);

        // δᴸ = (P - Y) / N
        let mut delta = softmax_cross_entropy_grad(&pass.output, targets);
        for l in (0..n_layers).rev() {
            // ∂L/∂Wˡ = (Aˡ⁻¹)ᵀ · δˡ   and   ∂L/∂bˡ = Σ_rows δˡ
            grad_w.push(pass.inputs[l].transpose().matmul(&delta));
            grad_b.push(delta.column_sums());
            if l > 0 {
                // δˡ⁻¹ = (δˡ · (Wˡ)ᵀ) ⊙ f'(Zˡ⁻¹)
                let upstream = delta.matmul(&self.layers[l].weights.transpose());
                let local = self
                    .hidden_activation
                    .derivative_matrix(&pass.pre_activations[l - 1]);
                delta = upstream.hadamard(&local);
            }
        }
        // Gradients were collected output-first; flip to input-first.
        grad_w.reverse();
        grad_b.reverse();
        Gradients {
            weights: grad_w,
            biases: grad_b,
        }
    }

    /// One gradient descent step: `θ ← θ - η · ∂L/∂θ`.
    ///
    /// # Panics
    ///
    /// Panics if `grads` does not match the network's shape.
    pub fn apply_gradients(&mut self, grads: &Gradients, learning_rate: f64) {
        assert_eq!(
            grads.weights.len(),
            self.layers.len(),
            "gradient layer count"
        );
        for ((layer, gw), gb) in self
            .layers
            .iter_mut()
            .zip(&grads.weights)
            .zip(&grads.biases)
        {
            layer.weights.add_scaled_in_place(gw, -learning_rate);
            layer.biases.add_scaled_in_place(gb, -learning_rate);
        }
    }

    /// Convenience: forward + backward + update on one mini-batch.
    /// Returns the batch loss measured *before* the update.
    pub fn train_step(&mut self, x: &Matrix, targets: &Matrix, learning_rate: f64) -> f64 {
        let pass = self.forward(x);
        let loss = cross_entropy(&pass.output, targets);
        let grads = self.backward(&pass, targets);
        self.apply_gradients(&grads, learning_rate);
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::one_hot;

    fn random_batch(rng: &mut Rng, n: usize, features: usize, classes: usize) -> (Matrix, Matrix) {
        let x = Matrix::from_vec(
            n,
            features,
            (0..n * features).map(|_| rng.normal()).collect(),
        )
        .unwrap();
        let labels: Vec<usize> = (0..n).map(|_| rng.below(classes)).collect();
        (x, one_hot(&labels, classes))
    }

    #[test]
    fn forward_shapes_and_probabilities() {
        let mut rng = Rng::new(0);
        let net = Mlp::new(&[4, 8, 3], Activation::Tanh, &mut rng);
        let (x, _) = random_batch(&mut rng, 5, 4, 3);
        let pass = net.forward(&x);
        assert_eq!(pass.inputs.len(), 2);
        assert_eq!(pass.pre_activations[0].shape(), (5, 8));
        assert_eq!(pass.output.shape(), (5, 3));
        for row in pass.output.row_iter() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        assert_eq!(net.layer_sizes(), vec![4, 8, 3]);
        assert_eq!(net.n_parameters(), 4 * 8 + 8 + 8 * 3 + 3);
    }

    /// Mutable reference to one scalar parameter of the network.
    fn param_mut(net: &mut Mlp, layer: usize, is_bias: bool, k: usize) -> &mut f64 {
        let layer = &mut net.layers[layer];
        let m = if is_bias {
            &mut layer.biases
        } else {
            &mut layer.weights
        };
        &mut m.as_mut_slice()[k]
    }

    /// The single most important test in the project: compares the analytic
    /// gradients from `backward` with numerical ones from central finite
    /// differences, `(L(θ+h) - L(θ-h)) / 2h`, for **every** parameter.
    #[test]
    fn backprop_matches_numerical_gradients() {
        for activation in Activation::ALL {
            let mut rng = Rng::new(11);
            let mut net = Mlp::new(&[3, 5, 4, 2], activation, &mut rng);
            // Random (non-zero) biases: with zero biases a sample whose ReLU
            // units are all inactive gives Z = 0 exactly, i.e. it sits on the
            // ReLU kink where finite differences are meaningless.
            for layer in &mut net.layers {
                for b in layer.biases.as_mut_slice() {
                    *b = 0.1 * rng.normal();
                }
            }
            let (x, y) = random_batch(&mut rng, 6, 3, 2);
            let grads = net.backward(&net.forward(&x), &y);
            let h = 1e-5;

            for l in 0..net.layers.len() {
                for is_bias in [false, true] {
                    let analytic = if is_bias {
                        &grads.biases[l]
                    } else {
                        &grads.weights[l]
                    };
                    for k in 0..analytic.as_slice().len() {
                        let original = *param_mut(&mut net, l, is_bias, k);
                        *param_mut(&mut net, l, is_bias, k) = original + h;
                        let loss_plus = net.loss(&x, &y);
                        *param_mut(&mut net, l, is_bias, k) = original - h;
                        let loss_minus = net.loss(&x, &y);
                        *param_mut(&mut net, l, is_bias, k) = original;

                        let numeric = (loss_plus - loss_minus) / (2.0 * h);
                        let a = analytic.as_slice()[k];
                        let rel = (a - numeric).abs() / (a.abs() + numeric.abs()).max(1e-8);
                        assert!(
                            rel < 1e-5,
                            "{activation} layer {l} {} #{k}: analytic={a:e} numeric={numeric:e}",
                            if is_bias { "bias" } else { "weight" }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gradient_descent_reduces_loss_on_a_fixed_batch() {
        let mut rng = Rng::new(5);
        let mut net = Mlp::new(&[4, 6, 3], Activation::Sigmoid, &mut rng);
        let (x, y) = random_batch(&mut rng, 12, 4, 3);
        let before = net.loss(&x, &y);
        for _ in 0..200 {
            net.train_step(&x, &y, 0.5);
        }
        let after = net.loss(&x, &y);
        assert!(after < before * 0.5, "loss {before} -> {after}");
    }

    #[test]
    fn same_seed_builds_identical_networks() {
        let a = Mlp::new(&[4, 8, 3], Activation::Relu, &mut Rng::new(1));
        let b = Mlp::new(&[4, 8, 3], Activation::Relu, &mut Rng::new(1));
        assert_eq!(a, b);
    }
}
