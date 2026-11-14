//! The training loop: mini-batch gradient descent over a number of epochs,
//! recording train and validation metrics after every epoch.

use crate::activation::Activation;
use crate::data::{Dataset, one_hot};
use crate::metrics::accuracy;
use crate::network::Mlp;
use crate::rng::Rng;

/// Hyper-parameters for one training run.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    /// Neurons in each hidden layer, e.g. `[8]` or `[16, 8]`.
    pub hidden_layers: Vec<usize>,
    /// Activation used by every hidden layer.
    pub activation: Activation,
    /// Gradient descent step size η.
    pub learning_rate: f64,
    /// Number of passes over the training set.
    pub epochs: usize,
    /// Samples per gradient step. Use the training set size for full-batch GD.
    pub batch_size: usize,
    /// Seed for weight initialization and shuffling.
    pub seed: u64,
    /// Early stopping: stop after this many epochs without a new best
    /// validation loss, and return the weights from the best epoch.
    /// `None` trains for all `epochs`.
    pub patience: Option<usize>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![8],
            activation: Activation::Tanh,
            learning_rate: 0.05,
            epochs: 300,
            batch_size: 16,
            seed: 42,
            patience: None,
        }
    }
}

impl TrainConfig {
    /// Full layer sizes for a dataset: `[inputs, hidden..., classes]`.
    #[must_use]
    pub fn layer_sizes(&self, n_features: usize, n_classes: usize) -> Vec<usize> {
        let mut sizes = Vec::with_capacity(self.hidden_layers.len() + 2);
        sizes.push(n_features);
        sizes.extend_from_slice(&self.hidden_layers);
        sizes.push(n_classes);
        sizes
    }
}

/// Metrics measured at the end of one epoch (epoch 0 = before training).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochMetrics {
    /// Epoch number, starting at 0 for the untrained network.
    pub epoch: usize,
    /// Mean cross-entropy on the full training set.
    pub train_loss: f64,
    /// Mean cross-entropy on the validation set.
    pub val_loss: f64,
    /// Accuracy on the training set.
    pub train_accuracy: f64,
    /// Accuracy on the validation set.
    pub val_accuracy: f64,
}

/// The learning curves of a training run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    /// One entry per epoch, starting with epoch 0.
    pub epochs: Vec<EpochMetrics>,
    /// `true` if the loss became NaN or infinite and training was stopped.
    pub diverged: bool,
    /// With early stopping: the epoch whose weights were returned.
    pub restored_epoch: Option<usize>,
}

impl History {
    /// Metrics after the last completed epoch.
    ///
    /// # Panics
    ///
    /// Panics if the history is empty.
    #[must_use]
    pub fn last(&self) -> &EpochMetrics {
        self.epochs.last().expect("history always contains epoch 0")
    }

    /// The epoch with the lowest validation loss.
    #[must_use]
    pub fn best_by_val_loss(&self) -> Option<&EpochMetrics> {
        self.epochs
            .iter()
            .filter(|m| m.val_loss.is_finite())
            .min_by(|a, b| a.val_loss.total_cmp(&b.val_loss))
    }
}

/// Measures loss and accuracy of `net` on `data`.
fn evaluate(net: &Mlp, data: &Dataset) -> (f64, f64) {
    let targets = one_hot(data.labels(), data.n_classes());
    let probs = net.predict_proba(data.features());
    let loss = crate::loss::cross_entropy(&probs, &targets);
    let acc = accuracy(data.labels(), &probs.argmax_rows());
    (loss, acc)
}

/// Builds a fresh network from `config` and trains it on `train`, measuring
/// `validation` after every epoch. Both datasets must already be standardized.
///
/// Each epoch shuffles the training set and walks it in mini-batches of
/// `config.batch_size`, taking one gradient descent step per batch.
///
/// # Panics
///
/// Panics if `batch_size` or `epochs` is zero, or the datasets are empty.
#[must_use]
pub fn train(config: &TrainConfig, train: &Dataset, validation: &Dataset) -> (Mlp, History) {
    assert!(config.batch_size > 0, "batch_size must be positive");
    assert!(!train.is_empty() && !validation.is_empty(), "empty dataset");

    let mut rng = Rng::new(config.seed);
    let sizes = config.layer_sizes(train.n_features(), train.n_classes());
    let mut net = Mlp::new(&sizes, config.activation, &mut rng);
    let targets = one_hot(train.labels(), train.n_classes());

    let mut history = History::default();
    let record = |net: &Mlp, epoch: usize| {
        let (train_loss, train_accuracy) = evaluate(net, train);
        let (val_loss, val_accuracy) = evaluate(net, validation);
        EpochMetrics {
            epoch,
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
        }
    };
    history.epochs.push(record(&net, 0));

    let mut order: Vec<usize> = (0..train.len()).collect();
    // Best weights seen so far, only tracked when early stopping is enabled.
    let mut best: Option<(f64, usize, Mlp)> = None;
    for epoch in 1..=config.epochs {
        rng.shuffle(&mut order);
        for batch in order.chunks(config.batch_size) {
            let x = train.features().select_rows(batch);
            let y = targets.select_rows(batch);
            net.train_step(&x, &y, config.learning_rate);
        }
        let metrics = record(&net, epoch);
        let finite = metrics.train_loss.is_finite() && metrics.val_loss.is_finite();
        history.epochs.push(metrics);
        if !finite {
            history.diverged = true;
            break;
        }
        if let Some(patience) = config.patience {
            let improved = best
                .as_ref()
                .is_none_or(|(loss, _, _)| metrics.val_loss < *loss);
            if improved {
                best = Some((metrics.val_loss, epoch, net.clone()));
            } else if epoch - best.as_ref().map_or(0, |b| b.1) >= patience {
                break;
            }
        }
    }
    if let Some((_, epoch, best_net)) = best {
        net = best_net;
        history.restored_epoch = Some(epoch);
    }
    (net, history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    /// Two well-separated Gaussian blobs.
    fn blobs(n_per_class: usize, seed: u64) -> Dataset {
        blobs_with(n_per_class, seed, 2.0, 0.5)
    }

    /// Two Gaussian blobs centred at ±`centre` with standard deviation `noise`.
    fn blobs_with(n_per_class: usize, seed: u64, centre: f64, noise: f64) -> Dataset {
        let mut rng = Rng::new(seed);
        let mut values = Vec::new();
        let mut labels = Vec::new();
        for class in 0..2 {
            let c = if class == 0 { -centre } else { centre };
            for _ in 0..n_per_class {
                values.push(c + noise * rng.normal());
                values.push(c + noise * rng.normal());
                labels.push(class);
            }
        }
        let features = Matrix::from_vec(labels.len(), 2, values).unwrap();
        Dataset::new(
            features,
            labels,
            vec!["neg".into(), "pos".into()],
            vec!["x".into(), "y".into()],
        )
    }

    #[test]
    fn learns_a_separable_problem() {
        let config = TrainConfig {
            epochs: 50,
            ..TrainConfig::default()
        };
        let (_, history) = train(&config, &blobs(40, 1), &blobs(20, 2));
        assert_eq!(history.epochs.len(), 51);
        assert!(!history.diverged);
        assert!(history.last().train_loss < history.epochs[0].train_loss);
        assert!(history.last().val_accuracy > 0.95);
    }

    #[test]
    fn early_stopping_stops_and_restores_best_epoch() {
        // Heavily overlapping classes and a large network: the validation
        // loss soon starts rising while the training loss keeps falling.
        let config = TrainConfig {
            hidden_layers: vec![64],
            epochs: 2000,
            learning_rate: 0.1,
            patience: Some(20),
            ..TrainConfig::default()
        };
        let (train_set, val_set) = (blobs_with(20, 1, 0.3, 1.0), blobs_with(20, 2, 0.3, 1.0));
        let (net, history) = train(&config, &train_set, &val_set);
        let restored = history.restored_epoch.expect("early stopping was enabled");
        let best = history.best_by_val_loss().unwrap();
        assert_eq!(restored, best.epoch);
        assert!(history.last().epoch < 2000, "should stop early");
        // The returned network really is the one from the best epoch.
        let targets = one_hot(val_set.labels(), 2);
        assert!((net.loss(val_set.features(), &targets) - best.val_loss).abs() < 1e-12);
    }

    #[test]
    fn training_is_reproducible_for_a_fixed_seed() {
        let config = TrainConfig {
            epochs: 5,
            ..TrainConfig::default()
        };
        let (a, ha) = train(&config, &blobs(10, 1), &blobs(5, 2));
        let (b, hb) = train(&config, &blobs(10, 1), &blobs(5, 2));
        assert_eq!(a, b);
        assert_eq!(ha, hb);
    }
}
