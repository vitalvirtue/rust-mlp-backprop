//! Hyper-parameter grid search: runs every combination of activation,
//! learning rate and hidden-layer size over several seeds, in parallel,
//! and aggregates the results.

use std::thread;

use crate::activation::Activation;
use crate::data::Split;
use crate::train::{History, TrainConfig, train};

/// The search space.
#[derive(Debug, Clone, PartialEq)]
pub struct GridSpec {
    /// Hidden activations to compare.
    pub activations: Vec<Activation>,
    /// Learning rates to compare.
    pub learning_rates: Vec<f64>,
    /// Neurons in the single hidden layer.
    pub hidden_sizes: Vec<usize>,
    /// Seeds per configuration (initialization + shuffling); results are
    /// averaged over them so one lucky initialization cannot win.
    pub seeds: Vec<u64>,
    /// Epochs per run.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
}

impl Default for GridSpec {
    /// The grid required by the assignment, 3 × 4 × 5 configurations × 5 seeds.
    fn default() -> Self {
        Self {
            activations: Activation::ALL.to_vec(),
            learning_rates: vec![0.001, 0.01, 0.05, 0.1],
            hidden_sizes: vec![2, 4, 8, 16, 32],
            seeds: vec![1, 2, 3, 4, 5],
            epochs: 300,
            batch_size: 16,
        }
    }
}

impl GridSpec {
    /// Every training configuration in the grid, in a fixed order.
    #[must_use]
    pub fn configs(&self) -> Vec<TrainConfig> {
        let mut out = Vec::new();
        for &activation in &self.activations {
            for &learning_rate in &self.learning_rates {
                for &hidden in &self.hidden_sizes {
                    for &seed in &self.seeds {
                        out.push(TrainConfig {
                            hidden_layers: vec![hidden],
                            activation,
                            learning_rate,
                            epochs: self.epochs,
                            batch_size: self.batch_size,
                            seed,
                        });
                    }
                }
            }
        }
        out
    }
}

/// One finished training run.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// The configuration that was trained.
    pub config: TrainConfig,
    /// Its learning curves.
    pub history: History,
}

/// Trains every configuration, spreading the work over all CPU cores.
///
/// Each run owns its own seeded RNG, so the results are identical no matter
/// how many threads are used or in which order they finish.
#[must_use]
pub fn run_grid(spec: &GridSpec, split: &Split) -> Vec<RunResult> {
    let configs = spec.configs();
    let n_threads = thread::available_parallelism().map_or(1, |n| n.get());
    let chunk_size = configs.len().div_ceil(n_threads).max(1);

    // `thread::scope` lets the worker threads borrow `split` and `configs`
    // safely: the compiler guarantees they finish before the scope ends.
    thread::scope(|scope| {
        let handles: Vec<_> = configs
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|config| {
                            let (_, history) = train(config, &split.train, &split.validation);
                            RunResult {
                                config: config.clone(),
                                history,
                            }
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("training thread panicked"))
            .collect()
    })
}

/// Mean and (population) standard deviation.
#[must_use]
pub fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Learning curves averaged over seeds, epoch by epoch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeanCurves {
    /// Mean training loss per epoch.
    pub train_loss: Vec<f64>,
    /// Mean validation loss per epoch.
    pub val_loss: Vec<f64>,
    /// Mean training accuracy per epoch.
    pub train_accuracy: Vec<f64>,
    /// Mean validation accuracy per epoch.
    pub val_accuracy: Vec<f64>,
}

/// Aggregated results of one (activation, learning rate, hidden size) cell.
#[derive(Debug, Clone, PartialEq)]
pub struct CellSummary {
    /// Hidden activation.
    pub activation: Activation,
    /// Learning rate.
    pub learning_rate: f64,
    /// Hidden layer size.
    pub hidden: usize,
    /// Number of seeds aggregated.
    pub n_runs: usize,
    /// Runs whose loss became NaN/inf.
    pub n_diverged: usize,
    /// Final validation loss: mean and std over seeds.
    pub val_loss: (f64, f64),
    /// Final validation accuracy: mean and std over seeds.
    pub val_accuracy: (f64, f64),
    /// Final training loss: mean and std over seeds.
    pub train_loss: (f64, f64),
    /// Final training accuracy: mean and std over seeds.
    pub train_accuracy: (f64, f64),
    /// Learning curves averaged over seeds.
    pub curves: MeanCurves,
}

/// Groups runs by (activation, learning rate, hidden size) and averages
/// over seeds. Output order follows the grid order.
#[must_use]
pub fn aggregate(runs: &[RunResult]) -> Vec<CellSummary> {
    let mut cells: Vec<CellSummary> = Vec::new();
    let mut groups: Vec<Vec<&RunResult>> = Vec::new();
    for run in runs {
        let key = (
            run.config.activation,
            run.config.learning_rate,
            run.config.hidden_layers[0],
        );
        match cells
            .iter()
            .position(|c| (c.activation, c.learning_rate, c.hidden) == key)
        {
            Some(i) => groups[i].push(run),
            None => {
                cells.push(CellSummary {
                    activation: key.0,
                    learning_rate: key.1,
                    hidden: key.2,
                    n_runs: 0,
                    n_diverged: 0,
                    val_loss: (0.0, 0.0),
                    val_accuracy: (0.0, 0.0),
                    train_loss: (0.0, 0.0),
                    train_accuracy: (0.0, 0.0),
                    curves: MeanCurves::default(),
                });
                groups.push(vec![run]);
            }
        }
    }
    for (cell, group) in cells.iter_mut().zip(&groups) {
        let finals = |f: fn(&crate::train::EpochMetrics) -> f64| -> Vec<f64> {
            group.iter().map(|r| f(r.history.last())).collect()
        };
        cell.n_runs = group.len();
        cell.n_diverged = group.iter().filter(|r| r.history.diverged).count();
        cell.val_loss = mean_std(&finals(|m| m.val_loss));
        cell.val_accuracy = mean_std(&finals(|m| m.val_accuracy));
        cell.train_loss = mean_std(&finals(|m| m.train_loss));
        cell.train_accuracy = mean_std(&finals(|m| m.train_accuracy));
        cell.curves = mean_curves(group);
    }
    cells
}

/// Averages curves epoch by epoch over the runs that reached that epoch.
fn mean_curves(group: &[&RunResult]) -> MeanCurves {
    let longest = group
        .iter()
        .map(|r| r.history.epochs.len())
        .max()
        .unwrap_or(0);
    let mut curves = MeanCurves::default();
    for e in 0..longest {
        let at: Vec<_> = group
            .iter()
            .filter_map(|r| r.history.epochs.get(e))
            .collect();
        let mean = |f: fn(&crate::train::EpochMetrics) -> f64| {
            at.iter().map(|m| f(m)).sum::<f64>() / at.len() as f64
        };
        curves.train_loss.push(mean(|m| m.train_loss));
        curves.val_loss.push(mean(|m| m.val_loss));
        curves.train_accuracy.push(mean(|m| m.train_accuracy));
        curves.val_accuracy.push(mean(|m| m.val_accuracy));
    }
    curves
}

/// The cell with the lowest mean final validation loss among those that
/// satisfy `filter`. Validation loss is used rather than accuracy because
/// on a small validation set many configurations tie at 100 % accuracy,
/// while the loss still tells them apart.
#[must_use]
pub fn best_cell(
    cells: &[CellSummary],
    filter: impl Fn(&CellSummary) -> bool,
) -> Option<&CellSummary> {
    cells
        .iter()
        .filter(|c| filter(c) && c.n_diverged == 0 && c.val_loss.0.is_finite())
        .min_by(|a, b| a.val_loss.0.total_cmp(&b.val_loss.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::matrix::Matrix;
    use crate::rng::Rng;

    fn tiny_split() -> Split {
        let mut rng = Rng::new(0);
        let n = 60;
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let values = labels
            .iter()
            .flat_map(|&y| {
                [
                    y as f64 + 0.3 * rng.normal(),
                    -(y as f64) + 0.3 * rng.normal(),
                ]
            })
            .collect();
        let ds = Dataset::new(
            Matrix::from_vec(n, 2, values).unwrap(),
            labels,
            vec!["a".into(), "b".into(), "c".into()],
            vec!["x".into(), "y".into()],
        );
        ds.stratified_split(0.6, 0.2, &mut rng).standardized().0
    }

    fn small_spec() -> GridSpec {
        GridSpec {
            activations: vec![Activation::Tanh, Activation::Relu],
            learning_rates: vec![0.01, 0.1],
            hidden_sizes: vec![4],
            seeds: vec![1, 2],
            epochs: 10,
            batch_size: 8,
        }
    }

    #[test]
    fn configs_cover_the_full_grid() {
        assert_eq!(GridSpec::default().configs().len(), 3 * 4 * 5 * 5);
        assert_eq!(small_spec().configs().len(), 2 * 2 * 2);
    }

    #[test]
    fn parallel_grid_is_deterministic_and_aggregates_per_cell() {
        let split = tiny_split();
        let a = run_grid(&small_spec(), &split);
        let b = run_grid(&small_spec(), &split);
        assert_eq!(a.len(), 8);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.history, y.history);
        }
        let cells = aggregate(&a);
        assert_eq!(cells.len(), 4);
        assert!(
            cells
                .iter()
                .all(|c| c.n_runs == 2 && c.curves.val_loss.len() == 11)
        );
        let best = best_cell(&cells, |_| true).unwrap();
        assert!(cells.iter().all(|c| best.val_loss.0 <= c.val_loss.0));
    }

    #[test]
    fn mean_std_of_known_values() {
        let (m, s) = mean_std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((m - 5.0).abs() < 1e-12);
        assert!((s - 2.0).abs() < 1e-12);
    }
}
