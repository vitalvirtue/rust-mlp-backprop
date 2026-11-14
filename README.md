# rust-mlp-backprop

A multilayer perceptron (MLP) written from scratch in Rust: hand-written
matrix operations, forward propagation, backpropagation and mini-batch
gradient descent, with **no machine learning or linear algebra libraries**.
It is trained and evaluated on two UCI datasets, **Iris** and **Breast Cancer
Wisconsin (Diagnostic)**, with a full hyper-parameter study and interactive
Plotly charts for each.

Course assignment: BM5011 Artificial Neural Networks, Assignment 2.

[![CI](https://github.com/vitalvirtue/rust-mlp-backprop/actions/workflows/ci.yml/badge.svg)](https://github.com/vitalvirtue/rust-mlp-backprop/actions/workflows/ci.yml)

## Highlights

- Everything that touches the network is implemented by hand: `Matrix`,
  activations and their derivatives, softmax, cross-entropy, backprop, the
  gradient descent update, metrics and the data pipeline.
- **Gradient check**: a unit test compares backprop's gradients against
  central finite differences for every parameter, for every activation.
- Fully reproducible: a seeded SplitMix64 PRNG drives the split, weight
  initialization and shuffling.
- Early stopping with best-weight restore, majority-class baselines and
  macro/weighted metrics for imbalanced data.
- The 300-run grid search runs in parallel with `std::thread::scope` in a few
  seconds.
- Zero `unsafe`, `clippy -D warnings` clean, CI on every push.

## Requirements

- Rust 1.85 or newer (edition 2024). Install it from <https://rustup.rs>.
- An internet connection when **viewing** the HTML charts (Plotly.js is loaded
  from its CDN).

The only dependencies are [`clap`](https://crates.io/crates/clap) for command
line parsing and [`plotly`](https://crates.io/crates/plotly) for chart
rendering; neither is involved in any network computation.

## Datasets

| | Iris | Breast Cancer Wisconsin (Diagnostic) |
|---|---|---|
| File | `data/iris.data` (included) | `data/wdbc.data` (download, see below) |
| Samples | 150 | 569 |
| Features | 4 | 30 (10 nucleus measurements × mean / SE / worst) |
| Classes | 3, balanced (50 / 50 / 50) | 2, imbalanced: benign 357 (62.7 %), malignant 212 (37.3 %) |
| Split (70/15/15, stratified) | 105 / 24 / 21 | 398 / 86 / 85 |
| Majority-class baseline (test) | 0.333 | 0.624 |

To get the Breast Cancer file, download the zip from
<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>,
extract `wdbc.data` and place it at `data/wdbc.data`. Each line is
`ID, diagnosis (M/B), 30 features`; the loader skips the ID and maps
`B → benign (class 0)`, `M → malignant (class 1)`.

## Quick start

```bash
git clone https://github.com/vitalvirtue/rust-mlp-backprop.git
cd rust-mlp-backprop

cargo test                                         # 61 unit tests + 1 doc test
cargo run --release                                # train one model on Iris, print test metrics
cargo run --release -- --dataset breast-cancer     # same for Breast Cancer
cargo run --release --bin experiments              # full study -> results/iris/
cargo run --release --bin experiments -- --dataset breast-cancer   # -> results/breast_cancer/
cargo run --release --bin worked_example           # numeric walkthrough -> results/iris/worked_example.md
```

Always use `--release` for training: it is roughly 10 to 50 times faster than a
debug build.

### Training a single model

```bash
cargo run --release -- --hidden 16 --activation relu --lr 0.05 --epochs 300
cargo run --release -- --dataset breast-cancer --hidden 16 --activation relu --patience 30
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | `iris` | `iris` or `breast-cancer` |
| `--data` | `data/iris.data` / `data/wdbc.data` | Path to the data file |
| `--hidden` | `8` | Hidden layer sizes, comma separated (`16,8` = two hidden layers) |
| `--activation` | `tanh` | `sigmoid`, `tanh` or `relu` |
| `--lr` | `0.05` | Learning rate η |
| `--epochs` | `300` | Passes over the training set |
| `--batch-size` | `16` | Mini-batch size |
| `--seed` | `42` | Seed for split, initialization and shuffling |
| `--patience` | off | Early stopping: stop after N epochs without a better validation loss and keep the best weights |
| `--log-every` | `50` | Print progress every N epochs |

### Running the experiments

```bash
cargo run --release --bin experiments -- --dataset iris --epochs 300 --seeds 5
cargo run --release --bin experiments -- --dataset breast-cancer
```

Trains every combination of 3 activations (sigmoid, tanh, ReLU) × 4 learning
rates (0.001, 0.01, 0.05, 0.1) × 5 hidden sizes (2, 4, 8, 16, 32) × 5 seeds,
selects the best configuration by mean validation loss, and evaluates it once
on the held-out test set. Outputs go to `results/<dataset>/`:

| File | Content |
|---|---|
| `dashboard.html` | All charts and tables on one interactive page |
| `plots/*.html` | Each chart on its own page; use the camera icon to save a PNG |
| `runs.csv` | One row per run (configuration × seed) |
| `summary.csv` | One row per configuration, mean ± std over seeds |
| `test_report.md` | Markdown tables: dataset summary, baseline, grid results, test metrics, confusion matrix |

### Numerical walkthrough

```bash
cargo run --release --bin worked_example -- --activation sigmoid --hidden 3 --lr 0.1
cargo run --release --bin worked_example -- --dataset breast-cancer
```

Takes a mini-batch of two training samples through one complete step and
prints every matrix: `Z¹`, `A¹`, `Z²`, softmax `P`, the loss, `δ²`, `∂L/∂W²`,
`δ¹`, `∂L/∂W¹`, the updated weights, and the loss before and after the update.

## Method

| | |
|---|---|
| Split | Stratified 70 / 15 / 15, fixed seed |
| Preprocessing | Z-score standardization fitted on the training set only |
| Architecture | inputs → hidden (activation) → classes (softmax); 4→h→3 for Iris, 30→h→2 for Breast Cancer |
| Initialization | He for ReLU, Xavier/Glorot for sigmoid and tanh; zero biases |
| Loss | Mean categorical cross-entropy |
| Optimizer | Mini-batch gradient descent, reshuffled every epoch |
| Model selection | Lowest mean validation loss over 5 seeds; the test set is used once |

Forward pass for a batch `X` (N × n₀):

```text
Zˡ = Aˡ⁻¹ · Wˡ + bˡ        A⁰ = X
Aˡ = f(Zˡ)                 hidden layers
P  = softmax(Zᴸ)           output layer
L  = -(1/N) Σ Σ Y ⊙ ln P
```

Backpropagation:

```text
δᴸ     = (P − Y) / N
∂L/∂Wˡ = (Aˡ⁻¹)ᵀ · δˡ
∂L/∂bˡ = Σ_rows δˡ
δˡ⁻¹   = (δˡ · (Wˡ)ᵀ) ⊙ f′(Zˡ⁻¹)
```

Update: `W ← W − η · ∂L/∂W`, `b ← b − η · ∂L/∂b`.

## Project layout

```text
src/
├── lib.rs                library root
├── matrix.rs             dense row-major matrix and its operations
├── rng.rs                seeded SplitMix64 PRNG, Box–Muller, Fisher–Yates
├── data.rs               CSV formats, dataset kinds, stratified split, standardization, one-hot
├── activation.rs         sigmoid, tanh, ReLU (+ derivatives), softmax
├── loss.rs               cross-entropy and its gradient
├── network.rs            the MLP: forward, backward, update (+ gradient check test)
├── train.rs              mini-batch training loop, learning curves, early stopping
├── metrics.rs            confusion matrix, accuracy, precision, recall, F1
├── experiment.rs         parallel grid search and aggregation
├── plots.rs              Plotly charts and the HTML dashboard
├── main.rs               CLI: train and evaluate one model
└── bin/
    ├── experiments.rs    the full study
    └── worked_example.rs numeric walkthrough for the report
data/iris.data            UCI Iris dataset
data/wdbc.data            UCI Breast Cancer Wisconsin (Diagnostic), downloaded separately
results/<dataset>/        generated reports and charts
```

## Results at a glance

| | Iris | Breast Cancer |
|---|---|---|
| Best configuration | ReLU, 16 hidden, lr 0.1 | ReLU, 16 hidden, lr 0.1 |
| Test accuracy | 0.952 | 0.976 (0.988 with early stopping) |
| Macro F1 | 0.952 | 0.975 |
| Majority-class baseline | 0.333 | 0.624 |

Full tables: `results/iris/test_report.md` and `results/breast_cancer/test_report.md`.
Open `results/<dataset>/dashboard.html` in a browser for the interactive charts.

## Development

```bash
cargo fmt                                  # format
cargo clippy --all-targets -- -D warnings  # lint
cargo test                                 # test
cargo doc --open                           # browse the API documentation
```

## Dataset citations

- R. A. Fisher, *Iris*, UCI Machine Learning Repository, 1936.
  <https://doi.org/10.24432/C56C76>. CC BY 4.0.
- W. Wolberg, O. Mangasarian, N. Street, W. Street, *Breast Cancer Wisconsin
  (Diagnostic)*, UCI Machine Learning Repository, 1993.
  <https://doi.org/10.24432/C5DW2B>. CC BY 4.0.
