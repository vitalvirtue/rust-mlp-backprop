//! Command-line entry point: trains one MLP configuration on a dataset and
//! reports its performance on the held-out test set.
//!
//! Examples:
//! ```text
//! cargo run --release -- --hidden 16 --activation relu --lr 0.05 --epochs 300
//! cargo run --release -- --dataset breast-cancer --hidden 16 --lr 0.01
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use mlp::activation::Activation;
use mlp::data::{Dataset, DatasetKind};
use mlp::metrics::ConfusionMatrix;
use mlp::rng::Rng;
use mlp::train::{TrainConfig, train};

/// Train a from-scratch MLP on Iris or Breast Cancer Wisconsin.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Dataset: iris or breast-cancer.
    #[arg(long, default_value = "iris")]
    dataset: DatasetKind,

    /// Path to the data file [default: data/iris.data or data/wdbc.data].
    #[arg(long)]
    data: Option<PathBuf>,

    /// Hidden layer sizes, comma separated (e.g. `8` or `16,8`).
    #[arg(long, value_delimiter = ',', default_value = "8")]
    hidden: Vec<usize>,

    /// Hidden activation: sigmoid, tanh or relu.
    #[arg(long, default_value = "tanh")]
    activation: Activation,

    /// Learning rate η.
    #[arg(long, default_value_t = 0.05)]
    lr: f64,

    /// Number of epochs.
    #[arg(long, default_value_t = 300)]
    epochs: usize,

    /// Mini-batch size.
    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    /// Random seed for the data split, initialization and shuffling.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Early stopping: stop after N epochs without a better validation loss
    /// and keep the best weights.
    #[arg(long)]
    patience: Option<usize>,

    /// Print progress every N epochs.
    #[arg(long, default_value_t = 50)]
    log_every: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let path = args
        .data
        .clone()
        .unwrap_or_else(|| PathBuf::from(args.dataset.default_path()));
    let dataset = match args.dataset.load(&path) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!(
                "hint: download the {} data file from UCI and place it at {}",
                args.dataset.display_name(),
                path.display()
            );
            return ExitCode::FAILURE;
        }
    };
    println!("{}", args.dataset.display_name());
    print_dataset_summary(&dataset);

    let split = dataset.stratified_split(0.7, 0.15, &mut Rng::new(args.seed));
    let (split, _scaler) = split.standardized();
    println!(
        "\nSplit: train {} / validation {} / test {}",
        split.train.len(),
        split.validation.len(),
        split.test.len()
    );

    let config = TrainConfig {
        hidden_layers: args.hidden,
        activation: args.activation,
        learning_rate: args.lr,
        epochs: args.epochs,
        batch_size: args.batch_size,
        seed: args.seed,
        patience: args.patience,
    };
    let sizes = config.layer_sizes(dataset.n_features(), dataset.n_classes());
    println!(
        "\nTraining {sizes:?} {} | lr {} | batch {} | {} epochs",
        config.activation, config.learning_rate, config.batch_size, config.epochs
    );

    let (net, history) = train(&config, &split.train, &split.validation);
    println!(
        "{:>6} {:>11} {:>9} {:>11} {:>9}",
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc"
    );
    for m in history
        .epochs
        .iter()
        .filter(|m| m.epoch % args.log_every.max(1) == 0 || m.epoch == history.last().epoch)
    {
        println!(
            "{:>6} {:>11.5} {:>9.4} {:>11.5} {:>9.4}",
            m.epoch, m.train_loss, m.train_accuracy, m.val_loss, m.val_accuracy
        );
    }
    if let Some(epoch) = history.restored_epoch {
        println!(
            "Early stopping: restored the weights from epoch {epoch} (lowest validation loss)"
        );
    }
    if history.diverged {
        eprintln!("warning: training diverged (loss became NaN/inf); try a smaller --lr");
    }

    let (majority, _) = split.train.majority_class();
    let baseline = split
        .test
        .labels()
        .iter()
        .filter(|&&y| y == majority)
        .count() as f64
        / split.test.len() as f64;
    println!(
        "\nBaseline (always predict `{}`): test accuracy {baseline:.4}",
        dataset.class_names()[majority]
    );

    let predicted = net.predict(split.test.features());
    let cm = ConfusionMatrix::new(split.test.labels(), &predicted, dataset.n_classes());
    println!("\nTest set results:\n{}", cm.report(dataset.class_names()));
    println!("\nConfusion matrix (rows = actual, columns = predicted):\n{cm}");
    for (i, name) in dataset.class_names().iter().enumerate() {
        println!("  {i} = {name}");
    }
    ExitCode::SUCCESS
}

fn print_dataset_summary(dataset: &Dataset) {
    println!(
        "Dataset: {} samples, {} features, {} classes",
        dataset.len(),
        dataset.n_features(),
        dataset.n_classes()
    );
    for (name, count) in dataset.class_names().iter().zip(dataset.class_counts()) {
        let share = count as f64 / dataset.len() as f64 * 100.0;
        println!("  {name:<16} {count:>4} ({share:.1}%)");
    }
    println!(
        "\n  {:<24} {:>9} {:>9} {:>9} {:>9}",
        "feature", "min", "max", "mean", "std"
    );
    for s in dataset.feature_stats() {
        println!(
            "  {:<24} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
            s.name, s.min, s.max, s.mean, s.std
        );
    }
}
