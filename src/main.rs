//! Command-line entry point: trains one MLP configuration on Iris and
//! reports its performance on the held-out test set.
//!
//! Example:
//! ```text
//! cargo run --release -- --hidden 16 --activation relu --lr 0.05 --epochs 300
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use mlp::activation::Activation;
use mlp::data::Dataset;
use mlp::metrics::ConfusionMatrix;
use mlp::rng::Rng;
use mlp::train::{TrainConfig, train};

/// Train a from-scratch MLP on the Iris dataset.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Path to the UCI `iris.data` file.
    #[arg(long, default_value = "data/iris.data")]
    data: PathBuf,

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

    /// Print progress every N epochs.
    #[arg(long, default_value_t = 50)]
    log_every: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let dataset = match Dataset::iris(&args.data) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!(
                "hint: place the UCI iris.data file at {}",
                args.data.display()
            );
            return ExitCode::FAILURE;
        }
    };
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
    if history.diverged {
        eprintln!("warning: training diverged (loss became NaN/inf); try a smaller --lr");
    }

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
        println!("  {name:<16} {count}");
    }
    println!(
        "\n  {:<16} {:>6} {:>6} {:>6} {:>6}",
        "feature", "min", "max", "mean", "std"
    );
    for s in dataset.feature_stats() {
        println!(
            "  {:<16} {:>6.2} {:>6.2} {:>6.2} {:>6.2}",
            s.name, s.min, s.max, s.mean, s.std
        );
    }
}
