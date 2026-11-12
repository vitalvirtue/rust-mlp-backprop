//! Command-line entry point for training and evaluating the MLP.

use std::process::ExitCode;

use mlp::activation::Activation;
use mlp::data::{Dataset, Standardizer, one_hot};
use mlp::network::Mlp;
use mlp::rng::Rng;

fn main() -> ExitCode {
    let dataset = match Dataset::iris("data/iris.data") {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("hint: download iris.data from UCI and place it at data/iris.data");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "Loaded {} samples, {} features, {} classes",
        dataset.len(),
        dataset.n_features(),
        dataset.n_classes()
    );
    for (name, count) in dataset.class_names().iter().zip(dataset.class_counts()) {
        println!("  {name:<16} {count}");
    }

    println!("\nFeature statistics:");
    println!(
        "  {:<16} {:>6} {:>6} {:>6} {:>6}",
        "feature", "min", "max", "mean", "std"
    );
    for s in dataset.feature_stats() {
        println!(
            "  {:<16} {:>6.2} {:>6.2} {:>6.2} {:>6.2}",
            s.name, s.min, s.max, s.mean, s.std
        );
    }

    let split = dataset.stratified_split(0.7, 0.15, &mut Rng::new(42));
    println!("\nStratified split (70/15/15):");
    for (name, part) in [
        ("train", &split.train),
        ("validation", &split.validation),
        ("test", &split.test),
    ] {
        println!(
            "  {name:<10} {:>3} samples, per class {:?}",
            part.len(),
            part.class_counts()
        );
    }

    // Quick smoke test: full-batch gradient descent on a 4-8-3 tanh network.
    let scaler = Standardizer::fit(split.train.features());
    let train = split.train.standardized(&scaler);
    let val = split.validation.standardized(&scaler);
    let y_train = one_hot(train.labels(), train.n_classes());

    let mut net = Mlp::new(&[4, 8, 3], Activation::Tanh, &mut Rng::new(0));
    println!(
        "\nTraining 4-8-3 tanh network ({} parameters):",
        net.n_parameters()
    );
    for epoch in 1..=200 {
        let loss = net.train_step(train.features(), &y_train, 0.1);
        if epoch % 50 == 0 {
            let predictions = net.predict(val.features());
            let correct = predictions
                .iter()
                .zip(val.labels())
                .filter(|(p, y)| p == y)
                .count();
            println!(
                "  epoch {epoch:>3}  train loss {loss:.4}  val acc {correct}/{}",
                val.len()
            );
        }
    }
    ExitCode::SUCCESS
}
