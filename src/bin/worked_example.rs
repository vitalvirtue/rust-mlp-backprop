//! Prints one complete training step on a 2-sample mini-batch with every
//! intermediate number: forward pass, loss, backpropagated gradients and the
//! weight update. This is the "numerical example" the report requires.
//!
//! The output is written to `results/<dataset>/worked_example.md` and echoed
//! to stdout.
//!
//! ```text
//! cargo run --release --bin worked_example
//! ```

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use clap::Parser;

use mlp::activation::Activation;
use mlp::data::{DatasetKind, one_hot};
use mlp::loss::cross_entropy;
use mlp::matrix::Matrix;
use mlp::network::Mlp;
use mlp::rng::Rng;

/// Print a fully worked forward/backward/update step.
#[derive(Debug, Parser)]
#[command(about)]
struct Args {
    /// Dataset: iris or breast-cancer.
    #[arg(long, default_value = "iris")]
    dataset: DatasetKind,
    /// Path to the data file [default: data/iris.data or data/wdbc.data].
    #[arg(long)]
    data: Option<PathBuf>,
    /// Output Markdown file [default: results/<dataset>/worked_example.md].
    #[arg(long)]
    out: Option<PathBuf>,
    /// Hidden neurons (kept small so the matrices stay readable).
    #[arg(long, default_value_t = 3)]
    hidden: usize,
    /// Hidden activation.
    #[arg(long, default_value = "sigmoid")]
    activation: Activation,
    /// Learning rate η.
    #[arg(long, default_value_t = 0.1)]
    lr: f64,
    /// Seed for the split and the initial weights.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Collects Markdown output.
struct Doc(String);

impl Doc {
    fn heading(&mut self, text: &str) {
        let _ = writeln!(self.0, "\n## {text}\n");
    }
    fn para(&mut self, text: &str) {
        let _ = writeln!(self.0, "{text}\n");
    }
    fn matrix(&mut self, name: &str, m: &Matrix) {
        let (r, c) = m.shape();
        let _ = writeln!(self.0, "**{name}** ({r}×{c})\n\n```text\n{m:.4}```\n");
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let data_path = args
        .data
        .clone()
        .unwrap_or_else(|| PathBuf::from(args.dataset.default_path()));
    let out = args.out.clone().unwrap_or_else(|| {
        PathBuf::from("results")
            .join(args.dataset.slug())
            .join("worked_example.md")
    });
    let dataset = args.dataset.load(&data_path)?;
    let mut rng = Rng::new(args.seed);
    let (split, scaler) = dataset.stratified_split(0.7, 0.15, &mut rng).standardized();

    // Pick two training samples of different classes so the example is not trivial.
    let first = 0;
    let second = (1..split.train.len())
        .find(|&i| split.train.labels()[i] != split.train.labels()[first])
        .expect("training set has more than one class");
    let batch = split.train.subset(&[first, second]);
    let x = batch.features().clone();
    let y = one_hot(batch.labels(), batch.n_classes());

    let mut net = Mlp::new(
        &[dataset.n_features(), args.hidden, dataset.n_classes()],
        args.activation,
        &mut rng,
    );
    let (w1, b1) = (
        net.layers()[0].weights().clone(),
        net.layers()[0].biases().clone(),
    );
    let (w2, b2) = (
        net.layers()[1].weights().clone(),
        net.layers()[1].biases().clone(),
    );
    let act = args.activation;
    let n = x.rows() as f64;

    let mut doc = Doc(String::new());
    let _ = writeln!(
        doc.0,
        "# Worked example: one training step ({})\n\nNetwork {:?}, hidden activation **{act}**, \
         softmax output, cross-entropy loss, learning rate η = {}, mini-batch of N = 2 samples.",
        args.dataset.display_name(),
        net.layer_sizes(),
        args.lr
    );

    doc.heading("0. Input mini-batch");
    let names: Vec<&str> = batch
        .labels()
        .iter()
        .map(|&c| dataset.class_names()[c].as_str())
        .collect();
    doc.para(&format!(
        "Samples are taken from the training set and z-score standardized with the training \
         mean μ = {:.4?} and standard deviation σ = {:.4?}. Classes: {names:?}.",
        scaler.mean(),
        scaler.std()
    ));
    doc.matrix("X", &x);
    doc.matrix("Y (one-hot)", &y);

    doc.heading("1. Initial parameters");
    doc.para(&format!(
        "Weights ~ N(0, σ²) with {} initialization; biases are zero.",
        if act == Activation::Relu {
            "He, σ = √(2/fan_in)"
        } else {
            "Xavier, σ = √(2/(fan_in+fan_out))"
        }
    ));
    doc.matrix("W¹", &w1);
    doc.matrix("b¹", &b1);
    doc.matrix("W²", &w2);
    doc.matrix("b²", &b2);

    // ---- Forward pass ----
    let pass = net.forward(&x);
    let (z1, a1, z2, p) = (
        &pass.pre_activations[0],
        &pass.inputs[1],
        &pass.pre_activations[1],
        &pass.output,
    );
    doc.heading("2. Forward propagation");
    doc.para("Z¹ = X·W¹ + b¹");
    let terms: Vec<String> = (0..x.cols())
        .map(|k| format!("({:.4})·({:.4})", x[(0, k)], w1[(k, 0)]))
        .collect();
    doc.para(&format!(
        "For example, the first hidden neuron for sample 1: z¹₁₁ = {} + {:.4} = **{:.4}**",
        terms.join(" + "),
        b1[(0, 0)],
        z1[(0, 0)]
    ));
    doc.matrix("Z¹", z1);
    doc.para(&format!(
        "A¹ = {act}(Z¹), e.g. {act}({:.4}) = **{:.4}**",
        z1[(0, 0)],
        a1[(0, 0)]
    ));
    doc.matrix("A¹", a1);
    doc.para("Z² = A¹·W² + b²");
    doc.matrix("Z²", z2);
    let row: Vec<f64> = z2.row(0).to_vec();
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<String> = row
        .iter()
        .map(|z| format!("{:.4}", (z - max).exp()))
        .collect();
    let probs: Vec<String> = p.row(0).iter().map(|v| format!("{v:.4}")).collect();
    doc.para(&format!(
        "P = softmax(Z²), row-wise: pₖ = e^(zₖ − max z) / Σⱼ e^(zⱼ − max z). For sample 1 the \
         shifted exponentials are [{}], which normalize to [{}].",
        exps.join(", "),
        probs.join(", ")
    ));
    doc.matrix("P", p);

    // ---- Loss ----
    let loss = cross_entropy(p, &y);
    let picked: Vec<String> = (0..x.rows())
        .map(|i| {
            let c = batch.labels()[i];
            format!("ln({:.4})", p[(i, c)])
        })
        .collect();
    doc.heading("3. Loss");
    doc.para(&format!(
        "L = −(1/N) Σₙ Σₖ yₙₖ ln pₙₖ = −(1/2)·[{}] = **{loss:.4}**",
        picked.join(" + ")
    ));

    // ---- Backward pass ----
    let grads = net.backward(&pass, &y);
    let delta2 = p.sub(&y).scale(1.0 / n);
    let delta1 = delta2
        .matmul(&w2.transpose())
        .hadamard(&act.derivative_matrix(z1));
    // The manual δ¹ above must reproduce the library's gradient exactly.
    let manual_dw1 = x.transpose().matmul(&delta1);
    assert!(
        manual_dw1
            .as_slice()
            .iter()
            .zip(grads.weights[0].as_slice())
            .all(|(a, b)| (a - b).abs() < 1e-12),
        "manual backprop disagrees with Mlp::backward"
    );
    doc.heading("4. Backpropagation");
    doc.para("Output layer: δ² = ∂L/∂Z² = (P − Y) / N");
    doc.matrix("δ²", &delta2);
    doc.para("∂L/∂W² = (A¹)ᵀ · δ²");
    doc.para(&format!(
        "For example, ∂L/∂w²₁₁ = a¹₁₁·δ²₁₁ + a¹₂₁·δ²₂₁ = ({:.4})·({:.4}) + ({:.4})·({:.4}) = **{:.4}**",
        a1[(0, 0)],
        delta2[(0, 0)],
        a1[(1, 0)],
        delta2[(1, 0)],
        grads.weights[1][(0, 0)]
    ));
    doc.matrix("∂L/∂W²", &grads.weights[1]);
    doc.para("∂L/∂b² = Σ over rows of δ²");
    doc.matrix("∂L/∂b²", &grads.biases[1]);
    doc.para(&format!(
        "Hidden layer: δ¹ = (δ² · (W²)ᵀ) ⊙ {act}′(Z¹){}",
        match act {
            Activation::Sigmoid => ", where σ′(z) = σ(z)(1 − σ(z))",
            Activation::Tanh => ", where tanh′(z) = 1 − tanh²(z)",
            Activation::Relu => ", where ReLU′(z) = 1 if z > 0 else 0",
        }
    ));
    doc.matrix(&format!("{act}′(Z¹)"), &act.derivative_matrix(z1));
    doc.matrix("δ¹", &delta1);
    doc.para("∂L/∂W¹ = Xᵀ · δ¹ and ∂L/∂b¹ = Σ over rows of δ¹");
    doc.matrix("∂L/∂W¹", &grads.weights[0]);
    doc.matrix("∂L/∂b¹", &grads.biases[0]);

    // ---- Update ----
    net.apply_gradients(&grads, args.lr);
    doc.heading("5. Gradient descent update");
    doc.para(&format!(
        "θ ← θ − η·∂L/∂θ with η = {}. For example, w²₁₁ ← {:.4} − {}·({:.4}) = **{:.4}**",
        args.lr,
        w2[(0, 0)],
        args.lr,
        grads.weights[1][(0, 0)],
        net.layers()[1].weights()[(0, 0)]
    ));
    doc.matrix("W¹ (updated)", net.layers()[0].weights());
    doc.matrix("b¹ (updated)", net.layers()[0].biases());
    doc.matrix("W² (updated)", net.layers()[1].weights());
    doc.matrix("b² (updated)", net.layers()[1].biases());

    let new_loss = net.loss(&x, &y);
    doc.heading("6. Check");
    doc.para(&format!(
        "Loss on the same mini-batch after the update: {loss:.4} → **{new_loss:.4}** \
         ({:+.4}). One gradient step moved the parameters downhill.",
        new_loss - loss
    ));

    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(&out, &doc.0)?;
    println!("{}", doc.0);
    eprintln!("Wrote {}", out.display());
    Ok(())
}
