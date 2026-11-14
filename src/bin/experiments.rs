//! Runs the full hyper-parameter study required by the assignment and writes
//! every artefact the report needs into `results/<dataset>/`:
//!
//! - `runs.csv`        one row per training run (configuration × seed)
//! - `summary.csv`     one row per configuration, averaged over seeds
//! - `test_report.md`  Markdown tables ready to paste into the report
//! - `dashboard.html`  all charts and tables on one interactive page
//! - `plots/*.html`    each chart on its own page (use the camera icon to save a PNG)
//!
//! ```text
//! cargo run --release --bin experiments
//! cargo run --release --bin experiments -- --dataset breast-cancer
//! ```

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use mlp::activation::Activation;
use mlp::data::{Dataset, DatasetKind, Split};
use mlp::experiment::{CellSummary, GridSpec, aggregate, best_cell, run_grid};
use mlp::metrics::{ClassificationReport, ConfusionMatrix};
use mlp::plots::{
    self, CurvePair, Dashboard, ErrorBarSeries, PALETTE, activation_color, html_table,
    learning_curves, metric_with_error_bars,
};
use mlp::rng::Rng;
use mlp::train::{History, TrainConfig, train};

/// Seed for the train/validation/test split. Fixed for every run so that all
/// configurations are compared on exactly the same data.
const SPLIT_SEED: u64 = 42;

/// Run the learning-rate × hidden-size × activation study.
#[derive(Debug, Parser)]
#[command(about)]
struct Args {
    /// Dataset: iris or breast-cancer.
    #[arg(long, default_value = "iris")]
    dataset: DatasetKind,
    /// Path to the data file [default: data/iris.data or data/wdbc.data].
    #[arg(long)]
    data: Option<PathBuf>,
    /// Output directory [default: results/<dataset>].
    #[arg(long)]
    out: Option<PathBuf>,
    /// Epochs per run.
    #[arg(long, default_value_t = 300)]
    epochs: usize,
    /// Seeds per configuration.
    #[arg(long, default_value_t = 5)]
    seeds: u64,
    /// Mini-batch size.
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let kind = args.dataset;
    let data_path = args
        .data
        .unwrap_or_else(|| PathBuf::from(kind.default_path()));
    let out = args
        .out
        .unwrap_or_else(|| Path::new("results").join(kind.slug()));
    let dataset = kind
        .load(&data_path)
        .map_err(|e| format!("{}: {e}", data_path.display()))?;
    println!("Dataset: {}", kind.display_name());
    let (split, _) = dataset
        .stratified_split(0.7, 0.15, &mut Rng::new(SPLIT_SEED))
        .standardized();

    let spec = GridSpec {
        seeds: (1..=args.seeds).collect(),
        epochs: args.epochs,
        batch_size: args.batch_size,
        ..GridSpec::default()
    };
    let n_runs = spec.configs().len();
    println!(
        "Running {n_runs} training runs ({} epochs each)...",
        spec.epochs
    );
    let started = Instant::now();
    let runs = run_grid(&spec, &split);
    println!("Done in {:.2?}", started.elapsed());

    let cells = aggregate(&runs);
    fs::create_dir_all(out.join("plots"))?;
    write_runs_csv(&out.join("runs.csv"), &runs)?;
    write_summary_csv(&out.join("summary.csv"), &cells)?;

    let best = best_cell(&cells, |_| true).ok_or("every configuration diverged")?;
    println!(
        "Best configuration: {} | hidden {} | lr {} (val loss {:.4} ± {:.4}, val acc {:.4})",
        best.activation,
        best.hidden,
        best.learning_rate,
        best.val_loss.0,
        best.val_loss.1,
        best.val_accuracy.0
    );

    // Final models: the overall best and the best per activation, each
    // retrained once and evaluated on the untouched test set.
    let mut finals = Vec::new();
    for activation in &spec.activations {
        let cell = best_cell(&cells, |c| c.activation == *activation)
            .ok_or("an activation diverged for every setting")?;
        finals.push(FinalModel::fit(cell, &spec, &split, &dataset));
    }
    let overall = finals
        .iter()
        .find(|f| f.cell == *best)
        .expect("best cell is the best of its own activation");

    println!("\nTest set, best overall model:\n{}", overall.report);
    println!("\n{}", overall.confusion);

    // Majority-class baseline: the accuracy of a "model" that always predicts
    // the most common training class. With imbalanced data this is far above
    // chance, so a good accuracy must be judged against it.
    let (majority, _) = split.train.majority_class();
    let baseline_accuracy = split
        .test
        .labels()
        .iter()
        .filter(|&&y| y == majority)
        .count() as f64
        / split.test.len() as f64;
    let study = Study {
        kind,
        spec: &spec,
        cells: &cells,
        best,
        finals: &finals,
        dataset: &dataset,
        split: &split,
        n_runs,
        baseline_class: &dataset.class_names()[majority],
        baseline_accuracy,
    };
    println!(
        "Baseline (always `{}`): test accuracy {baseline_accuracy:.4}",
        study.baseline_class
    );
    write_markdown(&out.join("test_report.md"), &study)?;
    build_dashboard(&out, &study)?;
    println!("\nWrote results to {}", out.display());
    Ok(())
}

/// Everything the report writers need, bundled to keep signatures short.
struct Study<'a> {
    kind: DatasetKind,
    spec: &'a GridSpec,
    cells: &'a [CellSummary],
    best: &'a CellSummary,
    finals: &'a [FinalModel],
    dataset: &'a Dataset,
    split: &'a Split,
    n_runs: usize,
    baseline_class: &'a str,
    baseline_accuracy: f64,
}

/// A configuration retrained once and evaluated on the test set.
struct FinalModel {
    cell: CellSummary,
    history: History,
    confusion: ConfusionMatrix,
    report: ClassificationReport,
}

impl FinalModel {
    fn fit(cell: &CellSummary, spec: &GridSpec, split: &Split, dataset: &Dataset) -> Self {
        let config = TrainConfig {
            hidden_layers: vec![cell.hidden],
            activation: cell.activation,
            learning_rate: cell.learning_rate,
            epochs: spec.epochs,
            batch_size: spec.batch_size,
            seed: spec.seeds[0],
            patience: None,
        };
        let (net, history) = train(&config, &split.train, &split.validation);
        let predicted = net.predict(split.test.features());
        let confusion = ConfusionMatrix::new(split.test.labels(), &predicted, dataset.n_classes());
        let report = confusion.report(dataset.class_names());
        Self {
            cell: cell.clone(),
            history,
            confusion,
            report,
        }
    }
}

fn write_runs_csv(path: &Path, runs: &[mlp::experiment::RunResult]) -> std::io::Result<()> {
    let mut csv = String::from(
        "activation,learning_rate,hidden,seed,epochs_run,diverged,train_loss,val_loss,train_accuracy,val_accuracy,best_val_loss,best_val_epoch\n",
    );
    for r in runs {
        let last = r.history.last();
        let best = r.history.best_by_val_loss().copied().unwrap_or(*last);
        let _ = writeln!(
            csv,
            "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            r.config.activation,
            r.config.learning_rate,
            r.config.hidden_layers[0],
            r.config.seed,
            last.epoch,
            r.history.diverged,
            last.train_loss,
            last.val_loss,
            last.train_accuracy,
            last.val_accuracy,
            best.val_loss,
            best.epoch
        );
    }
    fs::write(path, csv)
}

fn write_summary_csv(path: &Path, cells: &[CellSummary]) -> std::io::Result<()> {
    let mut csv = String::from(
        "activation,learning_rate,hidden,runs,diverged,val_loss_mean,val_loss_std,val_acc_mean,val_acc_std,train_loss_mean,train_acc_mean\n",
    );
    for c in cells {
        let _ = writeln!(
            csv,
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            c.activation,
            c.learning_rate,
            c.hidden,
            c.n_runs,
            c.n_diverged,
            c.val_loss.0,
            c.val_loss.1,
            c.val_accuracy.0,
            c.val_accuracy.1,
            c.train_loss.0,
            c.train_accuracy.0
        );
    }
    fs::write(path, csv)
}

fn find(cells: &[CellSummary], a: Activation, lr: f64, hidden: usize) -> &CellSummary {
    cells
        .iter()
        .find(|c| c.activation == a && c.learning_rate == lr && c.hidden == hidden)
        .expect("cell exists in the grid")
}

fn cell_label(c: &CellSummary) -> String {
    format!("{} · h={} · lr={}", c.activation, c.hidden, c.learning_rate)
}

fn write_markdown(path: &Path, study: &Study<'_>) -> std::io::Result<()> {
    let Study {
        spec,
        cells,
        best,
        finals,
        dataset,
        split,
        ..
    } = *study;
    let mut md = format!("# Experiment results: {}\n\n", study.kind.display_name());
    let counts = |d: &Dataset| -> String {
        d.class_names()
            .iter()
            .zip(d.class_counts())
            .map(|(n, c)| format!("{n} {c}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(
        md,
        "{} samples, {} features, {} classes ({}). Stratified split: train {} ({}), validation {} ({}), test {} ({}).\n",
        dataset.len(),
        dataset.n_features(),
        dataset.n_classes(),
        counts(dataset),
        split.train.len(),
        counts(&split.train),
        split.validation.len(),
        counts(&split.validation),
        split.test.len(),
        counts(&split.test)
    );
    let _ = writeln!(
        md,
        "Majority-class baseline (always predict `{}`): test accuracy **{:.4}**.\n",
        study.baseline_class, study.baseline_accuracy
    );
    let _ = writeln!(
        md,
        "Grid: activations {:?}, learning rates {:?}, hidden sizes {:?}; {} seeds × {} epochs, batch size {}.\n",
        spec.activations
            .iter()
            .map(|a| a.name())
            .collect::<Vec<_>>(),
        spec.learning_rates,
        spec.hidden_sizes,
        spec.seeds.len(),
        spec.epochs,
        spec.batch_size
    );

    md.push_str("## Final validation loss (mean ± std over seeds)\n\n");
    for &a in &spec.activations {
        let _ = writeln!(md, "### {a}\n");
        md.push_str("| lr \\ hidden |");
        for h in &spec.hidden_sizes {
            let _ = write!(md, " {h} |");
        }
        md.push_str("\n|---|");
        md.push_str(&"---|".repeat(spec.hidden_sizes.len()));
        md.push('\n');
        for &lr in &spec.learning_rates {
            let _ = write!(md, "| {lr} |");
            for &h in &spec.hidden_sizes {
                let c = find(cells, a, lr, h);
                let mark = if c == best { "**" } else { "" };
                let _ = write!(
                    md,
                    " {mark}{:.4} ± {:.4}{mark} |",
                    c.val_loss.0, c.val_loss.1
                );
            }
            md.push('\n');
        }
        md.push('\n');
    }

    md.push_str("## Best configuration per activation, evaluated on the test set\n\n");
    md.push_str("| activation | hidden | lr | val loss | val acc | test acc | macro precision | macro recall | macro F1 |\n");
    md.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for f in finals {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            f.cell.activation,
            f.cell.hidden,
            f.cell.learning_rate,
            f.cell.val_loss.0,
            f.cell.val_accuracy.0,
            f.report.accuracy,
            f.report.macro_precision,
            f.report.macro_recall,
            f.report.macro_f1
        );
    }

    let overall = finals
        .iter()
        .find(|f| f.cell == *best)
        .expect("best present");
    let _ = writeln!(md, "\n## Best overall model: {}\n", cell_label(best));
    md.push_str("| class | precision | recall | F1 | support |\n|---|---|---|---|---|\n");
    for m in &overall.report.per_class {
        let _ = writeln!(
            md,
            "| {} | {:.4} | {:.4} | {:.4} | {} |",
            m.name, m.precision, m.recall, m.f1, m.support
        );
    }
    let _ = writeln!(
        md,
        "| **macro avg** | {:.4} | {:.4} | {:.4} | {} |\n| **weighted avg** | {:.4} | {:.4} | {:.4} | {} |\n\nAccuracy: **{:.4}** (majority-class baseline {:.4})\n",
        overall.report.macro_precision,
        overall.report.macro_recall,
        overall.report.macro_f1,
        overall.confusion.total(),
        overall.report.weighted_precision,
        overall.report.weighted_recall,
        overall.report.weighted_f1,
        overall.confusion.total(),
        overall.report.accuracy,
        study.baseline_accuracy
    );
    md.push_str("### Confusion matrix (rows = actual, columns = predicted)\n\n| |");
    for name in dataset.class_names() {
        let _ = write!(md, " {name} |");
    }
    md.push_str("\n|---|");
    md.push_str(&"---|".repeat(dataset.n_classes()));
    md.push('\n');
    for (a, name) in dataset.class_names().iter().enumerate() {
        let _ = write!(md, "| **{name}** |");
        for p in 0..dataset.n_classes() {
            let _ = write!(md, " {} |", overall.confusion.count(a, p));
        }
        md.push('\n');
    }
    fs::write(path, md)
}

fn build_dashboard(out: &Path, study: &Study<'_>) -> std::io::Result<()> {
    let Study {
        spec,
        cells,
        best,
        finals,
        dataset,
        split,
        n_runs,
        ..
    } = *study;
    let mut dash = Dashboard::new(
        format!("MLP from scratch: {}", study.kind.display_name()),
        format!(
            "{} samples · {} features · {} classes · split {}/{}/{}. \
             {n_runs} training runs · {} activations × {} learning rates × {} hidden sizes × {} seeds · {} epochs · batch {}. \
             Curves are averaged over seeds; dashed = train, solid = validation.",
            dataset.len(),
            dataset.n_features(),
            dataset.n_classes(),
            split.train.len(),
            split.validation.len(),
            split.test.len(),
            spec.activations.len(),
            spec.learning_rates.len(),
            spec.hidden_sizes.len(),
            spec.seeds.len(),
            spec.epochs,
            spec.batch_size
        ),
    );
    let save =
        |dash: &mut Dashboard, file: &str, heading: &str, caption: &str, plot: plotly::Plot| {
            plot.write_html(out.join("plots").join(format!("{file}.html")));
            dash.add_plot(heading, caption, &plot);
        };

    // 1. Learning rate comparison at the best activation and hidden size.
    let lr_cells: Vec<&CellSummary> = spec
        .learning_rates
        .iter()
        .map(|&lr| find(cells, best.activation, lr, best.hidden))
        .collect();
    let lr_labels: Vec<String> = lr_cells
        .iter()
        .map(|c| format!("lr = {}", c.learning_rate))
        .collect();
    let pairs: Vec<CurvePair<'_>> = lr_cells
        .iter()
        .zip(&lr_labels)
        .enumerate()
        .map(|(i, (c, label))| CurvePair {
            label: label.clone(),
            color: PALETTE[i % PALETTE.len()],
            train: &c.curves.train_loss,
            validation: &c.curves.val_loss,
        })
        .collect();
    save(
        &mut dash,
        "lr_comparison",
        "Effect of the learning rate",
        &format!(
            "Cross-entropy loss per epoch for {} with {} hidden neurons. Small rates converge slowly; larger rates converge faster.",
            best.activation, best.hidden
        ),
        learning_curves(
            "Loss vs. epoch for different learning rates",
            "Cross-entropy loss",
            &pairs,
        ),
    );

    // 2. Activation comparison at the best learning rate and hidden size.
    let act_cells: Vec<&CellSummary> = spec
        .activations
        .iter()
        .map(|&a| find(cells, a, best.learning_rate, best.hidden))
        .collect();
    let pairs: Vec<CurvePair<'_>> = act_cells
        .iter()
        .map(|c| CurvePair {
            label: c.activation.to_string(),
            color: activation_color(c.activation),
            train: &c.curves.train_loss,
            validation: &c.curves.val_loss,
        })
        .collect();
    save(
        &mut dash,
        "activation_comparison",
        "Effect of the activation function",
        &format!(
            "Loss per epoch for each activation at lr = {} and {} hidden neurons.",
            best.learning_rate, best.hidden
        ),
        learning_curves(
            "Loss vs. epoch for different activations",
            "Cross-entropy loss",
            &pairs,
        ),
    );

    // 3. Hidden size comparison: each activation at its own best learning rate.
    let hidden_labels: Vec<String> = spec.hidden_sizes.iter().map(ToString::to_string).collect();
    let mut acc_series = Vec::new();
    let mut loss_series = Vec::new();
    for &a in &spec.activations {
        let lr =
            best_cell(cells, |c| c.activation == a).map_or(best.learning_rate, |c| c.learning_rate);
        let row: Vec<&CellSummary> = spec
            .hidden_sizes
            .iter()
            .map(|&h| find(cells, a, lr, h))
            .collect();
        let name = format!("{a} (lr = {lr})");
        acc_series.push(ErrorBarSeries {
            name: name.clone(),
            color: activation_color(a),
            values: row.iter().map(|c| c.val_accuracy).collect(),
        });
        loss_series.push(ErrorBarSeries {
            name,
            color: activation_color(a),
            values: row.iter().map(|c| c.val_loss).collect(),
        });
    }
    save(
        &mut dash,
        "hidden_size_val_loss",
        "Effect of the number of hidden neurons: validation loss",
        "Final validation loss (mean ± 1 std over seeds) for each activation at its best learning rate.",
        metric_with_error_bars(
            "Validation loss vs. hidden neurons",
            "Hidden neurons",
            "Validation loss",
            &hidden_labels,
            &loss_series,
        ),
    );
    save(
        &mut dash,
        "hidden_size_val_accuracy",
        "Effect of the number of hidden neurons: validation accuracy",
        "Final validation accuracy (mean ± 1 std over seeds). The validation set has 24 samples, so accuracy moves in steps of 1/24 ≈ 0.042 and lines that overlap have identical values; the loss chart above separates them better.",
        metric_with_error_bars(
            "Validation accuracy vs. hidden neurons",
            "Hidden neurons",
            "Validation accuracy",
            &hidden_labels,
            &acc_series,
        ),
    );

    // 4. Full grid as heatmaps, one per activation.
    let lr_axis: Vec<String> = spec
        .learning_rates
        .iter()
        .map(ToString::to_string)
        .collect();
    for &a in &spec.activations {
        let z: Vec<Vec<f64>> = spec
            .learning_rates
            .iter()
            .map(|&lr| {
                spec.hidden_sizes
                    .iter()
                    .map(|&h| find(cells, a, lr, h).val_loss.0)
                    .collect()
            })
            .collect();
        save(
            &mut dash,
            &format!("grid_{a}"),
            &format!("Grid search: {a}"),
            "Mean final validation loss for every learning rate × hidden size. Darker is better.",
            plots::heatmap(
                &plots::HeatmapSpec {
                    title: &format!("Validation loss, {a}"),
                    x_title: "Hidden neurons",
                    y_title: "Learning rate",
                    x_labels: &hidden_labels,
                    y_labels: &lr_axis,
                    z: &z,
                    reverse: true,
                },
                |v| format!("{v:.3}"),
            ),
        );
    }

    // 5. Best model: learning curves and confusion matrix.
    let overall = finals
        .iter()
        .find(|f| f.cell == *best)
        .expect("best present");
    let h = &overall.history;
    let col =
        |f: fn(&mlp::train::EpochMetrics) -> f64| -> Vec<f64> { h.epochs.iter().map(f).collect() };
    let (tl, vl, ta, va) = (
        col(|m| m.train_loss),
        col(|m| m.val_loss),
        col(|m| m.train_accuracy),
        col(|m| m.val_accuracy),
    );
    let color = activation_color(best.activation);
    save(
        &mut dash,
        "best_model_loss",
        "Best model: training and validation loss",
        &format!("{} (seed {}).", cell_label(best), spec.seeds[0]),
        learning_curves(
            "Best model: loss",
            "Cross-entropy loss",
            &[CurvePair {
                label: "loss".into(),
                color,
                train: &tl,
                validation: &vl,
            }],
        ),
    );
    save(
        &mut dash,
        "best_model_accuracy",
        "Best model: training and validation accuracy",
        "Accuracy per epoch. A large gap between the two would signal over-fitting.",
        learning_curves(
            "Best model: accuracy",
            "Accuracy",
            &[CurvePair {
                label: "accuracy".into(),
                color,
                train: &ta,
                validation: &va,
            }],
        ),
    );
    save(
        &mut dash,
        "confusion_matrix",
        "Best model: confusion matrix on the test set",
        &format!(
            "{} test samples. Rows are true classes, columns are predictions.",
            overall.confusion.total()
        ),
        plots::confusion_matrix(
            "Confusion matrix (test set)",
            &overall.confusion,
            dataset.class_names(),
        ),
    );

    // 6. Tables.
    let rows: Vec<Vec<String>> = finals
        .iter()
        .map(|f| {
            vec![
                f.cell.activation.to_string(),
                f.cell.hidden.to_string(),
                f.cell.learning_rate.to_string(),
                format!("{:.4}", f.cell.val_loss.0),
                format!("{:.4}", f.report.accuracy),
                format!("{:.4}", f.report.macro_precision),
                format!("{:.4}", f.report.macro_recall),
                format!("{:.4}", f.report.macro_f1),
            ]
        })
        .collect();
    let best_row: Vec<usize> = finals
        .iter()
        .position(|f| f.cell == *best)
        .into_iter()
        .collect();
    dash.add_html(
        "Best configuration per activation (test set)",
        "Each activation's best (lr, hidden) by validation loss, retrained and evaluated once on the test set. Bold = best overall.",
        html_table(
            &["activation", "hidden", "lr", "val loss", "test acc", "precision", "recall", "F1"],
            &rows,
            &best_row,
        ),
    );
    let rows: Vec<Vec<String>> = overall
        .report
        .per_class
        .iter()
        .map(|m| {
            vec![
                m.name.clone(),
                format!("{:.4}", m.precision),
                format!("{:.4}", m.recall),
                format!("{:.4}", m.f1),
                m.support.to_string(),
            ]
        })
        .chain([
            vec![
                "macro avg".into(),
                format!("{:.4}", overall.report.macro_precision),
                format!("{:.4}", overall.report.macro_recall),
                format!("{:.4}", overall.report.macro_f1),
                overall.confusion.total().to_string(),
            ],
            vec![
                "weighted avg".into(),
                format!("{:.4}", overall.report.weighted_precision),
                format!("{:.4}", overall.report.weighted_recall),
                format!("{:.4}", overall.report.weighted_f1),
                overall.confusion.total().to_string(),
            ],
        ])
        .collect();
    dash.add_html(
        "Best model: classification report (test set)",
        &format!(
            "Accuracy {:.4}, against a majority-class baseline of {:.4} (always predicting `{}`). \
             Macro averages weight every class equally; weighted averages weight them by support.",
            overall.report.accuracy, study.baseline_accuracy, study.baseline_class
        ),
        html_table(
            &["class", "precision", "recall", "F1", "support"],
            &rows,
            &[],
        ),
    );

    dash.write(out.join("dashboard.html"))
}
