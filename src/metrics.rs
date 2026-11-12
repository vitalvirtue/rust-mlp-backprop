//! Classification metrics: confusion matrix, accuracy, precision, recall, F1.

use std::fmt;

/// Fraction of predictions that equal the true label.
///
/// # Panics
///
/// Panics if the slices have different lengths or are empty.
#[must_use]
pub fn accuracy(actual: &[usize], predicted: &[usize]) -> f64 {
    assert_eq!(actual.len(), predicted.len(), "length mismatch");
    assert!(!actual.is_empty(), "accuracy of an empty set is undefined");
    let correct = actual.iter().zip(predicted).filter(|(a, p)| a == p).count();
    correct as f64 / actual.len() as f64
}

/// `counts[actual][predicted]`: rows are true classes, columns are predictions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfusionMatrix {
    n_classes: usize,
    counts: Vec<usize>,
}

impl ConfusionMatrix {
    /// Tallies predictions against true labels.
    ///
    /// # Panics
    ///
    /// Panics if lengths differ or a label is `>= n_classes`.
    #[must_use]
    pub fn new(actual: &[usize], predicted: &[usize], n_classes: usize) -> Self {
        assert_eq!(actual.len(), predicted.len(), "length mismatch");
        let mut counts = vec![0; n_classes * n_classes];
        for (&a, &p) in actual.iter().zip(predicted) {
            assert!(a < n_classes && p < n_classes, "label out of range");
            counts[a * n_classes + p] += 1;
        }
        Self { n_classes, counts }
    }

    /// Number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Number of samples of class `actual` predicted as `predicted`.
    #[must_use]
    pub fn count(&self, actual: usize, predicted: usize) -> usize {
        self.counts[actual * self.n_classes + predicted]
    }

    /// Total number of samples.
    #[must_use]
    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }

    /// True positives for class `c`: the diagonal entry.
    #[must_use]
    pub fn true_positives(&self, c: usize) -> usize {
        self.count(c, c)
    }

    /// Samples whose true class is `c` (row sum).
    #[must_use]
    pub fn support(&self, c: usize) -> usize {
        (0..self.n_classes).map(|p| self.count(c, p)).sum()
    }

    /// Samples predicted as `c` (column sum).
    #[must_use]
    pub fn predicted_count(&self, c: usize) -> usize {
        (0..self.n_classes).map(|a| self.count(a, c)).sum()
    }

    /// Overall accuracy: trace / total.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        let correct: usize = (0..self.n_classes).map(|c| self.true_positives(c)).sum();
        safe_div(correct, self.total())
    }

    /// Precision for class `c`: TP / (TP + FP). Of everything predicted as `c`,
    /// how much really was `c`? Defined as 0 when nothing was predicted as `c`.
    #[must_use]
    pub fn precision(&self, c: usize) -> f64 {
        safe_div(self.true_positives(c), self.predicted_count(c))
    }

    /// Recall for class `c`: TP / (TP + FN). Of all real `c` samples, how many
    /// did we find? Defined as 0 when the class has no samples.
    #[must_use]
    pub fn recall(&self, c: usize) -> f64 {
        safe_div(self.true_positives(c), self.support(c))
    }

    /// F1 for class `c`: harmonic mean of precision and recall.
    #[must_use]
    pub fn f1(&self, c: usize) -> f64 {
        let (p, r) = (self.precision(c), self.recall(c));
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Builds a full per-class and macro-averaged report.
    ///
    /// # Panics
    ///
    /// Panics if `class_names.len() != self.n_classes()`.
    #[must_use]
    pub fn report(&self, class_names: &[String]) -> ClassificationReport {
        assert_eq!(class_names.len(), self.n_classes, "class name count");
        let per_class: Vec<ClassMetrics> = (0..self.n_classes)
            .map(|c| ClassMetrics {
                name: class_names[c].clone(),
                precision: self.precision(c),
                recall: self.recall(c),
                f1: self.f1(c),
                support: self.support(c),
            })
            .collect();
        let k = self.n_classes as f64;
        ClassificationReport {
            accuracy: self.accuracy(),
            macro_precision: per_class.iter().map(|m| m.precision).sum::<f64>() / k,
            macro_recall: per_class.iter().map(|m| m.recall).sum::<f64>() / k,
            macro_f1: per_class.iter().map(|m| m.f1).sum::<f64>() / k,
            per_class,
        }
    }

    /// Row-major counts, for plotting as a heatmap.
    #[must_use]
    pub fn as_rows(&self) -> Vec<Vec<usize>> {
        self.counts
            .chunks_exact(self.n_classes.max(1))
            .map(<[usize]>::to_vec)
            .collect()
    }
}

fn safe_div(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// Metrics for a single class.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassMetrics {
    /// Class name.
    pub name: String,
    /// TP / (TP + FP).
    pub precision: f64,
    /// TP / (TP + FN).
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
    /// Number of true samples of this class.
    pub support: usize,
}

/// Per-class metrics plus macro averages (unweighted mean over classes).
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationReport {
    /// One entry per class.
    pub per_class: Vec<ClassMetrics>,
    /// Overall accuracy.
    pub accuracy: f64,
    /// Mean precision over classes.
    pub macro_precision: f64,
    /// Mean recall over classes.
    pub macro_recall: f64,
    /// Mean F1 over classes.
    pub macro_f1: f64,
}

impl fmt::Display for ClassificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<18} {:>9} {:>9} {:>9} {:>8}",
            "class", "precision", "recall", "f1", "support"
        )?;
        for m in &self.per_class {
            writeln!(
                f,
                "{:<18} {:>9.4} {:>9.4} {:>9.4} {:>8}",
                m.name, m.precision, m.recall, m.f1, m.support
            )?;
        }
        let total: usize = self.per_class.iter().map(|m| m.support).sum();
        writeln!(
            f,
            "{:<18} {:>9.4} {:>9.4} {:>9.4} {:>8}",
            "macro avg", self.macro_precision, self.macro_recall, self.macro_f1, total
        )?;
        write!(f, "{:<18} {:>29.4} {:>8}", "accuracy", self.accuracy, total)
    }
}

impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:>12}", "actual\\pred")?;
        for p in 0..self.n_classes {
            write!(f, "{p:>6}")?;
        }
        for a in 0..self.n_classes {
            write!(f, "\n{a:>12}")?;
            for p in 0..self.n_classes {
                write!(f, "{:>6}", self.count(a, p))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // actual:    0 0 0 1 1 2
    // predicted: 0 0 1 1 2 2
    fn example() -> ConfusionMatrix {
        ConfusionMatrix::new(&[0, 0, 0, 1, 1, 2], &[0, 0, 1, 1, 2, 2], 3)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn counts_land_in_the_right_cells() {
        let cm = example();
        assert_eq!(
            cm.as_rows(),
            vec![vec![2, 1, 0], vec![0, 1, 1], vec![0, 0, 1]]
        );
        assert_eq!(cm.total(), 6);
    }

    #[test]
    fn precision_recall_f1_by_hand() {
        let cm = example();
        // Class 1: TP=1, predicted as 1 = 2, actual 1 = 2.
        assert!(close(cm.precision(1), 0.5));
        assert!(close(cm.recall(1), 0.5));
        assert!(close(cm.f1(1), 0.5));
        // Class 0: TP=2, predicted 2, actual 3.
        assert!(close(cm.precision(0), 1.0));
        assert!(close(cm.recall(0), 2.0 / 3.0));
        assert!(close(cm.f1(0), 0.8));
    }

    #[test]
    fn accuracy_agrees_with_free_function() {
        let cm = example();
        assert!(close(cm.accuracy(), 4.0 / 6.0));
        assert!(close(
            accuracy(&[0, 0, 0, 1, 1, 2], &[0, 0, 1, 1, 2, 2]),
            4.0 / 6.0
        ));
    }

    #[test]
    fn class_never_predicted_has_zero_precision_not_nan() {
        let cm = ConfusionMatrix::new(&[0, 1], &[0, 0], 2);
        assert_eq!(cm.precision(1), 0.0);
        assert_eq!(cm.f1(1), 0.0);
    }

    #[test]
    fn macro_average_is_unweighted_mean() {
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let r = example().report(&names);
        let mean_f1 = r.per_class.iter().map(|m| m.f1).sum::<f64>() / 3.0;
        assert!(close(r.macro_f1, mean_f1));
    }
}
