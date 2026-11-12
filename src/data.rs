//! Dataset loading and preprocessing: CSV parsing, stratified splitting,
//! z-score standardization and one-hot encoding.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::matrix::Matrix;
use crate::rng::Rng;

/// Names of the four Iris features, in file order.
pub const IRIS_FEATURES: [&str; 4] = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
];

/// Errors that can occur while loading a dataset.
#[derive(Debug)]
pub enum DataError {
    /// The file could not be read.
    Io(std::io::Error),
    /// A line could not be parsed.
    Parse {
        /// 1-based line number in the file.
        line: usize,
        /// Human-readable reason.
        message: String,
    },
    /// The file contained no samples.
    Empty,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::Io(e) => write!(f, "could not read dataset: {e}"),
            DataError::Parse { line, message } => write!(f, "line {line}: {message}"),
            DataError::Empty => write!(f, "dataset contains no samples"),
        }
    }
}

impl std::error::Error for DataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DataError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Lets the `?` operator convert I/O errors into `DataError` automatically.
impl From<std::io::Error> for DataError {
    fn from(e: std::io::Error) -> Self {
        DataError::Io(e)
    }
}

/// A labelled classification dataset.
///
/// Invariant: `features.rows() == labels.len()` and every label is
/// `< class_names.len()`.
#[derive(Debug, Clone)]
pub struct Dataset {
    features: Matrix,
    labels: Vec<usize>,
    class_names: Vec<String>,
    feature_names: Vec<String>,
}

impl Dataset {
    /// Creates a dataset, validating its invariants.
    ///
    /// # Panics
    ///
    /// Panics if the number of rows and labels differ or a label is out of range.
    #[must_use]
    pub fn new(
        features: Matrix,
        labels: Vec<usize>,
        class_names: Vec<String>,
        feature_names: Vec<String>,
    ) -> Self {
        assert_eq!(features.rows(), labels.len(), "rows and labels differ");
        assert_eq!(features.cols(), feature_names.len(), "feature name count");
        assert!(
            labels.iter().all(|&y| y < class_names.len()),
            "label out of range"
        );
        Self {
            features,
            labels,
            class_names,
            feature_names,
        }
    }

    /// Parses a headerless CSV file whose last column is a string class label
    /// and whose other columns are numeric features (the UCI Iris format).
    /// Blank lines are skipped. Classes are numbered in order of first
    /// appearance.
    ///
    /// # Errors
    ///
    /// Returns a [`DataError`] if the file cannot be read, a line is malformed,
    /// or there are no samples.
    pub fn from_csv(path: impl AsRef<Path>, feature_names: &[&str]) -> Result<Self, DataError> {
        let text = fs::read_to_string(path)?;
        let n_features = feature_names.len();

        let mut values = Vec::new();
        let mut labels = Vec::new();
        let mut class_names: Vec<String> = Vec::new();
        let mut class_index: HashMap<String, usize> = HashMap::new();

        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() != n_features + 1 {
                return Err(DataError::Parse {
                    line: line_no + 1,
                    message: format!(
                        "expected {} columns, found {}",
                        n_features + 1,
                        fields.len()
                    ),
                });
            }
            for field in &fields[..n_features] {
                let v: f64 = field.parse().map_err(|_| DataError::Parse {
                    line: line_no + 1,
                    message: format!("`{field}` is not a number"),
                })?;
                values.push(v);
            }
            let name = fields[n_features];
            let next_id = class_names.len();
            let id = *class_index.entry(name.to_string()).or_insert_with(|| {
                class_names.push(name.to_string());
                next_id
            });
            labels.push(id);
        }

        if labels.is_empty() {
            return Err(DataError::Empty);
        }
        let features = Matrix::from_vec(labels.len(), n_features, values)
            .expect("every parsed row has exactly n_features values");
        Ok(Self::new(
            features,
            labels,
            class_names,
            feature_names.iter().map(|s| (*s).to_string()).collect(),
        ))
    }

    /// Loads the UCI Iris dataset (`iris.data`).
    ///
    /// # Errors
    ///
    /// See [`Dataset::from_csv`].
    pub fn iris(path: impl AsRef<Path>) -> Result<Self, DataError> {
        Self::from_csv(path, &IRIS_FEATURES)
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// `true` if there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Feature matrix, one sample per row.
    #[must_use]
    pub fn features(&self) -> &Matrix {
        &self.features
    }

    /// Class label of each sample.
    #[must_use]
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Class names, indexed by label.
    #[must_use]
    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }

    /// Feature names, indexed by column.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Number of distinct classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.class_names.len()
    }

    /// Number of input features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.features.cols()
    }

    /// Number of samples in each class.
    #[must_use]
    pub fn class_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.n_classes()];
        for &y in &self.labels {
            counts[y] += 1;
        }
        counts
    }

    /// Returns a new dataset with only the given samples, in that order.
    #[must_use]
    pub fn subset(&self, indices: &[usize]) -> Self {
        Self {
            features: self.features.select_rows(indices),
            labels: indices.iter().map(|&i| self.labels[i]).collect(),
            class_names: self.class_names.clone(),
            feature_names: self.feature_names.clone(),
        }
    }

    /// Returns a copy whose features have been transformed by `scaler`.
    #[must_use]
    pub fn standardized(&self, scaler: &Standardizer) -> Self {
        Self {
            features: scaler.transform(&self.features),
            ..self.clone()
        }
    }

    /// Splits into train / validation / test sets while keeping the class
    /// proportions equal in every split (stratification). The test fraction
    /// is whatever remains after `train_frac + val_frac`.
    ///
    /// # Panics
    ///
    /// Panics if the fractions are not in `(0, 1)` or sum to `>= 1`.
    #[must_use]
    pub fn stratified_split(&self, train_frac: f64, val_frac: f64, rng: &mut Rng) -> Split {
        assert!(
            train_frac > 0.0 && val_frac > 0.0 && train_frac + val_frac < 1.0,
            "invalid split fractions"
        );
        let (mut train, mut val, mut test) = (Vec::new(), Vec::new(), Vec::new());
        for class in 0..self.n_classes() {
            let mut idx: Vec<usize> = (0..self.len())
                .filter(|&i| self.labels[i] == class)
                .collect();
            rng.shuffle(&mut idx);
            let n = idx.len() as f64;
            let n_train = (n * train_frac).round() as usize;
            let n_val = (n * val_frac).round() as usize;
            train.extend_from_slice(&idx[..n_train]);
            val.extend_from_slice(&idx[n_train..n_train + n_val]);
            test.extend_from_slice(&idx[n_train + n_val..]);
        }
        // Mix the classes so that consecutive samples are not all the same class.
        rng.shuffle(&mut train);
        rng.shuffle(&mut val);
        rng.shuffle(&mut test);
        Split {
            train: self.subset(&train),
            validation: self.subset(&val),
            test: self.subset(&test),
        }
    }

    /// Per-feature summary statistics, useful for the report.
    #[must_use]
    pub fn feature_stats(&self) -> Vec<FeatureStats> {
        let scaler = Standardizer::fit(&self.features);
        (0..self.n_features())
            .map(|j| {
                let column = self.features.row_iter().map(|row| row[j]);
                let (min, max) = column.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                    (lo.min(x), hi.max(x))
                });
                FeatureStats {
                    name: self.feature_names[j].clone(),
                    min,
                    max,
                    mean: scaler.mean[j],
                    std: scaler.std[j],
                }
            })
            .collect()
    }
}

/// Summary statistics for a single feature column.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureStats {
    /// Feature name.
    pub name: String,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean.
    pub mean: f64,
    /// Population standard deviation.
    pub std: f64,
}

/// The three disjoint parts of a dataset.
#[derive(Debug, Clone)]
pub struct Split {
    /// Used to fit the weights.
    pub train: Dataset,
    /// Used to compare hyper-parameters and watch for over-fitting.
    pub validation: Dataset,
    /// Touched only once, to report the final model's performance.
    pub test: Dataset,
}

/// Z-score standardization: `x' = (x - mean) / std`, per feature.
///
/// Must be fitted on the training set only and then applied to every split;
/// fitting on all data would leak information from the test set.
#[derive(Debug, Clone, PartialEq)]
pub struct Standardizer {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Standardizer {
    /// Computes per-column mean and standard deviation.
    #[must_use]
    pub fn fit(x: &Matrix) -> Self {
        let n = x.rows() as f64;
        let mean: Vec<f64> = x.column_sums().as_slice().iter().map(|s| s / n).collect();
        let mut var = vec![0.0; x.cols()];
        for row in x.row_iter() {
            for (j, &v) in row.iter().enumerate() {
                var[j] += (v - mean[j]).powi(2);
            }
        }
        let std = var
            .iter()
            .map(|v| {
                let s = (v / n).sqrt();
                // A constant column would divide by zero; leave it unscaled.
                if s < 1e-12 { 1.0 } else { s }
            })
            .collect();
        Self { mean, std }
    }

    /// Applies the fitted transformation.
    ///
    /// # Panics
    ///
    /// Panics if `x` has a different number of columns than the fitted data.
    #[must_use]
    pub fn transform(&self, x: &Matrix) -> Matrix {
        assert_eq!(x.cols(), self.mean.len(), "standardizer column mismatch");
        let mut out = x.clone();
        for i in 0..x.rows() {
            for j in 0..x.cols() {
                out[(i, j)] = (x[(i, j)] - self.mean[j]) / self.std[j];
            }
        }
        out
    }

    /// Per-feature means learned by [`Standardizer::fit`].
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Per-feature standard deviations learned by [`Standardizer::fit`].
    #[must_use]
    pub fn std(&self) -> &[f64] {
        &self.std
    }
}

/// One-hot encodes labels into an `n x n_classes` matrix.
///
/// # Panics
///
/// Panics if a label is `>= n_classes`.
#[must_use]
pub fn one_hot(labels: &[usize], n_classes: usize) -> Matrix {
    let mut out = Matrix::zeros(labels.len(), n_classes);
    for (i, &y) in labels.iter().enumerate() {
        out[(i, y)] = 1.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("mlp_test_{name}_{}.csv", std::process::id()));
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path
    }

    fn toy() -> Dataset {
        let path = write_temp("toy", "1,2,a\n3,4,b\n\n5,6,a\r\n7,8,c\n9,10,b\n11,12,c\n");
        let ds = Dataset::from_csv(&path, &["x", "y"]).unwrap();
        fs::remove_file(path).ok();
        ds
    }

    #[test]
    fn parses_csv_skipping_blank_lines_and_crlf() {
        let ds = toy();
        assert_eq!(ds.len(), 6);
        assert_eq!(ds.class_names(), &["a", "b", "c"]);
        assert_eq!(ds.labels(), &[0, 1, 0, 2, 1, 2]);
        assert_eq!(ds.features().row(2), &[5.0, 6.0]);
    }

    #[test]
    fn reports_bad_number_with_line_number() {
        let path = write_temp("bad", "1,2,a\n1,oops,b\n");
        let err = Dataset::from_csv(&path, &["x", "y"]).unwrap_err();
        fs::remove_file(path).ok();
        assert!(matches!(err, DataError::Parse { line: 2, .. }), "{err}");
    }

    #[test]
    fn reports_wrong_column_count() {
        let path = write_temp("cols", "1,2,3,a\n");
        let err = Dataset::from_csv(&path, &["x", "y"]).unwrap_err();
        fs::remove_file(path).ok();
        assert!(matches!(err, DataError::Parse { line: 1, .. }));
    }

    #[test]
    fn missing_file_is_io_error() {
        let err = Dataset::from_csv("definitely/not/here.csv", &["x"]).unwrap_err();
        assert!(matches!(err, DataError::Io(_)));
    }

    #[test]
    fn stratified_split_preserves_class_balance_and_is_disjoint() {
        // 3 classes x 20 samples.
        let n = 60;
        let features = Matrix::from_vec(n, 1, (0..n).map(|i| i as f64).collect()).unwrap();
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let ds = Dataset::new(
            features,
            labels,
            vec!["a".into(), "b".into(), "c".into()],
            vec!["id".into()],
        );
        let split = ds.stratified_split(0.7, 0.15, &mut Rng::new(0));
        assert_eq!(split.train.class_counts(), vec![14, 14, 14]);
        assert_eq!(split.validation.class_counts(), vec![3, 3, 3]);
        assert_eq!(split.test.class_counts(), vec![3, 3, 3]);

        // Every original sample appears in exactly one split.
        let mut ids: Vec<i64> = [&split.train, &split.validation, &split.test]
            .iter()
            .flat_map(|d| d.features().as_slice().iter().map(|&x| x as i64))
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..n as i64).collect::<Vec<_>>());
    }

    #[test]
    fn standardizer_gives_zero_mean_unit_std() {
        let x = Matrix::from_vec(4, 2, vec![1., 10., 2., 20., 3., 30., 4., 40.]).unwrap();
        let z = Standardizer::fit(&x).transform(&x);
        let refit = Standardizer::fit(&z);
        for j in 0..2 {
            assert!(refit.mean()[j].abs() < 1e-12);
            assert!((refit.std()[j] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn standardizer_leaves_constant_columns_finite() {
        let x = Matrix::from_vec(3, 1, vec![5., 5., 5.]).unwrap();
        let z = Standardizer::fit(&x).transform(&x);
        assert!(z.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn one_hot_encodes_labels() {
        let y = one_hot(&[2, 0], 3);
        assert_eq!(y.as_slice(), &[0., 0., 1., 1., 0., 0.]);
    }
}
