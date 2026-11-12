//! A minimal dense matrix type used by the network.

use std::fmt;
use std::ops::{Index, IndexMut};

/// Errors that can occur when constructing a [`Matrix`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    /// The flat data length does not equal `rows * cols`.
    DataLengthMismatch {
        /// Requested number of rows.
        rows: usize,
        /// Requested number of columns.
        cols: usize,
        /// Actual length of the provided data.
        len: usize,
    },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::DataLengthMismatch { rows, cols, len } => write!(
                f,
                "expected {} elements for a {rows}x{cols} matrix, got {len}",
                rows * cols
            ),
        }
    }
}

impl std::error::Error for MatrixError {}

/// A dense 2D matrix of `f64` values stored in row-major order.
///
/// Invariant: `data.len() == rows * cols`. Fields are private so that
/// this invariant can only be established through the constructors.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Creates a `rows x cols` matrix filled with zeros.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a matrix from a flat, row-major vector.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixError::DataLengthMismatch`] if `data.len() != rows * cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlp::matrix::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// assert_eq!(m.shape(), (2, 3));
    /// assert_eq!(m[(1, 0)], 4.0);
    /// ```
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::DataLengthMismatch {
                rows,
                cols,
                len: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Shape as `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Read-only view of the underlying row-major data.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Converts a `(row, col)` pair into a flat index, checking bounds.
    fn offset(&self, i: usize, j: usize) -> usize {
        assert!(
            i < self.rows && j < self.cols,
            "index ({i}, {j}) out of bounds for {}x{} matrix",
            self.rows,
            self.cols
        );
        i * self.cols + j
    }

    /// Returns the transpose: a `cols x rows` matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// Matrix product `self · other`.
    ///
    /// Uses the cache-friendly `i-k-j` loop order: the innermost loop walks
    /// one row of `other` and one row of `out` sequentially in memory, which
    /// is much faster than the textbook `i-j-k` order on row-major data.
    ///
    /// # Panics
    ///
    /// Panics if `self.cols() != other.rows()`.
    #[must_use]
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "matmul shape mismatch: {}x{} · {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let (n, m, p) = (self.rows, self.cols, other.cols);
        let mut out = Self::zeros(n, p);
        for i in 0..n {
            let out_row = &mut out.data[i * p..(i + 1) * p];
            for k in 0..m {
                let a_ik = self.data[i * m + k];
                let other_row = &other.data[k * p..(k + 1) * p];
                for (o, &b_kj) in out_row.iter_mut().zip(other_row) {
                    *o += a_ik * b_kj;
                }
            }
        }
        out
    }

    /// Creates a `rows x cols` matrix filled with `value`.
    #[must_use]
    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    /// Returns row `i` as a slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.rows()`.
    #[must_use]
    pub fn row(&self, i: usize) -> &[f64] {
        assert!(
            i < self.rows,
            "row {i} out of bounds for {} rows",
            self.rows
        );
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Iterates over the rows of the matrix as slices.
    pub fn row_iter(&self) -> impl Iterator<Item = &[f64]> {
        // `max(1)` keeps `chunks_exact` valid for zero-column matrices.
        self.data.chunks_exact(self.cols.max(1))
    }

    /// Builds a new matrix from the given row indices, in that order.
    /// Used for mini-batching and train/validation/test splits.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    #[must_use]
    pub fn select_rows(&self, indices: &[usize]) -> Self {
        let mut data = Vec::with_capacity(indices.len() * self.cols);
        for &i in indices {
            data.extend_from_slice(self.row(i));
        }
        Self {
            rows: indices.len(),
            cols: self.cols,
            data,
        }
    }

    /// Applies `f` to every element, returning a new matrix.
    #[must_use]
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Combines two same-shaped matrices element by element with `f`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    #[must_use]
    pub fn zip_map(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "element-wise shape mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }

    /// Element-wise sum `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.zip_map(other, |a, b| a + b)
    }

    /// Element-wise difference `self - other`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.zip_map(other, |a, b| a - b)
    }

    /// Element-wise (Hadamard) product `self ⊙ other`.
    #[must_use]
    pub fn hadamard(&self, other: &Self) -> Self {
        self.zip_map(other, |a, b| a * b)
    }

    /// Multiplies every element by `factor`.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        self.map(|x| x * factor)
    }

    /// Adds a `1 x cols` row vector to every row (bias broadcasting).
    ///
    /// # Panics
    ///
    /// Panics if `row` is not `1 x self.cols()`.
    #[must_use]
    pub fn add_row_vector(&self, row: &Self) -> Self {
        assert_eq!(
            row.shape(),
            (1, self.cols),
            "row vector must be 1x{}, got {:?}",
            self.cols,
            row.shape()
        );
        let mut out = self.clone();
        for out_row in out.data.chunks_exact_mut(self.cols.max(1)) {
            for (o, &b) in out_row.iter_mut().zip(&row.data) {
                *o += b;
            }
        }
        out
    }

    /// Sums over rows, producing a `1 x cols` row vector.
    /// This is the gradient of a broadcast bias.
    #[must_use]
    pub fn column_sums(&self) -> Self {
        let mut out = Self::zeros(1, self.cols);
        for row in self.row_iter() {
            for (o, &x) in out.data.iter_mut().zip(row) {
                *o += x;
            }
        }
        out
    }

    /// In-place `self += alpha * other`. With `alpha = -learning_rate` this
    /// is exactly one gradient descent step.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    pub fn add_scaled_in_place(&mut self, other: &Self, alpha: f64) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "add_scaled shape mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );
        for (a, &b) in self.data.iter_mut().zip(&other.data) {
            *a += alpha * b;
        }
    }

    /// Index of the largest element in each row (the predicted class).
    #[must_use]
    pub fn argmax_rows(&self) -> Vec<usize> {
        self.row_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold((0, f64::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
                        if v > best_v { (i, v) } else { (best_i, best_v) }
                    })
                    .0
            })
            .collect()
    }
}

/// Pretty-prints the matrix. Respects a precision flag, e.g. `{:.3}`.
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(4);
        for (i, row) in self.row_iter().enumerate() {
            write!(f, "{}", if i == 0 { "[" } else { " " })?;
            for (j, x) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{x:>width$.precision$}", width = precision + 4)?;
            }
            writeln!(f, "{}", if i + 1 == self.rows { "]" } else { "" })?;
        }
        Ok(())
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        let idx = self.offset(i, j);
        &self.data[idx]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        let idx = self.offset(i, j);
        &mut self.data[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: builds a matrix and panics on invalid input.
    fn mat(rows: usize, cols: usize, data: &[f64]) -> Matrix {
        Matrix::from_vec(rows, cols, data.to_vec()).expect("valid test matrix")
    }

    #[test]
    fn from_vec_rejects_wrong_length() {
        let result = Matrix::from_vec(2, 3, vec![1.0, 2.0]);
        assert_eq!(
            result,
            Err(MatrixError::DataLengthMismatch {
                rows: 2,
                cols: 3,
                len: 2
            })
        );
    }

    #[test]
    fn index_reads_row_major_layout() {
        let m = mat(2, 3, &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    fn index_mut_writes_to_correct_cell() {
        let mut m = Matrix::zeros(2, 3);
        m[(1, 2)] = 7.0;
        assert_eq!(m.as_slice(), &[0., 0., 0., 0., 0., 7.]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_rejects_column_overflow() {
        let m = Matrix::zeros(2, 3);
        let _ = m[(0, 3)];
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        let a = mat(2, 3, &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(a.transpose(), mat(3, 2, &[1., 4., 2., 5., 3., 6.]));
    }

    #[test]
    fn transpose_twice_is_identity() {
        let a = mat(2, 3, &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(a.transpose().transpose(), a);
    }

    #[test]
    fn matmul_known_result() {
        let a = mat(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let b = mat(3, 2, &[7., 8., 9., 10., 11., 12.]);
        assert_eq!(a.matmul(&b), mat(2, 2, &[58., 64., 139., 154.]));
    }

    #[test]
    fn matmul_with_identity_returns_same_matrix() {
        let a = mat(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let eye = mat(3, 3, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
        assert_eq!(a.matmul(&eye), a);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn matmul_rejects_mismatched_shapes() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 3);
        let _ = a.matmul(&b);
    }

    #[test]
    fn matmul_matches_naive_triple_loop() {
        let mut rng = crate::rng::Rng::new(9);
        let a = Matrix::from_vec(4, 5, (0..20).map(|_| rng.normal()).collect()).unwrap();
        let b = Matrix::from_vec(5, 3, (0..15).map(|_| rng.normal()).collect()).unwrap();
        let fast = a.matmul(&b);
        for i in 0..4 {
            for j in 0..3 {
                let naive: f64 = (0..5).map(|k| a[(i, k)] * b[(k, j)]).sum();
                assert!((fast[(i, j)] - naive).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn select_rows_picks_rows_in_given_order() {
        let a = mat(3, 2, &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(a.select_rows(&[2, 0]), mat(2, 2, &[5., 6., 1., 2.]));
    }

    #[test]
    fn elementwise_ops() {
        let a = mat(1, 3, &[1., 2., 3.]);
        let b = mat(1, 3, &[4., 5., 6.]);
        assert_eq!(a.add(&b), mat(1, 3, &[5., 7., 9.]));
        assert_eq!(b.sub(&a), mat(1, 3, &[3., 3., 3.]));
        assert_eq!(a.hadamard(&b), mat(1, 3, &[4., 10., 18.]));
        assert_eq!(a.scale(2.0), mat(1, 3, &[2., 4., 6.]));
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn elementwise_ops_reject_different_shapes() {
        let _ = Matrix::zeros(2, 2).add(&Matrix::zeros(2, 3));
    }

    #[test]
    fn add_row_vector_broadcasts_over_rows() {
        let a = mat(2, 2, &[1., 2., 3., 4.]);
        let bias = mat(1, 2, &[10., 20.]);
        assert_eq!(a.add_row_vector(&bias), mat(2, 2, &[11., 22., 13., 24.]));
    }

    #[test]
    fn column_sums_reduces_over_rows() {
        let a = mat(3, 2, &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(a.column_sums(), mat(1, 2, &[9., 12.]));
    }

    #[test]
    fn add_scaled_in_place_performs_gradient_step() {
        let mut w = mat(1, 2, &[1.0, 1.0]);
        let grad = mat(1, 2, &[0.5, -0.5]);
        w.add_scaled_in_place(&grad, -0.1);
        assert!((w[(0, 0)] - 0.95).abs() < 1e-12);
        assert!((w[(0, 1)] - 1.05).abs() < 1e-12);
    }

    #[test]
    fn argmax_rows_returns_column_of_max() {
        let a = mat(2, 3, &[0.1, 0.7, 0.2, 0.5, 0.3, 0.2]);
        assert_eq!(a.argmax_rows(), vec![1, 0]);
    }
}
