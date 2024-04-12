use ndarray::{array, Array1, Array2};
use rand::random;

pub struct Examples;

impl Examples {
    pub fn vector_n(n: usize) -> Array1<f64> {
        (0..n).map(|i| (i * i) as f64).collect()
    }

    pub fn vector_n_same(n: usize, value: f64) -> Array1<f64> {
        (0..n).map(|_| value).collect()
    }

    pub fn bad_matrix2() -> Array2<f64> {
        array![[1.0, 0.99], [0.99, 0.98]]
    }
    pub fn bad_vector2() -> Array1<f64> {
        array![1.99, 1.97]
    }

    pub fn hilbert(n: usize) -> Array2<f64> {
        Self::generate_matrix(n, |row, column| 1.0 / (row + column + 1) as f64)
    }

    pub fn tridiagonal(n: usize) -> Array2<f64> {
        Self::generate_matrix(n, |row, column| {
            if row.abs_diff(column) == 1 || row == column {
                (row * column + 1) as f64 * random::<f64>()
            } else {
                0.0
            }
        })
    }

    pub fn diagonal(n: usize) -> Array2<f64> {
        Self::generate_matrix(n, |row, column| {
            if row == column {
                60.43 * (row + 1) as f64 * random::<f64>() * 40.0
            } else {
                0.0
            }
        })
    }

    pub fn random_matrix(n: usize) -> Array2<f64> {
        Self::generate_matrix(n, |_, _| random::<f64>() * 100.0)
    }

    pub fn random_vector(n: usize) -> Array1<f64> {
        (0..n).map(|_| random::<f64>() * 70.0).collect()
    }

    fn generate_matrix(n: usize, generator: fn(usize, usize) -> f64) -> Array2<f64> {
        let matrix = (0..n)
            .map(|row| (0..n).map(|column| generator(row, column)).collect())
            .collect::<Vec<Vec<f64>>>();

        let dimensions = (n, n);
        Array2::from_shape_vec(
            dimensions,
            matrix.into_iter().flatten().collect::<Vec<f64>>(),
        )
        .unwrap()
    }
}
