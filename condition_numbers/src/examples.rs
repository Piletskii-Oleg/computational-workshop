use ndarray::{array, Array1, Array2};

pub struct Examples;

impl Examples {
    pub fn vector() -> Array1<f64> {
        array![200.0, -600.0]
    }

    pub fn vector_changed() -> Array1<f64> {
        array![199.0, -601.0]
    }

    pub fn simple() -> Array2<f64> {
        array![[-400.6, 199.8], [1198.80, -600.4]]
    }

    pub fn hilbert(n: usize) -> Array2<f64> {
        let mut matrix = vec![vec![0.0; n]; n];

        for row in 0..n {
            for column in 0..n {
                matrix[row][column] = 1.0 / (row + column + 1) as f64;
            }
        }

        let dimensions = (n, n);
        Array2::from_shape_vec(
            dimensions,
            matrix.into_iter().flatten().collect::<Vec<f64>>(),
        )
        .unwrap()
    }
}
