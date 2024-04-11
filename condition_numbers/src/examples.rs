use ndarray::{array, Array1, Array2};

pub struct Examples;

impl Examples {
    pub fn vector2() -> Array1<f64> {
        array![200.0, -600.0]
    }

    pub fn vector_n(n: usize) -> Array1<f64> {
        (0..n).map(|i| (i * i) as f64).collect()
    }

    pub fn vector_changed() -> Array1<f64> {
        array![199.0, -601.0]
    }

    pub fn bad_matrix() -> Array2<f64> {
        array![[-400.6, 199.8], [1198.80, -600.4]]
    }

    pub fn hilbert(n: usize) -> Array2<f64> {
        let matrix = (0..n)
            .map(|row| {
                (0..n)
                    .map(|column| 1.0 / (row + column + 1) as f64)
                    .collect()
            })
            .collect::<Vec<Vec<f64>>>();

        let dimensions = (n, n);
        Array2::from_shape_vec(
            dimensions,
            matrix.into_iter().flatten().collect::<Vec<f64>>(),
        )
        .unwrap()
    }

    pub fn tridiagonal(n: usize) -> Array2<f64> {
        let matrix = (0..n)
            .map(|row| {
                (0..n)
                    .map(|column| {
                        if row.abs_diff(column) == 1 {
                            (row * column + 1) as f64
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect::<Vec<Vec<f64>>>();

        let dimensions = (n, n);
        Array2::from_shape_vec(
            dimensions,
            matrix.into_iter().flatten().collect::<Vec<f64>>(),
        )
        .unwrap()
    }
}
