use ndarray::Array2;

pub use examples::*;

mod examples;

pub fn generate_matrix<T, F: Fn(usize, usize) -> T>(n: usize, generator: F) -> Array2<T> {
    let matrix = (0..n)
        .map(|row| (0..n).map(|column| generator(row, column)).collect())
        .collect::<Vec<Vec<T>>>();

    let dimensions = (n, n);
    Array2::from_shape_vec(dimensions, matrix.into_iter().flatten().collect::<Vec<T>>()).unwrap()
}

pub fn rotation_matrix_default(
    n: usize,
    (cos, sin): (f64, f64),
    (i, j): (usize, usize),
) -> Array2<f64> {
    generate_matrix(n, |row, column| {
        if (row == i && column == i) || (row == j && column == j) {
            cos
        } else if row == i && column == j {
            -sin
        } else if row == j && column == i {
            sin
        } else if row == column {
            1.0
        } else {
            0.0
        }
    })
}
