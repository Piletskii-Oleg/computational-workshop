use ndarray::Array2;

pub use examples::*;

mod examples;

pub fn generate_matrix<F: Fn(usize, usize) -> f64>(n: usize, generator: F) -> Array2<f64> {
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
