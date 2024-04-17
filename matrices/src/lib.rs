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
