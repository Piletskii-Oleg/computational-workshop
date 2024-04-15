use ndarray::{Array, ArrayView, Dimension};

pub use condition::*;

mod condition;

pub fn add_number<D: Dimension>(matrix: ArrayView<f64, D>, number: f64) -> Array<f64, D> {
    matrix.map(|&a| a + number)
}
