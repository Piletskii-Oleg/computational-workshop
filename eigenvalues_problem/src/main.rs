use ndarray_linalg::{random_hermite, EigVals};

use eigenvalues_problem::{jacobi_method, OptimalElement};

fn main() {
    let matrix = random_hermite::<f64, ndarray::OwnedRepr<f64>>(2);
    let eigenvalues = jacobi_method(matrix.view(), OptimalElement::new(matrix.view()), 1e-2);
    println!("{eigenvalues:.2?}");
    println!("{:.2?}", matrix.eigvals())
}
