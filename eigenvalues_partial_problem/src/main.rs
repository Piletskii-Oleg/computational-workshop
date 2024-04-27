use ndarray_linalg::EigVals;

use eigenvalues_partial_problem::{dot_product_method, power_iteration};
use matrices::Examples;

fn main() {
    let matrix = Examples::random_matrix(4);
    let vector = Examples::random_vector(4);

    let lambda = power_iteration(matrix.view(), vector.view(), 1.0);
    println!("{lambda:?}");
    println!();
    println!("{:.2}", matrix.eigvals().unwrap());

    println!(
        "{:?}",
        dot_product_method(matrix.view(), vector.view(), 1e-6)
    );
}
