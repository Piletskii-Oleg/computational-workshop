use ndarray::{Array1, Array2};
use ndarray_linalg::EigVals;

use eigenvalues_partial_problem::{dot_product_method, power_iteration};
use matrices::Examples;

const EPSILONS: [f64; 3] = [1e-5, 1e-7, 1e-9];

fn main() {
    let random_matrix = Examples::random_matrix(4);
    let random_vector = Examples::random_vector(4);
    examine(random_matrix, random_vector, "-- Random matrix --");

    let hilbert_matrix = Examples::hilbert(9);
    let random_vector = Examples::random_vector(9);
    examine(hilbert_matrix, random_vector, "-- Hilbert matrix --");
}

fn examine(matrix: Array2<f64>, vector: Array1<f64>, message: &str) {
    println!("{message}");
    for epsilon in EPSILONS {
        println!("Epsilon: {epsilon:e}");
        println!("Power iteration.");
        let lambda = power_iteration(matrix.view(), vector.view(), epsilon);
        println!(
            "found eigenvalue {:.6} with error {:.10}\nvector: {:?}\nsteps: {}",
            lambda.eigenvalue, lambda.error, lambda.eigenvector, lambda.steps
        );
        println!("Actual eigenvalues: {:.2}", matrix.eigvals().unwrap());

        println!("Dot product.");
        let lambda_dot = dot_product_method(matrix.view(), vector.view(), epsilon);
        println!(
            "found eigenvalue {:.6} with error {:.8}\nvector: {:?}\nsteps: {}",
            lambda_dot.eigenvalue, lambda_dot.error, lambda_dot.eigenvector, lambda_dot.steps
        );
        println!("--------------------------------")
    }
    println!()
}
