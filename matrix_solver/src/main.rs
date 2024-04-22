use ndarray::{array, Array1, Array2};
use ndarray_linalg::Solve;

use condition_numbers::ConditionNumbers;
use matrices::Examples;
use matrix_solver::{q_matrix, r_matrix, solve_qr};

fn main() {
    let matrix = array![[2.0, 2.0, -1.0], [3.0, 4.0, 2.0], [5.0, 6.4, 1.1]];
    let vector = array![4.0, 5.0, -5.4];
    examine(matrix, vector, "3x3 matrix");
    println!("-----------------------------------------------------------");

    let hilbert_matrix = Examples::hilbert(9);
    let hilbert_vector = Examples::random_vector(9);
    examine(hilbert_matrix, hilbert_vector, "Hilbert matrix");
    println!("-----------------------------------------------------------");

    let diagonal = Examples::diagonal(11);
    let diag_vector = Examples::random_vector(11);
    examine(diagonal, diag_vector, "Diagonal matrix");
    println!("-----------------------------------------------------------");
}

fn examine(matrix: Array2<f64>, vector: Array1<f64>, message: &str) {
    println!("{message}");

    let x_correct = matrix.solve(&vector).unwrap();
    let x_qr = solve_qr(matrix.view(), vector.view()).unwrap();

    println!("Matrix:\n{matrix:.3}");
    println!("Vector:\n{vector:.3}");
    println!("Solution:\n{x_correct:.3}");
    println!("QR solution:\n{x_qr:.3}");

    let r_matrix = r_matrix(matrix.view());
    let q_matrix = q_matrix(matrix.view());

    println!("Matrix:");
    let matrix_numbers = ConditionNumbers::new(matrix.view())
        .unwrap()
        .condition_table();
    matrix_numbers.printstd();

    println!("R matrix:");
    let r_numbers = ConditionNumbers::new(r_matrix.view())
        .unwrap()
        .condition_table();
    r_numbers.printstd();

    println!("Q matrix:");
    let q_numbers = ConditionNumbers::new(q_matrix.view())
        .unwrap()
        .condition_table();
    q_numbers.printstd()
}
