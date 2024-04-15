use ndarray::{ArrayView1, ArrayView2};
use ndarray_linalg::{Norm, Solve};
use prettytable::{row, Cell, Row, Table};

use condition_numbers::{add_number, ConditionNumbers};
use matrices::Examples;

const VARIATIONS: [f64; 8] = [10.0, 1.0, 0.1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10];
const NEGATIVE_VARIATIONS: [f64; 8] = [-10.0, -1.0, -0.1, -1e-2, -1e-4, -1e-6, -1e-8, -1e-10];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bad_matrix = Examples::bad_matrix2();
    let vector = Examples::bad_vector2();

    const HILBERT_SIZE: usize = 5;
    let hilbert = Examples::hilbert(HILBERT_SIZE);
    let hilbert_vec = Examples::random_vector(HILBERT_SIZE);

    const TRIDIAGONAL_SIZE: usize = 7;
    let tridiagonal = Examples::tridiagonal(TRIDIAGONAL_SIZE);
    let vector_tridiagonal = Examples::random_vector(TRIDIAGONAL_SIZE);

    const DIAGONAL_SIZE: usize = 7;
    let diagonal = Examples::diagonal(DIAGONAL_SIZE);
    let vector_diagonal = Examples::random_vector(DIAGONAL_SIZE);

    let random_matrix = Examples::random_matrix(8);
    let random_vector = Examples::random_vector(8);

    examine(bad_matrix.view(), vector.view(), "------ Bad matrix ------");
    examine(
        hilbert.view(),
        hilbert_vec.view(),
        "------ Hilbert matrix ------",
    );
    examine(
        tridiagonal.view(),
        vector_tridiagonal.view(),
        "------ Tridiagonal matrix ------",
    );
    examine(
        diagonal.view(),
        vector_diagonal.view(),
        "------ Diagonal matrix ------",
    );
    examine(
        random_matrix.view(),
        random_vector.view(),
        "------ Random matrix ------",
    );
    Ok(())
}

fn examine(matrix: ArrayView2<f64>, vector: ArrayView1<f64>, text: &str) {
    println!("{}", text);
    println!("Matrix");
    println!("{:.3}", matrix);

    println!("Vector: {vector:.3}");
    let x = matrix.solve(&vector).unwrap();
    println!("Solution: {x:.2}");
    let numbers = ConditionNumbers::new(matrix.view()).unwrap();

    let mut table = Table::new();
    table.add_row(row!["spectre", "volume", "angle"]);
    table.add_row(numbers.into());
    table.printstd();

    let mut table = Table::new();
    let headers = push_back_to_row(
        VARIATIONS.map(|num| format!("{num:e}")).to_vec(),
        "".to_string(),
    );

    let matrix_variation = matrix_variations(matrix, vector, VARIATIONS.to_vec());
    let matrix_neg_variation = matrix_variations(matrix, vector, NEGATIVE_VARIATIONS.to_vec());

    let vector_variation = vector_variations(matrix, vector, VARIATIONS.to_vec());
    let vector_neg_variation = vector_variations(matrix, vector, NEGATIVE_VARIATIONS.to_vec());

    table.add_row(headers);
    table.add_row(matrix_variation);
    table.add_row(matrix_neg_variation);
    table.add_row(vector_variation);
    table.add_row(vector_neg_variation);
    table.printstd()
}

fn vector_variations(
    matrix: ArrayView2<f64>,
    vector: ArrayView1<f64>,
    variations: Vec<f64>,
) -> Row {
    let vector_variations = variations
        .iter()
        .map(|&var| variate_vector_solve(matrix.view(), vector.view(), var))
        .map(|variance| format!("{:.4}", variance))
        .collect();

    let row_name = if variations[0].is_sign_negative() {
        "Vector (-)"
    } else {
        "Vector (+)"
    };

    push_back_to_row(vector_variations, row_name.to_string())
}

fn matrix_variations(
    matrix: ArrayView2<f64>,
    vector: ArrayView1<f64>,
    variations: Vec<f64>,
) -> Row {
    let matrix_variations = variations
        .iter()
        .map(|&var| variate_matrix_solve(matrix.view(), vector.view(), var))
        .map(|variance| format!("{:.4}", variance))
        .collect();

    let row_name = if variations[0].is_sign_negative() {
        "Matrix (-)"
    } else {
        "Matrix (+)"
    };

    push_back_to_row(matrix_variations, row_name.to_string())
}

fn push_back_to_row(mut vec: Vec<String>, text: String) -> Row {
    vec.insert(0, text);
    Row::new(vec.iter().map(|string| Cell::new(string)).collect())
}

fn variate_matrix_solve(matrix: ArrayView2<f64>, vector: ArrayView1<f64>, var: f64) -> f64 {
    let x = matrix.solve(&vector).unwrap();
    let var_matrix = add_number(matrix, var);
    let x_var = var_matrix.solve(&vector).unwrap();
    (x - x_var).norm_l1()
}

fn variate_vector_solve(matrix: ArrayView2<f64>, vector: ArrayView1<f64>, var: f64) -> f64 {
    let x = matrix.solve(&vector).unwrap();
    let var_vector = add_number(vector, var);
    let x_var = matrix.solve(&var_vector).unwrap();
    (x - x_var).norm_l1()
}
