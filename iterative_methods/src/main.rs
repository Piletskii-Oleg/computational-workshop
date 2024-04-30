use ndarray::{array, Array1, Array2};
use ndarray_linalg::Solve;

use iterative_methods::seidel;
use matrices::Examples;

const EPSILONS: [f64; 5] = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9];

fn main() {
    let mut matrix = array![[3, 4, 1], [5, 5, 1], [6, 6, 9]].mapv(|value| value as f64);
    add_to_diagonal(&mut matrix, 40.0);
    let vector = Examples::random_vector(3);
    examine(&matrix, &vector, "3x3 Matrix", &EPSILONS);

    let symmetric = Examples::sparse_diagonal_dominance(200);
    let vector = Examples::random_vector(200);
    examine(&symmetric, &vector, "Symmetric 200x200", &EPSILONS);

    let symmetric = Examples::sparse_diagonal_dominance(230);
    let vector = Examples::random_vector(230);
    examine(&symmetric, &vector, "Symmetric 230x230", &EPSILONS);
}

fn add_to_diagonal(matrix: &mut Array2<f64>, num: f64) {
    for i in 0..matrix.nrows() {
        matrix[(i, i)] += num;
    }
}

fn examine(matrix: &Array2<f64>, vector: &Array1<f64>, message: &str, epsilons: &[f64]) {
    println!("{message}");

    let x_correct = matrix.solve(&vector).unwrap();
    println!("Correct: {x_correct:.3}");

    for epsilon in epsilons {
        println!("Epsilon: {epsilon:e}");

        let x = iterative_methods::iterate(matrix.view(), vector.view(), *epsilon);
        println!("Iterative: {:.3} in {} steps", x.x(), x.iteration_count());

        let x = seidel(matrix.view(), vector.view(), *epsilon);
        println!("Seidel: {:.3} in {} steps", x.x(), x.iteration_count());
    }
}
