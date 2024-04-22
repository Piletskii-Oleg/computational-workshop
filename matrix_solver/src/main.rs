use ndarray::{array, Array2};
use ndarray_linalg::Solve;

use matrix_solver::solve_qr;

fn main() {
    let matrix = array![[2.0, 2.0, -1.0], [3.0, 4.0, 2.0], [5.0, 6.4, 1.1]];
    let unit: Array2<f64> = Array2::eye(matrix.nrows());
    let matrix_by_unit = matrix.clone().dot(&unit);
    assert_eq!(matrix.clone(), matrix_by_unit);
    let vector = array![4.0, 5.0, -5.4];
    let x_correct = matrix.solve(&vector);
    let x_qr = solve_qr(matrix.view(), vector.view()).unwrap();

    println!("{x_correct:.3?}\n{x_qr:.3?}")
}
