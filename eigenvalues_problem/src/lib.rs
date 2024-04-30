use std::fmt::{Display, Formatter};

use ndarray::{Array2, ArrayView2};
use ndarray_linalg::Inverse;

pub use choose_max::*;
pub use gershgorin::*;
use matrices::rotation_matrix_default;

mod choose_max;
mod gershgorin;

#[derive(Debug)]
pub struct JacobiResult {
    eigenvalues: Vec<f64>,
    steps: u32,
}

fn rotation_matrix((i, j): (usize, usize), matrix: ArrayView2<f64>) -> Array2<f64> {
    let x = -2.0 * matrix[(i, j)];
    let y = matrix[(i, i)] - matrix[(j, j)];

    let angle = (x / y).atan() / 2.0;
    let cos = angle.cos();
    let sin = angle.sin();
    // let (cos, sin) = if y < f64::EPSILON {
    //     let value = 1.0 / 2.0_f64.sqrt();
    //     (value, value)
    // } else {
    //     let distance = (x * x + y * y).sqrt();
    //     let cos = ((1.0 + y.abs() / distance) / 2.0).sqrt();
    //     let sin = x.abs() * (x * y).signum() / (2.0 * cos * distance);
    //     (cos, sin)
    // };

    let n = matrix.nrows();
    rotation_matrix_default(n, (cos, sin), (i, j))
}

pub fn jacobi_method<C: ChooseMax>(
    matrix: ArrayView2<f64>,
    mut finder: C,
    epsilon: f64,
) -> JacobiResult {
    let n = matrix.nrows();

    let mut steps = 0;
    let mut matrix = matrix.to_owned();
    while !is_diagonal_new(matrix.view(), epsilon) {
        let max = finder.choose(matrix.view());

        let rotation_matrix = rotation_matrix(max, matrix.view());
        let rotation_matrix_inv = rotation_matrix.inv().unwrap();
        matrix = rotation_matrix.dot(&matrix).dot(&rotation_matrix_inv);

        steps += 1;
        //println!("{matrix:.2}");
        // finder.update(max, matrix.view());
    }

    let eigenvalues = (0..n).map(|i| matrix[(i, i)]).collect();
    JacobiResult { eigenvalues, steps }
}

fn is_diagonal(matrix: ArrayView2<f64>, epsilon: f64) -> bool {
    let n = matrix.nrows();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            if matrix[(i, j)].abs() > epsilon {
                return false;
            }
        }
    }

    true
}

fn is_diagonal_new(matrix: ArrayView2<f64>, epsilon: f64) -> bool {
    let sums = matrix
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| row.map(|x| x.abs()).sum() - row[i].abs())
        .collect::<Vec<_>>();
    //println!("R_i: {sum:?}");
    sums.iter().all(|sum| *sum < epsilon)
}

impl Display for JacobiResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2?} in {} steps", self.eigenvalues, self.steps)
    }
}
