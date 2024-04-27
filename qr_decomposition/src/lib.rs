use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::Solve;

use matrices::rotation_matrix_default;

pub fn rotation_matrix((i, j): (usize, usize), vector: ArrayView1<f64>) -> Array2<f64> {
    let denominator = (vector[i] * vector[i] + vector[j] * vector[j]).sqrt();
    let cos = vector[i] / denominator;
    let sin = -vector[j] / denominator;

    rotation_matrix_default(vector.len(), (cos, sin), (i, j))
}

pub fn rotation_matrices(matrix: ArrayView2<f64>) -> Array2<Array2<f64>> {
    let n = matrix.nrows();
    matrices::generate_matrix(n, |row, column| {
        rotation_matrix((row, column), matrix.column(row))
    })
}

pub fn r_matrix(matrix: ArrayView2<f64>) -> Array2<f64> {
    let n = matrix.nrows();

    (0..n).fold(matrix.to_owned(), |acc, i| {
        (i + 1..n).fold(acc, |acc2, j| {
            let rotation_matrix = rotation_matrix((i, j), matrix.column(i));
            rotation_matrix.dot(&acc2)
        })
    })
}

pub fn q_matrix(matrix: ArrayView2<f64>) -> Array2<f64> {
    let n = matrix.nrows();
    let unit = Array2::eye(n);

    (0..n).fold(unit, |acc, i| {
        (i + 1..n).fold(acc, |acc2, j| {
            let rotation_matrix = rotation_matrix((i, j), matrix.column(i));
            let transposed = rotation_matrix.t();
            acc2.dot(&transposed)
        })
    })
}

pub fn dissolve(matrix: ArrayView2<f64>, vector: ArrayView1<f64>) -> (Array2<f64>, Array1<f64>) {
    let n = matrix.nrows();

    let rotation_matrices = rotation_matrices(matrix);
    let mut r = matrix.to_owned();
    let mut y = vector.to_owned();
    let mut q = Array2::eye(n);
    for i in 0..n {
        for j in i + 1..n {
            let rotation_matrix = rotation_matrices[(i, j)].view();

            r = rotation_matrix.dot(&r);
            y = rotation_matrix.dot(&y);
            q = q.dot(&rotation_matrix.t());
        }
    }

    (r, y)
}

pub fn r_vector(matrix: ArrayView2<f64>, vector: ArrayView1<f64>) -> Array1<f64> {
    let n = matrix.nrows();
    (0..n - 1).fold(vector.to_owned(), |acc, i| {
        (i + 1..n).fold(acc, |acc2, j| {
            let rotation_matrix = rotation_matrix((i, j), matrix.column(i));
            rotation_matrix.dot(&acc2)
        })
    })
}

pub fn solve_qr(matrix: ArrayView2<f64>, vector: ArrayView1<f64>) -> Option<Array1<f64>> {
    let r = r_matrix(matrix);
    let y = r_vector(matrix, vector);
    Some(r.solve(&y).unwrap())
}

#[cfg(test)]
mod tests {
    use ndarray_linalg::Solve;

    use matrices::Examples;

    use crate::{q_matrix, r_matrix, solve_qr};

    #[test]
    fn qr_multiplied_is_original_matrix() {
        let matrix = Examples::random_matrix(10);
        let r_matrix = r_matrix(matrix.view());
        let q_matrix = q_matrix(matrix.view());

        let qr = q_matrix.dot(&r_matrix);
        assert!(matrix.abs_diff_eq(&qr, 1e-10));
    }

    #[test]
    fn qr_decomposition_solution_is_correct() {
        let matrix = Examples::random_matrix(10);
        let vector = Examples::random_vector(10);

        let x_qr = solve_qr(matrix.view(), vector.view()).unwrap();
        let x_normal = matrix.solve(&vector).unwrap();

        assert!(x_normal.abs_diff_eq(&x_qr, 1e-10))
    }
}
