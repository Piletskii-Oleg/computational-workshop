use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::Solve;

pub fn rotation_matrix((i, j): (usize, usize), vector: ArrayView1<f64>) -> Array2<f64> {
    let denominator = (vector[i] * vector[i] + vector[j] * vector[j]).sqrt();
    let cos = vector[i] / denominator;
    let sin = -vector[j] / denominator;

    matrices::generate_matrix(vector.len(), |row, column| {
        if (row == i && column == i) || (row == j && column == j) {
            cos
        } else if row == i && column == j {
            -sin
        } else if row == j && column == i {
            sin
        } else if row == column {
            1.0
        } else {
            0.0
        }
    })
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
            transposed.dot(&acc2)
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

    println!("r: {r:.3?}\ny: {y:.3?}");
    Some(r.solve((&y).into()).unwrap())
}
