use ndarray::{Array2, ArrayView1, ArrayView2};

pub fn rotation_matrix((i, j): (usize, usize), vector: ArrayView1<f64>) -> Array2<f64> {
    let denominator = 1.0 / (vector[i] * vector[i] + vector[j] * vector[j]).sqrt();
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

pub fn rotation_table(matrix: ArrayView2<f64>) -> Option<Array2<Array2<f64>>> {
    if matrix.ncols() != matrix.nrows() {
        return None;
    }

    let n = matrix.nrows();
    let matrices_matrix = matrices::generate_matrix(n, |row, column| {
        if row < column {
            rotation_matrix((row, column), matrix.column(row)) // is this correct?
        } else {
            Array2::default((n, n))
        }
    });

    Some(matrices_matrix)
}
