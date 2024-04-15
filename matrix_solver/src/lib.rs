use ndarray::Array2;

pub fn rotation_matrix((i, j): (usize, usize), n: usize, phi: f64) -> Array2<f64> {
    matrices::generate_matrix(n, |row, column| {
        if (row == i && column == i) || (row == j && column == j) {
            phi.cos()
        } else if row == i && column == j {
            -phi.sin()
        } else if row == j && column == i {
            phi.sin()
        } else if row == column {
            1.0
        } else {
            0.0
        }
    })
}
