use ndarray::{Array2, ArrayView2};

use matrices::rotation_matrix_default;

fn rotation_matrix((i, j): (usize, usize), matrix: ArrayView2<f64>) -> Array2<f64> {
    let x = -2.0 * matrix[(i, j)];
    let y = matrix[(i, i)] - matrix[(j, j)];

    let (cos, sin) = if y - 0.0 < f64::EPSILON {
        let value = 1.0 / 2.0_f64.sqrt();
        (value, value)
    } else {
        let distance = (x * x + y * y).sqrt();
        let cos = ((1.0 + y.abs() / distance) / 2.0).sqrt();
        let sin = x.abs() * (x * y).signum() / (2.0 * cos * distance);
        (cos, sin)
    };

    let n = matrix.nrows();
    rotation_matrix_default(n, (cos, sin), (i, j))
}
