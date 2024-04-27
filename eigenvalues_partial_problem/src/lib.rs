use ndarray::{Array1, ArrayView1, ArrayView2};
use ndarray_linalg::Norm;

#[derive(Debug)]
pub struct EigenvalueResult {
    eigenvalue: f64,
    eigenvector: Array1<f64>,
    error: f64,
    steps: u32,
}

pub fn power_iteration(
    matrix: ArrayView2<f64>,
    vector: ArrayView1<f64>,
    epsilon: f64,
) -> EigenvalueResult {
    let mut prev = vector.to_owned();
    let mut next = matrix.dot(&prev);
    let mut eigenvalue = (next.dot(&next) / prev.dot(&prev)).sqrt();

    let mut error = posterior_error(prev.view(), next.view(), eigenvalue);
    let mut steps = 0;
    while error > epsilon {
        let buf = matrix.dot(&next);

        prev = next;
        next = buf;
        eigenvalue = (next.dot(&next) / prev.dot(&prev)).sqrt();

        error = posterior_error(prev.view(), next.view(), eigenvalue);
        steps += 1;
    }

    EigenvalueResult {
        eigenvalue,
        eigenvector: next,
        error,
        steps,
    }
}

pub fn dot_product_method(
    matrix: ArrayView2<f64>,
    vector: ArrayView1<f64>,
    epsilon: f64,
) -> EigenvalueResult {
    let matrix_t = matrix.t();

    let mut x_prev = vector.to_owned();
    let mut x_next = matrix.dot(&x_prev);

    let mut y_prev = vector.to_owned();
    let mut y_next = matrix_t.dot(&y_prev);

    let mut eigenvalue = (x_next.dot(&y_next)) / (x_prev.dot(&y_next));
    let mut error = posterior_error(x_prev.view(), x_next.view(), eigenvalue);
    let mut steps = 0;

    while error > epsilon {
        let x_buf = matrix.dot(&x_next);
        x_prev = x_next;
        x_next = x_buf;

        let y_buf = matrix_t.dot(&y_next);
        y_prev = y_next;
        y_next = y_buf;

        eigenvalue = (x_next.dot(&y_next)) / (x_prev.dot(&y_next));
        error = posterior_error(x_prev.view(), x_next.view(), eigenvalue);
        steps += 1;
    }

    EigenvalueResult {
        eigenvalue,
        eigenvector: x_next,
        error,
        steps,
    }
}

fn posterior_error(prev: ArrayView1<f64>, next: ArrayView1<f64>, eigenvalue: f64) -> f64 {
    (&next - eigenvalue * &prev).norm() / prev.norm()
}
