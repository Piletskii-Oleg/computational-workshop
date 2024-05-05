use ndarray::{Array1, Array2, ShapeBuilder};
use ndarray_linalg::{Norm, Scalar, Solve};

/// p(x)u'' + q(x)u' + r(x)u = f(x), a < x < b
///
/// alpha1 * u(a) - alpha2 * u'(a) = alpha
///
/// beta1 * u(b) + beta2 * u'(b) = beta
pub struct Equation {
    pub p: fn(f64) -> f64,
    pub q: fn(f64) -> f64,
    pub r: fn(f64) -> f64,
    pub f: fn(f64) -> f64,
    pub a: f64,
    pub b: f64,
    pub alpha1: f64,
    pub alpha2: f64,
    pub alpha: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub beta: f64,
}

#[derive(Debug)]
pub struct Solution {
    pub u: Array1<f64>,
    pub grid: Vec<f64>,
    pub errors: Vec<f64>,
    pub grid_sizes: Vec<usize>,
}

pub fn solve_grid(equation: Equation, mut n: usize, epsilon: f64) -> Solution {
    let mut errors = Vec::new();
    let mut grid_sizes = vec![n];

    let (A, b) = generate_matrix(&equation, n);
    let mut prev = A.solve(&b).unwrap();

    n *= 2;
    let (A, b) = generate_matrix(&equation, n);
    let mut next = A.solve(&b).unwrap();
    grid_sizes.push(n);

    let delta = delta(&prev, &next);
    next += &delta;
    let mut error = delta.norm_max();
    errors.push(error);
    while error > epsilon {
        prev = next;
        n *= 2;

        let (A, b) = generate_matrix(&equation, n);
        next = A.solve(&b).unwrap();

        let delta = crate::delta(&prev, &next);
        next += &delta;
        error = delta.norm_max();
        errors.push(error);
        grid_sizes.push(n)
    }

    Solution {
        u: next,
        grid: generate_grid((equation.a, equation.b), n),
        errors,
        grid_sizes,
    }
}

fn delta(prev: &Array1<f64>, next: &Array1<f64>) -> Array1<f64> {
    let mut prev_modified = Array1::default(next.len().f());

    for i in 0..prev.len() {
        prev_modified[2 * i] = prev[i];
        if i != prev.len() - 1 {
            prev_modified[2 * i + 1] = (prev[i + 1] - prev[i]) / 2.0;
        }
    }

    (next - prev_modified) / 3.0
}

fn generate_matrix(equation: &Equation, n: usize) -> (Array2<f64>, Array1<f64>) {
    let n = n + 1;
    let grid = generate_grid((equation.a, equation.b), n);
    let h = (equation.a - equation.b).abs() / (n - 1) as f64;

    let mut matrix = Array2::default((n, n).f());
    let mut vector = Array1::default(n.f());

    matrix[(0, 0)] = h * equation.alpha1 + 3.0 * equation.alpha2 / 2.0;
    matrix[(0, 1)] = -2.0 * equation.alpha2;
    matrix[(0, 2)] = 1.0 * equation.alpha2 / 2.0;
    vector[0] = h * equation.alpha;

    for i in 1..n - 1 {
        let x = grid[i];
        matrix[(i, i - 1)] = (equation.p)(x) - (equation.q)(x) * h / 2.0;
        matrix[(i, i + 1)] = (equation.p)(x) + (equation.q)(x) * h / 2.0;
        matrix[(i, i)] = (matrix[(i, i - 1)] + matrix[(i, i + 1)]) - (equation.r)(x) * h.square();
        vector[i] = (equation.f)(x) * h.square();
    }

    matrix[(n - 1, n - 3)] = equation.beta2 / 2.0;
    matrix[(n - 1, n - 2)] = -2.0 * equation.beta2;
    matrix[(n - 1, n - 1)] = h * equation.beta1 + 3.0 * equation.beta2 / 2.0;
    vector[n - 1] = h * equation.beta;

    (matrix, vector)
}

fn generate_grid((a, b): (f64, f64), n: usize) -> Vec<f64> {
    Array1::linspace(a, b, n + 1).to_vec()
}
