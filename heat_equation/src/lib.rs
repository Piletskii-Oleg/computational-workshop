use ndarray::{Array1, Array2, ShapeBuilder};
use ndarray_linalg::{Scalar, Solve};

/// u_t(x, t) = kappa * u_xx(x, t) + f(x, t)
///
/// k > 0, 0 < x < a, 0 < t <= T
///
/// u(x, 0) = mu(x), 0 <= x <= a
///
/// u(0, t) = mu1(t), u(a, t) = mu2(t), 0 <= t <= T
pub struct Equation {
    pub kappa: f64,
    pub f: fn((f64, f64)) -> f64,
    pub a: f64,
    pub T: f64,
    pub mu: fn(f64) -> f64,
    pub mu1: fn(f64) -> f64,
    pub mu2: fn(f64) -> f64,
}

pub struct Solution {
    pub x_grid: Array1<f64>,
    pub t_grid: Array1<f64>,
    pub approximate_solution: Array2<f64>,
}

pub fn solve_explicit(equation: &Equation, n: usize, m: usize) -> Solution {
    let x_grid = Array1::linspace(0.0, equation.a, n + 1);
    let t_grid = Array1::linspace(0.0, equation.T, m + 1);

    let h = equation.a / n as f64;
    let tau = equation.T / m as f64;

    let mut solution = Array2::zeros((m + 1, n + 1).f());

    for i in 0..n + 1 {
        solution[(0, i)] = (equation.mu)(x_grid[i]);
    }

    for j in 1..m + 1 {
        solution[(j, 0)] = (equation.mu1)(t_grid[j]);
        solution[(j, n)] = (equation.mu2)(t_grid[j]);
    }

    for j in 1..m + 1 {
        for i in 1..n {
            let c = tau * equation.kappa / (h * h);

            solution[(j, i)] = c * solution[(j - 1, i - 1)]
                + (1.0 - 2.0 * c) * solution[(j - 1, i)]
                + c * solution[(j - 1, i + 1)]
                + tau * (equation.f)((x_grid[i], t_grid[j - 1]));
        }
    }

    Solution {
        x_grid,
        t_grid,
        approximate_solution: solution,
    }
}

pub fn solve_implicit(equation: &Equation, n: usize, m: usize) -> Solution {
    let x_grid = Array1::linspace(0.0, equation.a, n + 1);
    let t_grid = Array1::linspace(0.0, equation.T, m + 1);

    let _h = equation.a / n as f64;
    let tau = equation.T / m as f64;

    let mut solution = Array2::zeros((m + 1, n + 1).f());

    for i in 0..n + 1 {
        solution[(0, i)] = (equation.mu)(x_grid[i]);
    }

    for j in 1..m + 1 {
        solution[(j, 0)] = (equation.mu1)(t_grid[j]);
        solution[(j, n)] = (equation.mu2)(t_grid[j]);
    }

    let a_matrix = local_matrix(equation, n, m);

    for j in 1..m + 1 {
        let mut vector = Array1::zeros((n + 1).f());
        vector[0] = (equation.mu1)(t_grid[j]);
        for i in 1..n {
            vector[i] = solution[(j - 1, i)] + tau * (equation.f)((x_grid[i], t_grid[j]));
        }
        vector[n] = (equation.mu2)(t_grid[j]);

        let approx_vector = a_matrix.solve(&vector).unwrap();
        for i in 0..x_grid.len() {
            solution[(j, i)] = approx_vector[i];
        }
    }

    Solution {
        x_grid,
        t_grid,
        approximate_solution: solution,
    }
}

fn local_matrix(equation: &Equation, n: usize, m: usize) -> Array2<f64> {
    let h = equation.a / n as f64;
    let tau = equation.T / m as f64;

    let mut matrix = Array2::zeros((n + 1, n + 1).f());

    matrix[(0, 0)] = 1.0;
    for i in 1..n {
        matrix[(i, i - 1)] = -tau * equation.kappa / (h.square());
        matrix[(i, i)] = 2.0 * tau * equation.kappa / (h.square()) + 1.0;
        matrix[(i, i + 1)] = -tau * equation.kappa / (h.square());
    }
    matrix[(n, n)] = 1.0;

    matrix
}
