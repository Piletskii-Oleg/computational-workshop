use std::rc::Rc;

use ndarray::Array1;
use ndarray_linalg::Solve;
use peroxide::fuga::Integral;
use pyo3::prelude::*;
use pyo3::Python;
use reikna::func;
use reikna::func::Function;

use grid_method::Equation;
use matrices::generate_matrix;

#[derive(Copy, Clone)]
pub struct OrthogonalFunction {
    pub function: fn(f64) -> f64,
    pub prime: fn(f64) -> f64,
    pub second_prime: fn(f64) -> f64,
}

pub fn solve_ritz(
    equation: Equation,
    orthogonal_system: Vec<OrthogonalFunction>,
) -> impl Fn(f64) -> f64 {
    let n = orthogonal_system.len();
    let matrix = generate_matrix(n, |i, j| {
        bilinear_form(&equation, (orthogonal_system[i], orthogonal_system[j]))
    });

    let vector = (0..n)
        .map(|index| {
            let integral = |x: f64| (equation.f)(x) * (orthogonal_system[index].function)(x);
            peroxide::numerical::integral::integrate(
                integral,
                (-1.0, 1.0),
                Integral::NewtonCotes(10000),
            )
        })
        .collect::<Array1<f64>>();

    let coefficients = matrix.solve(&vector).unwrap();

    move |x: f64| {
        coefficients
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (orthogonal_system[i].function)(x))
            .sum()
    }
}

pub fn solve_collocation(
    equation: Equation,
    roots: Vec<f64>,
    orthogonal_system: Vec<OrthogonalFunction>,
) -> impl Fn(f64) -> f64 {
    let n = orthogonal_system.len();
    let matrix = generate_matrix(n, |i, j| {
        let x = roots[i];
        (equation.p)(x) * (orthogonal_system[j].second_prime)(x)
            + (equation.q)(x) * (orthogonal_system[j].prime)(x)
            + (equation.r)(x) * (orthogonal_system[j].function)(x)
    });
    let vector = roots
        .iter()
        .map(|&root| (equation.f)(root))
        .collect::<Array1<f64>>();

    let coefficients = matrix.solve(&vector).unwrap();

    move |x: f64| {
        coefficients
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (orthogonal_system[i].function)(x))
            .sum()
    }
}

fn bilinear_form(equation: &Equation, (y, z): (OrthogonalFunction, OrthogonalFunction)) -> f64 {
    let q_l = if equation.alpha1 < f64::EPSILON || equation.alpha2 < f64::EPSILON {
        0.0
    } else {
        equation.alpha1 / equation.alpha2
            * (equation.p)(-1.0)
            * (y.function)(-1.0)
            * (z.function)(-1.0)
    };

    let q_r = if equation.alpha1 < f64::EPSILON || equation.alpha2 < f64::EPSILON {
        0.0
    } else {
        equation.alpha1 / equation.alpha2
            * (equation.p)(1.0)
            * (y.function)(1.0)
            * (z.function)(1.0)
    };

    let y_derivative = reikna::derivative::derivative(&func!(move |x| (y.function)(x)));
    let z_derivative = reikna::derivative::derivative(&func!(move |x| (z.function)(x)));

    let integral = |x: f64| {
        (equation.p)(x) * y_derivative(x) * z_derivative(x)
            + (equation.r)(x) * (y.function)(x) * (z.function)(x)
    };
    peroxide::numerical::integral::integrate(integral, (-1.0, 1.0), Integral::NewtonCotes(10000))
        + q_l
        + q_r
}

pub fn jacobi_function(i: usize, alpha: usize, beta: usize, x: f64) -> f64 {
    let x = Python::with_gil(|py| -> PyResult<f64> {
        let scipy = PyModule::import_bound(py, "scipy").unwrap();
        let x = scipy
            .getattr("special")
            .unwrap()
            .getattr("eval_jacobi")
            .unwrap()
            .call1((i, alpha, beta, x))
            .unwrap();
        Ok(x.extract().unwrap())
    });

    x.unwrap()
}
