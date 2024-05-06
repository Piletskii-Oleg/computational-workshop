use std::f64::consts::PI;
use std::rc::Rc;

use ndarray_linalg::Scalar;
use plotters::prelude::full_palette::GREY;
use plotters::prelude::*;
use plotters::style::full_palette::BROWN;

use grid_method::Equation;
use projection_methods::{
    jacobi_function, solve_collocation, solve_galerkin, solve_ritz, OrthogonalFunction,
};

fn main() {
    pyo3::prepare_freethreaded_python();

    // Вариант 2
    let equation = Equation {
        p: |x: f64| (2.0 + x) / (3.0 + x),
        q: |x: f64| -1.0 / ((3.0 + x).square()),
        r: |x: f64| 1.0 + x.sin(),
        f: |x: f64| 1.0 - x,
        a: 0.0,
        b: 0.0,
        alpha1: 0.0,
        alpha2: -1.0,
        alpha: 0.0,
        beta1: 1.0,
        beta2: 1.0,
        beta: 0.0,
    };

    //draw_ritz(&equation);
    draw_collocation(&equation);
    draw_galerkin(&equation);
}

fn draw_galerkin(equation: &Equation) {
    let function_root_area = SVGBackend::new("task_8_galerkin.svg", (600, 400)).into_drawing_area();
    function_root_area.fill(&WHITE).unwrap();

    let mut galerkin_ctx = ChartBuilder::on(&function_root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Ritz Method", ("sans-serif", 40))
        .build_cartesian_2d(-1.0..1.0, 0.0..2.0)
        .unwrap();
    galerkin_ctx.configure_mesh().draw().unwrap();

    for (n, color) in [2, 4, 8, 16, 32, 64]
        .into_iter()
        .zip([GREY, GREEN, RED, BROWN, BLUE, BLACK])
    {
        let galerkin = solve_galerkin(&equation, orthogonal_system(n));
        galerkin_ctx
            .draw_series(LineSeries::new(
                (-10000..=10000)
                    .map(|x| x as f64 * 0.0001)
                    .map(|x| (x, galerkin(x))),
                &color,
            ))
            .unwrap()
            .label(&format!("{n}"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    galerkin_ctx
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();
}

fn draw_ritz(equation: &Equation) {
    let function_root_area = SVGBackend::new("task_8_ritz.svg", (600, 400)).into_drawing_area();
    function_root_area.fill(&WHITE).unwrap();

    let mut ritz_ctx = ChartBuilder::on(&function_root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Ritz Method", ("sans-serif", 40))
        .build_cartesian_2d(-1.0..1.0, 0.0..2.0)
        .unwrap();
    ritz_ctx.configure_mesh().draw().unwrap();

    for (n, color) in [2, 4, 8, 16, 32, 64]
        .into_iter()
        .zip([GREY, GREEN, RED, BROWN, BLUE, BLACK])
    {
        let ritz = solve_ritz(&equation, orthogonal_system(n));
        ritz_ctx
            .draw_series(LineSeries::new(
                (-10000..=10000)
                    .map(|x| x as f64 * 0.0001)
                    .map(|x| (x, ritz(x))),
                &color,
            ))
            .unwrap()
            .label(&format!("{n}"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    ritz_ctx
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();
}

fn draw_collocation(equation: &Equation) {
    let function_root_area =
        SVGBackend::new("task_8_collocation.svg", (600, 400)).into_drawing_area();
    function_root_area.fill(&WHITE).unwrap();

    let mut collocation_ctx = ChartBuilder::on(&function_root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Collocation Method", ("sans-serif", 40))
        .build_cartesian_2d(-1.0..1.0, 0.0..2.0)
        .unwrap();
    collocation_ctx.configure_mesh().draw().unwrap();

    for (n, color) in [64].into_iter().zip([BLACK]) {
        let collocation = solve_collocation(equation, roots(n), orthogonal_system(n));
        collocation_ctx
            .draw_series(LineSeries::new(
                (-10000..=10000)
                    .map(|x| x as f64 * 0.0001)
                    .map(|x| (x, collocation(x))),
                &color,
            ))
            .unwrap()
            .label(&format!("{n}"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    collocation_ctx
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();
}

fn roots(n: usize) -> Vec<f64> {
    (1..=n)
        .map(|i| ((2.0 * (i as f64) - 1.0) / (2.0 * (n as f64)) * PI).cos())
        .collect()
}

fn orthogonal_system(n: usize) -> Vec<OrthogonalFunction> {
    let first = OrthogonalFunction {
        function: Rc::new(|_| 1.0),
        prime: Rc::new(|_| 0.0),
        second_prime: Box::new(|_| 0.0),
    };

    let second = OrthogonalFunction {
        function: Rc::new(|x: f64| x),
        prime: Rc::new(|_| 1.0),
        second_prime: Box::new(|_| 0.0),
    };

    let mut orthogonal_system = vec![first, second];
    for i in 0..n - 2 {
        orthogonal_system.push(jacobi(i));
    }
    orthogonal_system
}

fn jacobi(i: usize) -> OrthogonalFunction {
    OrthogonalFunction {
        function: Rc::new(move |x: f64| (1.0 - x * x) * jacobi_function(i, 1, 1, x)),
        prime: Rc::new(move |x: f64| -2.0 * ((i + 1) as f64) * jacobi_function(i + 1, 0, 0, x)),
        second_prime: Box::new(move |x: f64| {
            -(((i + 1) * (i + 2)) as f64) * jacobi_function(i, 1, 1, x)
        }),
    }
}
