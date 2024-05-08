use plotters::prelude::*;

use grid_method::{solve_grid, Equation};

fn main() {
    // Вариант 9
    let equation = Equation {
        p: |x: f64| -(6.0 + x) / (7.0 + 3.0 * x),
        q: |x: f64| -(1.0 - x / 2.0),
        r: |x: f64| 1.0 + x.cos() / 2.0,
        f: |x: f64| 1.0 - x / 3.0,
        a: -1.0,
        b: 1.0,
        alpha1: -2.0,
        alpha2: -1.0,
        alpha: 0.0,
        beta1: 0.0,
        beta2: 1.0,
        beta: 0.0,
    };

    let solution = solve_grid(equation, 2, 1e-6);

    let err_root_area = SVGBackend::new("task_7.svg", (600, 400)).into_drawing_area();
    err_root_area.fill(&WHITE).unwrap();

    let _min_error = solution
        .errors
        .iter()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap();
    let max_error = solution
        .errors
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();

    let min_size = solution.grid_sizes.iter().min().unwrap();
    let max_size = *solution.grid_sizes.iter().max().unwrap();

    let mut error_ctx = ChartBuilder::on(&err_root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Error to grid size ratio", ("sans-serif", 40))
        .build_cartesian_2d(
            (*min_size as u32..(max_size * 100) as u32).log_scale(),
            (0.0..max_error * 1.01).log_scale(),
        )
        .unwrap();
    error_ctx.configure_mesh().draw().unwrap();

    error_ctx
        .draw_series(LineSeries::new(
            solution
                .grid_sizes
                .iter()
                .map(|a| *a as u32)
                .zip(solution.errors.iter().cloned()),
            &GREEN,
        ))
        .unwrap();
}
