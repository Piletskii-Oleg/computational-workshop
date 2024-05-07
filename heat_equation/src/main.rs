use std::f64::consts::PI;

use ndarray_linalg::Scalar;
use plotters::prelude::*;

use heat_equation::{solve_explicit, solve_implicit, Equation, Solution};

fn main() {
    let equation = Equation {
        kappa: 1.0 / PI.square(),
        f: |(x, t)| 0.0,
        a: 1.0,
        T: 1.0,
        mu: |t| (PI * t).sin() / PI.square(),
        mu1: |_| 0.0,
        mu2: |_| 0.0,
    };

    let explicit = solve_explicit(&equation, 15, 30);
    let implicit = solve_implicit(&equation, 20, 30);
    draw_solution("explicit_solution", explicit);
    draw_solution("implicit_solution", implicit);
}

fn draw_solution(file: &str, solution: Solution) {
    println!("{}", solution.approximate_solution);
    let path = format!("{file}.png");
    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(file, ("sans-serif", 40))
        .build_cartesian_3d(0.0..1.0, 0.0..0.3, 0.0..1.0)
        .unwrap();

    chart.with_projection(|mut pb| {
        pb.pitch = 0.4;
        pb.yaw = 2.4;
        pb.scale = 0.9;
        pb.into_matrix()
    });

    chart.configure_axes().draw().unwrap();

    let n = solution.approximate_solution.ncols();
    let m = solution.approximate_solution.nrows();
    let mut data = vec![];

    for i in (0..n) {
        let mut row = vec![];
        for j in (0..m) {
            row.push((
                solution.x_grid[i],
                solution.approximate_solution[(j, i)],
                solution.t_grid[j],
            ));
        }
        data.push(row);
    }

    chart
        .draw_series(
            (0..n - 1)
                .map(|x| std::iter::repeat(x).zip(0..m - 1))
                .flatten()
                .map(|(x, z)| {
                    Polygon::new(
                        vec![
                            data[x][z],
                            data[x + 1][z],
                            data[x + 1][z + 1],
                            data[x][z + 1],
                        ],
                        &BLUE.mix(0.3),
                    )
                }),
        )
        .unwrap();
}
