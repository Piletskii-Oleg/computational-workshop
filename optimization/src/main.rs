use ndarray::array;

use optimization::{minimize_gradient, minimize_heavy_ball, minimize_nesterov, Task};

fn main() {
    let task = Task {
        f: |x| x[0] * x[0] + x[1] * x[1] - 3.0,
        start_point: array![4.0, -1.0 / 4.0],
        alpha: 0.3,
        beta: 0.4,
    };

    let gradient = minimize_gradient(&task, 1e-6);
    let heavy_ball = minimize_heavy_ball(&task, 1e-6);
    let nesterov = minimize_nesterov(&task, 1e-9);
    println!("{:.8} in {} steps", gradient.min, gradient.steps);
    println!("{:.8} in {} steps", heavy_ball.min, heavy_ball.steps);
    println!("{:.10} in {} steps", nesterov.min, nesterov.steps);
}
