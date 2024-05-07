use ndarray::array;
use ndarray_linalg::Scalar;

use optimization::{
    minimize_gradient, minimize_heavy_ball, minimize_nesterov, minimize_newton, Answer, Task,
};

const EPSILONS: [f64; 5] = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12];

fn main() {
    let task = Task {
        f: |x| x[0] * x[0] + x[1] * x[1],
        start_point: array![1.0, -1.0 / 4.0],
        alpha: 0.1,
        beta: 0.4,
    };
    examine(task);

    println!("----------------------------------");

    let task = Task {
        f: |x| x[0].cos() * x[1].sin().square(),
        start_point: array![1.0, -1.0],
        alpha: 0.1,
        beta: 0.4,
    };
    examine(task);
}

fn examine(task: Task) {
    for epsilon in EPSILONS {
        println!("Current epsilon: {epsilon:e}");
        let gradient = minimize_gradient(&task, epsilon);
        let heavy_ball = minimize_heavy_ball(&task, epsilon);
        let nesterov = minimize_nesterov(&task, epsilon);
        let newton = minimize_newton(&task, epsilon);

        examine_answer(gradient, epsilon);
        examine_answer(heavy_ball, epsilon);
        examine_answer(nesterov, epsilon);
        examine_answer(newton, epsilon);
    }
}

fn examine_answer(answer: Answer, epsilon: f64) {
    println!(
        "{:?}: {:.*} in {} steps",
        answer.method,
        -epsilon.log10() as usize + 2,
        answer.min,
        answer.steps
    );
}
