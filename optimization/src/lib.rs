use finitediff::FiniteDiff;
use ndarray::Array1;
use ndarray_linalg::{Inverse, Norm};

#[derive(Debug)]
pub enum Method {
    Gradient,
    HeavyBall,
    Nesterov,
    Newton,
}

pub struct Task {
    pub f: fn(&Array1<f64>) -> f64,
    pub start_point: Array1<f64>,
    pub alpha: f64,
    pub beta: f64,
}

pub struct Answer {
    pub method: Method,
    pub min: Array1<f64>,
    pub points: Vec<Array1<f64>>,
    pub steps: u32,
}

pub fn minimize_gradient(task: &Task, epsilon: f64) -> Answer {
    let mut x = task.start_point.clone();
    let mut steps = 0;
    let mut points = vec![x.clone()];
    while x.forward_diff(&task.f).norm_max() > epsilon {
        x = &x - task.alpha * x.forward_diff(&task.f);
        steps += 1;
        points.push(x.clone());
    }

    Answer {
        method: Method::Gradient,
        min: x,
        points,
        steps,
    }
}

pub fn minimize_heavy_ball(task: &Task, epsilon: f64) -> Answer {
    let mut prev = task.start_point.clone();
    let mut next = &prev - task.alpha * prev.forward_diff(&task.f);

    let mut points = vec![prev.clone(), next.clone()];
    let mut steps = 0;

    while next.forward_diff(&task.f).norm_max() > epsilon {
        steps += 1;
        let x = &next - task.alpha * next.forward_diff(&task.f) + task.beta * (&next - &prev);
        prev = next;
        next = x;
        points.push(next.clone());
    }

    Answer {
        method: Method::HeavyBall,
        min: next,
        points,
        steps,
    }
}

pub fn minimize_nesterov(task: &Task, epsilon: f64) -> Answer {
    let mut steps = 0;
    let mut x_prev = task.start_point.clone();
    let mut y_prev = x_prev.clone();
    steps += 1;

    let mut x_next = &y_prev - task.alpha * y_prev.forward_diff(&task.f);
    let mut y_next = &x_next + task.beta * (&x_next - &x_prev);
    steps += 1;

    let mut points = vec![
        x_prev.clone(),
        y_prev.clone(),
        x_next.clone(),
        y_next.clone(),
    ];

    while y_next.forward_diff(&task.f).norm_max() > epsilon {
        let x = &y_prev - task.alpha * y_prev.forward_diff(&task.f);
        let y = &x + task.beta * (&x_next - &x_prev);

        x_prev = x_next;
        y_prev = y_next;

        x_next = x;
        y_next = y;

        points.push(x_next.clone());
        points.push(y_next.clone());
        steps += 2;
    }

    Answer {
        method: Method::Nesterov,
        min: y_next,
        points,
        steps,
    }
}

pub fn minimize_newton(task: &Task, epsilon: f64) -> Answer {
    let mut x = task.start_point.clone();
    let mut steps = 0;
    let mut points = vec![x.clone()];
    while x.forward_diff(&task.f).norm_max() > epsilon {
        x = &x
            - x.forward_hessian_nograd(&task.f)
                .inv()
                .unwrap()
                .dot(&x.forward_diff(&task.f));
        steps += 1;
        points.push(x.clone());
    }

    Answer {
        method: Method::Newton,
        min: x,
        points,
        steps,
    }
}
