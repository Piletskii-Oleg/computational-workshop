use finitediff::FiniteDiff;
use ndarray::Array1;
use ndarray_linalg::Norm;

pub struct Task {
    pub f: fn(&Array1<f64>) -> f64,
    pub start_point: Array1<f64>,
    pub alpha: f64,
}

#[derive(Debug)]
pub struct Answer {
    pub min: Array1<f64>,
    pub start_point: Array1<f64>,
    pub steps: u32,
}

pub fn minimize_gradient(task: Task, epsilon: f64) -> Answer {
    let mut x = task.start_point.clone();
    let mut steps = 0;
    while x.forward_diff(&task.f).norm() > epsilon {
        x = &x - task.alpha * x.forward_diff(&task.f);
        steps += 1;
    }

    Answer {
        min: x,
        start_point: task.start_point,
        steps,
    }
}
