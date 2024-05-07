use ndarray::array;

use optimization::{minimize_gradient, Task};

fn main() {
    let task = Task {
        f: |x| x[0] * x[0] + x[1] * x[1],
        start_point: array![1.0, -1.0 / 4.0],
        alpha: 0.3,
    };

    let gradient = minimize_gradient(task, 1e-6);
    println!("{gradient:.8?}");
}
