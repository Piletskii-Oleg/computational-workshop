use ndarray::array;
use ndarray_linalg::Solve;

use matrices::Examples;

fn main() {
    let mut matrix = array![[3, 4, 1], [5, 5, 1], [6, 6, 9]].mapv(|value| value as f64);

    for i in 0..matrix.nrows() {
        matrix[(i, i)] += 40.0;
    }

    let vector = Examples::random_vector(3);

    let x = iterative_methods::iterate(matrix.view(), vector.view(), 10e-9);
    let x_correct = matrix.solve(&vector).unwrap();
    println!(
        "{} in {} steps\ncorrect: {x_correct}",
        x.x(),
        x.iteration_count()
    );
}
