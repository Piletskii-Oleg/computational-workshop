use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};
use ndarray_linalg::Norm;

#[derive(Debug)]
pub struct IterationResult {
    x: Array1<f64>,
    iteration_count: u32,
}

impl IterationResult {
    pub fn x(&self) -> ArrayView1<f64> {
        self.x.view()
    }

    pub fn iteration_count(&self) -> u32 {
        self.iteration_count
    }
}

pub fn iterate(matrix: ArrayView2<f64>, vector: ArrayView1<f64>, epsilon: f64) -> IterationResult {
    let n = matrix.nrows();

    let (b, c) = iterative_matrices(matrix, vector);

    let mut prev = Array1::zeros(n.f());
    let mut this = b.dot(&prev) + &c;

    let mut iteration_count = 0;
    while (&this - &prev).norm() > epsilon {
        let buf = b.dot(&this) + &c;
        prev = this;
        this = buf;
        iteration_count += 1;
    }

    IterationResult {
        x: this,
        iteration_count,
    }
}

pub fn seidel(matrix: ArrayView2<f64>, vector: ArrayView1<f64>, epsilon: f64) -> IterationResult {
    let n = matrix.nrows();

    let mut prev = Array1::from_vec(vec![0.0; n]);
    let mut next = Array1::from_vec(vec![1.0; n]);

    let mut iteration_count = 0;

    while (&prev - &next).norm() > epsilon {
        let buf = (0..n)
            .map(|i| {
                let first = (0..i).fold(0.0, |acc, j| {
                    acc + matrix[(i, j)] * next[j] / matrix[(i, i)]
                });
                let second = (i + 1..n).fold(0.0, |acc, j| {
                    acc + matrix[(i, j)] * prev[j] / matrix[(i, i)]
                });
                let third = vector[i] / matrix[(i, i)];
                -first - second + third
            })
            .collect();

        prev = next;
        next = buf;

        iteration_count += 1;
    }

    IterationResult {
        x: next,
        iteration_count,
    }
}

fn iterative_matrices(
    matrix: ArrayView2<f64>,
    vector: ArrayView1<f64>,
) -> (Array2<f64>, Array1<f64>) {
    let n = matrix.nrows();

    let b = matrices::generate_matrix(n, |row, column| {
        if row == column {
            0.0
        } else {
            -matrix[(row, column)] / matrix[(row, row)]
        }
    });

    let c = (0..n).map(|i| vector[i] / matrix[(i, i)]).collect();
    (b, c)
}
