mod condition;
mod examples;

pub use condition::*;
pub use examples::*;

use ndarray::{Array, ArrayView, ArrayView1, ArrayView2, Dimension};
use prettytable::{Cell, Row, Table};

pub fn euclidean_norm<D: Dimension>(matrix: ArrayView<f64, D>) -> f64 {
    matrix.iter().map(|&a| a * a).sum::<f64>().sqrt()
}

pub fn add_number<D: Dimension>(matrix: ArrayView<f64, D>, number: f64) -> Array<f64, D> {
    matrix.map(|&a| a + number)
}

pub fn print_matrix(matrix: ArrayView2<f64>) {
    let mut table = Table::new();
    matrix
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .map(|num| Cell::new(&format!("{:.4}", num)))
                .collect::<Vec<Cell>>()
        })
        .map(Row::new)
        .for_each(|row| {
            table.add_row(row);
        });
    table.printstd()
}

pub fn print_vector(vector: ArrayView1<f64>) {
    let mut table = Table::new();
    vector
        .iter()
        .map(|num| Cell::new(&format!("{:.4}", num)))
        .for_each(|cell| {
            table.add_row(Row::new(vec![cell]));
        });
    table.printstd()
}

#[cfg(test)]
mod tests {
    use crate::euclidean_norm;
    use assert_approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn euclidean_norm_test() {
        let vector = array![3.0, 4.0];
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];

        let vector_norm = euclidean_norm(vector.view());
        let matrix_norm = euclidean_norm(matrix.view());

        assert_approx_eq!(5.0_f64, vector_norm, f64::EPSILON);
        assert_approx_eq!(30.0_f64.sqrt(), matrix_norm, f64::EPSILON)
    }
}
