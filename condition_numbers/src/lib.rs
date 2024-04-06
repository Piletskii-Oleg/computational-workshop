use ndarray::{Array, Array1, Array2, ArrayView, Dimension};

pub fn euclidean_norm<D: Dimension>(matrix: &ArrayView<f64, D>) -> f64 {
    matrix.iter().map(|&a| a * a).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use crate::euclidean_norm;
    use assert_approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn euclidean_norm_test() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = euclidean_norm(&matrix.view());
        assert_approx_eq!(30.0_f64.sqrt(), norm, f64::EPSILON)
    }
}
