use crate::euclidean_norm;
use ndarray::{ArrayView2, Axis};
use ndarray_linalg::{Determinant, Inverse};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ConditionNumbers {
    spectre: f64,
    volume: f64,
    angle: f64,
}

impl ConditionNumbers {
    pub fn new(matrix: ArrayView2<f64>) -> Result<Self, Box<dyn Error>> {
        let spectre = spectre_criterion(matrix)?;
        let volume = volume_criterion(matrix)?;
        let angle = angle_criterion(matrix)?;
        Ok(Self {
            spectre,
            volume,
            angle,
        })
    }
}

impl Display for ConditionNumbers {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Спектральное число обусловленности: {:.3}\nОбъемное число обусловленности: {:.3}\nУгловое число обусловленности: {:.3}",
        self.spectre, self.volume, self.angle)
    }
}

fn spectre_criterion(matrix: ArrayView2<f64>) -> Result<f64, Box<dyn Error>> {
    let inverse = matrix.inv()?;
    let spectre = euclidean_norm(matrix) * euclidean_norm(inverse.view());
    Ok(spectre)
}

fn volume_criterion(matrix: ArrayView2<f64>) -> Result<f64, Box<dyn Error>> {
    let det = matrix.det()?;
    Ok(matrix
        .axis_iter(Axis(0))
        .map(|row| row.iter().map(|a| a * a).sum::<f64>().sqrt())
        .product::<f64>()
        / det.abs())
}

fn angle_criterion(matrix: ArrayView2<f64>) -> Result<f64, Box<dyn Error>> {
    let inverse = matrix.inv()?;
    matrix
        .axis_iter(Axis(0))
        .zip(inverse.axis_iter(Axis(0)))
        .map(|(row, inv)| euclidean_norm(row) * euclidean_norm(inv))
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(
            "angle criterion - something went wrong... probably NaN found"
                .to_string()
                .into(),
        )
}

#[cfg(test)]
mod tests {
    use crate::condition::{angle_criterion, spectre_criterion, volume_criterion};
    use assert_approx_eq::assert_approx_eq;
    use ndarray::array;
    use ndarray_linalg::Determinant;

    #[test]
    fn ortega_test() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let det: f64 = matrix.det().unwrap();
        let det = det.abs();

        let ortega = volume_criterion(matrix.view()).unwrap();
        assert_approx_eq!(5.0 * 5.0_f64.sqrt() / det, ortega, f64::EPSILON)
    }

    #[test]
    fn angle_test() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let _inverse = array![[-2.0, 1.0], [1.5, -0.5]];

        let angle = angle_criterion(matrix.view()).unwrap();
        assert_approx_eq!(angle, 5.0 * 2.5_f64.sqrt(), f64::EPSILON)
    }
}
