use std::error::Error;
use std::fmt::{Display, Formatter};

use ndarray::{ArrayView2, Axis};
use ndarray_linalg::{Determinant, Inverse, Norm};
use prettytable::{row, Row, Table};

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

    pub fn condition_table(&self) -> Table {
        let mut table = Table::new();
        table.add_row(row!["spectre", "volume", "angle"]);
        table.add_row((*self).into());
        table
    }
}

impl Display for ConditionNumbers {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Спектральное число обусловленности: {:.3}\nОбъемное число обусловленности: {:.3}\nУгловое число обусловленности: {:.3}",
        self.spectre, self.volume, self.angle)
    }
}

impl From<ConditionNumbers> for Row {
    fn from(value: ConditionNumbers) -> Self {
        row![
            format!("{:.3}", value.spectre),
            format!("{:.3}", value.volume),
            format!("{:.3}", value.angle)
        ]
    }
}

fn spectre_criterion(matrix: ArrayView2<f64>) -> Result<f64, Box<dyn Error>> {
    let inverse = matrix.inv()?;
    let spectre = matrix.norm() * inverse.view().norm();
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
        .map(|(row, inv)| row.norm() * inv.norm())
        .max_by(|a, b| a.total_cmp(b))
        .ok_or(
            "angle criterion - something went wrong... probably NaN found"
                .to_string()
                .into(),
        )
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use ndarray::array;
    use ndarray_linalg::Determinant;

    use crate::condition::{angle_criterion, volume_criterion};

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
