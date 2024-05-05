use std::fmt::{Display, Formatter};

use ndarray::ArrayView2;

struct Circle {
    center: f64,
    radius: f64,
}

pub struct Circles(Vec<Circle>);

impl Circles {
    pub fn new(matrix: ArrayView2<f64>) -> Self {
        let n = matrix.nrows();

        let centers = (0..n).map(|i| matrix[(i, i)]).collect::<Vec<_>>();
        let radii = Self::radii(matrix);

        let circles = centers
            .into_iter()
            .zip(radii.iter())
            .map(|(center, &radius)| Circle { center, radius })
            .collect();
        Self(circles)
    }

    fn radii(matrix: ArrayView2<f64>) -> Vec<f64> {
        let n = matrix.nrows();
        (0..n)
            .map(|i| matrix.row(i).map(|x| x.abs()).sum() - matrix[(i, i)])
            .collect()
    }
}

impl Display for Circles {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let print = self
            .0
            .iter()
            .map(|circle| {
                let left = circle.center - circle.radius;
                let right = circle.center + circle.radius;
                format!("{left:.2} <= z <= {right:.2}\n")
            })
            .collect::<Vec<_>>()
            .concat();
        write!(f, "{print}")
    }
}
