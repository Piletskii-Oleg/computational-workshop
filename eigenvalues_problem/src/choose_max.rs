use std::fmt::{Debug, Formatter};

use ndarray::{ArrayView1, ArrayView2};
use ndarray_linalg::Scalar;

pub trait ChooseMax: Debug {
    fn choose(&mut self, matrix: ArrayView2<f64>) -> (usize, usize);
    fn update(&mut self, pos: (usize, usize), matrix: ArrayView2<f64>);
}

pub struct MaxModule;

impl ChooseMax for MaxModule {
    fn choose(&mut self, matrix: ArrayView2<f64>) -> (usize, usize) {
        let n = matrix.nrows();

        let mut max = f64::MIN;
        let mut max_index = (0, 0);

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                if matrix[(i, j)].abs() > max {
                    max = matrix[(i, j)].abs();
                    //println!("max: {max:.2} at ({}, {})", i, j);
                    max_index = (i, j);
                }
            }
        }

        max_index
    }

    fn update(&mut self, _: (usize, usize), _: ArrayView2<f64>) {}
}

impl Debug for MaxModule {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MaxModule")
    }
}

pub struct CyclicChoice {
    i: usize,
    j: usize,
    n: usize,
}

impl CyclicChoice {
    pub fn new(matrix: ArrayView2<f64>) -> Self {
        Self {
            i: 0,
            j: 0,
            n: matrix.nrows(),
        }
    }
}

impl Debug for CyclicChoice {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CyclicChoice")
    }
}

impl ChooseMax for CyclicChoice {
    fn choose(&mut self, _: ArrayView2<f64>) -> (usize, usize) {
        self.j = (self.j + 1) % self.n;
        if self.j == 0 {
            self.i = (self.i + 1) % self.n;
        }

        if self.i == self.j {
            if self.i == self.n - 1 && self.j == self.n - 1 {
                self.i = 0;
                self.j = 1;
            } else {
                self.j = (self.j + 1) % self.n;
            }
        }

        //println!("{}, {}", self.i, self.j);
        (self.i, self.j)
    }

    fn update(&mut self, _: (usize, usize), _: ArrayView2<f64>) {}
}

pub struct OptimalElement {
    sums: Vec<f64>,
}

impl OptimalElement {
    fn sums(matrix: ArrayView2<f64>) -> Vec<f64> {
        (0..matrix.nrows())
            .map(|i| Self::sum(matrix.row(i), i))
            .collect()
    }

    fn sum(row: ArrayView1<f64>, i: usize) -> f64 {
        row.iter().map(|x| x.square()).sum::<f64>() - row[i].square()
    }

    pub fn new(matrix: ArrayView2<f64>) -> Self {
        Self {
            sums: Self::sums(matrix),
        }
    }
}

impl Debug for OptimalElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "OptimalElement")
    }
}

impl ChooseMax for OptimalElement {
    fn choose(&mut self, matrix: ArrayView2<f64>) -> (usize, usize) {
        self.sums = Self::sums(matrix);

        let row_index = self
            .sums
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();
        //println!("row sums: {:?}", self.sums);
        //println!("max row: {}", matrix.row(row_index));

        let element_index = matrix
            .row(row_index)
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != row_index)
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .map(|(index, _)| index)
            .unwrap();
        //println!("max element in row: {}", matrix[(row_index, element_index)]);

        //println!("{}, {}", row_index, element_index);
        (row_index, element_index)
    }

    fn update(&mut self, (i, j): (usize, usize), matrix: ArrayView2<f64>) {
        let i_row = Self::sum(matrix.row(i), i);
        let j_row = Self::sum(matrix.row(j), j);
        self.sums[i] = i_row;
        self.sums[j] = j_row;
    }
}
