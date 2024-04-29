use ndarray::ArrayView2;
use ndarray_linalg::Scalar;

pub trait ChooseMax {
    fn choose(&mut self, matrix: ArrayView2<f64>) -> (usize, usize);
    fn update(&mut self, (i, j): (usize, usize), matrix: ArrayView2<f64>) {}
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
                    max_index = (i, j);
                }
            }
        }

        max_index
    }
}

pub struct OptimalElement {
    sums: Vec<f64>,
}

impl OptimalElement {
    fn sums(matrix: ArrayView2<f64>) -> Vec<f64> {
        (0..matrix.nrows())
            .map(|i| matrix.row(i).map(|&x| x.square()).sum() - matrix[(i, i)].square())
            .collect()
    }

    pub fn new(matrix: ArrayView2<f64>) -> Self {
        Self {
            sums: Self::sums(matrix),
        }
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

        let element_index = matrix
            .row(row_index)
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != row_index)
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .map(|(index, _)| index)
            .unwrap();

        (row_index, element_index)
    }

    fn update(&mut self, (i, j): (usize, usize), matrix: ArrayView2<f64>) {
        let i_row = matrix.row(i).map(|&x| x.square()).sum() - matrix[(i, i)].square();
        let j_row = matrix.row(j).map(|&x| x.square()).sum() - matrix[(j, j)].square();
        self.sums[i] = i_row;
        self.sums[j] = j_row;
    }
}
