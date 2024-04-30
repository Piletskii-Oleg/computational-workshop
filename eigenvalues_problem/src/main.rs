use ndarray::{ArrayBase, Ix2, OwnedRepr};
use ndarray_linalg::{random_hermite, EigVals};

use eigenvalues_problem::{jacobi_method, Circles, MaxModule, OptimalElement};
use matrices::Examples;

fn main() {
    let matrix = random_hermite::<f64, OwnedRepr<f64>>(3);
    examine(matrix);

    let hilbert = Examples::hilbert(90);
    examine(hilbert);
}

fn examine(matrix: ArrayBase<OwnedRepr<f64>, Ix2>) {
    let epsilons = vec![1e-5, 1e-7, 1e-9];
    for epsilon in epsilons {
        println!("{epsilon}");
        println!("MaxModule");
        let eigenvalues = jacobi_method(matrix.view(), MaxModule, epsilon);
        println!("{eigenvalues:.2}");
        println!("{:.2?}", matrix.eigvals());
        let circles = Circles::new(matrix.view());
        println!("{circles}");

        println!("OptimalElement");
        let eigenvalues = jacobi_method(matrix.view(), OptimalElement::new(matrix.view()), epsilon);
        println!("{eigenvalues:.2}");
        println!("{:.2?}", matrix.eigvals());
        let circles = Circles::new(matrix.view());
        println!("{circles}");
    }
}
