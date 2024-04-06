use condition_numbers::ConditionNumbers;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix = array![[-1.0, 2.0], [2.0, -1.0]];
    let numbers = ConditionNumbers::new(matrix.view())?;
    println!("{}", numbers);
    Ok(())
}
