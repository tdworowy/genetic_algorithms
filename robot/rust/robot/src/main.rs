use rand::distributions::WeightedIndex;
use rand::prelude::*;


fn generate_gird_random(width: usize, height: usize, weights: Vec<usize>) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(&weights).unwrap();

    let grid: Vec<Vec<usize>> = (0..height)
        .map(|_| (0..width).map(|_| [0, 1][dist.sample(&mut rng)]).collect())
        .collect();
    grid
}

fn main() {
    println!("{:?}", generate_gird_random(4, 4, vec![6, 4]));
}
