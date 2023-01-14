use std::collections::HashMap;

use itertools::Itertools;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;

fn generate_gird_random(width: usize, height: usize, weights: Vec<usize>) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(&weights).unwrap();

    let grid: Vec<Vec<usize>> = (0..height)
        .map(|_| (0..width).map(|_| [0, 1][dist.sample(&mut rng)]).collect())
        .collect();
    grid
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum State {
    Wall,
    Empty,
    Point,
}

#[derive(Debug)]
enum Action {
    GoUp,
    GoDown,
    GoRirht,
    GoLeft,
    TakePoint,
}

fn get_random_action() -> Action {
    match rand::thread_rng().gen_range(0..=4) {
        // rand 0.8
        0 => Action::GoUp,
        1 => Action::GoDown,
        2 => Action::GoRirht,
        3 => Action::GoLeft,
        _ => Action::TakePoint,
    }
}
fn generate_strategy() -> HashMap<(State, State, State, State, State), Action> {
    /*0: above
    1: right
    2: below
    3: left
    4: current*/
    let mut strategy: HashMap<(State, State, State, State, State), Action> = HashMap::new();

    let possible_states = vec![State::Wall, State::Empty, State::Point];
    let all_states = possible_states.iter().combinations_with_replacement(5);

    let mut all_possible_states: Vec<Vec<&State>> = Vec::new();

    all_states.for_each(|state| match state.as_slice() {
        [State::Wall, _, State::Wall, _, _] => {}
        [_, _, _, _, State::Wall] => {}
        [_, State::Wall, _, State::Wall, _] => {}
        _ => all_possible_states.push(state.to_owned()),
    });

    all_possible_states.iter().for_each(|state| {
        strategy.insert(
            (*state[0], *state[1], *state[2], *state[3], *state[4]),
            get_random_action(),
        );
    });
    strategy
}
struct Robot {
    points: isize,
    grid: Vec<Vec<usize>>,
}

fn main() {
    println!("{:?}", generate_strategy());
}
