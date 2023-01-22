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
    GoRight,
    GoLeft,
    TakePoint,
}

fn get_random_action() -> Action {
    match rand::thread_rng().gen_range(0..=4) {
        0 => Action::GoUp,
        1 => Action::GoDown,
        2 => Action::GoRight,
        3 => Action::GoLeft,
        _ => Action::TakePoint,
    }
}
// TODO not all possible states are generated
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
fn display_strategy(strategy: &HashMap<(State, State, State, State, State), Action>) {
    for (k, v) in strategy {
        println!("{:?} {:?}", k, v)
    }
}

struct Robot {
    points: isize,
    grid: Vec<Vec<usize>>,
    x: usize,
    y: usize,
}

impl Robot {
    fn new(grid: Vec<Vec<usize>>, x: Option<usize>, y: Option<usize>) -> Robot {
        Robot {
            points: 0,
            grid,
            x: x.unwrap_or(0),
            y: y.unwrap_or(0),
        }
    }

    fn get_state(&self) -> (State, State, State, State, State) {
        let up: State = if self.y + 1 > self.grid.len() {
            State::Wall
        } else {
            match self.grid[self.y + 1][self.x] {
                1 => State::Point,
                _ => State::Empty,
            }
        };

        let down: State = if self.y as isize - 1 < 0 {
            State::Wall
        } else {
            match self.grid[self.y + 1][self.x] {
                1 => State::Point,
                _ => State::Empty,
            }
        };

        let right: State = if self.x + 1 > self.grid[0].len() {
            State::Wall
        } else {
            match self.grid[self.y + 1][self.x] {
                1 => State::Point,
                _ => State::Empty,
            }
        };

        let left: State = if self.x as isize - 1 < 0 {
            State::Wall
        } else {
            match self.grid[self.y + 1][self.x] {
                1 => State::Point,
                _ => State::Empty,
            }
        };

        let current = match self.grid[self.y][self.x] {
            1 => State::Point,
            _ => State::Empty,
        };
        (up, right, down, left, current)
    }

    fn play_strategy(
        &mut self,
        strategy: HashMap<(State, State, State, State, State), Action>,
        steps: usize,
    ) {
        for _ in 0..steps {
            let state = self.get_state();

            println!("State {:?}", state);

            let action = strategy.get(&state);
            match action.unwrap() {
                Action::GoUp => self.y += 1,
                Action::GoDown => self.y -= 1,
                Action::GoLeft => self.x -= 1,
                Action::GoRight => self.x += 1,
                Action::TakePoint => {
                    self.points += 1;
                    self.grid[self.x][self.y] = 0;
                }
            }
        }
    }
}

fn main() {
    let grid = generate_gird_random(500, 500, vec![3, 7]);
    let mut robot = Robot::new(grid, Some(0), Some(0));
    let strategy = generate_strategy();

    display_strategy(&strategy);
    robot.play_strategy(strategy, 100);

    println!("Points: {}", robot.points);
}
