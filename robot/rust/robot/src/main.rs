use std::collections::HashMap;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;

//from https://stackoverflow.com/questions/71420176/permutations-with-replacement-in-rust
pub struct PermutationsReplacementIter<I> {
    items: Vec<I>,
    permutation: Vec<usize>,
    group_len: usize,
    finished: bool,
}

impl<I: Copy> PermutationsReplacementIter<I> {
    fn increment_permutation(&mut self) -> bool {
        let mut idx = 0;

        loop {
            if idx >= self.permutation.len() {
                return true;
            }

            self.permutation[idx] += 1;

            if self.permutation[idx] >= self.items.len() {
                self.permutation[idx] = 0;
                idx += 1;
            } else {
                return false;
            }
        }
    }

    fn build_vec(&self) -> Vec<I> {
        let mut vec = Vec::with_capacity(self.group_len);

        for idx in &self.permutation {
            vec.push(self.items[*idx]);
        }

        vec
    }
}

impl<I: Copy> Iterator for PermutationsReplacementIter<I> {
    type Item = Vec<I>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let item = self.build_vec();

        if self.increment_permutation() {
            self.finished = true;
        }

        Some(item)
    }
}

pub trait ToPermutationsWithReplacement {
    type Iter;
    fn permutations_with_replacement(self, group_len: usize) -> Self::Iter;
}

impl<I: Iterator> ToPermutationsWithReplacement for I {
    type Iter = PermutationsReplacementIter<<I as Iterator>::Item>;

    fn permutations_with_replacement(self, group_len: usize) -> Self::Iter {
        let items = self.collect::<Vec<_>>();
        PermutationsReplacementIter {
            permutation: vec![0; group_len],
            group_len,
            finished: group_len == 0 || items.len() == 0,
            items,
        }
    }
}

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

fn generate_strategy() -> HashMap<(State, State, State, State, State), Action> {
    /*0: above
    1: right
    2: below
    3: left
    4: current*/
    let mut strategy: HashMap<(State, State, State, State, State), Action> = HashMap::new();

    let possible_states = vec![State::Wall, State::Empty, State::Point];
    let all_states = possible_states.iter().permutations_with_replacement(5);

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
    width: usize,
    height: usize,
    x: usize,
    y: usize,
}

impl Robot {
    fn new(
        grid: Vec<Vec<usize>>,
        width: usize,
        height: usize,
        x: Option<usize>,
        y: Option<usize>,
    ) -> Robot {
        Robot {
            points: 0,
            grid,
            width,
            height,
            x: x.unwrap_or(0),
            y: y.unwrap_or(0),
        }
    }

    fn get_state(&self) -> (State, State, State, State, State) {
        let up: State = if self.y + 1 > self.height {
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

        let right: State = if self.x + 1 > self.width {
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

            let action = strategy.get(&state);
            match action.unwrap() {
                // TODO don't go about/bewlow grid limits
                // TODO handle different ways to count points (eq penalties for hitting wals, moves, picking on empty space)
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
    let width: usize = 500;
    let height: usize = 500;
    let grid = generate_gird_random(width, height, vec![3, 7]);
    let mut robot = Robot::new(grid, width, height, Some(0), Some(0));
    let strategy = generate_strategy();

    // display_strategy(&strategy);
    robot.play_strategy(strategy, 100);
    println!("Points: {}", robot.points);
}
