use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::HashMap;

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

#[derive(Debug, Clone, Copy)]
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

struct Penalties {
    move_: isize,
    wall: isize,
    empty_pick_up: isize,
}

#[derive(Debug, Clone)]
struct Specimen {
    strategy: HashMap<(State, State, State, State, State), Action>,
    points: isize,
}

struct Evolution {
    width: usize,
    height: usize,
    grid: Vec<Vec<usize>>,
    population: Vec<Specimen>,
    steps: usize,
    penalties: Penalties,
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
        strategy: &HashMap<(State, State, State, State, State), Action>,
        steps: usize,
        penalties: &Penalties,
    ) {
        for _ in 0..steps {
            let state = self.get_state();

            let action = strategy.get(&state);
            match action.unwrap() {
                Action::GoUp => {
                    if self.y < self.height {
                        self.y += 1;
                        self.points -= penalties.move_
                    } else {
                        self.points -= penalties.wall;
                    }
                }
                Action::GoDown => {
                    if self.y > 0 {
                        self.y -= 1;
                        self.points -= penalties.move_;
                    } else {
                        self.points -= penalties.wall;
                    }
                }
                Action::GoLeft => {
                    if self.x > 0 {
                        self.x -= 1;
                        self.points -= penalties.move_;
                    } else {
                        self.points -= penalties.wall;
                    }
                }
                Action::GoRight => {
                    if self.x < self.width {
                        self.x += 1;
                        self.points -= penalties.move_;
                    } else {
                        self.points -= penalties.wall;
                    }
                }
                Action::TakePoint => {
                    if self.grid[self.x][self.y] != 0 {
                        self.points += 1;
                        self.grid[self.x][self.y] = 0;
                    } else {
                        self.points -= penalties.empty_pick_up;
                    }
                }
            }
        }
    }
}

fn generate_population(population_size: usize) -> Vec<Specimen> {
    (0..population_size)
        .map(|_| Specimen {
            strategy: generate_strategy(),
            points: 0,
        })
        .collect()
}

impl Evolution {
    fn new(width: usize, height: usize, population_size: usize) -> Evolution {
        Evolution {
            width,
            height,
            grid: generate_gird_random(width, height, vec![3, 7]),
            population: generate_population(population_size),
            penalties: Penalties {
                move_: 1,
                wall: 5,
                empty_pick_up: 3,
            },
            steps: 300,
        }
    }

    fn get_n_best(&self, best: usize) -> Vec<Specimen> {
        get_n_best(self.population.clone(), best)
    }

    fn play_population(&mut self) {
        let mut new_population: Vec<Specimen> = Vec::new();
        for spiceman in &self.population {
            let mut robot =
                Robot::new(self.grid.clone(), self.width, self.height, Some(0), Some(0));
            robot.play_strategy(&spiceman.strategy, self.steps, &self.penalties);

            new_population.push(Specimen {
                strategy: spiceman.strategy.clone(),
                points: robot.points,
            });
        }
        self.population = new_population;
    }
}

fn get_n_best(mut population: Vec<Specimen>, best: usize) -> Vec<Specimen> {
    population.sort_by_key(|s| Reverse(s.points));
    population[0..best].to_vec()
}

#[test]
fn test_get_n_best() {
    let population = vec![
        Specimen {
            strategy: generate_strategy(),
            points: 10,
        },
        Specimen {
            strategy: generate_strategy(),
            points: 20,
        },
        Specimen {
            strategy: generate_strategy(),
            points: 8,
        },
    ];

    let best = get_n_best(population, 2);
    assert!(best[0].points == 20);
    assert!(best[1].points == 10);
}

fn main() {
    let width: usize = 500;
    let height: usize = 500;
    let mut evolution = Evolution::new(width, height, 2000);
    evolution.play_population();

    let top = evolution.get_n_best(5);
    // TODO why all top speciment have same number of points ? 
    for s in top {
        println!("{:?}", s);
        println!("##########################",);
    }
}
