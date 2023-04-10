use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::thread;

use std::sync::{Arc, Mutex};

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
            finished: group_len == 0 || items.is_empty(),
            items,
        }
    }
}

fn generate_gird_random(width: usize, height: usize, weights: [usize; 2]) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(&weights).unwrap();

    let grid: Vec<Vec<usize>> = (0..height)
        .map(|_| (0..width).map(|_| [0, 1][dist.sample(&mut rng)]).collect())
        .collect();
    grid
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
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

fn generate_strategy() -> BTreeMap<(State, State, State, State, State), Action> {
    /*0: above
    1: right
    2: below
    3: left
    4: current*/
    let mut strategy: BTreeMap<(State, State, State, State, State), Action> = BTreeMap::new();

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

fn display_strategy(strategy: &BTreeMap<(State, State, State, State, State), Action>) {
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

#[derive(Debug, Clone, Copy)]
struct Penalties {
    move_: isize,
    wall: isize,
    empty_pick_up: isize,
}

#[derive(Debug, Clone)]
struct Specimen {
    strategy: BTreeMap<(State, State, State, State, State), Action>,
    points: isize,
}

struct Evolution {
    width: usize,
    height: usize,
    weights: [usize; 2],
    grid: Vec<Vec<usize>>,
    population: Vec<Specimen>,
    steps: usize,
    penalties: Penalties,
    generations: usize,
    population_size: usize,
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
        let up: State = if self.y + 1 >= self.height {
            State::Wall
        } else {
            match self.grid[self.y + 1][self.x] {
                1 => State::Point,
                _ => State::Empty,
            }
        };

        let down: State = if self.y as isize - 1 <= 0 {
            State::Wall
        } else {
            match self.grid[self.y - 1][self.x] {
                1 => State::Point,
                _ => State::Empty,
            }
        };

        let right: State = if self.x + 1 >= self.width {
            State::Wall
        } else {
            match self.grid[self.y][self.x + 1] {
                1 => State::Point,
                _ => State::Empty,
            }
        };

        let left: State = if self.x as isize - 1 <= 0 {
            State::Wall
        } else {
            match self.grid[self.y][self.x - 1] {
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
        strategy: &BTreeMap<(State, State, State, State, State), Action>,
        steps: usize,
        penalties: &Penalties,
    ) {
        for _ in 0..steps {
            let state = self.get_state();

            let action = strategy.get(&state);
            match action.unwrap() {
                Action::GoUp => {
                    if self.y + 1 < self.height {
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
                    if self.x + 1 < self.width {
                        self.x += 1;
                        self.points -= penalties.move_;
                    } else {
                        self.points -= penalties.wall;
                    }
                }
                Action::TakePoint => {
                    if self.grid[self.x][self.y] != 0 {
                        self.points += 5;
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
    fn new(width: usize, height: usize, weights: [usize; 2], population_size: usize) -> Evolution {
        Evolution {
            width,
            height,
            weights,
            grid: generate_gird_random(width, height, weights),
            population: generate_population(population_size),
            steps: 300,
            penalties: Penalties {
                move_: 1,
                wall: 5,
                empty_pick_up: 3,
            },
            generations: 800,
            population_size,
        }
    }

    fn get_n_best(&self, best: usize) -> Vec<Specimen> {
        get_n_best(self.population.clone(), best)
    }

    fn cross_spicemans(&self, population_to_corss: Vec<Specimen>) -> Vec<Specimen> {
        cross_spicemans(population_to_corss)
    }

    fn play_population_multi_thread(&mut self) {
        let num_threads = 12;
        let chunk_size = (self.population.len() + num_threads - 1) / num_threads;

        let population_arc = Arc::new(Mutex::new(self.population.clone()));
        let (tx, rx) = crossbeam_channel::bounded(num_threads);

        let width = self.width;
        let height = self.height;
        let steps = self.steps;
        let penalties = self.penalties;
        let weights = self.weights;

        for i in 0..num_threads {
            let population_arc = population_arc.clone();
            let tx = tx.clone();

            thread::spawn(move || {
                let mut new_population = Vec::new();

                let start = i * chunk_size;
                let end = std::cmp::min(start + chunk_size, population_arc.lock().unwrap().len());

                for spiceman in &population_arc.lock().unwrap()[start..end] {
                    let mut robot = Robot::new(
                        generate_gird_random(width, height, weights),
                        width,
                        height,
                        Some(0),
                        Some(0),
                    );
                    robot.play_strategy(&spiceman.strategy, steps, &penalties);

                    new_population.push(Specimen {
                        strategy: spiceman.strategy.clone(),
                        points: robot.points,
                    });
                }

                tx.send(new_population).unwrap();
            });
        }

        drop(tx);

        let mut new_population = Vec::with_capacity(self.population.len());
        for chunk_population in rx {
            new_population.extend(chunk_population);
        }
        self.population = new_population;
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

    fn evolv(&mut self) {
        let n_best: usize = 100;
        for _i in 0..self.generations {
            self.play_population_multi_thread();

            let best = &self.get_n_best(n_best);
            let new_generation = &self.cross_spicemans(best.clone());
            let mut new_specimens = generate_population(self.population_size - n_best * 3);

            new_specimens.append(&mut best.clone());
            new_specimens.append(&mut new_generation.clone());

            self.population = new_specimens;
        }
    }
}

fn get_n_best(mut population: Vec<Specimen>, best: usize) -> Vec<Specimen> {
    population.sort_by_key(|s| Reverse(s.points));
    population[0..best].to_vec()
}

fn cross_spicemans(population: Vec<Specimen>) -> Vec<Specimen> {
    let mut new_population: Vec<Specimen> = Vec::new();
    for i in (0..population.len()).step_by(2) {
        let strategy_len = population[i].strategy.len();
        let mut chunk1 = get_strategy_chunk(&population[i].strategy, 0, strategy_len / 2);
        let chunk2 =
            get_strategy_chunk(&population[i + 1].strategy, strategy_len / 2, strategy_len);

        chunk1.extend(chunk2);

        new_population.push(Specimen {
            strategy: chunk1,
            points: 0,
        })
    }
    new_population
}

fn get_strategy_chunk(
    strategy: &BTreeMap<(State, State, State, State, State), Action>,
    from: usize,
    to: usize,
) -> BTreeMap<(State, State, State, State, State), Action> {
    strategy
        .iter()
        .skip(from)
        .take(to - from)
        .map(|(&k, &v)| (k, v))
        .collect()
}

#[test]
fn test_get_strategy_chunk() {
    let strategy = generate_strategy();
    let chunk1 = get_strategy_chunk(&strategy, 1, 10);
    let chunk2 = get_strategy_chunk(&strategy, 0, 1);

    assert!(chunk1.len() == 9);
    assert!(chunk2.len() == 1);
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
    let width: usize = 20;
    let height: usize = 20;
    let weights = [3, 7];
    let mut evolution = Evolution::new(width, height, weights, 4500);
    evolution.evolv();

    evolution.play_population_multi_thread();

    let n_bests = evolution.get_n_best(1);

    for specimen in n_bests {
        println!("{:?}", specimen);
    }
}
