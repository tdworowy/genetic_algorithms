import statistics

import time
from functools import partial
from itertools import product
from multiprocessing.dummy import Pool as ThreadPool
from os import cpu_count

from pebble import ProcessPool
from random import choices, randrange

from matplotlib import pyplot as plt


states = ["empty", "point", "wall"]
actions = ["go_up", "go_down", "go_left", "go_right", "take_point"]


def generate_grid(
    width: int, height: int, states: list, weights: list, random_start: bool = True
) -> tuple:
    grid = [
        [(choices(states, weights)[0], 0) for _ in range(width)] for _ in range(height)
    ]
    if random_start:
        x = randrange(0, height)
        y = randrange(0, width)
        grid[x][y] = (states[0], 1)
    else:
        x = 0
        y = 0
        grid[0][0] = (states[0], 1)
    return grid, x, y


def save_strategy(strategy: list):
    with open("../last_strategy.txt", "a") as f:
        f.write("\n" + str(strategy))


class Robot:
    def __init__(self, width: int, height: int, grid: tuple, rewards: dict):
        self.width = width
        self.height = height
        self.grid = [[value for value in row] for row in grid[0]]

        self.x = grid[1]
        self.y = grid[2]

        self.actions = {
            "go_up": lambda x, y: self.move(x - 1, y),
            "go_down": lambda x, y: self.move(x + 1, y),
            "go_left": lambda x, y: self.move(x, y - 1),
            "go_right": lambda x, y: self.move(x, y + 1),
            "take_point": lambda x, y: self.take_point(),
        }
        self.states = states
        self.points = 0

        self.wall_penalty = rewards["wall_penalty"]
        self.pickup_empty_penalty = rewards["pickup_empty_penalty"]
        self.step_penalty = rewards["step_penalty"]
        self.pickup_reward = rewards["pickup_reward"]

    def move(self, new_x: int, new_y: int):
        if 0 <= new_x < self.height and 0 <= new_y < self.width:
            self.grid[self.x][self.y] = (self.grid[self.x][self.y][0], 0)
            self.grid[new_x][new_y] = (self.grid[new_x][new_y][0], 1)

            self.x = new_x
            self.y = new_y
            self.points -= self.step_penalty
        else:
            self.points -= self.wall_penalty

    def take_point(self):
        if self.grid[self.x][self.y][0] == "point":
            self.points += self.pickup_reward
            self.grid[self.x][self.y] = ("empty", self.grid[self.x][self.y][1])
        else:
            self.points -= self.pickup_empty_penalty

    def play_strategy(self, strategy: list) -> list:
        current_situation = {}
        if self.x - 1 < 0:
            current_situation["up"] = "wall"
        else:
            current_situation["up"] = self.grid[self.x - 1][self.y][0]

        if self.x + 1 >= self.height:
            current_situation["down"] = "wall"
        else:
            current_situation["down"] = self.grid[self.x + 1][self.y][0]

        if self.y < 0:
            current_situation["left"] = "wall"
        else:
            current_situation["left"] = self.grid[self.x][self.y - 1][0]

        if self.y + 1 >= self.width:
            current_situation["right"] = "wall"
        else:
            current_situation["right"] = self.grid[self.x][self.y + 1][0]

        current_situation["current"] = self.grid[self.x][self.y][0]

        for situation in strategy:
            if situation[0] == current_situation:
                self.actions[situation[1]["action"]](self.x, self.y)
                break

        return self.grid


def generation_threed(evolution, key: int) -> tuple:
    population = evolution.population.copy()
    env_per_strategy = evolution.env_per_strategy
    width = evolution.width
    height = evolution.height
    grid_states = evolution.grid_states
    rewards = evolution.rewards
    moves = evolution.moves
    # print(key)
    thread_pool = ThreadPool(env_per_strategy)

    def env_thread(number: int) -> int:
        robot = Robot(
            width,
            height,
            generate_grid(width, height, grid_states, [0.7, 0.3]),
            rewards,
        )

        [robot.play_strategy(population[key]) for _ in range(moves)]
        return robot.points

    points = thread_pool.map(env_thread, range(env_per_strategy))
    return key, statistics.mean(points), population[key]


class Evolution:
    def __init__(
        self,
        width: int,
        height: int,
        init_pop_count: int,
        generation_count: int,
        env_per_strategy,
        keep_parents: bool = True,
        keep_best: int = 1000,
        moves: int = 200,
        mutation_rate: float = 0.03,
        rewards: dict = {
            "wall_penalty": 10,
            "pickup_empty_penalty": 5,
            "step_penalty": 1,
            "pickup_reward": 5,
        },
    ):
        self.grid_states = ["empty", "point"]
        self.width = width
        self.height = height

        self.states = states
        self.actions = actions
        self.moves = moves
        self.env_per_strategy = env_per_strategy
        self.keep_parents = keep_parents
        self.init_pop_count = init_pop_count
        self.generation_count = generation_count
        self.keep_best = keep_best

        self.mutation_rate = mutation_rate

        self.population = {}
        self.results = {}

        self.rewards = rewards

    @staticmethod
    def plot_learning_curve(generations: list, result: list):
        plt.plot(generations, result)
        plt.show()

    def generate_init_population(self, population_size: int):
        for i in range(population_size):
            strategy = [
                (
                    {
                        "up": state[0],
                        "down": state[1],
                        "left": state[2],
                        "right": state[3],
                        "current": state[4],
                    },
                    {"action": choices(self.actions)[0]},
                )
                for state in product(self.states, repeat=len(self.actions))
            ]
            self.population[i] = strategy

    def play_generation(self):

        with ProcessPool(
            max_workers=cpu_count() - 1
        ) as pool:  # TODO it take to match time, does it really run concurrently?
            generation_thread_partial = partial(generation_threed, self)
            future = pool.map(
                generation_thread_partial, list(self.population.keys()), timeout=60 * 5
            )

            iterator = future.result()
            print(list(iterator))
            while True:
                try:
                    result = next(iterator)
                    self.results[result[0]] = result[1], result[2]
                    print("*", end="")
                except StopIteration:
                    print("\n" + "_" * 50)
                    break
                except TimeoutError as error:
                    print(
                        f"function took longer than {error.args[1]} seconds", flush=True
                    )

    @staticmethod
    def _get_best(results: dict) -> tuple:
        best = -100000
        strategy_id = 0
        for key in results:
            if results[key][0] > best:
                best = results[key][0]
                strategy_id = key
        return strategy_id, best

    def get_best(self) -> list:
        strategy_id, points = self._get_best(self.results)
        return self.population[strategy_id]

    def selection(self, get_best: int) -> tuple:
        results_temp = self.results.copy()
        best_ids = []
        best_points = []
        for i in range(get_best):
            id, res = self._get_best(results_temp)
            best_ids.append(id)
            best_points.append(res)
            del results_temp[id]
        return best_ids, best_points

    def generate_new_population(self, get_best: int):
        self.best = self.selection(get_best)
        best = self.best[0]
        new_population = {}
        i = 0

        if self.keep_parents:
            for id in best:
                new_population[i] = self.population[id]
                i += 1

        while len(new_population) < len(self.population):

            for j in range(get_best - 1):

                split_place = randrange(0, len(self.population[best[j]]))
                first_half = self.population[best[j]][:split_place]
                second_half = self.population[best[j + 1]][split_place:]
                new_population[i] = first_half + second_half

                if choices(
                    [0, 1], weights=[1 - self.mutation_rate, self.mutation_rate]
                )[0]:
                    x = randrange(0, len(new_population[i]))
                    new_population[i][x] = (
                        new_population[i][x][0],
                        {"action": choices(self.actions)[0]},
                    )  # random mutation
                i += 1

        self.population = new_population

    def evolve(self):
        self.generate_init_population(self.init_pop_count)

        generations_ = []
        results = []

        for i in range(self.generation_count):
            start = time.time()
            self.play_generation()
            self.generate_new_population(get_best=self.keep_best)

            fife_best = sorted(self.best[1], reverse=True)[0:5]
            print(f"generation:{i} best 5:{fife_best}", flush=True)

            generations_.append(i)
            results.append(fife_best[0])

            end = time.time()
            print(f"Generation time: {end - start}")

        strategy = self.get_best()
        save_strategy(strategy)
        self.plot_learning_curve(generations_, results)
