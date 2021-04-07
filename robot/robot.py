from collections import Counter

import numpy as np
from itertools import product
from random import choice, randrange, choices
from matplotlib import pyplot as plt


def generate_grid(width: int, height: int, weights: list) -> np.ndarray:
    """
    1 <- point
    0 <- empty field
    """
    return np.random.choice(a=[0, 1], size=(width, height), p=weights)


class Robot:
    def __init__(self, start_x: int, start_y: int, rewards: dict, width: int, height: int, weights: list):
        self.x = self.start_x = start_x
        self.y = self.start_y = start_y

        self.points = 0
        self.grid = generate_grid(width=width, height=height, weights=weights)

        self.width, self.height = self.grid.shape

        self.wall_penalty = rewards["wall_penalty"]
        self.pickup_empty_penalty = rewards["pickup_empty_penalty"]
        self.step_penalty = rewards["step_penalty"]
        self.pickup_reward = rewards["pickup_reward"]

        self.actions = {
            1: self.go_up,
            2: self.go_down,
            3: self.go_left,
            4: self.go_right,
            5: self.take_point
        }
        self.actions_debug = {
            1: "go_up",
            2: "go_down",
            3: "go_left",
            4: "go_right",
            5: "take_point"
        }

    def go_up(self):
        new_y = self.y - 1
        if new_y <= self.width:
            self.points -= self.wall_penalty
        else:
            self.y = new_y
            self.points -= self.step_penalty

    def go_down(self):
        new_y = self.y + 1
        if new_y >= self.width:
            self.points -= self.wall_penalty
        else:
            self.y = new_y
            self.points -= self.step_penalty

    def go_left(self):
        new_x = self.x - 1
        if new_x <= self.height:
            self.points -= self.wall_penalty
        else:
            self.x = new_x
            self.points -= self.step_penalty

    def go_right(self):
        new_x = self.x + 1
        if new_x >= self.height:
            self.points -= self.wall_penalty
        else:
            self.x = new_x
            self.points -= self.step_penalty

    def take_point(self):
        if self.grid[self.y][self.x] == 1:
            self.points += self.pickup_reward
            self.grid[self.y][self.x] = 0
        else:
            self.points -= self.pickup_empty_penalty

    def check_state(self):
        """
        -1 <- wall
        0 <- empty space
        1 <- point

        0: above
        1: right
        2: below
        3: left
        4: current
        """
        state = []
        if self.width <= self.x - 1:
            state.append(-1)
        else:
            state.append(self.grid[self.y][self.x - 1])

        if self.height >= self.y + 1:
            state.append(-1)
        else:
            state.append(self.grid[self.y + 1][self.x])

        if self.width >= self.x + 1:
            state.append(-1)
        else:
            state.append(self.grid[self.y][self.x + 1])

        if self.height <= self.y - 1:
            state.append(-1)
        else:
            state.append(self.grid[self.y - 1][self.x])

        state.append(self.grid[self.y][self.x])

        return tuple(state)

    def play_strategy(self, strategy: dict, moves_count: int, debug=False):
        for _ in range(moves_count):
            action = strategy[self.check_state()]
            self.actions[
                action
            ]()
            if debug:
                print(self.actions_debug[action])

    def reset(self):
        self.points = 0
        self.grid = generate_grid(width=20, height=20, weights=[0.7, 0.3])
        self.x = self.start_x
        self.y = self.start_y


def generate_strategy() -> dict:
    """
     -1 <- wall
      0 <- empty space
      1 <- point
      key:
       0: above
       1: right
       2: below
       3: left
       4: current
      value:
       action (1 to 5)
       warning: also generate impossible states
    """
    return {state: choice([1, 2, 3, 4, 5]) for state in product([-1, 0, 1], repeat=5)}


class Evolution:
    STRATEGY_LEN = 243  # it is constant

    def __init__(self, init_pop_count: int, generation_count: int, env_per_strategy: int = 1,
                 keep_parents: bool = True, keep_best: int = 200, moves: int = 200, mutation_rate: float = 0.03,
                 rewards: dict = None, width: int = 20, height: int = 20, weights=None):

        if weights is None:
            weights = [0.7, 0.3]
        if rewards is None:
            rewards = {"wall_penalty": 10, "pickup_empty_penalty": 5, "step_penalty": 1,
                       "pickup_reward": 5}

        self.moves = moves
        self.env_per_strategy = env_per_strategy
        self.init_pop_count = init_pop_count
        self.generation_count = generation_count
        self.keep_best = keep_best
        self.keep_parents = keep_parents
        self.mutation_rate = mutation_rate

        self.population = {}
        self.results = {}

        self.rewards = rewards
        self.width = width
        self.height = height
        self.weights = weights

        self.robot = Robot(start_x=0,
                           start_y=0,
                           rewards=self.rewards,
                           width=self.width,
                           height=self.height,
                           weights=self.weights)

    @staticmethod
    def plot_learning_curve(generations: list, result: list):
        plt.plot(generations, result)
        plt.show()

    def _get_best(self, top: int) -> dict:
        return dict(Counter(self.results).most_common(top))

    def generate_init_population(self):
        for i in range(self.init_pop_count):
            self.population[i] = generate_strategy()

    def play_generation(self):
        for number, strategy in self.population.items():
            points_per_env = []
            for env in range(self.env_per_strategy):
                self.robot.play_strategy(strategy=strategy, moves_count=self.moves)
                points_per_env.append(self.robot.points)
                self.robot.reset()
            self.results[number] = np.mean(points_per_env)

    def generate_new_population(self):
        new_population = {}
        pop_count = 0
        best = self._get_best(self.keep_best)
        best_keys = list(best.keys())

        if self.keep_parents:
            for i, key in enumerate(best_keys):
                new_population[i] = self.population[key]
            pop_count = self.keep_best

        for j in range(pop_count, self.init_pop_count):
            key1 = choices(best_keys)[0]
            key2 = choices([key for key in best_keys if key != key1])[0]

            split_place = randrange(0, self.STRATEGY_LEN)

            first_half = dict(list(self.population[key1].items())[:split_place])
            second_half = dict(list(self.population[key2].items())[split_place:])

            new_population[j] = first_half | second_half

            if choices([0, 1], weights=[1 - self.mutation_rate, self.mutation_rate])[0]:
                key = choices(list(new_population[j].keys()))[0]
                new_population[j][key] = choices([1, 2, 3, 4, 5])[0]
        self.population = new_population

    def evolve(self):
        generations = []
        results = []
        for i in range(self.generation_count):
            self.play_generation()
            generations.append(i)
            results.append(list(self._get_best(1).values())[0])
            self.generate_new_population()

        self.plot_learning_curve(generations, results)

    def get_best_strategy(self):
        return self.population[list(self._get_best(1).keys())[0]]


if __name__ == '__main__':
    # strategy = generate_strategy()
    # print(strategy)
    # grid = generate_grid(width=20, height=20, weights=[0.7, 0.3])
    # robot = Robot(start_x=0, start_y=0, grid=grid, rewards=rewards)
    #
    # robot.play_strategy(strategy=strategy, moves_count=200)
    #
    # print(robot.points)

    rewards: dict = {"wall_penalty": 10, "pickup_empty_penalty": 5, "step_penalty": 1,
                     "pickup_reward": 5}

    evolution_parameters: dict = {
        "width": 20,
        "height": 20,
        "init_pop_count": 500,  # 2000
        "generation_count": 200,  # 401
        "env_per_strategy": 5,  # 25
        "keep_parents": True,
        "keep_best": 50,  # 300
        "moves": 200,
        "mutation_rate": 0.04,
        "rewards": rewards
    }

    evolution = Evolution(**evolution_parameters
                          )
    evolution.generate_init_population()
    evolution.evolve()
    best_strategy = evolution.get_best_strategy()
    # print(best_strategy)

    robot = Robot(start_x=0,
                  start_y=0,
                  rewards=rewards,
                  width=20,
                  height=20,
                  weights=[0.3, 0.7])
    robot.play_strategy(strategy=best_strategy, moves_count=200)
