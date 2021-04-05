import numpy as np
from itertools import product
from random import choice
from matplotlib import pyplot as plt


def generate_grid(width: int, height: int, weights: list) -> np.ndarray:
    """
    1 <- point
    0 <- empty field
    """
    return np.random.choice(a=[0, 1], size=(width, height), p=weights)


class Robot:
    def __init__(self, start_x: int, start_y: int, grid: np.ndarray, rewards: dict):
        self.x = start_x
        self.y = start_y

        self.points = 0
        self.grid = grid.copy()

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

    def go_up(self):
        new_x = self.x - 1
        if new_x <= self.width:
            self.points -= self.wall_penalty
        else:
            self.x = new_x
            self.points -= self.step_penalty

    def go_down(self):
        new_x = self.x + 1
        if new_x >= self.width:
            self.points -= self.wall_penalty
        else:
            self.x = new_x
            self.points -= self.step_penalty

    def go_left(self):
        new_y = self.y - 1
        if new_y <= self.height:
            self.points -= self.wall_penalty
        else:
            self.y = new_y
            self.points -= self.step_penalty

    def go_right(self):
        new_y = self.y + 1
        if new_y >= self.height:
            self.points -= self.wall_penalty
        else:
            self.y = new_y
            self.points -= self.step_penalty

    def take_point(self):
        if self.grid[self.x][self.y] == 1:
            self.points += self.pickup_reward
            self.grid[self.x][self.y] = 0
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
        if self.width < self.x - 1:
            state.append(-1)
        else:
            state.append(self.grid[self.x - 1][self.y])

        if self.height > self.y + 1:
            state.append(-1)
        else:
            state.append(self.grid[self.x][self.y + 1])

        if self.width > self.x + 1:
            state.append(-1)
        else:
            state.append(self.grid[self.x + 1][self.y])

        if self.height < self.y - 1:
            state.append(-1)
        else:
            state.append(self.grid[self.x][self.y - 1])

        state.append(self.grid[self.x][self.y])

        return tuple(state)

    def play_strategy(self, strategy: dict, moves_count: int):
        for _ in range(moves_count):
            self.actions[
                strategy[self.check_state()]
            ]()


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
    """
    return {state: choice([1, 2, 3, 4, 5]) for state in product([-1, 0, 1], repeat=5)}


class Evolution:
    def __init__(self, init_pop_count: int, generation_count: int, env_per_strategy,
                 keep_parents: bool = True, keep_best: int = 1000, moves: int = 200, mutation_rate: float = 0.03,
                 rewards: dict = {"wall_penalty": 10, "pickup_empty_penalty": 5, "step_penalty": 1,
                                  "pickup_reward": 5}):
        self.moves = moves
        self.env_per_strategy = env_per_strategy
        self.keep_parents = keep_parents
        self.init_pop_count = init_pop_count
        self.generation_count = generation_count
        self.keep_best = keep_best

        self.mutation_rate = mutation_rate

        self.population = []
        self.results = {}

        self.rewards = rewards

    @staticmethod
    def plot_learning_curve(generations: list, result: list):
        plt.plot(generations, result)
        plt.show()

    def generate_init_population(self, population_size: int):
        for i in range(population_size):
            self.population.append(generate_strategy())


if __name__ == '__main__':
    rewards = {"wall_penalty": 10, "pickup_empty_penalty": 5, "step_penalty": 1,
               "pickup_reward": 5}
    strategy = generate_strategy()
    grid = generate_grid(width=20, height=20, weights=[0.7, 0.3])
    robot = Robot(start_x=0, start_y=0, grid=grid, rewards=rewards)

    robot.play_strategy(strategy=strategy, moves_count=200)

    print(robot.points)
