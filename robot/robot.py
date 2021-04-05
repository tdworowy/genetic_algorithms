import numpy as np


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

    def check_situation(self):
        pass

    def play_strategy(self, strategy: dict):
        pass


if __name__ == '__main__':
    print(generate_grid(10, 10, [0.7, 0.3]))
