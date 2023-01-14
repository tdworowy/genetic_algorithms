import time
import tkinter
from collections import defaultdict
from doctest import master
import numpy as np
from python.robot import Robot
import ast


class GUI:
    def __init__(self, width: int = 1920, height: int = 1080, cell_size: int = 5):

        self.top = tkinter.Tk()
        self.top_frame = tkinter.Frame()
        self.button_frame = tkinter.Frame()

        self.colours = {
            0: "blue",
            1: "green"
        }
        self.width = width
        self.height = height
        self.canvas = tkinter.Canvas(master, width=self.width, height=self.height)
        self.cell_size = cell_size
        self.start_robot = 0
        self.cells= defaultdict(lambda: (-1, -1), {})

        self.prev_robot_x = -1
        self.prev_robot_y = -1

    def rectangle_coordinates(self, x: int, y: int) -> dict:
        dic = {'x1': x, 'y1': y, 'x2': self.cell_size + x, 'y2': self.cell_size + y}
        return dic

    def draw_robot(self, robot_x: int, robot_y: int) -> int:
        robot_coordinate = self.rectangle_coordinates(robot_x, robot_y)
        return self.canvas.create_rectangle(robot_coordinate["x1"],
                                            robot_coordinate["y1"],
                                            robot_coordinate["x2"],
                                            robot_coordinate["y2"],
                                            fill='red')

    def draw(self, grid: np.ndarray, prev_grid: np.ndarray, robot_x, robot_y):
        x = 0
        y = 0

        robot_x = robot_x * self.cell_size
        robot_y = robot_y * self.cell_size
        coordinate = {}

        for row, prev_row in zip(grid, prev_grid):
            for cell, prev_cell in zip(row, prev_row):
                coordinate = self.rectangle_coordinates(x, y)

                if cell != prev_cell or (x == self.prev_robot_x and y == self.prev_robot_y):

                    if self.cells[(x, y)] != (-1, -1):
                        self.canvas.delete(self.cells[(x, y)])

                    rectangle = self.canvas.create_rectangle(coordinate["x1"],
                                                             coordinate["y1"],
                                                             coordinate["x2"],
                                                             coordinate["y2"],
                                                             fill=self.colours[cell])
                    self.cells[(x, y)] = rectangle

                x = coordinate['x2']
            y = coordinate['y2']
            x = 0

        self.draw_robot(robot_x, robot_y)
        self.prev_robot_x = robot_x
        self.prev_robot_y = robot_y

        self.top.update()

    def main_loop(self):

        self.top_frame.pack(side="top", fill="both", expand=True)
        self.button_frame.pack(side="bottom", fill="both")
        self.canvas.pack(in_=self.button_frame)


if __name__ == "__main__":
    width: int = 400
    height: int = 400
    cell_size: int = 20
    rewards: dict = {"wall_penalty": 10, "pickup_empty_penalty": 5, "step_penalty": 1,
                     "pickup_reward": 5}

    with open("last_strategy.txt") as strategy_file:
        strategy = ast.literal_eval(strategy_file.readlines()[0])

    robot = Robot(
        start_x=0,
        start_y=0,
        width=width // cell_size,
        height=height // cell_size,
        rewards=rewards,
        weights=[0.7, 0.3])

    grid = robot.grid.copy()
    prev_grid = np.full(grid.shape, -1)

    gui = GUI(width, height, cell_size)
    gui.main_loop()

    gui.draw(grid, prev_grid, robot.start_x, robot.start_y)
    prev_grid = grid.copy()

    for i in range(300):
        robot.play_strategy(strategy, 1, debug=False)
        grid = robot.grid.copy()
        gui.draw(grid, prev_grid, robot.x, robot.y)
        prev_grid = grid.copy()
        time.sleep(0.3)

    print(f"Robot points: {robot.points}")
    while 1:
        pass
