from python.robot import Robot, Evolution, save_strategy

if __name__ == "__main__":
    width: int = 400
    height: int = 400
    cell_size: int = 20

    steps: int = 150  # 300

    rewards: dict = {
        "wall_penalty": 10,
        "pickup_empty_penalty": 5,
        "step_penalty": 1,
        "pickup_reward": 5,
    }

    evolution_parameters: dict = {
        "width": width // cell_size,
        "height": height // cell_size,
        "init_pop_count": 2500,  # 2000
        "generation_count": 1500,  # 401
        "env_per_strategy": 15,  # 25
        "keep_parents": True,
        "keep_best": 100,  # 300
        "moves": steps,
        "mutation_rate": 0.04,
        "rewards": rewards,
        "random_start": True,  # it makes evolution harder
    }

    evolution = Evolution(**evolution_parameters)
    evolution.generate_init_population()

    evolution.evolve()
    print(evolution.get_best(10))
    strategy = evolution.get_best_strategy()
    save_strategy(strategy)
