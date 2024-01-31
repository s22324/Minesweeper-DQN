import os
import time
import numpy as np
import torch

from minesweeper_env import MinesweeperEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

BOARD_WIDTH = 4
BOARD_HEIGHT = 4
MINES = 4

def calculate_mine_density(n_mines, nrows, ncols):
    total_tiles = nrows * ncols
    mine_density = n_mines / total_tiles

    if mine_density <= 0.25:
        return 'LOW'
    elif mine_density >= 0.75:
        return 'HIGH'
    else:
        return 'MED'


models_dir = f"models/BoardSize_{BOARD_WIDTH}x{BOARD_HEIGHT}_MineDensity{calculate_mine_density(MINES, BOARD_WIDTH,BOARD_HEIGHT)}_{int(time.time())}"
logdir = f"logs/BoardSize_{BOARD_WIDTH}x{BOARD_HEIGHT}_MineDensity{calculate_mine_density(MINES, BOARD_WIDTH,BOARD_HEIGHT)}_{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


if not os.path.exists(logdir):
    os.makedirs(logdir)

env = MinesweeperEnv(BOARD_WIDTH,BOARD_HEIGHT,MINES)
# check_env(env)


class CustomCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.progress_list = []
        self.wins_list = []

    def _on_step(self) -> bool:
        # Update lists with custom metrics from the environment
        self.progress_list.append(env.unwrapped.n_progress)
        self.wins_list.append(env.unwrapped.n_wins)

        if len(self.progress_list) >= self.check_freq:
            # Calculate custom statistics
            med_progress = round(np.median(self.progress_list[-self.check_freq:]), 2)
            win_total = max(self.wins_list)

            # Log to TensorBoard
            self.logger.record('custom/median_progress', med_progress)
            self.logger.record('custom/win_rate', win_total)

            # Reset lists
            self.progress_list = []
            self.wins_list = []

        return True

# Learning settings
LEARNING_RATE = 0.001 #The learning rate, it can be a function of the current progress remaining (from 1 to 0)
BUFFER_SIZE = 500_000 #Size of the replay buffer
BATCH_SIZE = 64 #Minibatch size for each gradient update
SOFT_UPDATE = 1.0 #The soft update coefficient (“Polyak update”, between 0 and 1) default 1 for hard update
DISCOUNT = 0.9 #gamma (the discount factor)

# Exploration settings
EPSILON = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

callback = CustomCallback(check_freq=100)

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[70,80,60,30])

model = DQN("MlpPolicy", env, verbose=1,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            tau=SOFT_UPDATE,
            gamma=DISCOUNT,
            learning_starts=20000,
            exploration_fraction=EPSILON_DECAY,
            exploration_initial_eps=EPSILON,
            exploration_final_eps=EPSILON_MIN,
            tensorboard_log=logdir,
            policy_kwargs=policy_kwargs,
            device=torch.device('cuda'))

TIME_STEPS = 1000000
model.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name="DQN", progress_bar=True,callback=callback)
model.save(f"{models_dir}/{TIME_STEPS}")

env.close()