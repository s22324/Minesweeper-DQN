import numpy as np

from minesweeper_env import MinesweeperEnv
from stable_baselines3 import DQN

env = MinesweeperEnv(4,4,4)
env.reset()

model = DQN.load('models/BoardSize_4x4_MineDensityLOW_1705836336/1000000.zip')

# model = DQN.load('models/BoardSize_4x4_MineDensityMED_1705840305/1000000.zip')

# model = DQN.load('models/BoardSize_4x4_MineDensityHIGH_1705845353/1000000.zip')

def main():
    obs, info = env.reset()
    episode_rewards = []
    total_wins = 0
    episode_count = 0
    total_reward = 0

    for episode in range(50000):
        episode_reward = 0
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        total_reward += reward
        env.render()
        if terminated or truncated or info.get("is_success", False):
            print("Reward:", episode_reward, "Success?", info.get("is_success", False))
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            if info.get("is_success", False):
                total_wins += 1
            obs, info = env.reset()
            episode_count += 1

    mean_episode_reward = total_reward / episode_count
    win_rate = (total_wins / episode_count) * 100

    print(f"Win rate = {win_rate}%")
    print(f"Mean Episode Reward = {mean_episode_reward}")
    print(f"Total Wins = {total_wins}")
            

if __name__ == "__main__":
    main()
