import gymnasium as gym
from stable_baselines3 import DQN
import time

# Create environment with display
env = gym.make("LunarLander-v3", render_mode="human")

# Load trained model
model = DQN.load("lunar_lander_dqn")

episodes_to_watch = 5  # watch multiple landings

for episode in range(episodes_to_watch):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    print(f"\n🚀 Episode {episode + 1} starting...")

    while not done:
        # Use deterministic=True for best learned behavior
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        time.sleep(0.01)  # slow down so you can watch

        if truncated:
            break

    print(f"🏁 Episode {episode + 1} finished | Total Reward: {total_reward:.2f}")

env.close()

