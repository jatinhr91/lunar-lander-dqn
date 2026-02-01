import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Create environment
env = gym.make("LunarLander-v3")
env = Monitor(env)  # Records episode rewards

# Create DQN model with better hyperparameters
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
)

print("🚀 Training started... This will take a while on CPU.")

# Train the agent
model.learn(total_timesteps=1_000_000)

# Save the trained model
model.save("lunar_lander_dqn")

print("✅ Training complete! Model saved as lunar_lander_dqn.zip")

env.close()


