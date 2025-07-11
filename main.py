from environment import PortfolioEnv
from agent import Agent
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


num_iterations=1000
S=100
T=1
mu=np.array([.1,.1])
C=np.array([[1,-1],
            [-1,1]])
steps=100


# Create an instance of the PortfolioEnv environment
env = PortfolioEnv(S, T, mu, C, steps)

# Create an instance of the agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = Agent(env, state_size, action_size, random_seed=0)

# Run the training loop
for i in range(num_iterations):
  # Reset the environment at the start of each episode
  state = env.reset()
  agent.reset()  # Reset noise
  
  episode_reward = 0
  while True:
    # Select an action using the agent and the current state
    action = agent.act(state)
    
    # Step the environment and receive the next state, reward, done flag, and info
    next_state, reward, done, info = env.step(action)
    
    # Save experience to replay buffer and learn
    agent.step(state, action, reward, next_state, done)
    
    # Set the state to the next state
    state = next_state
    episode_reward += reward
    
    # If the episode is done, break the loop
    if done:
      if i % 100 == 0:
        print(f"Episode {i}, Reward: {episode_reward}")
      break
