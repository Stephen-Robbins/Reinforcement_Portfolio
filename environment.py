import gym
import numpy as np
from BlackScholesPaths import generate_BS_paths
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class PortfolioEnv(gym.Env):
  def __init__(self, S, T, mu, C, steps):
    # Store the input variables
    self.S = S
    self.T = T
    self.mu = mu
    self.C = C
    self.steps = steps
    self.current_step=0
    self.path = generate_BS_paths(self.S, self.T, self.mu, self.C, self.steps, 1)
    
    # Define the state space (asset prices)
    self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(self.mu),))
    # Define the action space (portfolio weights)
    self.action_space = gym.spaces.Box(low=0, high=1, shape=(len(self.mu),))
    
  def reset(self):
    # Generate a new stock price path using the generate_BS_paths function
    self.path = generate_BS_paths(self.S, self.T, self.mu, self.C, self.steps, 1)
    
    # Initialize the cash and stock holdings to zero
    self.holdings = np.zeros(self.path.shape)
    self.current_step = 0
    
    # Return the initial state of the path
    return self.path[:, 0]
    
  def step(self, action):
    
    # Normalize the action to sum to 1 (portfolio weights)
    if np.sum(action) > 0:
      allocation = action / np.sum(action)
    else:
      allocation = np.ones_like(action) / len(action)
    
    # Update the holdings
    self.holdings[:, self.current_step] = allocation
    
    # Calculate the reward if not the first step
    if self.current_step > 0:
      price_returns = (self.path[:, self.current_step] - self.path[:, self.current_step-1]) / self.path[:, self.current_step-1]
      reward = np.dot(self.holdings[:, self.current_step-1], price_returns)
    else:
      reward = 0
    
    # Increment the current time step
    self.current_step += 1
    
    # Check if the episode is done
    done = self.current_step >= self.steps - 1
    
    # Return the next state, reward, and done flag
    if done:
      next_state = self.path[:, -1]
    else:
      next_state = self.path[:, self.current_step]
    
    return next_state, reward, done, {}
    
  def render(self):
    # Print the current time step and the current holdings for each asset
    print(f"Step: {self.current_step}")
    print(f"Holdings: {self.holdings[:, self.current_step]}")
    
    # Plot the stock price path and the holdings for each asset
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(self.path.T)
    plt.title('Asset Prices')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    
    plt.subplot(1, 2, 2)
    plt.plot(self.holdings.T)
    plt.title('Portfolio Holdings')
    plt.xlabel('Time Step')
    plt.ylabel('Holdings')
    plt.show()