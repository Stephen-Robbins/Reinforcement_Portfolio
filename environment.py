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
    
    # Define the state space
    self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.steps,))
    # Define the action space
    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.path.shape[0],))
    
  def reset(self):
    # Generate a new stock price path using the generate_BS_paths function
    self.path = generate_BS_paths(self.S, self.T, self.mu, self.C, self.steps, 1)
    
    # Initialize the cash and stock holdings to zero
    self.holdings = np.zeros(self.path.shape)
    
    # Return the initial state of the path
    return self.path[:,0]
    
  def step(self, action):
    
    # Calculate the total value of the portfolio at the current time step
    total_value = np.trace(self.holdings[:, 0:self.current_step-1] @ (self.path[:, 1:self.current_step]-self.path[:, 0:self.current_step-1]).T)
    
    allocation = action / np.sum(action)
    # Update the cash and stock holdings
   
    self.holdings[self.current_step] = allocation
    
    # Calculate the reward based on the change in portfolio value
    reward = self.holdings[:, self.current_step-1] @ (self.path[:, 1:self.current_step]-self.path[:, self.current_step-1])
    
    # Check if the episode is done
    done = self.current_step == self.steps - 1
    
    # Increment the current time step
    self.current_step += 1
    
    # Return the next state, reward, and done flag
    return self.path[self.current_step], reward, done, {}
    
def render(self):
  # Print the current time step and the current holdings for each asset
  print(f"Step: {self.current_step}")
  print(f"Holdings: {self.holdings[self.current_step]}")
  
  # Plot the stock price path and the holdings for each asset
  plt.plot(self.path)
  plt.plot(self.holdings)
  plt.show()