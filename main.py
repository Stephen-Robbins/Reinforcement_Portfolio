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
agent = Agent(env)
print('hi')

# Run the training loop
for i in range(num_iterations):
  print('yo')
  # Reset the environment at the start of each episode
  state = env.reset()
  
  while True:
    # Select an action using the agent and the current state
    action = agent.act(state)
    
    # Step the environment and receive the next state, reward, done flag, and info
    next_state, reward, done, info = env.step(action)
    
    # Update the agent based on the reward and next state
    agent.learn(reward, state, next_state)
    
    # Set the state to the next state
    state = next_state
    
    # If the episode is done, break the loop
    if done:
      break
