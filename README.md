# Portfolio Optimization using Deep Reinforcement Learning

A Deep Deterministic Policy Gradient (DDPG) implementation for dynamic portfolio allocation using reinforcement learning. This project demonstrates the application of advanced RL techniques to financial portfolio management with correlated assets.

## Overview

This project implements a reinforcement learning agent that learns optimal portfolio allocation strategies for multiple correlated assets. The agent uses the DDPG algorithm, which combines the benefits of both policy gradient methods and Q-learning to handle continuous action spaces - perfect for portfolio weight allocation.

### Key Features

- **Continuous Action Space**: Handles portfolio weights as continuous values
- **Correlated Assets**: Models asset correlations using multivariate geometric Brownian motion
- **Actor-Critic Architecture**: Implements separate networks for policy (Actor) and value function (Critic)
- **Experience Replay**: Improves sample efficiency and training stability
- **Soft Target Updates**: Ensures stable learning through gradual target network updates

## Technical Details

### Environment
The portfolio environment (`environment.py`) simulates:
- Multiple correlated assets following geometric Brownian motion
- Transaction costs and market impact (configurable)
- Risk-adjusted rewards based on portfolio performance

### Agent Architecture
The DDPG agent (`agent.py`) features:
- **Actor Network**: Maps states to portfolio allocations
- **Critic Network**: Evaluates state-action pairs
- **Target Networks**: Stabilizes training through soft updates
- **Experience Replay Buffer**: Stores and samples past experiences

### Market Simulation
The Black-Scholes path generator (`BlackScholesPaths.py`) creates:
- Realistic asset price paths with specified drift and volatility
- Correlated asset movements through covariance matrix
- Multiple scenarios for robust training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Reinforcement_Portfolio.git
cd Reinforcement_Portfolio

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
python main.py
```

### Custom Configuration

```python
from environment import PortfolioEnv
from agent import Agent
import numpy as np

# Configure environment parameters
S = 100  # Initial asset price
T = 1    # Time horizon
mu = np.array([0.1, 0.1])  # Expected returns
C = np.array([[1, -1], [-1, 1]])  # Covariance matrix
steps = 100  # Number of time steps

# Initialize environment and agent
env = PortfolioEnv(S, T, mu, C, steps)
agent = Agent(env)

# Train the agent
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(reward, state, next_state)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}, Total Reward: {total_reward}")
```

## Project Structure

```
Reinforcement_Portfolio/
│
├── main.py              # Main training script
├── agent.py             # DDPG agent implementation
├── environment.py       # Portfolio environment
├── model.py            # Neural network architectures
├── BlackScholesPaths.py # Asset price simulation
├── requirements.txt     # Project dependencies
└── README.md           # This file
```


## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Gym >= 0.17.0
- Matplotlib >= 3.3.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

Stephen Robbins - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/Reinforcement_Portfolio](https://github.com/yourusername/Reinforcement_Portfolio)
