# Reinforcement Learning: Q-Learning & Game AI

## Overview

This project demonstrates the fundamentals of reinforcement learning through two progressively complex environments. Starting with a custom grid-based maze game, the project shows how an intelligent agent learns to navigate efficiently using the Q-Learning algorithm. The agent discovers that random policies require 30-40 steps on average, but through learning improves to just 5-6 steps. The project then escalates to the CartPole balancing problem from OpenAI Gym, showing how the same Q-Learning principles apply to continuous physics simulations. Both projects illustrate the core principle of reinforcement learning: agents learn optimal behavior through trial, error, and reward signals.

## Environments

### 1. Custom Grid-Based Maze (reinfocement.ipynb)
A custom 8x8 grid environment where the agent (Peter) must navigate to find apples while avoiding hazards:
- **Terrain Types:** Ground (walkable), Water (fatal), Trees (rest spots), Apples (goal), Wolves (dangerous)
- **Goal:** Reach an apple efficiently
- **Agent Actions:** Move Up, Down, Left, Right
- **Starting Policy:** Random walk
- **Learned Policy:** Q-Learning with probability-weighted action selection

### 2. CartPole Balancing (openAiGYM.ipynb)
A physics-based simulation using OpenAI Gym where the agent must balance a vertical pole on a moving cart:
- **Observation Space:** 4 continuous values (cart position, velocity, pole angle, rotation rate)
- **Action Space:** 2 discrete actions (move left, move right)
- **Goal:** Keep pole balanced upright
- **Challenge:** Continuous state space requires discretization into bins
- **Training Scale:** 100,000 episodes to achieve stable balancing

## Methods

### Baseline: Random Policy
- **Grid Environment:** Random action selection yields average path length of 30-40 steps
- **CartPole:** Random actions fail to balance pole
- **Purpose:** Establishes performance baseline for comparison

### Q-Learning Algorithm
The core learning algorithm that finds optimal policies through iterative value updates:

**Key Components:**
1. **Q-Table/Q-Dictionary:** Stores learned value (Q-value) for each state-action pair
   - Grid: 8×8×4 table (position × actions)
   - CartPole: Dictionary with (state, action) keys for flexibility
   
2. **Reward Function:** Defines the goal structure
   - Move penalty: -0.1 (encourages short paths)
   - Goal reward: +10 (reaching apple)
   - End reward: -10 (death, failure)
   - CartPole: +1 per step alive

3. **Update Rule:** The core Q-Learning equation
   ```
   Q(s,a) = (1-α)·Q(s,a) + α·(r + γ·max(Q(s',a')))
   ```
   - α (alpha) = learning rate (0.3 for CartPole, decaying for grid)
   - γ (gamma) = discount factor (0.5 for grid, 0.9 for CartPole)
   - r = immediate reward
   - s,a = current state and action
   - s',a' = next state and best action

4. **Exploration-Exploitation:** ε-Greedy strategy
   - **Exploration:** Random actions (learn new state-action values)
   - **Exploitation:** Select actions with highest Q-values (use learned knowledge)
   - ε (epsilon) = exploration probability (90% for grid, decreases during CartPole training)

### State Discretization (CartPole)
Converts continuous observation space into discrete states for Q-Table compatibility:
- **Method 1 - Simple Scaling:** Divide continuous values by scaling factors
- **Method 2 - Bins:** Divide observation ranges into equally-spaced bins (20, 20, 10, 10)
- **Purpose:** Enables traditional Q-Table representation despite continuous observations

### Policy Extraction
Two approaches to convert Q-values into action selection:
1. **Greedy Policy:** Always choose the action with highest Q-value (may oscillate)
2. **Probabilistic Policy:** Sample actions with probability proportional to Q-values (more robust)

## Results

### Grid Environment (reinfocement.ipynb)
**Baseline (Random Policy):**
- Average path length: 30-40 steps
- Killed by wolves: Frequent
- Success rate: Low

**After Q-Learning Training (100 episodes):**
- Average path length: 5-6 steps
- Killed by wolves: Rare
- Success rate: 95%+
- **Improvement:** 5-8x more efficient

### CartPole Environment (openAiGYM.ipynb)
**Training Progress:**
- Episodes: 100,000
- Initial performance: Pole falls immediately
- Mid-training: Growing stability
- Final performance: Pole balanced 200-500 steps per episode
- **Running average:** Shows clear learning curve with convergence

**Training Insights:**
- Requires ~5,000-20,000 episodes to achieve basic stability
- ~50,000+ episodes for robust balancing
- Hyperparameter tuning essential for convergence

## Tech Stack

- **Python 3**
- **NumPy** - Arrays, mathematical operations, digitize for binning, statistics
- **Gymnasium (OpenAI Gym)** - CartPole environment with physics simulation
- **Matplotlib** - Training progress visualization, reward plotting
- **OpenCV (cv2)** - Image loading and drawing for grid visualization
- **Custom rlboard module** - Custom grid environment with rendering

## How to Run

### Prerequisites
Install required libraries:
```bash
pip install numpy matplotlib gymnasium opencv-python
```

### Project 1: Grid Maze with Q-Learning

1. Open `reinfocement.ipynb` in Jupyter Notebook
2. Ensure the `rlboard.py` file is in the same directory
3. Ensure the `images/` folder contains wolf.png, apple.png, human.png
4. Run all cells to:
   - Create and visualize the 8x8 board
   - Test random policy baseline (30-40 step average)
   - Define reward function for goals and hazards
   - Initialize Q-Table with uniform probabilities
   - Train Q-Learning algorithm for 5,000 epochs
   - Extract and test learned policy
   - Compare learned vs. random performance
5. Expected output: Show significant improvement from random to learned policy

### Project 2: CartPole Balancing with Q-Learning

1. Open `openAiGYM.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Initialize CartPole-v1 environment
   - Understand observation space (4 values) and action space (2 actions)
   - Explore state discretization methods
   - Create Q-Table as a dictionary for flexible state representation
   - Train Q-Learning algorithm for 100,000 episodes
   - Visualize training progress with reward plots
   - View trained agent balancing pole in real-time
3. Expected output: Agent learns to balance pole with increasing duration
4. Optional: Adjust hyperparameters (alpha, gamma, epsilon) to experiment with convergence speed

## Key Concepts Demonstrated

**Q-Learning:** Model-free, off-policy algorithm that learns value of actions in states
**Reward Engineering:** Designing reward functions to guide agent behavior
**Exploration vs. Exploitation:** Balancing discovery of new states with using known good actions
**State Representation:** Handling both discrete (grid) and continuous (CartPole) state spaces
**Convergence:** How agents improve through iterative training
**Hyperparameter Sensitivity:** Impact of learning rate, discount factor, exploration rate on learning

## Project Files

- **reinfocement.ipynb** - Grid-based maze environment with Q-Learning training and policy extraction
- **openAiGYM.ipynb** - CartPole environment with continuous state discretization and training
- **rlboard.py** - Custom grid environment implementation with visualization using OpenCV
- **images/wolf.png, apple.png, human.png** - Visual assets for grid rendering

## Notes & Extensions

- Q-Learning is off-policy, meaning it learns optimal policy while exploring with exploration strategy
- The algorithm is guaranteed to converge in finite state spaces under certain conditions
- Real-world applications include robotics, game AI, autonomous vehicles, and resource optimization
- Advanced extensions: Deep Q-Networks (DQN) for high-dimensional state spaces, Policy Gradient methods, Actor-Critic algorithms
- The CartPole environment can reach reward thresholds of 195+ with well-tuned hyperparameters
- Grid environment can be extended with larger boards, more hazards, or multiple agents
