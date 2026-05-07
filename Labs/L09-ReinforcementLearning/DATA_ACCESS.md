# Data Access and Environment — L09 Reinforcement Learning and AI Agents Lab

## Dataset Used

This lab does not use a traditional dataset such as images, text, or CSV files.

Instead, it uses the **CartPole-v1** environment from Gymnasium.

## Environment Used

CartPole-v1 is a classic reinforcement learning environment where an agent must balance a pole on a moving cart.

The environment is loaded directly from Gymnasium:

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
```

## State, Action, and Reward

### State

The CartPole state contains information about the cart and pole, including position and velocity values.

Conceptually, the state describes the current condition of the cart-pole system.

### Action

The action space has two possible actions:

```text
0 = move cart left
1 = move cart right
```

### Reward

The agent receives a reward for each timestep that it keeps the pole balanced.

```text
Reward: +1 for each timestep survived
Maximum score: 500
```

## Data Generated During the Lab

The notebook generates its own training data through interaction with the environment.

During each episode:

1. The environment provides a state.
2. The agent chooses an action.
3. The environment returns the next state and reward.
4. The agent updates its Q-table.
5. The process repeats until the pole falls or the maximum episode length is reached.

## External Data Download Required?

No external dataset download is required.

Everything needed for the lab is provided through Gymnasium once the package is installed.

## Files to Include in GitHub

For this lab folder, include:

```text
L09_Nicole_Marcial_ITAI_2376.ipynb
README.md
REQUIREMENTS.md
DATA_ACCESS.md
```

## Reproducibility Notes

To reproduce the lab:

1. Open the notebook in Google Colab.
2. Run the setup cell to install Gymnasium, Matplotlib, and NumPy.
3. Run all notebook cells from top to bottom.
4. Review the random agent baseline.
5. Review the Q-learning training curve and epsilon decay chart.
6. Review the exploration-rate comparison chart.
7. Read the final reflection answers connecting RL to the final AI agent project.

## Important Note About Results

The exact scores may change when the notebook is rerun because reinforcement learning training involves randomness. The general pattern should remain the same: the random agent performs poorly, while the Q-learning agent improves through reward-based learning.
