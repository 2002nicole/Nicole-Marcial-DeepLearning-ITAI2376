# Requirements and Dependencies — L09 Reinforcement Learning and AI Agents Lab

This lab was designed to run in **Google Colab**. It does not require a GPU because the lab uses a small Q-learning example rather than a deep reinforcement learning model.

## Recommended Environment

```text
Python 3.10 or later
Google Colab recommended
CPU is sufficient
```

## Main Dependencies

```text
gymnasium
matplotlib
numpy
```

## Installation Command

The notebook installation cell used:

```bash
pip install gymnasium matplotlib numpy --quiet
```

If running locally, install the required libraries with:

```bash
pip install gymnasium matplotlib numpy
```

## Libraries Used

### Gymnasium

Gymnasium was used to load the CartPole-v1 reinforcement learning environment.

The environment was created with:

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
```

### NumPy

NumPy was used for:

- Q-table creation
- Random action selection
- Binning/discretizing continuous states
- Calculating average scores
- Managing arrays of training results

### Matplotlib

Matplotlib was used to create:

- Random agent score charts
- Q-learning training curve
- Rolling average visualization
- Epsilon decay chart
- Exploration-rate comparison chart

## Hardware Notes

A GPU is not required for this lab. The notebook can run on a standard Google Colab CPU runtime.

## Generated Files

This lab does not require saving a trained model file. The Q-table is created and trained during notebook execution, and the results are shown through printed outputs and charts.

## Troubleshooting Notes

If the notebook does not run correctly:

1. Restart the runtime.
2. Run all cells from the top.
3. Make sure Gymnasium installed successfully.
4. Check that all charts are visible before downloading the notebook.
