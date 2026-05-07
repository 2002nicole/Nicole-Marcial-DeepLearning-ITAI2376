# L09 — Reinforcement Learning and AI Agents Lab

## Lab Overview

This folder contains my completed Module 09 lab for **ITAI 2376: Deep Learning**. This was a course lab, not a fully independent project. The notebook included starter code and guided sections provided by the course. My work included running the full notebook, analyzing the learning curves, completing the exploration-rate experiment, and writing reflection responses connecting reinforcement learning to my final AI agent project.

The purpose of this lab was to understand the reinforcement learning loop through the **CartPole-v1** environment and connect those ideas to modern AI agents and RLHF.

## Problem Statement

The task was to teach an agent to balance a pole on a moving cart using reinforcement learning concepts.

The lab used the CartPole environment, where:

- The **state** describes the cart and pole position/velocity information.
- The **actions** are moving the cart left or right.
- The **reward** is given for keeping the pole balanced longer.
- The goal is to survive as many timesteps as possible, up to a maximum score of 500.

This lab helped show the difference between an untrained random agent and a Q-learning agent that improves through experience.

## Source Code

The main source file for this lab is:

```text
L09_Nicole_Marcial_ITAI_2376.ipynb
```

The notebook includes:

- Gymnasium setup
- CartPole-v1 environment loading
- Random agent baseline
- Q-learning agent implementation
- Q-table creation
- State discretization/binning
- Epsilon-greedy exploration
- Training loop for 500 episodes
- Learning curve visualization
- Epsilon decay visualization
- Exploration-rate comparison experiment
- Final project reflection questions

## Course Lab Requirements Completed

This lab required three main parts:

### Part 1 — Random Agent

The notebook first ran a random agent in the CartPole environment. The random agent chose left or right with no strategy and failed quickly. This created a baseline for comparison.

The exercise questions asked about:

- Average score of the random agent
- What it means to reach the maximum score of 500
- State, action, and reward in CartPole

### Part 2 — Q-Learning Agent

The notebook then trained a Q-learning agent for 500 episodes. The agent used a Q-table to learn which actions were more useful in different states.

The lab required reading and analyzing:

- The learning curve
- The rolling average score
- The epsilon decay chart
- The difference between early exploration and later exploitation

### Part 2B — Exploration-Rate Experiment

The lab compared different epsilon decay rates:

```text
Agent A: epsilon_decay = 0.990  fast decay
Original: epsilon_decay = 0.995 medium decay
Agent B: epsilon_decay = 0.999  slow decay
```

I wrote a hypothesis before running the experiment, then compared the actual results.

### Part 3 — Final Project Reflection

The final section connected CartPole reinforcement learning to RLHF and my final AI agent project. My final project agent is a **Cookie Recipe Adaptation Agent** that helps users adapt cookie recipes by scaling ingredients, suggesting substitutions, and explaining likely baking effects.

## Approach and Methodology

### 1. Random Agent Baseline

The random agent used the CartPole environment but did not learn. It selected actions randomly:

```text
Action 0: move cart left
Action 1: move cart right
```

This showed what failure looked like before learning was introduced.

### 2. Q-Learning Agent

The Q-learning agent used a Q-table to store estimated action values.

CartPole has a continuous state space, so the notebook discretized the state into bins. This allowed the Q-learning algorithm to use a table-based approach.

The Q-table shape was:

```text
Q-table shape: (10, 10, 10, 10, 2)
Total Q-values to learn: 20,000
```

The agent used:

```text
Environment: CartPole-v1
Episodes: 500
Bins: 10
Learning rate: 0.1
Discount factor: 0.99
Initial epsilon: 1.0
Epsilon decay: 0.995
Minimum epsilon: 0.01
```

### 3. Epsilon-Greedy Exploration

The agent used epsilon-greedy exploration. Early in training, epsilon was high, so the agent explored more random actions. As epsilon decayed, the agent exploited the Q-table more often.

This demonstrated the exploration/exploitation tradeoff:

- **Exploration:** try uncertain actions to discover better strategies
- **Exploitation:** use the best-known action from the Q-table

### 4. Exploration-Rate Comparison

The notebook trained two additional agents with different epsilon decay rates.

The goal was to observe how changing the exploration rate affected learning speed and final performance.

## Results and Evaluation

### Random Agent Results

The random agent failed quickly. One notebook run showed:

```text
Random agent average score: 15.7
Best score: 25
Worst score: 9
```

In the later learning-curve comparison, the random-agent average displayed as:

```text
Random agent average score: 21.9
```

Because the random baseline is stochastic, the exact average can change when the cell is rerun.

### Original Q-Learning Agent Results

The original Q-learning agent trained for 500 episodes.

```text
Final average score over last 50 episodes: 47.7
Final epsilon: 0.0816
```

The learning curve improved compared with the random agent, but the model did not solve the environment. The rolling average increased over time, but it was not smooth, showing that learning was inconsistent.

The notebook comparison showed:

```text
Random agent average:          21.9
Q-agent first 50 episodes avg: 20.1
Q-agent last 50 episodes avg:  47.7
Improvement over random:       25.8 points
```

### Exploration-Rate Experiment Results

The exploration-rate experiment produced the following last-50-episode averages:

```text
Agent A, fast decay 0.990:     83.9
Original, medium decay 0.995:  47.7
Agent B, slow decay 0.999:     31.2
```

My original hypothesis was that the slow-decay agent would perform better because it would explore more. The result was the opposite. The fast-decay agent performed best.

This showed that too much exploration can hurt performance because the agent keeps trying random actions instead of using what it has already learned. In this CartPole run, the faster-decay agent benefited from switching sooner toward exploitation.

## Connection to My Final Project

This lab helped me connect reinforcement learning concepts to my final **Cookie Recipe Adaptation Agent**.

Although my final project does not train a Q-learning model or DQN, the RL loop still matters because AI agents need to make decisions, receive feedback, and improve their outputs. For my cookie agent, a useful “reward signal” would measure whether the adapted recipe actually meets the user’s request while still making baking sense.

For example:

- If a user asks for vegan cookies, a good output should not include eggs or butter.
- If a user asks for chewier cookies, the agent should suggest changes that support chewiness, not just random substitutions.
- If a user asks to scale a recipe, the ingredient ratios should remain reasonable.

This lab also helped me understand exploration versus exploitation in agent design. My cookie agent should usually exploit reliable baking knowledge, such as common egg replacements or standard ingredient scaling. However, it may need to explore less common options when the user has unusual constraints, such as wanting a cookie recipe that is both vegan and high protein.

## Learning Outcomes

Key takeaways from this lab:

- A random agent fails quickly because it has no learned strategy.
- Q-learning improves by updating a Q-table based on rewards.
- CartPole uses a small state/action structure, which makes it suitable for a table-based learning demonstration.
- Continuous state values need to be discretized before they can be used in a Q-table.
- Epsilon controls how much the agent explores versus exploits.
- More exploration is not always better.
- Reward design matters because the agent learns based on what is being rewarded.
- RLHF uses a similar high-level idea: model behavior is improved using feedback.
- Real-world LLM agents need models that can generalize across complex natural language inputs, not fixed Q-tables.

## Limitations

The main limitations of this lab were:

- The Q-learning agent did not solve CartPole.
- The final score was still far below the perfect score of 500.
- Results varied because of randomness in training.
- Q-learning with a table does not scale well to complex real-world tasks.
- Discretizing continuous states loses some detail.
- The lab was intended to teach the RL loop, not build a production-ready RL system.

## Notes About Authorship

This was a guided course lab with starter code provided as part of ITAI 2376. My contributions included running the notebook, interpreting the charts, completing the exercise responses, writing a hypothesis before the exploration experiment, analyzing the results, and connecting reinforcement learning concepts to my final Cookie Recipe Adaptation Agent.
