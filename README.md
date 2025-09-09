# LAB 3: Reinforcement Learning for Logic Optimization
# Due Date: July 30th 2025

## Setup

Clone the repository from GitLab and set up a Python virtual environment.

```zsh
mkdir -p $HOME/ece493t32-s25_ml-chip-design/labs
cd $HOME/ece493t32-s25_ml-chip-design/labs
git clone ist-git@git.uwaterloo.ca:ece493t32-s25_ml-chip-design/labs/y2fahmy-lab3.git
cd y2fahmy-lab3
```

### On Lab Machines

If you are using the lab machines we have a Python virtual environment already set up for you. Use this command to activate it.

```zsh
source /zfsspare/ml-playground-env/bin/activate
```

### On Other Machines

Create and activate a Python virtual environment, then install dependencies:

```zsh
python3 -m venv ~/ml-playground
source ~/ml-playground/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

**Every time you create a new terminal, you will need to activate the virtual environment.**

Check using `which python` to make sure it points to `ml-playground/bin/python`.

## Lab Overview

In this lab, you will explore reinforcement learning (RL) techniques applied to logic circuit optimization. You have access to a custom library called `opt_gym` that provides a reinforcement learning environment for optimizing Look-Up Table (LUT) netlists.

The goal is to minimize both the number of LUTs and the circuit depth through strategic LUT merging operations. This is a classic optimization problem in digital circuit design  called technology mapping that can benefit from machine learning approaches.

### The `opt_gym` Library

The `opt_gym` package provides the following modules:

* **`opt_gym.core`**: Contains the main data structure `LUTNetlist` that represents logic circuits as collections of LUTs
* **`opt_gym.envs`**: Provides `LUTEnv`, a class that wraps the `LUTNetlist` with a reinforcement learning interface
* **`opt_gym.epfl`**: Contains EPFL benchmark circuits and provides functions to load them as `LUTEnv` environments

### Lab Structure

This lab consists of **4 problems** with (the main goal being Problem 2):

There is a helpful guide [GUIDE.pdf](./GUIDE.pdf)

### Problem 0: Exhaustive Search and Value Functions (10%)

**Objective**: Implement an exhaustive search algorithm to compute the true value function V(s) = max_a Q(s, a) for small circuit instances.

**What you need to do**:
- Complete the implementation in `problem_0.py`
- Use the exhaustive search to understand the optimal policy for small benchmark
circuits.
- Look for TODOs in the Python code. We want you to fill in teh value function
through either value iteration or other approaches you see fit. The goal is to
make the value function correctly predict the reward for this exhaustively
searched design space.
- Analyze the relationship between state features and optimal values

**Key concepts**: Dynamic programming, value functions, state space exploration

**Deliverables**: Working exhaustive search implementation with analysis of small benchmark results

### Problem 1: Heuristic Agent Design (10%)

**Objective**: Design and implement a heuristic-based agent that can outperform
a random agent. This does not use RL and tries to estimate a way to select a
move using some circuit/node information.

**What you need to do**:
- Study the examples in `epfl_benchmarks.py` and `lutenv_explore.py` to understand the environment
- Design a heuristic strategy for selecting LUT merge operations
- Implement your agent in `problem_1.py`
- Test your agent against the random baseline and achieve better than random performance

**Deliverables**: Heuristic agent implementation with performance comparison against random agent

### Problem 2: REINFORCE Algorithm Optimization (70%)

**Objective**: Tune and improve the provided REINFORCE implementation for
optimal performance. REINFORCE was the big splashy OpenAI breakthrough paper!

**What you need to do**:
- Start with the code from `reinforce.py` (a working REINFORCE implementation)
- Experiment with different aspects of the RL pipeline:
  - **Feature engineering**: Modify `get_lut_features()` to improve state representation
  - **Network architecture**: Adjust the policy network structure and hyperparameters
  - **Training parameters**: Tune learning rates, episode lengths, and exploration strategies
  - **Generalization**: Test policy transfer between different benchmark circuits

Remember, the actual RL agent code is already provided to you! You only have to
make some changes.

**Deliverables**: Improved REINFORCE agent with documented experiments and performance analysis

We will run `problem_2_sweep.sh` across various reasonable-sized circuits.

### Problem 3: Gymnasium Integration (10%) - **Advanced Challenge, for folks who are really intersted in RL i.e. ECE457 crowd**

**Objective**: Create a Gymnasium-compatible wrapper for the LUTEnv and train using external RL libraries.

**What you need to do**:
- Implement a Gymnasium wrapper around `LUTEnv` with proper action/observation spaces
- Design effective state and action encodings for the environment
- Train an agent using Stable-Baselines3 RL algorithms
- Compare performance with your previous implementation and see if you can go
beyond REINFORCE and use AC (Actor-Critic), PPO (Proximal Policy Optimization),
etc algorithms. The big win is that they're already implemented for you.

**Deliverables**: Gymnasium wrapper and trained agent using external RL library

## Example Files

To help you get started, we provide several working code examples:

* **`epfl_benchmarks.py`**: Demonstrates how to list and load EPFL benchmark circuits
* **`lutenv_explore.py`**: Shows basic `LUTEnv` environment manipulation and reward tracking
* **`random_agent.py`**: Implements a simple random agent as a baseline for comparison
* **`reinforce.py`**: A complete REINFORCE implementation using JAX/Flax for solving the `LUTEnv`

## Getting Started

1. **First**, run `epfl_benchmarks.py` to familiarize yourself with the available benchmarks:
   ```zsh
   python epfl_benchmarks.py
   ```

2. **Then**, examine `lutenv_explore.py` to understand the environment interface:
   ```zsh
   python lutenv_explore.py
   ```

3. **Next**, run `random_agent.py` to see how a simple random agent works:
   ```zsh
   python random_agent.py --base arithmetic --name adder
   ```

4. **Finally**, study `reinforce.py` to understand the full RL implementation of REINFORCE algorithm:
   ```zsh
   python reinforce.py --base arithmetic --name adder
   ```
   There are many hyperparameters that you will have to play with.

Note: `random_agent.py` requires command-line arguments to specify which benchmark to use. `reinforce.py` is using the "arithmetic/adder" benchmark by default but our sweep will test other benchmarks. Check available benchmarks by running `epfl_benchmarks.py` first.

## Environment Details

### State Representation
Each state represents a LUT netlist with:
- Number of inputs, outputs, and intermediate LUTs
- LUT connectivity and truth tables
- Circuit depth and fan-out information

### Actions
Actions correspond to merging LUTs into a single LUT, identified by LUT_ID integers.
A LUT will merge into all its fan-out LUTs, effectively reducing the total number of LUTs in the netlist by 1.

### Rewards
The environment provides rewards based on:
- Reduction in total number of LUTs
- Changes in circuit depth
- Feasibility constraints (merged LUT must have ≤6 inputs)

There is no discount factor, you can provide rewards at each step, or consider only the final reward at episode termination.

### Episode Termination
Episodes end when no more valid merges are possible (all potential merges would exceed the 6-input limit).

Alternatively, an episode can be terminated after a fixed rollout.

## Submission Guidelines

For each problem, submit:
1. Your implementation file (`problem_X.py`)
2. A brief text file (`problem_X.txt`) summarizing your approach, results.

Include performance comparisons with baseline methods and analysis of what techniques worked best.
