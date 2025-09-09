"""
Problem 3: Gymnasium Integration - Advanced Challenge

Your task is to create a Gymnasium-compatible wrapper for the LUTEnv environment
and train an agent using external RL libraries like Stable-Baselines3.

INSTRUCTIONS:
1. Create a Gymnasium wrapper class that inherits from gymnasium.Env
2. Design appropriate action and observation spaces
3. Implement the required methods: step(), reset(), etc.
4. Train an agent using Stable-Baselines3 or another RL library
5. Compare performance with your previous implementations

KEY CHALLENGES:

A) ACTION SPACE DESIGN:
   - How to represent the variable number of possible LUT merges?
   - Options: Discrete space with masking, continuous embeddings, hierarchical actions

B) OBSERVATION SPACE DESIGN:
   - How to encode the netlist state in a fixed-size format?
   - Consider: adjacency matrices, feature vectors, graph embeddings
   - You could consider what was learned in Problem 2
   - Balance between information content and computational efficiency

C) REWARD ENGINEERING:
   - Design rewards that guide learning effectively
   - Rewards at the end of an episode vs. step-wise rewards

D) EPISODE MANAGEMENT:
   - Handle variable episode lengths
   - Deal with terminal states and early stopping
"""
