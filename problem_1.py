"""
Problem 1: Heuristic Agent Design

Your task is to implement a heuristic-based agent that can outperform a random agent
in the LUTEnv environment.

INSTRUCTIONS:
1. Study example_1.py and example_2.py to understand how to interact with the environment
2. Design a heuristic strategy for selecting which LUTs to merge
3. Implement your agent below in the HeuristicAgent class
4. Test your agent against the random baseline (example_2.py)
5. Aim to achieve better than random!

EVALUATION:
Run your agent on multiple benchmarks and compare against random agent performance.
Document your heuristic strategy and results.
"""

import opt_gym.epfl as epfl
import opt_gym.envs as envs
from opt_gym.core import LUT_ID
import numpy as np


class HeuristicAgent:
    """
    A heuristic-based agent for the LUTEnv environment.

    TODO: Implement your heuristic strategy in the select_move method.
    """

    def __init__(self, seed: int = 42):
        """Initialize the heuristic agent."""
        self.rng = np.random.default_rng(seed)

    def run_episode(self, env: envs.LUTEnv) -> list[LUT_ID]:
        """
        Run a complete episode using the heuristic agent.

        Returns:
            list of moves taken during the episode
        """
        moves_taken: list[LUT_ID] = []

        while True:
            moves_list = [
                move for move in env.get_moves()
                if env.observe_move(move).largest_new_k() <= 6
            ]
            if not moves_list: break
            moves_list.sort(key=lambda x: env.observe_move(x).largest_new_k())
            move = moves_list[0]

            delta = env.observe_move(move)
            if delta.largest_new_k() > 6:
                print(
                    f"Skipping move {move} due to large new LUT size: {delta.largest_new_k()}"
                )
                continue
            env.commit_move(delta)
            moves_taken.append(move)

        return moves_taken


def test_heuristic_agent(base="arithmetic", name="adder"):
    """
    Test function to evaluate your heuristic agent.

    Args:
        base (str): The benchmark base category.
        name (str): The benchmark name.
    """
    env = epfl.get_env(base, name)
    agent = HeuristicAgent(seed=42)

    print(f"Testing Heuristic Agent on '{base}/{name}' benchmark")
    print("=" * 50)

    # Get initial state
    initial_info = env.get_info()
    print(
        f"Initial state: {initial_info.num_luts} LUTs, depth {initial_info.max_depth}"
    )

    # Run agent
    moves = agent.run_episode(env)

    # Get final state
    final_info = env.get_info()
    print(f"Final state: {final_info.num_luts} LUTs, depth {final_info.max_depth}")
    print(f"Moves taken: {len(moves)}")
    print(f"LUT reduction: {initial_info.num_luts - final_info.num_luts}")
    print(f"Depth reduction: {initial_info.max_depth - final_info.max_depth}")
    print(f"Depth*LUTs: {final_info.max_depth * final_info.num_luts}")

    return moves


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Heuristic Agent on LUTEnv benchmarks.")
    parser.add_argument("--base", type=str, default="arithmetic", help="Benchmark base category (e.g., arithmetic, random_control)")
    parser.add_argument("--name", type=str, default="adder", help="Benchmark name (e.g., adder, bar, sin, etc.)")
    args = parser.parse_args()
    _ = test_heuristic_agent(base=args.base, name=args.name)
    
