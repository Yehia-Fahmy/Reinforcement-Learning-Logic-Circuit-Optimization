"""
Example 2: Random Agent Baseline

This example implements a simple random agent that serves as a baseline for comparison.
The agent randomly selects from valid moves until no more moves are possible.

Usage:
    python random_agent.py --base arithmetic --name adder
    python random_agent.py --base random_control --name arbiter --seed 123

This random agent provides a lower bound on performance - your heuristic and RL agents
should significantly outperform this baseline.
"""

import argparse
from typing import TextIO
import opt_gym.epfl as epfl
import opt_gym.envs as envs
from opt_gym.core import LUT_ID
import numpy as np


def write_info(f: TextIO, info: envs.NetlistInfo):
    f.write(f"num_inputs: {info.num_inputs}\n")
    f.write(f"num_intermediates: {info.num_intermediates}\n")
    f.write(f"num_outputs: {info.num_outputs}\n")
    f.write(f"num_luts: {info.num_luts}\n")
    f.write(f"max_depth: {info.max_depth}\n")
    f.write(f"lut_counts: {info.lut_counts}\n")
    f.write(f"fanout_counts: {info.fanout_counts}\n")
    f.write(f"depth_counts: {info.depth_counts}\n")
    f.write(f"depth_list: {info.depth_list}\n")


def main():
    parser = argparse.ArgumentParser(description="Random agent baseline for LUT optimization")
    parser.add_argument("--base", required=True, help="Benchmark base name (e.g., 'arithmetic', 'random_control')")
    parser.add_argument("--name", required=True, help="Benchmark name (e.g., 'adder', 'arbiter')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log", default="random_agent.log", help="Log file name for results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(args.log, "a") as f:
        env = epfl.get_env(args.base, args.name)
        f.write(f"\n\nBenchmark: {args.base}, {args.name}, Seed: {args.seed}\n")

        # Record initial state
        info = env.get_info()
        f.write("Initial Netlist Info:\n")
        write_info(f, info)
        f.write("\n")

        # Get all possible moves and shuffle them randomly
        moves = list(env.get_moves())
        actual_moves: list[LUT_ID] = []
        rng.shuffle(moves)

        print(f"Starting random agent on {args.base}/{args.name}")
        print(f"Initial state: {info.num_luts} LUTs, depth {info.max_depth}")
        print(f"Attempting up to {len(moves)} moves...")

        # Try each move in random order
        for move in moves:
            # Check if move is valid (doesn't exceed 6-input constraint)
            delta = env.observe_move(move)
            new_k = delta.largest_new_k()
            if new_k > 6:
                continue

            # Apply the valid move
            actual_moves.append(move)
            env.commit_move(delta)

        f.write(f"Actual moves applied: {actual_moves}\n")
        f.write(f"Number of moves: {len(actual_moves)}\n")

        # Record final state
        info = env.get_info()
        f.write("Final Netlist Info:\n")
        write_info(f, info)

        print(f"Final state: {info.num_luts} LUTs, depth {info.max_depth}")
        print(f"Applied {len(actual_moves)} moves")
        print(f"Results written to {args.log}")
        print("Done!")


if __name__ == "__main__":
    main()
