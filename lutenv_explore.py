"""
Basic LUTEnv Environment Usage

This example demonstrates the fundamental operations you can perform with the LUTEnv environment:
- Loading a benchmark circuit
- Inspecting circuit properties
- Exploring possible moves (LUT merge operations)
- Applying moves and observing their effects

Key concepts:
- LUT_ID: Integer identifiers for LUTs that can be merged
- Delta: Represents the prospective change from a move before committing
- Constraint checking: Merged LUTs must have ≤6 inputs (largest_new_k() ≤ 6)
"""

from opt_gym.core import LUT_ID
import opt_gym.epfl as epfl
import opt_gym.envs as envs


def print_info(info: envs.NetlistInfo):
    """Helper function to display netlist information in a readable format."""
    print(f"num_luts: {info.num_luts}")
    print(f"max_depth: {info.max_depth}")
    print(f"lut_counts: {info.lut_counts}")
    print(f"fanout_counts: {info.fanout_counts}")
    print(f"depth_counts: {info.depth_counts}")
    print()


def main():
    """Demonstrate basic environment usage."""
    print("Basic LUTEnv Environment Usage")
    print("=" * 50)

    # 1. Get the environment for a specific benchmark
    env = epfl.get_env("arithmetic", "adder")
    print("Loaded benchmark: arithmetic/adder")

    # 2. Explore available moves
    # Moves are LUT_IDs (integers) between 0 and env.netlist.max_id()-1
    moves = list(env.get_moves())  # All possible LUT merge operations
    print(f"Total possible moves: {len(moves)}")
    print(f"First 10 moves: {moves[:10]}")
    print()

    # 3. Select a few moves to demonstrate
    # Note: These specific moves may not always be valid for all benchmarks
    move_list = [LUT_ID(569), LUT_ID(888), LUT_ID(601), LUT_ID(705), LUT_ID(1000)]
    print(f"Selected moves to attempt: {move_list}")
    print()

    # 4. Print initial netlist information
    info = env.get_info()
    print("Initial Netlist State:")
    print_info(info)

    # 5. Apply the moves to the environment
    successful_moves: list[LUT_ID] = []
    for i, move in enumerate(move_list):
        print(f"Attempting move {i+1}/{len(move_list)}: LUT_ID {move}")

        # First, observe what would happen if we made this move
        delta = env.observe_move(move)
        new_k = delta.largest_new_k()
        print(f"  -> Largest new LUT would have {new_k} inputs")

        if new_k > 6:
            print("  -> SKIPPED: Would violate 6-input constraint")
            print()
            continue

        # The move is valid, so commit it to the environment
        env.commit_move(delta)
        successful_moves.append(move)
        print("  -> SUCCESS: Move applied")


        print()

    # 6. Show final state after all moves
    info = env.get_info()
    print("Final Netlist State:")
    print_info(info)

    print(f"Successfully applied {len(successful_moves)} out of {len(move_list)} attempted moves")
    print(f"Successful moves: {successful_moves}")
    print("\nDone! This demonstrates the basic environment interface.")


if __name__ == "__main__":
    main()
