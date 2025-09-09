import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
import chex
import numpy as np
from tqdm import tqdm
import os
import math
import time
from pathlib import Path
from dataclasses import dataclass, field

import opt_gym.epfl as epfl
import opt_gym.envs as envs
from opt_gym.core import LUT_ID

import argparse

os.makedirs("test", exist_ok=True)


def get_lut_features(env: envs.LUTEnv, info: envs.NetlistInfo) -> np.ndarray:
    """
    Generates features for each LUT in the netlist.

    This is slow since after each move you recompute the features for every LUT.

    In principle you could cache the features and only recompute them when the netlist changes.
    But for now, we recompute them every time to ensure correctness.
    """

    # TODO: Implement feature calculation
    # Feature 1: Normalized number of LUT inputs (K/6)
    # Feature 2: Fan-out (normalized by 8, capped at 1)
    # Feature 3: Normalized depth
    # Feature 4: Average depth of input LUTs
    # Feature 5: Max K of fan-out LUTs
    # Feature 6: Max K of fan-in LUTs
    # Feature 7: Is primary output (binary)
    # More features?

    # Stack all features
    features = np.column_stack(
        [
            feature_k,
            feature_fan_out,
            feature_depth,
            feature_avg_input_depth,
            feature_max_fan_out_k,
            feature_max_fan_in_k,
            feature_is_output,
        ]
    )

    return features


class PolicyNetwork(nnx.Module):
    """A simple repeated MLP for the policy."""

    def __init__(
        self, d_in: int, d_latent: int, d_middle: int, n_blocks: int, rngs: nnx.Rngs
    ):
        self.input_projection = nnx.Linear(d_in, d_latent, rngs=rngs)
        self.mlp_blocks = [
            nnx.Sequential(
                # TODO: Implement a neural network architecture, linear/relu, etc as you wish
            )
            for _ in range(n_blocks)
        ]
        self.output_projection = nnx.Linear(
            d_latent, 1, rngs=rngs
        )  # Single output for logits

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # TODO: Implement forward pass
        # 1. Input projection
        # 2. Multiple residual blocks
        # 3. Output projection
        return self.output_projection(x)


def action_sample(
    policy: PolicyNetwork, features: jnp.ndarray, mask: jnp.ndarray, key: chex.PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Samples an action from the policy given the features and mask.
    Returns the log probability of the action and the action itself.
    """
    # TODO: Implement action sampling
    # 1. Get logits from policy network
    # 2. Apply mask to invalid actions
    # 3. Sample from categorical distribution
    # 4. Compute log probability of chosen action
    return log_prob, chosen


# Returns ((log_prob, action), gradient state with respect to choice)
action_sample_grad = nnx.jit(nnx.value_and_grad(action_sample, has_aux=True))


@nnx.jit(donate_argnames=("x",))
def sum_gradients(x: nnx.State, y: nnx.State) -> nnx.State:
    return jax.tree.map(lambda a, b: a + b, x, y)


@nnx.jit(donate_argnames=("grads",))
def scale_gradients(grads: nnx.State, scale: float) -> nnx.State:
    return jax.tree.map(lambda g: g * scale, grads)


@nnx.jit
def abs_average(pytree: nnx.State) -> jnp.ndarray:
    acc = jnp.zeros(())
    num = 0
    for path, leaf in nnx.iter_graph(pytree):
        _ = path  # Ignore path, we only care about the values
        if not isinstance(leaf, jnp.ndarray):
            continue
        acc = acc + jnp.sum(jnp.abs(leaf))
        num = num + jnp.size(leaf)
    assert num > 0, "No valid gradients found in the pytree."
    return acc / num


@nnx.jit
def optimizer_update(optimizer: nnx.optimizer.Optimizer, grads: nnx.State) -> None:
    optimizer.update(grads)


@dataclass
class TrainingConfig:
    """Configuration for a training session."""

    num_episodes: int
    max_time: float
    # If None, no limit on steps per episode
    max_steps_per_episode: int | None = None
    # Warmup episodes update baseline but not policy
    warmup_episodes: int = 0
    baseline_alpha: float = 0.9
    advantage_alpha: float = 0.9
    description: str = ""
    # Optional seeded values for baseline and advantage tracking
    # If not provided, you should have some warmup episodes.
    initial_baseline: float | None = None
    initial_average_advantage: float | None = None


@dataclass
class EpisodeResult:
    """Results from a single episode."""

    episode_id: int
    reward: float
    baseline: float
    advantage: float
    scaled_advantage: float
    final_info: envs.NetlistInfo
    move_order: list[LUT_ID]
    timestamp: float
    average_grad: float
    env_name: str = ""


@dataclass
class TrainingResults:
    """Results from a training session."""

    episodes: list[EpisodeResult] = field(default_factory=list[EpisodeResult])
    best_reward: float = float("-inf")
    best_episode: EpisodeResult | None = None
    best_env: envs.LUTEnv | None = None
    baseline: float = float("nan")
    average_advantage: float = 1.0
    total_episodes: int = 0
    start_time: float = field(default_factory=time.time)

    def add_episode(self, episode: EpisodeResult, env: envs.LUTEnv):
        """Add an episode result and update best tracking."""
        self.episodes.append(episode)
        self.total_episodes += 1

        if episode.reward > self.best_reward:
            self.best_reward = episode.reward
            self.best_episode = episode
            self.best_env = env.copy()

    @property
    def rewards(self) -> list[float]:
        return [ep.reward for ep in self.episodes]

    @property
    def episode_timestamps(self) -> list[float]:
        return [ep.timestamp for ep in self.episodes]

    @property
    def total_time(self) -> float:
        return time.time() - self.start_time if self.episodes else 0.0


class RLAgent:
    def __init__(self, num_features: int, learning_rate: float, seed: int = 0):
        self.rng = nnx.Rngs(seed, params=seed + 1, action=seed + 2)
        self.policy = PolicyNetwork(
            d_in=num_features, d_latent=32, d_middle=64, n_blocks=4, rngs=self.rng
        )
        optimizer = optax.adam(learning_rate)
        self.optimizer = nnx.optimizer.Optimizer(self.policy, optimizer)

    def run_episode(
        self,
        env: envs.LUTEnv,
        config: TrainingConfig,
        episode_id: int,
        results: TrainingResults,
        env_name: str = "",
        update_policy: bool = True,
    ) -> tuple[EpisodeResult, envs.LUTEnv]:
        """Run a single episode and return results."""
        ep_env = env.copy()
        ep_grads: nnx.State | None = None
        episode_move_order: list[LUT_ID] = []

        # Mask for illegal actions
        mask = np.full(ep_env.netlist.max_id(), -np.inf, dtype=np.float32)
        legal_moves = ep_env.get_moves()
        np.put(mask, legal_moves, 0.0)

        total_steps = config.max_steps_per_episode
        if total_steps is None:
            total_steps = env.netlist.max_id()

        # Run episode steps
        for step in range(total_steps):
            if not np.any(mask == 0.0):
                break

            # First get features for the current state
            features = get_lut_features(ep_env, ep_env.get_info())

            # Sample an action, also get gradients for that action
            # (Note: Gradients are the direction to update the policy weights to increase the probability of this action)
            (log_prob, action), grads = action_sample_grad(
                self.policy, features, mask, self.rng.action()
            )

            action_id = LUT_ID(int(action))
            episode_move_order.append(action_id)

            # We mask the action so it can't be chosen again
            mask[action_id] = -np.inf
            delta = ep_env.observe_move(action_id)

            # Check if the move is valid (does not violate 6-input constraint)
            if delta.largest_new_k() <= 6:
                ep_env.commit_move(delta)
                if ep_grads is None:
                    ep_grads = grads
                else:
                    # Accumulate gradients for this episode
                    ep_grads = sum_gradients(ep_grads, grads)

        # After all moves, calculate rewards and advantages
        final_info = ep_env.get_info()

        # TODO: play with reward
        reward = -(final_info.max_depth * final_info.num_luts)

        # Update baseline (stored in results)
        # Exponentially moving average
        if math.isnan(results.baseline):
            results.baseline = reward
        else:
            results.baseline = #TODO -- compute exponentially moving average of rewards(
            )

        # Advatage is improvment over baseline
        advantage = #TODO -- calculate advantge over baseline

        # Rescale advantage (normalizes the variance)
        scaled_advantage = #TODO noramlize

        # Calculate average gradient magnitude for this episode
        # This is a measure of how much the policy change
        # Its important its not exploding or too small.
        average_grad = float(abs_average(ep_grads)) if ep_grads is not None else 0.0

        # Update policy if requested and past warmup
        if (
            update_policy
            and ep_grads is not None
            and episode_id >= config.warmup_episodes
        ):
            final_grads = scale_gradients(ep_grads, -scaled_advantage)
            optimizer_update(self.optimizer, final_grads)

        episode_result = EpisodeResult(
            episode_id=episode_id,
            reward=reward,
            baseline=results.baseline,
            advantage=advantage,
            scaled_advantage=scaled_advantage,
            final_info=final_info,
            move_order=episode_move_order,
            timestamp=time.time() - results.start_time,
            average_grad=average_grad,
            env_name=env_name,
        )

        return episode_result, ep_env

    def train_session(
        self,
        env: envs.LUTEnv,
        config: TrainingConfig,
        results: TrainingResults | None = None,
        env_name: str = "",
        verbose: bool = True,
    ) -> TrainingResults:
        """Run a training session, if results are provided, its a continuation of that session"""
        if results is None:
            results = TrainingResults()
            # Seed baseline and advantage if provided in config
            if config.initial_baseline is not None:
                results.baseline = config.initial_baseline
            if config.initial_average_advantage is not None:
                results.average_advantage = config.initial_average_advantage

        start_episode = results.total_episodes

        for episode in tqdm(
            range(config.num_episodes),
            desc=f"Training {config.description or env_name}",
            disable=not verbose,
        ):
            current_time = time.time()
            elapsed_time = current_time - results.start_time

            # Check time limit
            if elapsed_time > config.max_time and episode > 0:
                break

            episode_id = start_episode + episode
            episode_result, ep_env = self.run_episode(
                env, config, episode_id, results, env_name, update_policy=True
            )

            results.add_episode(episode_result, ep_env)

            if verbose:
                print(
                    f"Episode {episode_id}: Reward={episode_result.reward:.2f}, "
                    f"Baseline={episode_result.baseline:.2f}, "
                    f"Advantage={episode_result.advantage:.2f}, "
                    f"Depth={episode_result.final_info.max_depth}, "
                    f"LUTs={episode_result.final_info.num_luts}"
                )

        return results

    def evaluate(
        self, env: envs.LUTEnv, num_episodes: int = 1, env_name: str = ""
    ) -> TrainingResults:
        """Run evaluation episodes without policy updates."""
        config = TrainingConfig(
            num_episodes=num_episodes,
            max_time=float("inf"),
            description=f"Evaluation on {env_name}",
        )

        eval_results = TrainingResults()
        for episode in range(num_episodes):
            episode_result, ep_env = self.run_episode(
                env, config, episode, eval_results, env_name, update_policy=False
            )
            eval_results.add_episode(episode_result, ep_env)

        return eval_results

    def save(self, path: Path) -> None:
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        with ocp.StandardCheckpointer() as ckptr:
            policy_graph, policy_state = nnx.split(self.policy)
            opt_graph, opt_state = nnx.split(self.optimizer)
            ckptr.save(path / "policy", policy_state)
            ckptr.save(path / "optimizer", opt_state)

    def load(self, path: Path) -> None:
        with ocp.StandardCheckpointer() as ckptr:
            policy_graph, policy_state = nnx.split(self.policy)
            opt_graph, opt_state = nnx.split(self.optimizer)
            policy_state_t = ckptr.restore(path / "policy", target=policy_state)
            opt_state_t = ckptr.restore(path / "optimizer", target=opt_state)
            self.policy = nnx.merge(policy_graph, policy_state_t)
            self.optimizer = nnx.merge(opt_graph, opt_state_t)


def main():
    # Setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Benchmark base name (e.g., 'arithmetic', 'random_control')")
    parser.add_argument("--name", required=True, help="Benchmark name (e.g., 'adder', 'arbiter')")
    parser.add_argument("--episodes", type=int, default=10, help="Total number of training episodes")
    parser.add_argument("--warmup", type=int, default=4, help="Number of warmup episodes")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the RL agent")
    parser.add_argument("--max_time", type=float, default=120.0, help="Max time allowed per episode (in seconds)")
    parser.add_argument("--features", type=int, default=7, help="Number of input features to the agent")
    parser.add_argument("--description", type=str, default="Initial training", help="Description of training run")

    args = parser.parse_args()

    # Environment setup
    env = epfl.get_env(args.base, args.name)
    initial_info = env.get_info()

    print(f"Benchmark: {args.base} {args.name}")
    print(f"Initial state: {initial_info.num_luts} LUTs, {initial_info.max_depth} depth")

    # RL Agent
    agent = RLAgent(num_features=args.features, learning_rate=args.lr)

    # Training config
    initial_config = TrainingConfig(
        num_episodes=args.episodes,
        max_time=args.max_time,
        warmup_episodes=args.warmup,
        description=args.description,
    )

    # Continued config (example)
    continued_config = TrainingConfig(
        num_episodes=args.episodes,
        max_time=args.max_time,
        warmup_episodes=0,
        description="Continued training",
    )

    env_name = f"{args.base}_{args.name}"

    # Initial training
    results = agent.train_session(env, initial_config, env_name=env_name)

    # Continued training (same results object)
    results = agent.train_session(env, continued_config, results, env_name=env_name)

    # Evaluation (separate results = separate baseline, no policy updates)
    eval_results = agent.evaluate(env, num_episodes=3, env_name=env_name)
    _ = eval_results

    # Save and load, note, you will get error for overwriting!
    # agent.save(Path.cwd() / "checkpoints" / "test")
    # agent.load(Path.cwd() / "checkpoints" / "test")

    print("\nTraining finished.")
    if results.best_episode:
        print(
            f"Best result: Depth={results.best_episode.final_info.max_depth}, "
            f"LUTs={results.best_episode.final_info.num_luts}"
        )

    if results.best_env:
        results.best_env.emit_dotfile("reinforce.dot")

    # Write results
    with open("reinforce.log", "a") as f:
        f.write(f"Benchmark: {args.base} {args.name}\n")
        f.write(
            f"Initial: {initial_info.num_luts} LUTs, {initial_info.max_depth} depth\n"
        )
        if results.best_episode:
            f.write(
                f"Final: {results.best_episode.final_info.num_luts} LUTs, "
                f"{results.best_episode.final_info.max_depth} depth\n"
            )
        f.write(f"Total episodes: {results.total_episodes}\n")
        f.write(f"Total time: {results.total_time:.2f}s\n\n")


if __name__ == "__main__":
    main()
