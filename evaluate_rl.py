"""Evaluate a trained DQN agent on CV-derived LunarLander observations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from stable_baselines3 import DQN
except ImportError as exc:  # pragma: no cover - depends on optional package.
    raise SystemExit(
        "stable-baselines3 is required for RL evaluation. "
        "Install dependencies with: pip install -r requirements.txt",
    ) from exc

from lunar_lander_cvrl.envs import make_vision_lander_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-weights", required=True, help="Path to CV model weights.")
    parser.add_argument("--model-path", required=True, help="Path to a saved DQN model.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument("--device", default="auto", help="Torch/SB3 device: auto, cpu, or cuda.")
    parser.add_argument(
        "--obs-mode",
        choices=("hybrid", "cv-only"),
        default="hybrid",
        help="Observation mode for the vision wrapper.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=200.0,
        help="Episode reward threshold counted as a successful landing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_vision_lander_env(
        cv_weights=args.cv_weights,
        device=args.device,
        obs_mode=args.obs_mode,
        seed=args.seed,
    )
    model = DQN.load(args.model_path, env=env, device=args.device)

    rewards: list[float] = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            done = False
            episode_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                done = bool(terminated or truncated)

            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: reward={episode_reward:.2f}")
    finally:
        env.close()

    rewards_array = np.asarray(rewards, dtype=np.float32)
    successes = int(np.sum(rewards_array >= args.success_threshold))
    print(f"Mean reward: {float(np.mean(rewards_array)):.2f}")
    print(f"Std reward: {float(np.std(rewards_array)):.2f}")
    print(f"Successful landings: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
