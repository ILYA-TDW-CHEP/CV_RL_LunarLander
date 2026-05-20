"""Evaluate a trained DQN agent on CV-derived LunarLander observations."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import hydra
except ImportError:  # pragma: no cover - exercised only without optional deps.
    hydra = None

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
from lunar_lander_cvrl.models.cv import CV_MODEL_TYPES


def run_evaluation(args: SimpleNamespace) -> None:
    _validate_args(args)
    env = make_vision_lander_env(
        cv_weights=args.cv_weights,
        cv_model_type=args.cv_model_type,
        cv_metadata=args.cv_metadata,
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


def _config_to_args(cfg) -> SimpleNamespace:
    return SimpleNamespace(
        cv_weights=cfg.cv.weights,
        cv_model_type=cfg.cv.model_type,
        cv_metadata=cfg.cv.metadata,
        model_path=cfg.model.path,
        episodes=cfg.evaluation.episodes,
        seed=cfg.seed,
        device=cfg.device,
        obs_mode=cfg.env.obs_mode,
        success_threshold=cfg.evaluation.success_threshold,
    )


def _run_hydra_main() -> None:
    if hydra is None:
        raise SystemExit(
            "Hydra is required for config-driven RL evaluation. "
            "Install dependencies with: pip install -r requirements.txt",
        )

    @hydra.main(version_base=None, config_path="configs/rl", config_name="evaluate")
    def _main(cfg) -> None:
        run_evaluation(_config_to_args(cfg))

    _main()


def _validate_args(args: SimpleNamespace) -> None:
    if args.cv_model_type not in CV_MODEL_TYPES:
        raise ValueError(f"cv.model_type must be one of {CV_MODEL_TYPES}, got {args.cv_model_type!r}.")
    if args.obs_mode not in ("hybrid", "cv-only"):
        raise ValueError("env.obs_mode must be either 'hybrid' or 'cv-only'.")
    if args.episodes <= 0:
        raise ValueError("evaluation.episodes must be positive.")


def main() -> None:
    _run_hydra_main()


if __name__ == "__main__":
    main()
