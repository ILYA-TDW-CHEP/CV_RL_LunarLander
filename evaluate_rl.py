"""Evaluate a trained SB3 agent on CV-derived LunarLander observations."""

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

from lunar_lander_cvrl.envs import make_vision_lander_env
from lunar_lander_cvrl.models.rl.utils import get_algorithm_class, validate_evaluation_args


def run_evaluation(args: SimpleNamespace) -> None:
    validate_evaluation_args(args)
    env = make_vision_lander_env(
        cv_weights=args.cv_weights,
        cv_model_type=args.cv_model_type,
        cv_metadata=args.cv_metadata,
        device=args.device,
        obs_mode=args.obs_mode,
        seed=args.seed,
    )
    model_cls = get_algorithm_class(args.algorithm)
    model = model_cls.load(args.model_path, env=env, device=args.device)

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
        algorithm=cfg.model.algorithm,
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


def main() -> None:
    _run_hydra_main()


if __name__ == "__main__":
    main()
