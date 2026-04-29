"""Train a DQN agent on CV-derived LunarLander observations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover - depends on optional package.
    raise SystemExit(
        "stable-baselines3 is required for RL training. "
        "Install dependencies with: pip install -r requirements.txt",
    ) from exc

from lunar_lander_cvrl.envs import make_vision_lander_env
from lunar_lander_cvrl.visualization import TrainingVisualizationCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-weights", required=True, help="Path to CV model weights.")
    parser.add_argument("--save-path", required=True, help="Where to save the trained RL model.")
    parser.add_argument(
        "--load-path",
        default=None,
        help="Optional existing DQN model to continue training from.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load --save-path if it already exists, then train for more timesteps.",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action="store_true",
        help="When loading a model, reset SB3's timestep counter instead of continuing it.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="DQN timesteps to run in this session.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="Torch/SB3 device: auto, cpu, or cuda.")
    parser.add_argument(
        "--obs-mode",
        choices=("hybrid", "cv-only"),
        default="hybrid",
        help="Observation mode for the vision wrapper.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Periodically render the current policy and save GIFs/plots.",
    )
    parser.add_argument(
        "--vis-dir",
        default="runs/visualizations",
        help="Directory for training GIFs and reward plots.",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10_000,
        help="Save one visualization every N training timesteps.",
    )
    parser.add_argument(
        "--vis-max-steps",
        type=int,
        default=0,
        help=(
            "Maximum episode length used for each visualization rollout. "
            "Use 0 to record a full episode until terminated/truncated."
        ),
    )
    parser.add_argument(
        "--vis-fps",
        type=int,
        default=30,
        help="FPS for saved visualization GIFs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/rl/sb3_dqn/periodic",
        help="Directory for periodic SB3 checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Save an SB3 checkpoint every N environment steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--replay-buffer-path",
        default=None,
        help=(
            "Path for DQN replay buffer persistence. Defaults to the save path "
            "with .replay_buffer.pkl suffix."
        ),
    )
    parser.add_argument(
        "--no-save-replay-buffer",
        action="store_true",
        help="Do not save/load the DQN replay buffer between sessions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer_path = (
        Path(args.replay_buffer_path)
        if args.replay_buffer_path
        else save_path.with_suffix(".replay_buffer.pkl")
    )

    env = Monitor(
        make_vision_lander_env(
            cv_weights=args.cv_weights,
            device=args.device,
            obs_mode=args.obs_mode,
            seed=args.seed,
        ),
    )

    callbacks = []
    if args.visualize:
        callbacks.append(
            TrainingVisualizationCallback(
                eval_env=make_vision_lander_env(
                    cv_weights=args.cv_weights,
                    device=args.device,
                    obs_mode=args.obs_mode,
                    seed=args.seed + 10_000,
                ),
                output_dir=args.vis_dir,
                eval_freq=args.vis_freq,
                max_episode_steps=args.vis_max_steps or None,
                fps=args.vis_fps,
                seed=args.seed + 20_000,
                verbose=1,
            ),
        )

    if args.checkpoint_freq > 0:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=save_path.stem,
                save_replay_buffer=not args.no_save_replay_buffer,
                save_vecnormalize=True,
            ),
        )

    callback = CallbackList(callbacks) if callbacks else None

    load_path = _select_load_path(
        save_path=save_path,
        load_path=Path(args.load_path) if args.load_path else None,
        resume=args.resume,
    )

    try:
        if load_path is None:
            model = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                seed=args.seed,
                device=args.device,
            )
            reset_num_timesteps = True
            print("Starting a new DQN model.")
        else:
            model = DQN.load(load_path, env=env, device=args.device)
            if not args.no_save_replay_buffer:
                _load_replay_buffer_if_available(model, replay_buffer_path)
            reset_num_timesteps = args.reset_num_timesteps
            print(f"Loaded DQN model from {load_path}")

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(save_path)
        if not args.no_save_replay_buffer:
            replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_replay_buffer(replay_buffer_path)
            print(f"Saved replay buffer to {replay_buffer_path}")
        print(f"Saved RL model to {save_path}")
    finally:
        env.close()


def _select_load_path(save_path: Path, load_path: Path | None, resume: bool) -> Path | None:
    if resume and save_path.exists():
        return save_path
    if load_path is not None:
        if not load_path.exists():
            raise FileNotFoundError(f"Requested --load-path does not exist: {load_path}")
        return load_path
    if resume:
        print(f"--resume was set, but {save_path} does not exist yet. Starting from scratch.")
    return None


def _load_replay_buffer_if_available(model: DQN, replay_buffer_path: Path) -> None:
    if replay_buffer_path.exists():
        model.load_replay_buffer(replay_buffer_path)
        print(f"Loaded replay buffer from {replay_buffer_path}")
    else:
        print(f"Replay buffer not found at {replay_buffer_path}; continuing without it.")


if __name__ == "__main__":
    main()
