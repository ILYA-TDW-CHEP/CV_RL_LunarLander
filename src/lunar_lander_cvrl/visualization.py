"""Stable-Baselines3 callbacks for visualizing training progress."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:  # pragma: no cover - depends on optional package.
    raise ImportError(
        "stable-baselines3 is required for training visualization. "
        "Install dependencies with: pip install -r requirements.txt",
    ) from exc

from .envs import VisionStateLunarLanderWrapper


@dataclass(frozen=True)
class VisualizationRecord:
    """Evaluation result saved by the visualization callback."""

    timestep: int
    reward: float
    steps: int
    gif_path: Path


class TrainingVisualizationCallback(BaseCallback):
    """Periodically render the current SB3 policy and save GIF/progress plots."""

    def __init__(
        self,
        eval_env: VisionStateLunarLanderWrapper,
        output_dir: str | Path,
        eval_freq: int = 10_000,
        max_episode_steps: int | None = None,
        fps: int = 30,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive.")
        if max_episode_steps is not None and max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive or None.")
        if fps <= 0:
            raise ValueError("fps must be positive.")

        self.eval_env = eval_env
        self.output_dir = Path(output_dir)
        self.episode_dir = self.output_dir / "episodes"
        self.eval_freq = int(eval_freq)
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps is not None else None
        self.fps = int(fps)
        self.seed = int(seed)
        self.records: list[VisualizationRecord] = []

    def _on_training_start(self) -> None:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        self._save_visualization_record()
        return True

    def _on_training_end(self) -> None:
        if not self.records or self.records[-1].timestep != self.num_timesteps:
            self._save_visualization_record()
        self.eval_env.close()

    def _save_visualization_record(self) -> None:
        reward, steps, gif_path = self._record_episode()
        record = VisualizationRecord(
            timestep=int(self.num_timesteps),
            reward=float(reward),
            steps=steps,
            gif_path=gif_path,
        )
        self.records.append(record)
        self.logger.record("visualization/eval_reward", record.reward)
        self._write_csv()
        self._write_reward_plot()

        if self.verbose:
            print(
                f"Visualization saved at step {record.timestep}: "
                f"reward={record.reward:.2f}, episode_steps={record.steps}, "
                f"gif={record.gif_path}",
            )

    def _record_episode(self) -> tuple[float, int, Path]:
        frames: list[np.ndarray] = []
        episode_seed = self.seed + len(self.records)
        obs, _ = self.eval_env.reset(seed=episode_seed)
        frames.append(np.asarray(self.eval_env.render()))

        total_reward = 0.0
        episode_steps = 0
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.eval_env.step(_as_scalar_action(action))
            total_reward += float(reward)
            episode_steps += 1
            frames.append(np.asarray(self.eval_env.render()))
            if terminated or truncated:
                break
            if self.max_episode_steps is not None and episode_steps >= self.max_episode_steps:
                break

        gif_path = self.episode_dir / (
            f"step_{self.num_timesteps:08d}_len_{episode_steps:04d}_reward_{total_reward:+08.2f}.gif"
        )
        _save_gif(frames, gif_path, fps=self.fps)
        return total_reward, episode_steps, gif_path

    def _write_csv(self) -> None:
        csv_path = self.output_dir / "training_visualization.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "reward", "episode_steps", "gif_path"])
            for record in self.records:
                writer.writerow(
                    [
                        record.timestep,
                        record.reward,
                        record.steps,
                        record.gif_path.as_posix(),
                    ],
                )

    def _write_reward_plot(self) -> None:
        if not self.records:
            return

        timesteps = [record.timestep for record in self.records]
        rewards = [record.reward for record in self.records]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(timesteps, rewards, marker="o")
        ax.set_title("Vision LunarLander Training Visualization")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Deterministic episode reward")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_rewards.png", dpi=150)
        plt.close(fig)


def _as_scalar_action(action):
    array = np.asarray(action)
    if array.shape == ():
        return int(array.item())
    if array.size == 1:
        return int(array.reshape(-1)[0])
    return action


def _save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot save a GIF without frames.")

    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(_to_uint8_rgb(frame)) for frame in frames]
    duration_ms = max(1, int(1000 / fps))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA frame, got shape {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)
