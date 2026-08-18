from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .features import FEATURE_DIM, JOINTS_PER_HAND, pack_hands


PRIMITIVES = {"reach", "pick", "move", "align", "insert", "tighten", "inspect"}


def open_hand_template() -> np.ndarray:
    """Return a simple MediaPipe-order, wrist-relative 3D hand skeleton."""
    return np.asarray(
        [
            [0.000, 0.000, 0.000],
            [-0.025, 0.012, 0.000],
            [-0.045, 0.025, -0.002],
            [-0.065, 0.038, -0.004],
            [-0.082, 0.050, -0.006],
            [-0.032, 0.040, 0.000],
            [-0.035, 0.078, -0.003],
            [-0.036, 0.108, -0.006],
            [-0.037, 0.136, -0.008],
            [-0.005, 0.045, 0.000],
            [-0.005, 0.090, -0.004],
            [-0.005, 0.126, -0.008],
            [-0.005, 0.158, -0.011],
            [0.022, 0.041, 0.000],
            [0.024, 0.083, -0.003],
            [0.026, 0.116, -0.006],
            [0.028, 0.144, -0.009],
            [0.046, 0.033, 0.000],
            [0.052, 0.069, -0.002],
            [0.057, 0.097, -0.005],
            [0.061, 0.121, -0.007],
        ],
        dtype=np.float32,
    )


def bend_fingers(hand: np.ndarray, closure: float) -> np.ndarray:
    """Procedurally curl finger joints while preserving MediaPipe topology."""
    result = hand.copy()
    closure = float(np.clip(closure, 0.0, 1.0))
    for finger_start in (5, 9, 13, 17):
        base = result[finger_start].copy()
        for offset in range(1, 4):
            index = finger_start + offset
            extended = result[index].copy()
            curled = base + np.asarray(
                [0.004 * offset, 0.017 * offset, 0.020 * offset], dtype=np.float32
            )
            result[index] = (1.0 - closure) * extended + closure * curled
    for index in (2, 3, 4):
        result[index, 1] -= closure * 0.018 * (index - 1)
        result[index, 2] += closure * 0.012 * (index - 1)
    return result


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return points @ rotation.T


@dataclass(frozen=True)
class ActionSpec:
    name: str
    primitive: str
    hand_mode: str = "right"

    def __post_init__(self) -> None:
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"Unsupported primitive: {self.primitive}")
        if self.hand_mode not in {"right", "left", "both", "random"}:
            raise ValueError(f"Unsupported hand_mode: {self.hand_mode}")


class SyntheticTrajectoryGenerator:
    """Generate randomized two-hand landmark trajectories for one action."""

    def __init__(
        self,
        window_size: int = 45,
        noise_std: float = 0.002,
        occlusion_probability: float = 0.25,
        frame_drop_probability: float = 0.03,
        seed: int = 42,
    ) -> None:
        if window_size < 8:
            raise ValueError("window_size must be >= 8")
        self.window_size = window_size
        self.noise_std = noise_std
        self.occlusion_probability = occlusion_probability
        self.frame_drop_probability = frame_drop_probability
        self.rng = np.random.default_rng(seed)
        self.template = open_hand_template()

    def generate(self, spec: ActionSpec) -> np.ndarray:
        time_power = self.rng.uniform(0.72, 1.38)
        camera_scale = self.rng.uniform(0.86, 1.16)
        camera_roll = self.rng.uniform(-0.18, 0.18)
        shift = self.rng.uniform(-0.045, 0.045, size=3).astype(np.float32)
        shift[2] *= 0.3
        mirror = spec.hand_mode == "left" or (
            spec.hand_mode == "random" and bool(self.rng.integers(0, 2))
        )
        use_both = spec.hand_mode == "both" or (
            spec.hand_mode == "random" and self.rng.random() < 0.28
        )
        occluded_frames: set[int] = set()
        if self.rng.random() < self.occlusion_probability:
            length = int(self.rng.integers(2, max(3, self.window_size // 6)))
            start = int(self.rng.integers(1, self.window_size - length))
            occluded_frames = set(range(start, start + length))

        sequence: list[np.ndarray] = []
        previous: np.ndarray | None = None
        for frame_index in range(self.window_size):
            phase = (frame_index / (self.window_size - 1)) ** time_power
            wrist, closure, hand_angle = self._motion(spec.primitive, phase)
            primary = bend_fingers(self.template, closure)
            primary = _rotate_z(primary, hand_angle)
            primary *= camera_scale * self.rng.uniform(0.94, 1.06)
            primary += wrist

            support: np.ndarray | None = None
            if use_both:
                support_wrist = np.asarray(
                    [0.38 + 0.025 * np.sin(np.pi * phase), 0.55, -0.015],
                    dtype=np.float32,
                )
                support = bend_fingers(self.template, 0.45 + 0.25 * phase)
                support = _rotate_z(support, -0.25)
                support *= camera_scale
                support += support_wrist

            primary = self._camera_transform(primary, camera_roll, shift)
            if support is not None:
                support = self._camera_transform(support, camera_roll, shift)
            primary += self.rng.normal(0.0, self.noise_std, primary.shape).astype(np.float32)
            if support is not None:
                support += self.rng.normal(0.0, self.noise_std, support.shape).astype(np.float32)

            primary_visible = frame_index not in occluded_frames
            if mirror:
                left = self._mirror(primary) if primary_visible else None
                right = self._mirror(support) if support is not None else None
            else:
                left = support
                right = primary if primary_visible else None
            packed = pack_hands(left=left, right=right)
            if previous is not None and self.rng.random() < self.frame_drop_probability:
                packed = previous.copy()
            sequence.append(packed)
            previous = packed

        result = np.stack(sequence).astype(np.float32)
        if result.shape != (self.window_size, FEATURE_DIM):
            raise RuntimeError(f"Unexpected generated shape: {result.shape}")
        return result

    def _motion(self, primitive: str, phase: float) -> tuple[np.ndarray, float, float]:
        smooth = _smoothstep(phase)
        if primitive == "reach":
            wrist = self._lerp((0.74, 0.76, 0.02), (0.52, 0.53, -0.02), smooth)
            return wrist, 0.05 + 0.18 * smooth, -0.15 * smooth
        if primitive == "pick":
            approach = _smoothstep(min(phase / 0.58, 1.0))
            wrist = self._lerp((0.57, 0.62, 0.02), (0.52, 0.52, -0.04), approach)
            closure = _smoothstep(max((phase - 0.42) / 0.48, 0.0))
            return wrist, closure, -0.1
        if primitive == "move":
            wrist = self._lerp((0.43, 0.61, -0.02), (0.67, 0.39, -0.01), smooth)
            return wrist, 0.92, 0.08 + 0.2 * smooth
        if primitive == "align":
            decay = 1.0 - smooth
            wrist = np.asarray(
                [
                    0.61 + 0.035 * decay * np.sin(5.0 * np.pi * phase),
                    0.46 + 0.018 * decay * np.cos(4.0 * np.pi * phase),
                    -0.02,
                ],
                dtype=np.float32,
            )
            return wrist, 0.72, 0.12 * np.sin(4.0 * np.pi * phase) * decay
        if primitive == "insert":
            wrist = self._lerp((0.61, 0.43, 0.015), (0.61, 0.51, -0.075), smooth)
            return wrist, 0.9, 0.02
        if primitive == "tighten":
            angle = 4.0 * np.pi * phase
            radius = 0.025 * (1.0 - 0.35 * smooth)
            wrist = np.asarray(
                [0.61 + radius * np.cos(angle), 0.49 + radius * np.sin(angle), -0.04],
                dtype=np.float32,
            )
            return wrist, 1.0, angle
        if primitive == "inspect":
            wrist = np.asarray(
                [
                    0.55 + 0.1 * smooth,
                    0.47 + 0.025 * np.sin(2.0 * np.pi * phase),
                    -0.01 + 0.025 * np.sin(np.pi * phase),
                ],
                dtype=np.float32,
            )
            return wrist, 0.15, -0.3 + 0.6 * smooth
        raise ValueError(f"Unknown primitive: {primitive}")

    @staticmethod
    def _lerp(start: tuple[float, ...], end: tuple[float, ...], value: float) -> np.ndarray:
        start_array = np.asarray(start, dtype=np.float32)
        end_array = np.asarray(end, dtype=np.float32)
        return start_array + value * (end_array - start_array)

    @staticmethod
    def _camera_transform(points: np.ndarray, angle: float, shift: np.ndarray) -> np.ndarray:
        centered = points - np.asarray([0.5, 0.5, 0.0], dtype=np.float32)
        return _rotate_z(centered, angle) + np.asarray([0.5, 0.5, 0.0]) + shift

    @staticmethod
    def _mirror(points: np.ndarray | None) -> np.ndarray | None:
        if points is None:
            return None
        mirrored = points.copy()
        mirrored[:, 0] = 1.0 - mirrored[:, 0]
        return mirrored


def load_synthesis_config(path: Path) -> tuple[list[ActionSpec], dict[str, object]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    actions = [ActionSpec(**item) for item in config["actions"]]
    names = [action.name for action in actions]
    if len(names) != len(set(names)):
        raise ValueError("Action names must be unique")
    return actions, dict(config.get("generation", {}))


def _split_for_index(index: int, count: int, train: float, val: float) -> str:
    train_count = max(1, int(round(count * train)))
    val_count = max(1, int(round(count * val)))
    if train_count + val_count >= count:
        train_count = max(1, count - 2)
        val_count = 1
    if index < train_count:
        return "train"
    if index < train_count + val_count:
        return "val"
    return "test"


def write_trajectory_dataset(
    config_path: Path,
    output_dir: Path,
    samples_per_action: int | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    actions, generation = load_synthesis_config(config_path)
    count = (
        samples_per_action
        if samples_per_action is not None
        else int(generation.get("samples_per_action", 200))
    )
    actual_seed = seed if seed is not None else int(generation.get("seed", 42))
    splits = dict(generation.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15}))
    train_ratio = float(splits.get("train", 0.7))
    val_ratio = float(splits.get("val", 0.15))
    if count < 3:
        raise ValueError("samples_per_action must be >= 3")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid train/val/test split ratios")

    generator = SyntheticTrajectoryGenerator(
        window_size=int(generation.get("window_size", 45)),
        noise_std=float(generation.get("noise_std", 0.002)),
        occlusion_probability=float(generation.get("occlusion_probability", 0.25)),
        frame_drop_probability=float(generation.get("frame_drop_probability", 0.03)),
        seed=actual_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    class_counts: dict[str, dict[str, int]] = {}
    sample_number = 0
    for action in actions:
        class_counts[action.name] = {"train": 0, "val": 0, "test": 0}
        for index in range(count):
            split = _split_for_index(index, count, train_ratio, val_ratio)
            features = generator.generate(action)
            sample_path = output_dir / f"sample_{sample_number:07d}.npz"
            np.savez_compressed(
                sample_path,
                features=features,
                label=np.asarray(action.name),
                primitive=np.asarray(action.primitive),
                seed=np.asarray(actual_seed),
            )
            rows.append(
                {"sample_path": sample_path.as_posix(), "label": action.name, "split": split}
            )
            class_counts[action.name][split] += 1
            sample_number += 1

    index_path = output_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "seed": actual_seed,
        "feature_dim": FEATURE_DIM,
        "window_size": generator.window_size,
        "samples_per_action": count,
        "total_samples": sample_number,
        "actions": [action.name for action in actions],
        "class_counts": class_counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
