from pathlib import Path

import numpy as np

from assembly_validation.features import FEATURE_DIM
from assembly_validation.synthesis import (
    ActionSpec,
    SyntheticTrajectoryGenerator,
    write_trajectory_dataset,
)


def test_generated_trajectory_contract() -> None:
    generator = SyntheticTrajectoryGenerator(window_size=20, seed=4)
    features = generator.generate(ActionSpec("tighten", "tighten", "right"))
    assert features.shape == (20, FEATURE_DIM)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert set(np.unique(features[:, -2:])).issubset({0.0, 1.0})


def test_actions_have_different_wrist_trajectories() -> None:
    reach_generator = SyntheticTrajectoryGenerator(window_size=20, noise_std=0.0, seed=2)
    tighten_generator = SyntheticTrajectoryGenerator(window_size=20, noise_std=0.0, seed=2)
    reach = reach_generator.generate(ActionSpec("reach", "reach", "right"))
    tighten = tighten_generator.generate(ActionSpec("tighten", "tighten", "right"))
    right_wrist_slice = slice(129, 132)
    assert not np.allclose(reach[:, right_wrist_slice], tighten[:, right_wrist_slice])


def test_write_trajectory_dataset(tmp_path: Path) -> None:
    config = Path("configs/synthetic_actions.yaml")
    summary = write_trajectory_dataset(config, tmp_path, samples_per_action=3, seed=8)
    assert summary["total_samples"] == 21
    assert summary["feature_dim"] == FEATURE_DIM
    assert summary["class_counts"]["reach_component"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert (tmp_path / "index.csv").is_file()
    assert len(list(tmp_path.glob("*.npz"))) == 21
