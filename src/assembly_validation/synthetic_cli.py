from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .sequence_generator import write_sequence_dataset
from .synthesis import write_trajectory_dataset


def _expected_sequence(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    procedure = config.get("procedure", {})
    sequence = procedure.get("expected_sequence")
    if sequence:
        return list(sequence)
    return [item["name"] for item in config["actions"]]


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    output = Path(args.output)
    if args.command == "sequences":
        summary = write_sequence_dataset(
            output_dir=output,
            expected_sequence=_expected_sequence(config_path),
            samples_per_type=args.samples_per_type,
            seed=args.seed,
            timeout_ms=args.timeout_ms,
        )
    else:
        summary = write_trajectory_dataset(
            config_path=config_path,
            output_dir=output,
            samples_per_action=args.samples_per_action,
            seed=args.seed,
        )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic assembly datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sequences = subparsers.add_parser("sequences", help="Generate procedure anomalies")
    sequences.add_argument("--config", required=True)
    sequences.add_argument("--output", required=True)
    sequences.add_argument("--samples-per-type", type=int, default=100)
    sequences.add_argument("--seed", type=int, default=42)
    sequences.add_argument("--timeout-ms", type=int, default=5000)

    trajectories = subparsers.add_parser("trajectories", help="Generate hand trajectories")
    trajectories.add_argument("--config", required=True)
    trajectories.add_argument("--output", required=True)
    trajectories.add_argument("--samples-per-action", type=int, default=None)
    trajectories.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

