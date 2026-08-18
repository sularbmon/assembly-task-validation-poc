from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .features import FEATURE_DIM, MediaPipeHandExtractor


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video_path", "label", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires columns: {sorted(required)}")
    return rows


def extract_segment(
    video_path: Path,
    start_sec: float | None,
    end_sec: float | None,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int((start_sec or 0.0) * fps)
    end_frame = int(end_sec * fps) if end_sec is not None else None
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    frame_number = start_frame
    with MediaPipeHandExtractor() as extractor:
        while end_frame is None or frame_number < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(extractor.extract(frame).features)
            frame_number += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return np.stack(frames).astype(np.float32)


def make_windows(sequence: np.ndarray, size: int, stride: int) -> list[np.ndarray]:
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [time, {FEATURE_DIM}], got {sequence.shape}")
    if len(sequence) < size:
        padding = np.repeat(sequence[-1:], size - len(sequence), axis=0)
        return [np.concatenate([sequence, padding])]
    starts = list(range(0, len(sequence) - size + 1, stride))
    final_start = len(sequence) - size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [sequence[start : start + size] for start in starts]


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(read_manifest(Path(args.manifest))):
        video = Path(row["video_path"])
        sequence = extract_segment(
            video,
            _optional_float(row.get("start_sec")),
            _optional_float(row.get("end_sec")),
        )
        for window_number, window in enumerate(
            make_windows(sequence, args.window_size, args.stride)
        ):
            sample_name = f"sample_{row_number:05d}_{window_number:04d}.npz"
            sample_path = output / sample_name
            np.savez_compressed(sample_path, features=window)
            index_rows.append(
                {"sample_path": str(sample_path), "label": row["label"], "split": row["split"]}
            )

    with (output / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_path", "label", "split"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Prepared {len(index_rows)} windows in {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand windows")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size", type=int, default=45)
    parser.add_argument("--stride", type=int, default=10)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

