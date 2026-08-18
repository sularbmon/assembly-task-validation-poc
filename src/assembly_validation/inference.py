from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .features import MediaPipeHandExtractor
from .model import load_model
from .validator import EventType, SequenceValidator


class ProbabilitySmoother:
    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value: np.ndarray | None = None

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        if self.value is None:
            self.value = probabilities.astype(np.float32)
        else:
            self.value = self.alpha * probabilities + (1.0 - self.alpha) * self.value
        return self.value


def _source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def run(args: argparse.Namespace) -> None:
    procedure = yaml.safe_load(Path(args.procedure).read_text(encoding="utf-8"))
    settings = procedure.get("validation", {})
    expected_sequence = list(procedure["expected_sequence"])
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device if args.device else default_device)
    model, class_names = load_model(args.model, device)
    missing = set(expected_sequence) - set(class_names)
    if missing:
        raise ValueError(f"Procedure labels missing from checkpoint: {sorted(missing)}")

    validator = SequenceValidator(
        expected_sequence,
        confidence_threshold=float(settings.get("confidence_threshold", 0.7)),
        confirmation_windows=int(settings.get("confirmation_windows", 4)),
        step_timeout_ms=settings.get("step_timeout_ms"),
    )
    smoother = ProbabilitySmoother(float(settings.get("smoothing_alpha", 0.35)))
    stride = int(settings.get("inference_stride", 5))
    window_size = args.window_size
    window: deque[np.ndarray] = deque(maxlen=window_size)

    capture = cv2.VideoCapture(_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    events_path = output_dir / "events.jsonl"
    events_handle = events_path.open("w", encoding="utf-8")
    frame_number = 0
    current_label = "warming_up"
    current_confidence = 0.0
    last_event = "WAIT"
    last_timestamp_ms = 0
    started = time.perf_counter()

    with MediaPipeHandExtractor() as extractor:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            window.append(extractor.extract(frame).features)
            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = int(frame_number / fps * 1000)
            last_timestamp_ms = timestamp_ms
            if len(window) == window_size and frame_number % stride == 0:
                tensor = torch.from_numpy(np.stack(window)[None]).to(device)
                with torch.no_grad():
                    probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                smoothed = smoother.update(probabilities)
                class_id = int(smoothed.argmax())
                current_label = class_names[class_id]
                current_confidence = float(smoothed[class_id])
                event = validator.observe(current_label, current_confidence, timestamp_ms)
                if event is None:
                    event = validator.check_timeout(timestamp_ms)
                if event is not None:
                    last_event = event.event.value
                    events_handle.write(json.dumps(event.to_dict()) + "\n")
                    events_handle.flush()

            status_color = (0, 200, 0) if last_event in {
                EventType.STEP_OK.value,
                EventType.COMPLETE.value,
            } else (0, 165, 255) if last_event == "WAIT" else (0, 0, 255)
            lines = [
                f"Action: {current_label} ({current_confidence:.2f})",
                f"Expected: {validator.expected_action or 'complete'}",
                f"Step: {validator.position}/{len(expected_sequence)}",
                f"Status: {last_event}",
            ]
            for line_number, text in enumerate(lines):
                cv2.putText(
                    frame,
                    text,
                    (20, 40 + line_number * 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            if args.display:
                cv2.imshow("Assembly Task Validation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_number += 1

    final_event = validator.finalize(last_timestamp_ms)
    if final_event is not None:
        events_handle.write(json.dumps(final_event.to_dict()) + "\n")
    elapsed = time.perf_counter() - started
    events_handle.close()
    writer.release()
    capture.release()
    cv2.destroyAllWindows()
    summary = {
        "source": args.source,
        "frames": frame_number,
        "processing_fps": frame_number / max(elapsed, 1e-6),
        "completed_steps": validator.position,
        "total_steps": len(expected_sequence),
        "complete": validator.complete,
        "device": str(device),
        "class_names": class_names,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-time assembly validation")
    parser.add_argument("--source", required=True, help="Video path, RTSP URL or webcam index")
    parser.add_argument("--model", required=True)
    parser.add_argument("--procedure", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size", type=int, default=45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--display", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
