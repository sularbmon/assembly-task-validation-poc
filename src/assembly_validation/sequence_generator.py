from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class AnomalyType(str, Enum):
    CORRECT = "correct"
    SKIP = "skip"
    REPEAT = "repeat"
    REORDER = "reorder"
    UNEXPECTED = "unexpected"
    TIMEOUT = "timeout"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class ProcedureEvent:
    action: str
    start_ms: int
    end_ms: int
    confidence: float
    timed_out: bool = False

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["duration_ms"] = self.duration_ms
        return value


@dataclass(frozen=True)
class ProcedureRecord:
    sequence_id: str
    anomaly_type: AnomalyType
    is_valid: bool
    expected_sequence: list[str]
    observed_actions: list[str]
    events: list[ProcedureEvent]
    anomaly_index: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "anomaly_type": self.anomaly_type.value,
            "is_valid": self.is_valid,
            "expected_sequence": self.expected_sequence,
            "observed_actions": self.observed_actions,
            "events": [event.to_dict() for event in self.events],
            "anomaly_index": self.anomaly_index,
        }


class SequenceDatasetGenerator:
    """Generate correct and deliberately invalid assembly procedures."""

    def __init__(
        self,
        expected_sequence: list[str],
        seed: int = 42,
        duration_range_ms: tuple[int, int] = (500, 1800),
        gap_range_ms: tuple[int, int] = (80, 350),
        timeout_ms: int = 5000,
        unexpected_label: str = "unexpected_action",
    ) -> None:
        if len(expected_sequence) < 3:
            raise ValueError("At least three expected actions are required")
        self.expected_sequence = list(expected_sequence)
        self.random = random.Random(seed)
        self.duration_range_ms = duration_range_ms
        self.gap_range_ms = gap_range_ms
        self.timeout_ms = timeout_ms
        self.unexpected_label = unexpected_label

    def generate(self, anomaly_type: AnomalyType, sequence_id: str) -> ProcedureRecord:
        actions = list(self.expected_sequence)
        anomaly_index: int | None = None
        timeout_index: int | None = None

        if anomaly_type == AnomalyType.SKIP:
            anomaly_index = self.random.randrange(len(actions))
            actions.pop(anomaly_index)
        elif anomaly_type == AnomalyType.REPEAT:
            anomaly_index = self.random.randrange(len(actions))
            actions.insert(anomaly_index + 1, actions[anomaly_index])
        elif anomaly_type == AnomalyType.REORDER:
            anomaly_index = self.random.randrange(len(actions) - 1)
            actions[anomaly_index], actions[anomaly_index + 1] = (
                actions[anomaly_index + 1],
                actions[anomaly_index],
            )
        elif anomaly_type == AnomalyType.UNEXPECTED:
            anomaly_index = self.random.randrange(len(actions) + 1)
            actions.insert(anomaly_index, self.unexpected_label)
        elif anomaly_type == AnomalyType.TIMEOUT:
            anomaly_index = self.random.randrange(len(actions))
            timeout_index = anomaly_index
        elif anomaly_type == AnomalyType.INCOMPLETE:
            keep = self.random.randrange(1, len(actions))
            anomaly_index = keep
            actions = actions[:keep]

        events: list[ProcedureEvent] = []
        timestamp = self.random.randint(0, 250)
        for index, action in enumerate(actions):
            duration = self.random.randint(*self.duration_range_ms)
            timed_out = index == timeout_index
            if timed_out:
                duration = self.timeout_ms + self.random.randint(250, 2000)
            start = timestamp
            end = start + duration
            events.append(
                ProcedureEvent(
                    action=action,
                    start_ms=start,
                    end_ms=end,
                    confidence=round(self.random.uniform(0.82, 0.99), 4),
                    timed_out=timed_out,
                )
            )
            timestamp = end + self.random.randint(*self.gap_range_ms)

        return ProcedureRecord(
            sequence_id=sequence_id,
            anomaly_type=anomaly_type,
            is_valid=anomaly_type == AnomalyType.CORRECT,
            expected_sequence=list(self.expected_sequence),
            observed_actions=actions,
            events=events,
            anomaly_index=anomaly_index,
        )


def write_sequence_dataset(
    output_dir: Path,
    expected_sequence: list[str],
    samples_per_type: int = 100,
    seed: int = 42,
    timeout_ms: int = 5000,
) -> dict[str, object]:
    if samples_per_type < 1:
        raise ValueError("samples_per_type must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = SequenceDatasetGenerator(
        expected_sequence=expected_sequence,
        seed=seed,
        timeout_ms=timeout_ms,
    )
    records: list[ProcedureRecord] = []
    for anomaly_type in AnomalyType:
        for index in range(samples_per_type):
            sequence_id = f"{anomaly_type.value}_{index:05d}"
            records.append(generator.generate(anomaly_type, sequence_id))

    jsonl_path = output_dir / "procedures.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")

    index_path = output_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence_id", "anomaly_type", "is_valid", "num_events"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "sequence_id": record.sequence_id,
                "anomaly_type": record.anomaly_type.value,
                "is_valid": record.is_valid,
                "num_events": len(record.events),
            }
            for record in records
        )

    counts = {kind.value: samples_per_type for kind in AnomalyType}
    summary: dict[str, object] = {
        "seed": seed,
        "samples_per_type": samples_per_type,
        "total_sequences": len(records),
        "timeout_ms": timeout_ms,
        "expected_sequence": expected_sequence,
        "counts": counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

