from pathlib import Path

from assembly_validation.sequence_generator import (
    AnomalyType,
    SequenceDatasetGenerator,
    write_sequence_dataset,
)


EXPECTED = ["reach", "pick", "move", "align", "insert", "tighten", "inspect"]


def test_all_anomaly_types_have_expected_structure() -> None:
    generator = SequenceDatasetGenerator(EXPECTED, seed=5, timeout_ms=3000)
    records = {
        kind: generator.generate(kind, f"sample_{kind.value}") for kind in AnomalyType
    }
    assert records[AnomalyType.CORRECT].observed_actions == EXPECTED
    assert len(records[AnomalyType.SKIP].observed_actions) == len(EXPECTED) - 1
    assert len(records[AnomalyType.REPEAT].observed_actions) == len(EXPECTED) + 1
    assert records[AnomalyType.REORDER].observed_actions != EXPECTED
    assert "unexpected_action" in records[AnomalyType.UNEXPECTED].observed_actions
    assert any(event.timed_out for event in records[AnomalyType.TIMEOUT].events)
    assert len(records[AnomalyType.INCOMPLETE].observed_actions) < len(EXPECTED)


def test_write_sequence_dataset(tmp_path: Path) -> None:
    summary = write_sequence_dataset(tmp_path, EXPECTED, samples_per_type=3, seed=9)
    assert summary["total_sequences"] == len(AnomalyType) * 3
    assert (tmp_path / "procedures.jsonl").is_file()
    assert (tmp_path / "index.csv").is_file()

