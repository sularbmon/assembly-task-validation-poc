from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class EventType(str, Enum):
    WAIT = "WAIT"
    STEP_OK = "STEP_OK"
    REPEAT = "REPEAT"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    UNEXPECTED = "UNEXPECTED"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class ValidationEvent:
    timestamp_ms: int
    event: EventType
    recognized_action: str | None
    expected_action: str | None
    confidence: float
    sequence_position: int

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["event"] = self.event.value
        return value


class SequenceValidator:
    """Debounced deterministic validator for a known assembly procedure."""

    def __init__(
        self,
        expected_sequence: list[str],
        confidence_threshold: float = 0.7,
        confirmation_windows: int = 4,
    ) -> None:
        if not expected_sequence:
            raise ValueError("expected_sequence cannot be empty")
        if confirmation_windows < 1:
            raise ValueError("confirmation_windows must be >= 1")
        self.expected_sequence = expected_sequence
        self.confidence_threshold = confidence_threshold
        self.confirmation_windows = confirmation_windows
        self.position = 0
        self._candidate: str | None = None
        self._candidate_count = 0
        self._blocked_label: str | None = None
        self._release_count = 0

    @property
    def expected_action(self) -> str | None:
        if self.position >= len(self.expected_sequence):
            return None
        return self.expected_sequence[self.position]

    @property
    def complete(self) -> bool:
        return self.position >= len(self.expected_sequence)

    def observe(
        self,
        label: str,
        confidence: float,
        timestamp_ms: int,
    ) -> ValidationEvent | None:
        if self.complete:
            return None
        if confidence < self.confidence_threshold:
            self._candidate = None
            self._candidate_count = 0
            return None

        # An action normally remains visible for several windows after it is
        # accepted. Require a stable change before the same label can produce
        # another event, otherwise every completed step looks like a repeat.
        released = False
        if self._blocked_label is not None:
            if label == self._blocked_label:
                self._release_count = 0
                return None
            self._release_count += 1
            if self._release_count < self.confirmation_windows:
                return None
            self._blocked_label = None
            self._release_count = 0
            released = True

        if released:
            self._candidate = label
            self._candidate_count = self.confirmation_windows
        elif label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = label
            self._candidate_count = 1

        if self._candidate_count < self.confirmation_windows:
            return None

        self._candidate = None
        self._candidate_count = 0
        expected = self.expected_action

        if label == expected:
            current_position = self.position
            self.position += 1
            self._blocked_label = label
            event_type = EventType.COMPLETE if self.complete else EventType.STEP_OK
            return ValidationEvent(
                timestamp_ms,
                event_type,
                label,
                expected,
                confidence,
                current_position,
            )

        completed = self.expected_sequence[: self.position]
        remaining = self.expected_sequence[self.position + 1 :]
        if label in completed:
            event_type = EventType.REPEAT
        elif label in remaining:
            event_type = EventType.OUT_OF_ORDER
        else:
            event_type = EventType.UNEXPECTED
        return ValidationEvent(
            timestamp_ms,
            event_type,
            label,
            expected,
            confidence,
            self.position,
        )
