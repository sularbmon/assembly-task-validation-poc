from assembly_validation.validator import EventType, SequenceValidator


def confirm(validator: SequenceValidator, label: str, confidence: float = 0.9):
    event = None
    for timestamp in range(validator.confirmation_windows):
        event = validator.observe(label, confidence, timestamp * 100)
    return event


def test_correct_sequence_completes() -> None:
    validator = SequenceValidator(["pick", "place"], confirmation_windows=2)
    first = confirm(validator, "pick")
    second = confirm(validator, "place")
    assert first is not None and first.event == EventType.STEP_OK
    assert second is not None and second.event == EventType.COMPLETE
    assert validator.complete


def test_out_of_order_does_not_advance() -> None:
    validator = SequenceValidator(["pick", "place", "tighten"], confirmation_windows=2)
    event = confirm(validator, "tighten")
    assert event is not None and event.event == EventType.OUT_OF_ORDER
    assert validator.position == 0


def test_repeat_is_reported() -> None:
    validator = SequenceValidator(["pick", "place"], confirmation_windows=2)
    confirm(validator, "pick")
    confirm(validator, "unrelated")
    event = confirm(validator, "pick")
    assert event is not None and event.event == EventType.REPEAT


def test_lingering_action_is_not_a_repeat() -> None:
    validator = SequenceValidator(["pick", "place"], confirmation_windows=2)
    confirm(validator, "pick")
    assert confirm(validator, "pick") is None
    assert validator.position == 1


def test_low_confidence_never_confirms() -> None:
    validator = SequenceValidator(["pick"], confidence_threshold=0.8, confirmation_windows=2)
    assert confirm(validator, "pick", confidence=0.4) is None
    assert validator.position == 0


def test_timeout_is_reported_only_once_per_step() -> None:
    validator = SequenceValidator(["pick", "place"], step_timeout_ms=1000)
    assert validator.check_timeout(999) is None
    event = validator.check_timeout(1000)
    assert event is not None and event.event == EventType.TIMEOUT
    assert validator.check_timeout(1500) is None


def test_finalize_reports_incomplete() -> None:
    validator = SequenceValidator(["pick", "place"])
    event = validator.finalize(2000)
    assert event is not None and event.event == EventType.INCOMPLETE
