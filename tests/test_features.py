import numpy as np

from assembly_validation.features import FEATURE_DIM, normalize_hand, pack_hands


def test_pack_hands_shape_and_presence() -> None:
    hand = np.arange(63, dtype=np.float32).reshape(21, 3)
    result = pack_hands(left=hand)
    assert result.shape == (FEATURE_DIM,)
    assert np.allclose(result[-2:], [1.0, 0.0])


def test_normalization_is_translation_invariant() -> None:
    hand = np.arange(63, dtype=np.float32).reshape(21, 3)
    translated = hand + np.asarray([100.0, -50.0, 9.0], dtype=np.float32)
    assert np.allclose(normalize_hand(hand), normalize_hand(translated), atol=1e-5)

