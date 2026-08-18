from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINTS_PER_HAND = 21
COORDS_PER_JOINT = 3
MAX_HANDS = 2
FEATURE_DIM = MAX_HANDS * JOINTS_PER_HAND * COORDS_PER_JOINT + MAX_HANDS


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """Make 21 xyz landmarks translation- and approximately scale-invariant."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(JOINTS_PER_HAND, 3)
    centered = points - points[0]
    palm_indices = np.asarray([5, 9, 13, 17])
    scale = float(np.mean(np.linalg.norm(centered[palm_indices], axis=1)))
    if scale < 1e-6:
        return np.zeros_like(centered)
    return centered / scale


def pack_hands(
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
) -> np.ndarray:
    """Pack normalized left/right landmarks and two presence flags."""
    features: list[np.ndarray] = []
    presence: list[float] = []
    for hand in (left, right):
        if hand is None:
            features.append(np.zeros((JOINTS_PER_HAND, 3), dtype=np.float32))
            presence.append(0.0)
        else:
            features.append(normalize_hand(hand))
            presence.append(1.0)
    return np.concatenate([*(item.reshape(-1) for item in features), presence]).astype(
        np.float32
    )


@dataclass
class HandFrame:
    features: np.ndarray
    detected_hands: int


class MediaPipeHandExtractor:
    """Classic MediaPipe Hands wrapper with a stable numpy output contract."""

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("Install MediaPipe with: pip install mediapipe") from exc

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, bgr_frame: np.ndarray) -> HandFrame:
        import cv2

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        by_side: dict[str, np.ndarray] = {}
        if result.multi_hand_landmarks:
            handedness = result.multi_handedness or []
            for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                side = "Right"
                if index < len(handedness):
                    side = handedness[index].classification[0].label
                points = np.asarray(
                    [[point.x, point.y, point.z] for point in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                by_side[side] = points
        packed = pack_hands(by_side.get("Left"), by_side.get("Right"))
        return HandFrame(features=packed, detected_hands=len(by_side))

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> MediaPipeHandExtractor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
