from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2

APP_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
MATPLOTLIB_CACHE_DIR = APP_CACHE_DIR / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import mediapipe as mp


NormalizedPoint = Tuple[float, float, float]
PixelPoint = Tuple[int, int]
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
DEFAULT_TASK_MODEL = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class TrackedHand:
    normalized_landmarks: List[NormalizedPoint]
    pixel_landmarks: List[PixelPoint]
    handedness: str


@dataclass(frozen=True)
class HandTrackerConfig:
    static_image_mode: bool = False
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    task_model_path: Path = DEFAULT_TASK_MODEL
    task_model_url: str = TASK_MODEL_URL


class HandTracker:
    def __init__(self, config: HandTrackerConfig) -> None:
        self.config = config
        self._backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        self._timestamp_ms = 0

        if self._backend == "solutions":
            self._mp_hands = mp.solutions.hands
            self._drawing = mp.solutions.drawing_utils
            self._drawing_styles = mp.solutions.drawing_styles
            self._hands = self._mp_hands.Hands(
                static_image_mode=config.static_image_mode,
                max_num_hands=config.max_num_hands,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
            self._landmarker = None
        else:
            self._mp_hands = None
            self._drawing = None
            self._drawing_styles = None
            self._hands = None
            self._landmarker = self._create_tasks_landmarker(config)

    def process(self, frame_bgr) -> Tuple[Optional[TrackedHand], object]:
        if self._backend == "tasks":
            return self._process_with_tasks(frame_bgr)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None, results

        landmarks = results.multi_hand_landmarks[0]
        height, width = frame_bgr.shape[:2]
        normalized: List[NormalizedPoint] = []
        pixels: List[PixelPoint] = []

        for landmark in landmarks.landmark:
            normalized.append((landmark.x, landmark.y, landmark.z))
            pixels.append((int(landmark.x * width), int(landmark.y * height)))

        handedness = "Unknown"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        return TrackedHand(normalized, pixels, handedness), results

    def draw_landmarks(self, frame, results) -> None:
        if self._backend == "tasks":
            self._draw_task_landmarks(frame, results)
            return

        if not getattr(results, "multi_hand_landmarks", None):
            return

        for landmarks in results.multi_hand_landmarks:
            self._drawing.draw_landmarks(
                frame,
                landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._drawing_styles.get_default_hand_landmarks_style(),
                self._drawing_styles.get_default_hand_connections_style(),
            )

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def _create_tasks_landmarker(self, config: HandTrackerConfig):
        self._ensure_task_model(config.task_model_path, config.task_model_url)

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(config.task_model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_tracking_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _process_with_tasks(self, frame_bgr) -> Tuple[Optional[TrackedHand], object]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms = max(self._timestamp_ms + 1, int(time.monotonic() * 1000))
        results = self._landmarker.detect_for_video(image, self._timestamp_ms)

        if not results.hand_landmarks:
            return None, results

        landmarks = results.hand_landmarks[0]
        height, width = frame_bgr.shape[:2]
        normalized: List[NormalizedPoint] = []
        pixels: List[PixelPoint] = []

        for landmark in landmarks:
            normalized.append((landmark.x, landmark.y, landmark.z))
            pixels.append((int(landmark.x * width), int(landmark.y * height)))

        handedness = "Unknown"
        if results.handedness and results.handedness[0]:
            handedness = results.handedness[0][0].category_name

        return TrackedHand(normalized, pixels, handedness), results

    def _draw_task_landmarks(self, frame, results) -> None:
        if not getattr(results, "hand_landmarks", None):
            return

        height, width = frame.shape[:2]
        for landmarks in results.hand_landmarks:
            points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (80, 220, 140), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(frame, point, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, point, 5, (40, 180, 255), 1, cv2.LINE_AA)

    def _ensure_task_model(self, model_path: Path, model_url: str) -> None:
        if model_path.exists() and model_path.stat().st_size > 0:
            return

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Downloading MediaPipe hand model to {model_path} ...")
            urlretrieve(model_url, model_path)
        except (OSError, URLError) as exc:
            raise RuntimeError(
                "MediaPipe Tasks is installed, but the hand model could not be downloaded. "
                f"Download it manually from {model_url} and place it at {model_path}."
            ) from exc


def landmark_distance(points: Sequence[PixelPoint], first: int, second: int) -> float:
    x1, y1 = points[first]
    x2, y2 = points[second]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
