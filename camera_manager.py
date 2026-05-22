from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


@dataclass(frozen=True)
class CameraConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 60
    inference_width: int = 640
    inference_height: int = 480
    flip_horizontal: bool = True


class CameraManager:
    """Small wrapper around OpenCV capture tuned for macOS AVFoundation."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = time.perf_counter()
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Camera is not available. Check macOS Camera permissions and close "
                "other apps that may be using the FaceTime camera."
            )

    def read(self) -> Tuple[bool, Optional[object], Optional[object]]:
        if self.cap is None:
            raise RuntimeError("CameraManager.open() must be called before read().")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None, None

        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        inference_frame = cv2.resize(
            frame,
            (self.config.inference_width, self.config.inference_height),
            interpolation=cv2.INTER_AREA,
        )
        self._update_fps()
        return True, frame, inference_frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _update_fps(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_frame_time
        self._last_frame_time = now
        if elapsed <= 0:
            return

        current_fps = 1.0 / elapsed
        self._fps = current_fps if self._fps == 0 else (0.9 * self._fps) + (0.1 * current_fps)
