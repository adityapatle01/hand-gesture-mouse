from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from hand_tracker import PixelPoint, TrackedHand, landmark_distance


WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20


class Gesture(str, Enum):
    NONE = "No hand"
    MOVE = "Move"
    LEFT_CLICK = "Left click"
    DOUBLE_CLICK = "Double click"
    RIGHT_CLICK = "Right click"
    DRAG = "Drag"
    SCROLL = "Scroll"


@dataclass(frozen=True)
class GestureConfig:
    click_threshold: float = 35.0
    pinch_release_threshold: float = 50.0
    right_click_threshold: float = 40.0
    right_click_release_threshold: float = 52.0
    drag_hold_seconds: float = 0.70
    click_max_movement_pixels: float = 42.0
    click_debounce_seconds: float = 0.06
    pending_click_delay_seconds: float = 0.18
    double_click_window_seconds: float = 0.65
    scroll_deadzone_pixels: int = 1
    scroll_scale: float = 0.22
    scroll_smoothing: float = 0.22
    scroll_max_step: int = 5


@dataclass(frozen=True)
class GestureState:
    gesture: Gesture
    index_tip: Optional[PixelPoint]
    thumb_index_distance: float
    thumb_middle_distance: float
    fingers: Dict[str, bool]
    click_ready: bool = False
    double_click_ready: bool = False
    right_click_ready: bool = False
    drag_active: bool = False
    drag_started: bool = False
    drag_released: bool = False
    scroll_amount: int = 0


class GestureDetector:
    def __init__(self, config: GestureConfig) -> None:
        self.config = config
        self._last_click_time = -config.click_debounce_seconds
        self._last_actual_click_time = -config.double_click_window_seconds
        self._last_right_click_time = -config.click_debounce_seconds
        self._pinch_started_at: Optional[float] = None
        self._pinch_start_point: Optional[PixelPoint] = None
        self._pending_click_time: Optional[float] = None
        self._right_pinch_active = False
        self._dragging = False
        self._last_scroll_y: Optional[float] = None
        self._scroll_velocity = 0.0
        self._scroll_accumulator = 0.0

    def detect(self, hand: Optional[TrackedHand]) -> GestureState:
        if hand is None:
            return self._empty_state(Gesture.NONE)

        now = time.perf_counter()
        points = hand.pixel_landmarks
        fingers = self._extended_fingers(points)
        thumb_index_distance = landmark_distance(points, THUMB_TIP, INDEX_TIP)
        thumb_middle_distance = landmark_distance(points, THUMB_TIP, MIDDLE_TIP)

        if self._dragging:
            active_pinch_threshold = self.config.pinch_release_threshold
        elif self._pinch_started_at is not None:
            active_pinch_threshold = self.config.pinch_release_threshold
        else:
            active_pinch_threshold = self.config.click_threshold
        is_thumb_index_pinch = thumb_index_distance < active_pinch_threshold
        right_threshold = (
            self.config.right_click_release_threshold
            if self._right_pinch_active
            else self.config.right_click_threshold
        )
        is_thumb_middle_pinch = thumb_middle_distance < right_threshold
        if not is_thumb_middle_pinch:
            self._right_pinch_active = False

        click_ready = False
        double_click_ready = False
        right_click_ready = False
        drag_started = False
        drag_released = False
        scroll_amount = 0
        gesture = Gesture.NONE

        if self._pending_click_due(now) and not is_thumb_index_pinch and self._pinch_started_at is None:
            click_ready = True
            gesture = Gesture.LEFT_CLICK
            self._finish_pending_click(now)

        if click_ready:
            self._reset_scroll_tracking()
        elif is_thumb_middle_pinch and not self._right_pinch_active:
            if self._dragging:
                drag_released = True
            right_click_ready = True
            self._right_pinch_active = True
            gesture = Gesture.RIGHT_CLICK
            self._reset_pinch_tracking()
        elif is_thumb_index_pinch:
            if self._pinch_started_at is None:
                self._pinch_started_at = now
                self._pinch_start_point = points[INDEX_TIP]

            pinch_duration = now - self._pinch_started_at
            pinch_movement = self._pinch_movement(points[INDEX_TIP])
            if pinch_duration >= self.config.drag_hold_seconds:
                gesture = Gesture.DRAG
                if not self._dragging:
                    drag_started = True
                    self._dragging = True
            else:
                gesture = Gesture.LEFT_CLICK
        else:
            pinch_duration = 0.0 if self._pinch_started_at is None else now - self._pinch_started_at
            if self._dragging:
                drag_released = True
            elif (
                self._pinch_started_at is not None
                and pinch_duration < self.config.drag_hold_seconds
                and self._pinch_movement(points[INDEX_TIP]) <= self.config.click_max_movement_pixels
                and now - self._last_click_time > self.config.click_debounce_seconds
            ):
                self._last_click_time = now
                click_ready, double_click_ready = self._queue_or_emit_click(now)
                gesture = Gesture.DOUBLE_CLICK if double_click_ready else Gesture.LEFT_CLICK
            self._reset_pinch_tracking()

            if click_ready or double_click_ready:
                pass
            elif fingers["index"] and fingers["middle"]:
                gesture = Gesture.SCROLL
                scroll_amount = self._scroll_from_two_fingers(points)
            elif fingers["index"]:
                gesture = Gesture.MOVE
                self._reset_scroll_tracking()
            else:
                gesture = Gesture.NONE
                self._reset_scroll_tracking()

        return GestureState(
            gesture=gesture,
            index_tip=points[INDEX_TIP],
            thumb_index_distance=thumb_index_distance,
            thumb_middle_distance=thumb_middle_distance,
            fingers=fingers,
            click_ready=click_ready,
            double_click_ready=double_click_ready,
            right_click_ready=right_click_ready,
            drag_active=self._dragging,
            drag_started=drag_started,
            drag_released=drag_released,
            scroll_amount=scroll_amount,
        )

    def _extended_fingers(self, points: List[PixelPoint]) -> Dict[str, bool]:
        # In camera coordinates, smaller y means higher on the frame.
        return {
            "index": points[INDEX_TIP][1] < points[INDEX_PIP][1] < points[INDEX_MCP][1],
            "middle": points[MIDDLE_TIP][1] < points[MIDDLE_PIP][1] < points[MIDDLE_MCP][1],
            "ring": points[RING_TIP][1] < points[RING_PIP][1] < points[RING_MCP][1],
            "pinky": points[PINKY_TIP][1] < points[PINKY_PIP][1] < points[PINKY_MCP][1],
        }

    def _scroll_from_two_fingers(self, points: List[PixelPoint]) -> int:
        current_y = (points[INDEX_TIP][1] + points[MIDDLE_TIP][1]) / 2.0
        if self._last_scroll_y is None:
            self._last_scroll_y = current_y
            return 0

        delta_y = self._last_scroll_y - current_y
        self._last_scroll_y = current_y

        if abs(delta_y) < self.config.scroll_deadzone_pixels:
            delta_y = 0.0

        alpha = min(max(self.config.scroll_smoothing, 0.0), 1.0)
        self._scroll_velocity = (1.0 - alpha) * self._scroll_velocity + alpha * delta_y
        self._scroll_accumulator += self._scroll_velocity * self.config.scroll_scale

        scroll_amount = int(self._scroll_accumulator)
        if scroll_amount == 0:
            return 0

        self._scroll_accumulator -= scroll_amount
        return max(-self.config.scroll_max_step, min(self.config.scroll_max_step, scroll_amount))

    def _reset_scroll_tracking(self) -> None:
        self._last_scroll_y = None
        self._scroll_velocity = 0.0
        self._scroll_accumulator = 0.0

    def _reset_pinch_tracking(self) -> None:
        self._pinch_started_at = None
        self._pinch_start_point = None
        self._dragging = False

    def _pinch_movement(self, current_point: PixelPoint) -> float:
        if self._pinch_start_point is None:
            return 0.0

        start_x, start_y = self._pinch_start_point
        current_x, current_y = current_point
        return ((current_x - start_x) ** 2 + (current_y - start_y) ** 2) ** 0.5

    def _pending_click_due(self, now: float) -> bool:
        return (
            self._pending_click_time is not None
            and now - self._pending_click_time >= self.config.pending_click_delay_seconds
        )

    def _finish_pending_click(self, now: float) -> None:
        self._pending_click_time = None
        self._last_actual_click_time = now

    def _queue_or_emit_click(self, now: float) -> Tuple[bool, bool]:
        if (
            self._pending_click_time is not None
            and now - self._pending_click_time <= self.config.double_click_window_seconds
        ):
            self._pending_click_time = None
            self._last_actual_click_time = now
            return False, True

        if now - self._last_actual_click_time <= self.config.double_click_window_seconds:
            self._last_actual_click_time = now
            return True, False

        self._pending_click_time = now
        return False, False

    def _empty_state(self, gesture: Gesture) -> GestureState:
        now = time.perf_counter()
        click_ready = self._pending_click_due(now)
        if click_ready:
            self._finish_pending_click(now)

        was_dragging = self._dragging
        if self._dragging:
            self._dragging = False
        self._pinch_started_at = None
        self._pinch_start_point = None
        self._right_pinch_active = False
        self._reset_scroll_tracking()
        return GestureState(
            gesture=Gesture.LEFT_CLICK if click_ready else gesture,
            index_tip=None,
            thumb_index_distance=0.0,
            thumb_middle_distance=0.0,
            fingers={"index": False, "middle": False, "ring": False, "pinky": False},
            click_ready=click_ready,
            drag_released=was_dragging,
        )

    def reset(self) -> None:
        self._last_click_time = -self.config.click_debounce_seconds
        self._last_actual_click_time = -self.config.double_click_window_seconds
        self._last_right_click_time = -self.config.click_debounce_seconds
        self._pinch_started_at = None
        self._pinch_start_point = None
        self._pending_click_time = None
        self._right_pinch_active = False
        self._dragging = False
        self._reset_scroll_tracking()
