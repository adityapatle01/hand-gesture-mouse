from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
import time
from typing import Tuple

import numpy as np
import pyautogui

from smoothing_filter import ExponentialMovingAverage


@dataclass(frozen=True)
class ControlBox:
    left: int
    top: int
    right: int
    bottom: int

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass(frozen=True)
class MouseControllerConfig:
    smoothing_factor: float = 7.5
    drag_smoothing_factor: float = 4.0
    fast_smoothing_factor: float = 2.6
    movement_deadzone_pixels: float = 2.0
    fast_movement_pixels: float = 120.0
    click_press_seconds: float = 0.035
    pause_seconds: float = 0.0
    fail_safe: bool = True


class MouseController:
    def __init__(self, config: MouseControllerConfig) -> None:
        self.config = config
        pyautogui.PAUSE = config.pause_seconds
        pyautogui.FAILSAFE = config.fail_safe
        self.screen_width, self.screen_height = pyautogui.size()
        self._filter = ExponentialMovingAverage(config.smoothing_factor)
        self._is_mouse_down = False

    def move_from_camera_point(
        self,
        point: Tuple[int, int],
        control_box: ControlBox,
        dragging: bool = False,
    ) -> Tuple[int, int]:
        x, y = point
        target_x = np.clip(np.interp(x, (control_box.left, control_box.right), (0, self.screen_width)), 0, self.screen_width - 1)
        target_y = np.clip(np.interp(y, (control_box.top, control_box.bottom), (0, self.screen_height)), 0, self.screen_height - 1)
        smoothing = self._adaptive_smoothing(target_x, target_y, dragging)
        smooth_x, smooth_y = self._filter.update_with_factor(target_x, target_y, smoothing)
        final_x = int(np.clip(smooth_x, 0, self.screen_width - 1))
        final_y = int(np.clip(smooth_y, 0, self.screen_height - 1))
        pyautogui.moveTo(final_x, final_y, duration=0)
        return final_x, final_y

    def click(self) -> None:
        self._press(button="left")

    def double_click(self) -> None:
        self._press(button="left")
        time.sleep(0.05)
        self._press(button="left")

    def right_click(self) -> None:
        self._press(button="right")

    def scroll(self, amount: int) -> None:
        if amount != 0:
            pyautogui.scroll(amount)

    def mouse_down(self) -> None:
        if not self._is_mouse_down:
            pyautogui.mouseDown()
            self._is_mouse_down = True

    def mouse_up(self) -> None:
        if self._is_mouse_down:
            pyautogui.mouseUp()
            self._is_mouse_down = False

    def reset_smoothing(self) -> None:
        self._filter.reset()

    def release(self) -> None:
        self.mouse_up()

    def reset(self) -> None:
        self.release()
        self.reset_smoothing()

    def adjust_smoothing(self, delta: float) -> float:
        next_factor = float(np.clip(self.config.smoothing_factor + delta, 1.5, 14.0))
        self.config = replace(self.config, smoothing_factor=next_factor)
        self._filter.smoothing_factor = next_factor
        return next_factor

    def _adaptive_smoothing(self, target_x: float, target_y: float, dragging: bool) -> float:
        if self._filter.previous is None:
            return 1.0

        previous_x, previous_y = self._filter.previous
        distance = math.hypot(target_x - previous_x, target_y - previous_y)
        if distance < self.config.movement_deadzone_pixels:
            return 9999.0

        base = self.config.drag_smoothing_factor if dragging else self.config.smoothing_factor
        if distance >= self.config.fast_movement_pixels:
            return min(base, self.config.fast_smoothing_factor)

        blend = distance / self.config.fast_movement_pixels
        return base - (base - self.config.fast_smoothing_factor) * blend

    def _press(self, button: str) -> None:
        pyautogui.mouseDown(button=button)
        time.sleep(self.config.click_press_seconds)
        pyautogui.mouseUp(button=button)
