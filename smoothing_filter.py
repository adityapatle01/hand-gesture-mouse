from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ExponentialMovingAverage:
    smoothing_factor: float = 5.0
    _previous: Optional[Tuple[float, float]] = None

    @property
    def previous(self) -> Optional[Tuple[float, float]]:
        return self._previous

    def update(self, target_x: float, target_y: float) -> Tuple[float, float]:
        return self.update_with_factor(target_x, target_y, self.smoothing_factor)

    def update_with_factor(self, target_x: float, target_y: float, smoothing_factor: float) -> Tuple[float, float]:
        if smoothing_factor <= 1:
            self._previous = (target_x, target_y)
            return target_x, target_y

        if self._previous is None:
            self._previous = (target_x, target_y)
            return target_x, target_y

        previous_x, previous_y = self._previous
        smooth_x = previous_x + (target_x - previous_x) / smoothing_factor
        smooth_y = previous_y + (target_y - previous_y) / smoothing_factor
        self._previous = (smooth_x, smooth_y)
        return smooth_x, smooth_y

    def reset(self) -> None:
        self._previous = None
