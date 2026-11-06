"""Map image pixels to data coordinates."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

type Point = tuple[float, float]
type Tick = tuple[float, float]
"""A labelled tick: its pixel position along the axis, and its value."""

MIN_POINTS = 3


@dataclass(frozen=True, slots=True)
class Calibration:
    """Linear maps from pixel to data coordinates along each axis.

    Pixel coordinates are measured from the bottom-left corner of the image,
    with x increasing to the right and y increasing upwards.

    Attributes:
        x_scale: Data units per pixel along the x-axis.
        x_offset: Data x-coordinate of pixel column zero.
        y_scale: Data units per pixel along the y-axis.
        y_offset: Data y-coordinate of pixel row zero.
        origin: Pixel position where the axes cross.
    """

    x_scale: float
    x_offset: float
    y_scale: float
    y_offset: float
    origin: Point

    @classmethod
    def from_points(cls, data: Sequence[Point], pixels: Sequence[Point]) -> Self:
        """Fit a calibration to known points.

        Args:
            data: Data coordinates of at least three reference points, such as
                the origin and a labelled tick on each axis.
            pixels: Pixel coordinates of the same points, in the same order.

        Returns:
            The least-squares calibration through the points, whose origin is
            the leftmost and lowest of them.

        Raises:
            ValueError: If the points do not determine both axes.
        """
        if len(data) != len(pixels):
            msg = f"got {len(data)} data points but {len(pixels)} pixel locations"
            raise ValueError(msg)
        if len(data) < MIN_POINTS:
            msg = f"need at least {MIN_POINTS} points to calibrate, got {len(data)}"
            raise ValueError(msg)

        data_xy = np.asarray(data, dtype=np.float64)
        pixel_xy = np.asarray(pixels, dtype=np.float64)
        x_scale, x_offset = _fit_axis(pixel_xy[:, 0], data_xy[:, 0], "x", "points")
        y_scale, y_offset = _fit_axis(pixel_xy[:, 1], data_xy[:, 1], "y", "points")
        origin = (float(pixel_xy[:, 0].min()), float(pixel_xy[:, 1].min()))
        return cls(x_scale, x_offset, y_scale, y_offset, origin)

    @classmethod
    def from_ticks(
        cls, x_ticks: Sequence[Tick], y_ticks: Sequence[Tick], origin: Point
    ) -> Self:
        """Fit a calibration to the labelled ticks on each axis.

        Args:
            x_ticks: The pixel x-coordinate and value of each x-axis tick.
            y_ticks: The pixel y-coordinate and value of each y-axis tick.
            origin: Pixel position where the axes cross.

        Returns:
            The least-squares calibration through the ticks.

        Raises:
            ValueError: If the ticks on an axis do not span it.
        """
        x_pixels, x_values = np.asarray(x_ticks, dtype=np.float64).reshape(-1, 2).T
        y_pixels, y_values = np.asarray(y_ticks, dtype=np.float64).reshape(-1, 2).T
        x_scale, x_offset = _fit_axis(x_pixels, x_values, "x", "ticks")
        y_scale, y_offset = _fit_axis(y_pixels, y_values, "y", "ticks")
        return cls(x_scale, x_offset, y_scale, y_offset, origin)

    def to_data(self, pixels: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Convert pixel coordinates to data coordinates.

        Args:
            pixels: An ``(n, 2)`` array of pixel ``(x, y)`` coordinates.

        Returns:
            An ``(n, 2)`` array of data ``(x, y)`` coordinates.
        """
        xy = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        scale = np.array([self.x_scale, self.y_scale], dtype=np.float64)
        offset = np.array([self.x_offset, self.y_offset], dtype=np.float64)
        return xy * scale + offset


def _fit_axis(
    pixels: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    name: str,
    kind: str,
) -> tuple[float, float]:
    """Fit ``values = scale * pixels + offset`` for one axis, from points or ticks."""
    if len(pixels) == 0 or np.ptp(pixels) == 0 or np.ptp(values) == 0:
        msg = f"the {kind} must span the {name}-axis in both data and pixels"
        raise ValueError(msg)
    offset, scale = np.polynomial.polynomial.polyfit(pixels, values, 1)
    return float(scale), float(offset)
