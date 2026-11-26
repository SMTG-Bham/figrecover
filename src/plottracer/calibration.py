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
    """Maps from pixel to data coordinates along each axis.

    Each axis maps linearly from pixels to data, or, if it is logarithmic, to
    the base-10 logarithm of the data. Pixel coordinates are measured from the
    bottom-left corner of the image, with x increasing to the right and y
    increasing upwards, so the centre of the bottom-left pixel is at
    (0.5, 0.5).

    Attributes:
        x_scale: Data units per pixel along the x-axis, or decades per pixel
            if it is logarithmic.
        x_offset: Data x-coordinate of pixel column zero, or its logarithm.
        y_scale: Data units per pixel along the y-axis, or decades per pixel
            if it is logarithmic.
        y_offset: Data y-coordinate of pixel row zero, or its logarithm.
        origin: Pixel position where the axes cross.
        x_log: Whether the x-axis is logarithmic.
        y_log: Whether the y-axis is logarithmic.
    """

    x_scale: float
    x_offset: float
    y_scale: float
    y_offset: float
    origin: Point
    x_log: bool = False
    y_log: bool = False

    @classmethod
    def from_points(
        cls,
        data: Sequence[Point],
        pixels: Sequence[Point],
        *,
        x_log: bool = False,
        y_log: bool = False,
    ) -> Self:
        """Fit a calibration to known points.

        Args:
            data: Data coordinates of at least three reference points, such as
                the origin and a labelled tick on each axis.
            pixels: Pixel coordinates of the same points, in the same order.
            x_log: Whether the x-axis is logarithmic.
            y_log: Whether the y-axis is logarithmic.

        Returns:
            The least-squares calibration through the points, whose origin is
            the leftmost and lowest of them.

        Raises:
            ValueError: If the points do not determine both axes, or are not
                positive on a logarithmic axis.
        """
        if len(data) != len(pixels):
            msg = f"got {len(data)} data points but {len(pixels)} pixel locations"
            raise ValueError(msg)
        if len(data) < MIN_POINTS:
            msg = f"need at least {MIN_POINTS} points to calibrate, got {len(data)}"
            raise ValueError(msg)

        data_xy = np.asarray(data, dtype=np.float64)
        pixel_xy = np.asarray(pixels, dtype=np.float64)
        x_scale, x_offset = _fit_axis(
            pixel_xy[:, 0], data_xy[:, 0], "x", "points", log=x_log
        )
        y_scale, y_offset = _fit_axis(
            pixel_xy[:, 1], data_xy[:, 1], "y", "points", log=y_log
        )
        origin = (float(pixel_xy[:, 0].min()), float(pixel_xy[:, 1].min()))
        return cls(x_scale, x_offset, y_scale, y_offset, origin, x_log, y_log)

    @classmethod
    def from_ticks(
        cls,
        x_ticks: Sequence[Tick],
        y_ticks: Sequence[Tick],
        origin: Point,
        *,
        x_log: bool = False,
        y_log: bool = False,
    ) -> Self:
        """Fit a calibration to the labelled ticks on each axis.

        Args:
            x_ticks: The pixel x-coordinate and value of each x-axis tick.
            y_ticks: The pixel y-coordinate and value of each y-axis tick.
            origin: Pixel position where the axes cross.
            x_log: Whether the x-axis is logarithmic.
            y_log: Whether the y-axis is logarithmic.

        Returns:
            The least-squares calibration through the ticks.

        Raises:
            ValueError: If the ticks on an axis do not span it, or are not
                positive on a logarithmic axis.
        """
        x_pixels, x_values = np.asarray(x_ticks, dtype=np.float64).reshape(-1, 2).T
        y_pixels, y_values = np.asarray(y_ticks, dtype=np.float64).reshape(-1, 2).T
        x_scale, x_offset = _fit_axis(x_pixels, x_values, "x", "ticks", log=x_log)
        y_scale, y_offset = _fit_axis(y_pixels, y_values, "y", "ticks", log=y_log)
        return cls(x_scale, x_offset, y_scale, y_offset, origin, x_log, y_log)

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
        data = xy * scale + offset
        log = np.array([self.x_log, self.y_log])
        data[:, log] = 10.0 ** data[:, log]
        return data


def _fit_axis(
    pixels: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    name: str,
    kind: str,
    *,
    log: bool,
) -> tuple[float, float]:
    """Fit ``values = scale * pixels + offset`` for one axis, from points or ticks.

    On a logarithmic axis, the base-10 logarithms of the values are fitted.
    """
    if log:
        if np.any(values <= 0):
            msg = f"the {kind} on the logarithmic {name}-axis must be positive"
            raise ValueError(msg)
        values = np.log10(values)
    if len(pixels) == 0 or np.ptp(pixels) == 0 or np.ptp(values) == 0:
        msg = f"the {kind} must span the {name}-axis in both data and pixels"
        raise ValueError(msg)
    offset, scale = np.polynomial.polynomial.polyfit(pixels, values, 1)
    return float(scale), float(offset)
