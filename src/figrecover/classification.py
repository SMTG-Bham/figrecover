"""Classify visible plot objects in source pixel coordinates."""

from dataclasses import replace
from typing import cast

from .calibration import Calibration, Point
from .digitize import Image, _plot_area
from .plot_objects import PlotObject, references


def classify(
    image: Image, origin: Point = (0, 0), *, calibration: Calibration | None = None
) -> list[PlotObject]:
    """Find reference strokes inside the axes and report their pixel or data values."""
    plot, left, top = _plot_area(image, origin)
    objects = references(plot, left, top, image.shape[0])
    result = []
    for obj in objects:
        axis = int(obj.axis == "y")
        position = cast("float", obj.coordinate)
        value = (
            float(calibration.to_data([(position, position)])[0, axis])
            if calibration
            else position
        )
        result.append(replace(obj, value=value))
    return result
