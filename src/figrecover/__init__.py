"""Recover data from images of plots."""

from importlib.metadata import version

from .calibration import Calibration, Point
from .digitize import (
    Colour,
    Image,
    find_colours,
    load_image,
    remove_grid,
    trace_curve,
    trace_curves,
)

__version__ = version("figrecover")

__all__ = [
    "Calibration",
    "Colour",
    "Image",
    "Point",
    "__version__",
    "find_colours",
    "load_image",
    "remove_grid",
    "trace_curve",
    "trace_curves",
]
