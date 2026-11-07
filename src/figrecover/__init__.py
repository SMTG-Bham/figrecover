"""Recover data from images of plots."""

from importlib.metadata import version

from .axes import Axes, read_axes
from .calibration import Calibration, Point, Tick
from .digitize import (
    Colour,
    Image,
    find_colours,
    find_markers,
    load_image,
    remove_grid,
    trace_curve,
    trace_curves,
)

__version__ = version("figrecover")

__all__ = [
    "Axes",
    "Calibration",
    "Colour",
    "Image",
    "Point",
    "Tick",
    "__version__",
    "find_colours",
    "find_markers",
    "load_image",
    "read_axes",
    "remove_grid",
    "trace_curve",
    "trace_curves",
]
