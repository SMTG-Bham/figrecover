"""Recover data from images of plots."""

from importlib.metadata import version

from .calibration import Calibration, Point
from .digitize import Image, load_image, remove_grid, trace_curve

__version__ = version("figrecover")

__all__ = [
    "Calibration",
    "Image",
    "Point",
    "__version__",
    "load_image",
    "remove_grid",
    "trace_curve",
]
