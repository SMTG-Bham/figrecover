"""Recover data from images of plots."""

from importlib.metadata import version

from .axes import Axes, read_axes
from .calibration import Calibration, Point, Tick
from .digitize import (
    Colour,
    Image,
    find_colours,
    find_error_bars,
    find_marker_error_bars,
    find_markers,
    load_image,
    remove_grid,
    trace_curve,
    trace_curves,
)
from .errorbars import ErrorBars
from .panels import Panel, read_panels
from .pdf import (
    Box,
    Drawing,
    find_pdf_colours,
    find_pdf_error_bars,
    find_pdf_markers,
    load_pdf,
    read_pdf_axes,
    remove_pdf_grid,
    render_pdf,
    trace_pdf_curves,
)

__version__ = version("figrecover")

__all__ = [
    "Axes",
    "Box",
    "Calibration",
    "Colour",
    "Drawing",
    "ErrorBars",
    "Image",
    "Panel",
    "Point",
    "Tick",
    "__version__",
    "find_colours",
    "find_error_bars",
    "find_marker_error_bars",
    "find_markers",
    "find_pdf_colours",
    "find_pdf_error_bars",
    "find_pdf_markers",
    "load_image",
    "load_pdf",
    "read_axes",
    "read_panels",
    "read_pdf_axes",
    "remove_grid",
    "remove_pdf_grid",
    "render_pdf",
    "trace_curve",
    "trace_curves",
    "trace_pdf_curves",
]
