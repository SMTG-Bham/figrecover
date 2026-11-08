"""Recover data from images of plots."""

from importlib.metadata import version

from .axes import Axes, read_axes
from .calibration import Calibration, Point, Tick
from .digitize import (
    STYLES,
    Colour,
    Image,
    MarkerSeries,
    Series,
    Style,
    find_colours,
    find_marker_series,
    find_markers,
    find_series,
    load_image,
    remove_grid,
    trace_curve,
    trace_curves,
    trace_series,
)
from .level import Levelling, find_skew, level_image
from .panels import Panel, read_panels
from .pdf import (
    Box,
    Drawing,
    find_pdf_colours,
    find_pdf_markers,
    find_pdf_series,
    load_pdf,
    read_pdf_axes,
    remove_pdf_grid,
    render_pdf,
    trace_pdf_curves,
    trace_pdf_series,
)
from .svg import load_svg, render_svg

__version__ = version("figrecover")

__all__ = [
    "STYLES",
    "Axes",
    "Box",
    "Calibration",
    "Colour",
    "Drawing",
    "Image",
    "Levelling",
    "MarkerSeries",
    "Panel",
    "Point",
    "Series",
    "Style",
    "Tick",
    "__version__",
    "find_colours",
    "find_marker_series",
    "find_markers",
    "find_pdf_colours",
    "find_pdf_markers",
    "find_pdf_series",
    "find_series",
    "find_skew",
    "level_image",
    "load_image",
    "load_pdf",
    "load_svg",
    "read_axes",
    "read_panels",
    "read_pdf_axes",
    "remove_grid",
    "remove_pdf_grid",
    "render_pdf",
    "render_svg",
    "trace_curve",
    "trace_curves",
    "trace_pdf_curves",
    "trace_pdf_series",
    "trace_series",
]
