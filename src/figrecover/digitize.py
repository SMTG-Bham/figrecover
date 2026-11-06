"""Find the curve in an image of a plot."""

import os

import cv2
import numpy as np
import numpy.typing as npt

from .calibration import Point

type Image = npt.NDArray[np.uint8]

AXIS_MARGIN = 3
"""Pixels beyond the axes that are ignored, so the axis lines are not traced."""

GRID_LINE_LENGTH = 120
"""Shortest straight run of pixels that ``remove_grid`` treats as a line."""

_LEVELS = 25  # grey-level histogram bins, about 10 levels wide
_MIN_CURVE_FRACTION = 0.75  # a curve has at least this many pixels per column
_MID_GREY = 128  # images darker than this on average are inverted
_CONTRAST = 0.5  # the curve is at most this fraction of the background level
_LUMA = np.array([0.114, 0.587, 0.299])  # weights of B, G and R in grey, as in OpenCV


def load_image(path: str | os.PathLike[str]) -> Image:
    """Read an image as greyscale, stretched to the full 0-255 range.

    Transparent pixels are placed on a white background, and images that are
    mostly dark are inverted, so the background is always light.

    Args:
        path: The image file, in any format OpenCV reads.

    Returns:
        A two-dimensional ``uint8`` array.

    Raises:
        ValueError: If the file cannot be read or has a single colour.
    """
    image = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        msg = f"cannot read image {os.fspath(path)!r}"
        raise ValueError(msg)

    pixels = image.astype(np.float64)
    match pixels.shape:
        case (_, _, 4):  # BGRA: place on white
            full = np.iinfo(image.dtype.str).max  # opaque
            alpha = pixels[..., 3:] / full
            pixels = (pixels[..., :3] * alpha + full * (1 - alpha)) @ _LUMA
        case (_, _, 3):  # BGR
            pixels = pixels @ _LUMA
        case _:  # already greyscale
            pass

    low, high = pixels.min(), pixels.max()
    if low == high:
        msg = f"image {os.fspath(path)!r} has a single colour"
        raise ValueError(msg)
    grey = np.round((pixels - low) * (255 / (high - low))).astype(np.uint8)
    return 255 - grey if np.median(grey) < _MID_GREY else grey


def remove_grid(image: Image) -> Image:
    """Paint over long horizontal and vertical lines, such as grid lines.

    Any straight, axis-aligned stretch of the curve at least
    ``GRID_LINE_LENGTH`` pixels long is removed as well.

    Args:
        image: Greyscale image with a light background.

    Returns:
        A copy of the image with the lines painted white.
    """
    _, ink = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    horizontal, vertical = (
        cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, shape)
        )
        for shape in ((GRID_LINE_LENGTH, 1), (1, GRID_LINE_LENGTH))
    )
    lines = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))

    cleaned = image.copy()
    cleaned[lines > 0] = 255
    return cleaned


def trace_curve(image: Image, origin: Point = (0, 0)) -> npt.NDArray[np.float64]:
    """Find the curve in a plot.

    The curve is the most common dark grey level in the plot area, the part of
    the image above and right of ``origin``. Excluding the rest removes the
    axes, their ticks and labels. Each image column that the curve crosses
    gives one point, at the median height of the curve's pixels there.

    Args:
        image: Greyscale image with a light background.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        An ``(n, 2)`` array of pixel ``(x, y)`` coordinates, measured from the
        bottom-left corner and sorted by x.

    Raises:
        ValueError: If no curve is found.
    """
    height = image.shape[0]
    left = max(0, int(origin[0]) + AXIS_MARGIN)
    bottom = max(0, height - int(origin[1]) - AXIS_MARGIN)
    plot = image[:bottom, left:]
    if plot.size == 0:
        msg = "the axes leave no plot area to trace"
        raise ValueError(msg)

    lower, upper = _curve_levels(plot)
    in_curve = (plot >= lower) & (plot < upper)
    columns, rows = np.nonzero(in_curve.T)  # sorted by column
    xs, starts = np.unique(columns, return_index=True)
    ys = [np.median(group) for group in np.split(rows, starts[1:])]
    return np.column_stack((xs + left, height - np.array(ys)))


def _curve_levels(plot: Image) -> tuple[float, float]:
    """Return the range of grey levels that belong to the curve."""
    counts, edges = np.histogram(plot, bins=_LEVELS, range=(0, 256))
    counts[counts < plot.shape[1] * _MIN_CURVE_FRACTION] = 0

    background = np.argmax(counts)
    dark = np.flatnonzero((counts > 0) & (edges[:-1] < _CONTRAST * edges[background]))
    if dark.size == 0:
        msg = "no curve found"
        raise ValueError(msg)
    curve = dark[np.argmax(counts[dark])]
    return float(edges[curve]), float(edges[curve + 1])
