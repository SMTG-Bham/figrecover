"""Find the curves in an image of a plot."""

import os
from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from .calibration import Point

type Image = npt.NDArray[np.uint8]
type Colour = tuple[int, int, int]

AXIS_MARGIN = 3
"""Pixels beyond the axes that are ignored, so the axis lines are not traced."""

GRID_LINE_LENGTH = 120
"""Shortest straight run of pixels that ``remove_grid`` treats as a line."""

_BIN_WIDTH = 32  # colour histogram bins are this many levels wide in each channel
_SAME_COLOUR = 48  # colours closer than this are shades of one colour
_MATCH_DISTANCE = 96  # a requested colour picks a curve colour at most this far away
_MIN_CURVE_FRACTION = 0.75  # a curve has at least this many pixels per column
_CONTRAST = 128  # a curve differs from the background by this much in some channel
_COVERAGE_COST = 0.25  # how much a pixel's distance from full coverage counts
_MIN_COVERAGE = 0.5  # less covered pixels belong to the background
_MIN_PIXELS = 4  # smaller specks are noise, such as where two other curves cross


def load_image(path: str | os.PathLike[str]) -> Image:
    """Read an image in colour, stretched to the full 0-255 range.

    Transparent pixels are placed on a white background.

    Args:
        path: The image file, in any format OpenCV reads.

    Returns:
        A ``(height, width, 3)`` ``uint8`` array of red, green and blue levels.

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
            pixels = pixels[..., 2::-1] * alpha + full * (1 - alpha)
        case (_, _, 3):  # BGR
            pixels = pixels[..., ::-1]
        case _:  # greyscale
            pixels = np.repeat(pixels[..., np.newaxis], 3, axis=2)

    if np.all(pixels == pixels[0, 0]):
        msg = f"image {os.fspath(path)!r} has a single colour"
        raise ValueError(msg)
    low, high = pixels.min(), pixels.max()
    return np.round((pixels - low) * (255 / (high - low))).astype(np.uint8)


def remove_grid(image: Image) -> Image:
    """Paint over long horizontal and vertical lines, such as grid lines.

    Any straight, axis-aligned stretch of a curve at least
    ``GRID_LINE_LENGTH`` pixels long is removed as well.

    Args:
        image: Colour image of a plot.

    Returns:
        A copy of the image with the lines painted in the background colour.
    """
    ink = ink_mask(image)
    horizontal, vertical = (
        cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, shape)
        )
        for shape in ((GRID_LINE_LENGTH, 1), (1, GRID_LINE_LENGTH))
    )
    lines = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))

    cleaned = image.copy()
    cleaned[lines > 0] = np.median(image, axis=(0, 1))
    return cleaned


def ink_mask(image: Image) -> npt.NDArray[np.uint8]:
    """Find the pixels that are drawn on, such as lines and text.

    Pixels are split into light and dark ones by Otsu's method, and the less
    common of the two is taken to be ink, so a dark background works too.

    Args:
        image: Colour image of a plot.

    Returns:
        A ``(height, width)`` array, 255 at ink and 0 at the background.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size / 2:  # a dark background, so light ink
        ink = cv2.bitwise_not(ink)
    return np.asarray(ink, dtype=np.uint8)


def find_colours(image: Image, count: int = 1, origin: Point = (0, 0)) -> list[Colour]:
    """Find the colours of the curves in a plot.

    The plot area is the part of the image above and right of ``origin``. Its
    common colours are those with enough pixels to span it; the most common
    is the background. Curves are the common colours that differ from the
    background by at least half the full range in some channel, so light grid
    lines are not taken for curves.

    Args:
        image: Colour image of a plot.
        count: How many curves to find.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        The colours of the ``count`` curves with the most pixels, most common
        first.

    Raises:
        ValueError: If there are fewer than ``count`` curves.
    """
    palette = _palette(_plot_area(image, origin)[0])
    contrast = np.abs(palette[1:] - palette[0]).max(axis=1)
    curves = palette[1:][contrast >= _CONTRAST]
    if len(curves) < count:
        msg = f"found only {len(curves)} of {count} curves"
        raise ValueError(msg if len(curves) else "no curve found")
    return [_to_colour(colour) for colour in curves[:count]]


def trace_curves(
    image: Image, colours: Sequence[Colour], origin: Point = (0, 0)
) -> list[npt.NDArray[np.float64]]:
    """Find the curves drawn in the given colours.

    Each colour picks the nearest common colour in the plot area, the part of
    the image above and right of ``origin``. Excluding the rest removes the
    axes, their ticks and labels. Each pixel belongs to the common colour
    that it best matches when blended with the background, so the antialiased
    edges of a curve count towards it, and specks of fewer than four pixels
    are ignored. Each image column that a curve crosses gives one point, at
    the median height of the curve's pixels there.

    Args:
        image: Colour image of a plot.
        colours: The colour of each curve, such as ``find_colours`` returns.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        One ``(n, 2)`` array per colour, of pixel ``(x, y)`` coordinates
        measured from the bottom-left corner and sorted by x.

    Raises:
        ValueError: If no curve has one of the colours.
    """
    plot, left = _plot_area(image, origin)
    palette = _palette(plot)
    indices = [_match(palette, colour) for colour in colours]
    labels = _classify(plot, palette)
    return [_follow(labels == index, left, image.shape[0]) for index in indices]


def trace_curve(image: Image, origin: Point = (0, 0)) -> npt.NDArray[np.float64]:
    """Find the curve with the most common colour in a plot.

    See ``find_colours`` for how curves are told apart from the background,
    and ``trace_curves`` for how they are traced.

    Args:
        image: Colour image of a plot.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        An ``(n, 2)`` array of pixel ``(x, y)`` coordinates, measured from the
        bottom-left corner and sorted by x.

    Raises:
        ValueError: If no curve is found.
    """
    return trace_curves(image, find_colours(image, 1, origin), origin)[0]


def _plot_area(image: Image, origin: Point) -> tuple[Image, int]:
    """Return the part of the image above and right of the axes, and its left edge."""
    height = image.shape[0]
    left = max(0, int(origin[0]) + AXIS_MARGIN)
    bottom = max(0, height - int(origin[1]) - AXIS_MARGIN)
    plot = image[:bottom, left:]
    if plot.size == 0:
        msg = "the axes leave no plot area to trace"
        raise ValueError(msg)
    return plot, left


def _palette(plot: Image) -> npt.NDArray[np.float64]:
    """Return the background colour followed by the other common colours.

    Antialiasing blends the edges of each curve with the background, and a
    blend can be common enough to count as a colour of its own. It is dropped
    if most of its pixels touch those of a colour that differs more from the
    background.
    """
    colours = _common_colours(plot)
    labels = _classify(plot, colours)
    strength = np.linalg.norm(colours - colours[0], axis=1)
    keep = [0]
    for index in range(1, len(colours)):
        own = labels == index
        stronger = np.isin(labels, np.flatnonzero(strength > strength[index]))
        touching = cv2.dilate(stronger.astype(np.uint8), np.ones((3, 3), np.uint8))
        if 2 * np.count_nonzero(own & (touching > 0)) < np.count_nonzero(own):
            keep.append(index)
    return colours[keep]


def _common_colours(plot: Image) -> npt.NDArray[np.float64]:
    """Return the colours with enough pixels to span the plot, most common first.

    Pixels are counted in a coarse colour histogram. Starting from the fullest
    bin, each bin joins the colour of an earlier bin within ``_SAME_COLOUR``,
    so a colour that straddles the bin edges is counted once. The most common
    colour, the background, is always returned.
    """
    pixels = plot.reshape(-1, 3)
    bins = np.ravel_multi_index((pixels // _BIN_WIDTH).T, (256 // _BIN_WIDTH,) * 3)
    counts = np.bincount(bins)
    sums = np.column_stack([np.bincount(bins, weights=level) for level in pixels.T])

    full = np.argsort(-counts, kind="stable")[: np.count_nonzero(counts)]
    colours = [sums[full[0]] / counts[full[0]]]
    totals = [counts[full[0]]]
    for bin_ in full[1:]:
        colour = sums[bin_] / counts[bin_]
        distances = np.linalg.norm(np.array(colours) - colour, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < _SAME_COLOUR:
            totals[nearest] += counts[bin_]
        else:
            colours.append(colour)
            totals.append(counts[bin_])

    ranked = np.argsort(-np.array(totals), kind="stable")
    minimum = plot.shape[1] * _MIN_CURVE_FRACTION
    common = [ranked[0], *(i for i in ranked[1:] if totals[i] >= minimum)]
    return np.array(colours)[common]


def _classify(plot: Image, colours: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Label each pixel with the index of the colour it shows.

    Each pixel is compared with every blend of the background,
    ``colours[0]``, and another colour, at the coverage that fits it best.
    Partial coverage costs a little, so a pixel that matches a light colour is
    not taken for a blend of a darker one. Pixels less than half-way from the
    background to their best match are labelled 0, as background.
    """
    offsets = plot.astype(np.float64) - colours[0]
    directions = colours - colours[0]  # the first is zero: pure background
    lengths = np.linalg.norm(directions, axis=1)
    projections = offsets @ directions.T
    fits = np.divide(
        projections, lengths**2, out=np.ones_like(projections), where=lengths > 0
    )
    coverage = np.clip(fits, 0, 1)
    # The squared distance |offset - coverage * direction|^2, expanded.
    squared = (
        np.sum(offsets**2, axis=-1, keepdims=True)
        - 2 * coverage * projections
        + (coverage * lengths) ** 2
    )
    cost = np.sqrt(np.maximum(squared, 0)) + _COVERAGE_COST * (1 - coverage) * lengths

    best = np.argmin(cost, axis=-1)
    best_coverage = np.take_along_axis(coverage, best[..., np.newaxis], axis=-1)
    return np.where(best_coverage[..., 0] >= _MIN_COVERAGE, best, 0)


def _match(palette: npt.NDArray[np.float64], colour: Colour) -> int:
    """Return the index of the curve colour in the palette nearest to ``colour``."""
    distances = np.linalg.norm(palette[1:] - colour, axis=1)
    if not np.any(distances <= _MATCH_DISTANCE):
        msg = f"no curve of colour #{bytes(colour).hex()} found"
        raise ValueError(msg)
    return int(np.argmin(distances)) + 1


def _follow(
    in_colour: npt.NDArray[np.bool_], left: int, height: int
) -> npt.NDArray[np.float64]:
    """Return the median height of a curve's pixels in each column it crosses."""
    _, specks, stats, _ = cv2.connectedComponentsWithStats(in_colour.astype(np.uint8))
    areas = stats[:, cv2.CC_STAT_AREA]
    areas[0] = 0  # the pixels outside the curve
    in_curve = (areas >= _MIN_PIXELS)[specks]

    columns, rows = np.nonzero(in_curve.T)  # sorted by column
    xs, starts = np.unique(columns, return_index=True)
    ys = [np.median(group) for group in np.split(rows, starts)[1:]]
    return np.column_stack((xs + left, height - np.array(ys)))


def _to_colour(levels: npt.NDArray[np.float64]) -> Colour:
    """Round red, green and blue levels to a colour."""
    red, green, blue = (int(level) for level in np.round(levels))
    return red, green, blue
