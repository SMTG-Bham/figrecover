"""Find the curves in an image of a plot."""

import itertools
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

FRAME_REACH = 0.5
"""Fraction of the plot area that each line of a frame around the axes spans."""

FRAME_WIDTH = 8
"""Most pixels across a line of a frame around the axes, so a filled area is not one."""

GRID_LINE_LENGTH = 120
"""Shortest straight run of pixels that ``remove_grid`` treats as a line."""

_BIN_WIDTH = 32  # colour histogram bins are this many levels wide in each channel
_SAME_COLOUR = 48  # colours closer than this are shades of one colour
_FINE_BIN_WIDTH = 16  # a second look at each colour, in narrower bins,
_FINE_SAME_COLOUR = 20  # can split it into colours this far apart
_MATCH_DISTANCE = 96  # a requested colour picks a curve colour at most this far away
_MIN_CURVE_FRACTION = 0.75  # a curve has at least this many pixels per column
_CONTRAST = 64  # a curve differs from the background by this much in some channel
_AREA_DEPTH = 4  # a colour whose pixels lie this deep inside its shapes fills areas
_LINE_SPAN = 0.6  # straight lines across this fraction of the plot are grid lines
_BACKGROUND_SHARE = 1 / 3  # a background has this many pixels of a fill's, or more
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
    is the background, unless it is a filled area and the colour along the top
    edge of the plot area is the background instead. Curves are the common
    colours that differ from the background by at least a quarter of
    the full range in some channel, and that neither fill areas nor are drawn
    mostly in straight lines across the plot, as grid lines are.

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
    plot = _plot_area(image, origin)[0]
    palette = _palette(plot)
    labels = _classify(plot, palette)
    curves = [
        colour
        for index, colour in enumerate(palette[1:], start=1)
        if np.abs(colour - palette[0]).max() >= _CONTRAST
        and not _fills_areas(labels == index)
        and not _draws_grid(labels == index)
    ]
    if len(curves) < count:
        msg = f"found only {len(curves)} of {count} curves"
        raise ValueError(msg if curves else "no curve found")
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
        One ``(n, 2)`` array per colour, of the pixel ``(x, y)`` coordinates of
        pixel centres, measured from the bottom-left corner and sorted by x.

    Raises:
        ValueError: If no curve has one of the colours, two of them match the
            same curve, or one fills areas.
    """
    plot, left, top = _plot_area(image, origin)
    palette = _palette(plot)
    indices = [_match(palette, colour) for colour in colours]
    for (first, index), (second, other) in itertools.combinations(
        zip(colours, indices, strict=True), 2
    ):
        if index == other:
            msg = (
                f"colours #{bytes(first).hex()} and #{bytes(second).hex()} both "
                f"match the curve of colour #{bytes(_to_colour(palette[index])).hex()}"
            )
            raise ValueError(msg)
    labels = _classify(plot, palette)
    for colour, index in zip(colours, indices, strict=True):
        if _fills_areas(labels == index):
            msg = f"colour #{bytes(colour).hex()} fills areas rather than a curve"
            raise ValueError(msg)
    height = image.shape[0]
    return [_follow(labels == index, left, top, height) for index in indices]


def trace_curve(image: Image, origin: Point = (0, 0)) -> npt.NDArray[np.float64]:
    """Find the curve with the most common colour in a plot.

    See ``find_colours`` for how curves are told apart from the background,
    and ``trace_curves`` for how they are traced.

    Args:
        image: Colour image of a plot.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        An ``(n, 2)`` array of the pixel ``(x, y)`` coordinates of pixel
        centres, measured from the bottom-left corner and sorted by x.

    Raises:
        ValueError: If no curve is found.
    """
    return trace_curves(image, find_colours(image, 1, origin), origin)[0]


def _plot_area(image: Image, origin: Point) -> tuple[Image, int, int]:
    """Return the part of the image inside the axes, and its left and top edges.

    That is the part above and right of the axes, or if a frame closes them,
    as matplotlib draws by default, the part inside the frame too.
    """
    height = image.shape[0]
    left = max(0, int(origin[0]) + AXIS_MARGIN)
    bottom = max(0, height - int(origin[1]) - AXIS_MARGIN)
    if image[:bottom, left:].size == 0:
        msg = "the axes leave no plot area to trace"
        raise ValueError(msg)
    top, right = _inside_frame(ink_mask(image)[:bottom, left:] > 0)
    return image[top:bottom, left : left + right], left, top


def _inside_frame(ink: npt.NDArray[np.bool_]) -> tuple[int, int]:
    """Find the frame closing the axes in the part of an image beyond them.

    The frame's top is the topmost line that runs right from the left edge,
    next to the y-axis, across at least ``FRAME_REACH`` of the width, and is
    at most ``FRAME_WIDTH`` pixels thick. Its right side runs down from the
    end of the top, as far again of the rest of the height. Both are dropped
    with ``AXIS_MARGIN`` beyond them.

    Returns:
        The first row below the frame and the number of columns left of it,
        or 0 and the width if there is no frame.
    """
    height, width = ink.shape
    rows = ink.all(axis=1)
    runs = np.where(rows, width, np.argmin(ink, axis=1))  # ink from the left edge
    long = np.flatnonzero(runs >= width * FRAME_REACH)
    if not len(long):
        return 0, width
    first = last = long[0]
    while last + 1 < height and runs[last + 1] >= width * FRAME_REACH:
        last += 1  # a thick line
    if last - first >= FRAME_WIDTH:
        return 0, width  # an area, not a line
    end = int(runs[first])
    for column in range(max(end - AXIS_MARGIN, 0), end):
        below = ink[first:, column]
        run = len(below) if below.all() else int(np.argmin(below))
        if run >= (height - first) * FRAME_REACH:
            return last + 1 + AXIS_MARGIN, max(column - AXIS_MARGIN, 0)
    return 0, width


def _palette(plot: Image) -> npt.NDArray[np.float64]:
    """Return the background colour followed by the other common colours.

    The background is the most common colour, unless that is a filled area:
    when the colour most common along the top edge of the plot differs from it
    by at least ``_CONTRAST``, and has at least ``_BACKGROUND_SHARE`` as many
    pixels, the top edge's colour is the background.

    Antialiasing blends the edges of each curve with the background, and a
    blend can be common enough to count as a colour of its own. It is dropped
    if most of its pixels touch those of a colour that differs more from the
    background.
    """
    colours = _common_colours(plot)
    pixels = plot.reshape(-1, 3).astype(np.float64)
    nearest = np.linalg.norm(pixels[:, np.newaxis] - colours, axis=2).argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(colours))
    edge = int(np.bincount(nearest[: plot.shape[1]]).argmax())  # the top row
    if (
        np.abs(colours[edge] - colours[0]).max() >= _CONTRAST
        and counts[edge] >= counts[0] * _BACKGROUND_SHARE
    ):
        colours = colours[[edge, *(i for i in range(len(colours)) if i != edge)]]
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


def _fills_areas(in_colour: npt.NDArray[np.bool_]) -> bool:
    """Return whether a colour's pixels fill areas rather than draw lines.

    Most pixels of a line lie close to its edges, and most of an area's lie
    more than ``_AREA_DEPTH`` pixels inside it.
    """
    depth = cv2.distanceTransform(in_colour.astype(np.uint8), cv2.DIST_L2, 3)
    return bool(np.median(depth[in_colour]) > _AREA_DEPTH)


def _draws_grid(in_colour: npt.NDArray[np.bool_]) -> bool:
    """Return whether most of a colour's pixels lie on lines across the plot.

    The lines are horizontal or vertical, and run across at least
    ``_LINE_SPAN`` of the plot area, as grid lines do.
    """
    height, width = in_colour.shape
    mask = in_colour.astype(np.uint8)
    lines = [
        cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, shape)
        )
        for shape in ((round(width * _LINE_SPAN), 1), (1, round(height * _LINE_SPAN)))
    ]
    on_lines = np.count_nonzero(cv2.bitwise_or(*lines))
    return bool(2 * on_lines > np.count_nonzero(mask))


def _common_colours(plot: Image) -> npt.NDArray[np.float64]:
    """Return the colours with enough pixels to span the plot, most common first.

    Pixels are counted in a coarse colour histogram, so that a colour blurred
    by antialiasing or compression is counted once. Each common colour but the
    background is then counted again from its own pixels, in narrower bins,
    and split if that finds two or more colours with enough pixels, such as
    neighbouring colours of a colour map. The most common colour, the
    background, is always returned.
    """
    pixels = plot.reshape(-1, 3)
    minimum = plot.shape[1] * _MIN_CURVE_FRACTION
    colours, totals, owners = _cluster(pixels, _BIN_WIDTH, _SAME_COLOUR)
    found: list[tuple[npt.NDArray[np.float64], int]] = []
    for index in np.flatnonzero(totals[1:] >= minimum) + 1:
        parts, sizes, _ = _cluster(
            pixels[owners == index], _FINE_BIN_WIDTH, _FINE_SAME_COLOUR
        )
        if np.count_nonzero(sizes >= minimum) > 1:
            found += zip(parts[sizes >= minimum], sizes[sizes >= minimum], strict=True)
        else:
            found.append((colours[index], totals[index]))
    found.sort(key=lambda pair: -pair[1])
    return np.array([colours[0], *(colour for colour, _ in found)])


def _cluster(
    pixels: npt.NDArray[np.uint8], width: int, same: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Group pixels into colours with a histogram whose bins are ``width`` wide.

    Starting from the fullest bin, each bin joins the colour of an earlier bin
    closer than ``same``, so a colour that straddles the bin edges is counted
    once.

    Returns:
        The colours, most pixels first; how many pixels each has; and which
        colour each pixel joined.
    """
    bins = np.ravel_multi_index((pixels // width).T, (256 // width,) * 3)
    counts = np.bincount(bins)
    sums = np.column_stack([np.bincount(bins, weights=level) for level in pixels.T])

    full = np.argsort(-counts, kind="stable")[: np.count_nonzero(counts)]
    colours = [sums[full[0]] / counts[full[0]]]
    totals = [counts[full[0]]]
    owner = np.zeros(len(counts), dtype=np.intp)  # the colour each bin joined
    for bin_ in full[1:]:
        colour = sums[bin_] / counts[bin_]
        distances = np.linalg.norm(np.array(colours) - colour, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < same:
            totals[nearest] += counts[bin_]
            owner[bin_] = nearest
        else:
            owner[bin_] = len(colours)
            colours.append(colour)
            totals.append(counts[bin_])

    ranked = np.argsort(-np.array(totals), kind="stable")
    rank = np.empty_like(ranked)
    rank[ranked] = np.arange(len(ranked))
    return np.array(colours)[ranked], np.array(totals)[ranked], rank[owner[bins]]


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
    in_colour: npt.NDArray[np.bool_], left: int, top: int, height: int
) -> npt.NDArray[np.float64]:
    """Return the median height of a curve's pixels in each column it crosses.

    The curve's pixels are in a part of an image ``height`` pixels tall, from
    column ``left`` and row ``top``.
    """
    _, specks, stats, _ = cv2.connectedComponentsWithStats(in_colour.astype(np.uint8))
    areas = stats[:, cv2.CC_STAT_AREA]
    areas[0] = 0  # the pixels outside the curve
    in_curve = (areas >= _MIN_PIXELS)[specks]

    columns, rows = np.nonzero(in_curve.T)  # sorted by column
    xs, starts = np.unique(columns, return_index=True)
    ys = [np.median(group) for group in np.split(rows, starts)[1:]]
    # The centre of the pixel in column c and row r is at (c + 0.5, height - r - 0.5).
    return np.column_stack((xs + left + 0.5, height - top - np.array(ys) - 0.5))


def _to_colour(levels: npt.NDArray[np.float64]) -> Colour:
    """Round red, green and blue levels to a colour."""
    red, green, blue = (int(level) for level in np.round(levels))
    return red, green, blue
