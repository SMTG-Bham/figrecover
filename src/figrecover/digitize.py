"""Find the curves in an image of a plot."""

import itertools
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from .calibration import Point
from .errorbars import ErrorBars, _Bars, _filled_markers, _read_bars

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
_MATCH_DISTANCE = 96  # a colour, or a pixel, matches a curve colour this far away
_MIN_CURVE_FRACTION = 0.75  # a curve has at least this many pixels per column
_CONTRAST = 64  # a curve differs from the background by this much in some channel
_AREA_DEPTH = 4  # a colour whose pixels lie this deep inside its shapes fills areas
_LINE_SPAN = 0.6  # straight lines across this fraction of the plot are grid lines
_BACKGROUND_SHARE = 1 / 3  # a background has this many pixels of a fill's, or more
_COVERAGE_COST = 0.25  # how much a pixel's distance from full coverage counts
_MIN_COVERAGE = 0.5  # less covered pixels belong to the background
_MIN_PIXELS = 4  # smaller specks are noise, such as where two other curves cross
_RUN_GAP = 6  # a curve's pixels in a column this far apart or less are one run
_MIN_BRANCH = 0.05  # branches across less of the plot's width than this are minor
_FRAME_SIDE = 0.05  # frame lines span this fraction of the plot's shorter side
_FRAME_CONTRAST = 32  # its lines differ from the background by this much or more
_FRAME_LINED = 0.75  # its sides run along this fraction of its height or more
_KEY_REACH = 20  # a legend key's marker lies within this many pixels of its position
_KEY_LINE = 2  # a key this many times as wide as tall has a line through its marker
_MARKER_SCORE = 0.6  # a marker correlates with its key's this well or better
_MARKER_INK = 0.6  # a marker has at least this fraction of the ink of its key's
_MARKER_SPACING = 0.75  # markers are this many marker widths apart or more


@dataclass(frozen=True, slots=True)
class _Box:
    """A rectangle of an image, in rows from the top and columns from the left.

    ``bottom`` and ``right`` are one past its last row and column.
    """

    top: int
    bottom: int
    left: int
    right: int

    def holds(self, other: _Box) -> bool:
        """Return whether another box lies within this one."""
        return (
            self.top <= other.top
            and other.bottom <= self.bottom
            and self.left <= other.left
            and other.right <= self.right
        )

    def within(self, other: _Box) -> _Box:
        """Return this box measured from the top-left corner of another."""
        return _Box(
            self.top - other.top,
            self.bottom - other.top,
            self.left - other.left,
            self.right - other.left,
        )

    def placed(self, other: _Box) -> _Box:
        """Return this box, measured in another, measured as the other one is."""
        return _Box(
            self.top + other.top,
            self.bottom + other.top,
            self.left + other.left,
            self.right + other.left,
        )

    def turned(self) -> _Box:
        """Return this box in the image turned over its diagonal, rows for columns."""
        return _Box(self.left, self.right, self.top, self.bottom)


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

    The plot area is the part of the image above and right of ``origin``,
    less the frames inside it, such as those of insets and legends, with all
    they hold, as ``_frames`` finds them. Its common colours are those with
    enough pixels to span it; the most common is the background, unless it
    is a filled area and the colour along the top edge of the plot area is
    the background instead. Curves are the common colours that differ from
    the background by at least a quarter of the full range in some channel,
    and that neither fill areas nor are drawn mostly in straight lines across
    the plot, as grid lines are.

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
    curves = _curve_colours(image, origin)
    if len(curves) < count:
        msg = f"found only {len(curves)} of {count} curves"
        raise ValueError(msg if curves else "no curve found")
    return curves[:count]


def _curve_colours(image: Image, origin: Point) -> list[Colour]:
    """Return the colours of all the curves in a plot, most pixels first.

    See ``find_colours`` for which colours are curves.
    """
    plot = _plot_area(image, origin)[0]
    palette = _palette(plot)
    labels = _classify(plot, palette)
    return [
        _to_colour(colour)
        for index, colour in enumerate(palette[1:], start=1)
        if np.abs(colour - palette[0]).max() >= _CONTRAST
        and not _fills_areas(labels == index)
        and not _draws_grid(labels == index)
    ]


def trace_curves(
    image: Image, colours: Sequence[Colour], origin: Point = (0, 0)
) -> list[npt.NDArray[np.float64]]:
    """Find the curves drawn in the given colours.

    Each colour picks the nearest common colour in the plot area, the part of
    the image above and right of ``origin``. Excluding the rest removes the
    axes, their ticks and labels, and frames inside the plot area, such as
    those of insets and legends, are painted over with all they hold, as
    ``_frames`` finds them, so their curves and keys are not traced. Each
    pixel belongs to the common colour that it best matches when blended with
    the background, so the antialiased edges of a curve count towards it,
    unless it matches none, as black text among coloured curves does; and
    specks of fewer than four pixels are ignored, as are the error bars of
    filled markers, as ``find_error_bars`` finds them, so that they do not
    pull the curve away from its points. Each image column gives one
    point for each branch of a curve that crosses it, at the median height of
    the branch's pixels there, so both sides of a loop are traced. See
    ``_follow`` for what a branch is.

    Args:
        image: Colour image of a plot.
        colours: The colour of each curve, such as ``find_colours`` returns.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        One ``(n, 2)`` array per colour, of the pixel ``(x, y)`` coordinates of
        pixel centres, measured from the bottom-left corner and sorted by x,
        then y.

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
    return [
        _follow(_without_bars(labels, index), left, top, height) for index in indices
    ]


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


def find_markers(
    image: Image, keys: Sequence[Point], origin: Point = (0, 0)
) -> list[npt.NDArray[np.float64]]:
    """Find the markers of each series in a plot, by their shapes.

    Each key is the position of a series' marker in the legend. Its shape is
    cut out of the image, leaving out a line through it, as legends of line
    plots draw, and looked for in the plot area, the part of the image above
    and right of ``origin``, less the frames inside it, such as those of
    insets and legends, with all they hold, as ``_frames`` finds them. A
    marker is where the shape correlates with the image better than
    ``_MARKER_SCORE``, with at least ``_MARKER_INK`` of the key's ink. Where
    the shapes of several keys match, closer together than
    ``_MARKER_SPACING`` of a marker's width, the best match wins. The keys
    themselves are not markers, if the legend lies in the plot area.

    Args:
        image: Colour image of a plot.
        keys: The pixel position of the marker in each series' legend key,
            measured from the bottom-left corner. Each key's marker must have
            its own shape.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        One ``(n, 2)`` array per key, of the pixel ``(x, y)`` coordinates of
        the centres of the markers, measured from the bottom-left corner and
        sorted by x, then y. A series whose markers are all hidden gives an
        empty array.

    Raises:
        ValueError: If there is no marker at a key, or there is no plot area.
    """
    levels = _ink_levels(image)
    shapes = [_key_marker(levels, key) for key in keys]
    plot, left, top = _plot_area(image, origin)
    area = _ink_levels(plot)
    found = [
        (score, row, column, series)
        for series, (shape, mask) in enumerate(shapes)
        for score, row, column in _matches(area, shape, mask)
    ]
    sizes = [max(shape.shape) - 2 for shape, _ in shapes]  # less the margin
    spacing = _MARKER_SPACING * min(sizes)
    height = image.shape[0]
    centres: list[list[Point]] = [[] for _ in keys]
    kept: list[tuple[float, float]] = []
    for _, row, column, series in sorted(found, reverse=True):
        # The centre of the pixel in column c and row r is at
        # (c + 0.5, height - r - 0.5).
        x, y = column + left + 0.5, height - top - row - 0.5
        if any(math.dist((row, column), other) < spacing for other in kept) or any(
            math.dist((x, y), key) < size for key, size in zip(keys, sizes, strict=True)
        ):
            continue
        kept.append((row, column))
        centres[series].append((x, y))
    return [
        np.array(sorted(points), dtype=np.float64).reshape(-1, 2) for points in centres
    ]


def find_error_bars(
    image: Image, colours: Sequence[Colour], origin: Point = (0, 0)
) -> list[ErrorBars]:
    """Find the filled markers of each colour in a plot, and their error bars.

    The plot area is that which ``trace_curves`` traces, and each colour
    picks the nearest common colour in it, as there. A filled marker is a
    shape of its colour thicker than its lines and bars, by a pixel each
    side or more, and its point is the middle of its box. Its error bars run
    from the edges of the marker, up and down its middle column and across
    its middle row, in its colour, to their ends, or to the middles of their
    caps: short lines across their ends, wider than the bars. A bar may cross
    ink of other colours up to 6 pixels wide. Before a cap, a bar needs only
    a row; without one, it must run 5 pixels beyond its marker and be
    straight, and not be a line of the colour that leaves the bar's path at
    its end, as the marker's own line does. See ``_follow_bar``.

    Args:
        image: Colour image of a plot.
        colours: The colour of each series, such as ``find_colours`` returns.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        The error bars of each colour's markers, measured from the
        bottom-left corner, sorted by x, then y. A bar whose end is hidden,
        by a marker, another colour or the edge of the plot area, gives NaN
        for that end, as does a marker without a bar that way.

    Raises:
        ValueError: If no curve has one of the colours.
    """
    plot, left, top = _plot_area(image, origin)
    palette = _palette(plot)
    indices = [_match(palette, colour) for colour in colours]
    labels = _classify(plot, palette)
    found = []
    for index in indices:
        points, low, high = _placed(_bars_of(labels, index), left, top, image.shape[0])
        order = np.lexsort((points[:, 1], points[:, 0]))
        found.append(ErrorBars(points[order], low[order], high[order]))
    return found


def _bars_of(labels: npt.NDArray[np.intp], index: int) -> _Bars:
    """Return the error bars of the filled markers of a colour, by its index.

    ``labels`` is the index of the colour of each pixel, as ``_classify``
    labels them. See ``find_error_bars`` for which markers and bars are
    found.
    """
    in_colour = labels == index
    return _read_bars(in_colour, (labels > 0) & ~in_colour, _filled_markers(in_colour))


def _without_bars(labels: npt.NDArray[np.intp], index: int) -> npt.NDArray[np.bool_]:
    """Return where a colour lies, less the error bars of its filled markers.

    ``labels`` is as ``_bars_of`` takes it.
    """
    return np.asarray((labels == index) & ~_bars_of(labels, index).drawn)


def find_marker_error_bars(
    image: Image, keys: Sequence[Point], origin: Point = (0, 0)
) -> list[ErrorBars]:
    """Find the markers of each series by their shapes, and their error bars.

    The markers are those that ``find_markers`` finds, each the size of its
    key's marker. Their error bars are followed as ``find_error_bars``
    follows them, but in ink of any colour, 64 levels or more from the
    background, as series told apart by their markers often share one.

    Args:
        image: Colour image of a plot.
        keys: The pixel position of the marker in each series' legend key,
            measured from the bottom-left corner.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        The error bars of each key's markers, whose points are those that
        ``find_markers`` returns, with the ends of their bars as
        ``find_error_bars`` gives them.

    Raises:
        ValueError: If there is no marker at a key, or there is no plot area.
    """
    markers = find_markers(image, keys, origin)
    levels = _ink_levels(image)
    plot, left, top = _plot_area(image, origin)
    ink = _ink_levels(plot) >= _CONTRAST / 255
    height = image.shape[0]
    found = []
    for series, key in zip(markers, keys, strict=True):
        size = np.array(_key_marker(levels, key)[0].shape) - 2  # less the margin
        middles = np.column_stack([height - top - series[:, 1], series[:, 0] - left])
        near, far = np.round(middles - size / 2), np.round(middles + size / 2)
        boxes = np.column_stack([near[:, 0], far[:, 0], near[:, 1], far[:, 1]])
        bars = _read_bars(ink, np.zeros_like(ink), boxes.astype(np.intp))
        found.append(ErrorBars(series, *_placed(bars, left, top, height)[1:]))
    return found


def _placed(
    bars: _Bars, left: int, top: int, height: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the points and the ends of error bars found in a part of an image.

    The part starts at column ``left`` and row ``top`` of an image ``height``
    pixels tall, and positions are measured from its bottom-left corner: the
    middles of the markers, the left and lower ends of their bars, and their
    right and upper ends.
    """
    rows, columns = bars.middles.T
    return (
        np.column_stack([left + columns, height - top - rows]),
        np.column_stack([left + bars.low[:, 1], height - top - bars.high[:, 0]]),
        np.column_stack([left + bars.high[:, 1], height - top - bars.low[:, 0]]),
    )


def _plot_area(image: Image, origin: Point) -> tuple[Image, int, int]:
    """Return the part of the image inside the axes, and its left and top edges.

    That is the part that ``_framed_area`` finds, with the frames inside it,
    such as those of insets and legends, painted out, as ``_unframed``
    paints them.
    """
    area, frames = _framed_area(image, origin)
    plot = _unframed(image[area.top : area.bottom, area.left : area.right], frames)
    return plot, area.left, area.top


def _framed_area(image: Image, origin: Point) -> tuple[_Box, list[_Box]]:
    """Return the part of the image inside the axes, and the frames inside it.

    That is the part above and right of the axes, or if a frame closes them,
    as matplotlib draws by default, the part inside the frame too. The frames
    inside it, such as those of insets and legends, are those that
    ``_frames`` finds, measured in it.

    Raises:
        ValueError: If the axes leave no plot area.
    """
    height = image.shape[0]
    left = max(0, int(origin[0]) + AXIS_MARGIN)
    bottom = max(0, height - int(origin[1]) - AXIS_MARGIN)
    if image[:bottom, left:].size == 0:
        msg = "the axes leave no plot area to trace"
        raise ValueError(msg)
    ink = ink_mask(image)[:bottom, left:] > 0
    top, right = _inside_frame(ink)
    plot = image[top:bottom, left : left + right]
    background = np.median(plot.reshape(-1, 3), axis=0)
    frames = _frames(plot, ink[top:, :right], background)
    return _Box(top, bottom, left, left + right), frames


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
            return int(last) + 1 + AXIS_MARGIN, max(column - AXIS_MARGIN, 0)
    return 0, width


def _unframed(plot: Image, frames: Sequence[_Box]) -> Image:
    """Return a plot area with frames inside it painted in its background colour.

    Each frame, such as ``_frames`` finds, is painted with all it holds, and
    ``AXIS_MARGIN`` around it, so that an inset's curve or a legend's key is
    not traced as part of a curve of its colour. The background colour is
    the median.
    """
    background = np.median(plot.reshape(-1, 3), axis=0)
    painted = plot.copy()
    for frame in frames:
        painted[
            max(frame.top - AXIS_MARGIN, 0) : frame.bottom + AXIS_MARGIN,
            max(frame.left - AXIS_MARGIN, 0) : frame.right + AXIS_MARGIN,
        ] = background
    return painted


def _frames(
    plot: Image, ink: npt.NDArray[np.bool_], background: npt.NDArray[np.float64]
) -> list[_Box]:
    """Find the frames inside a plot area, such as those of insets and legends.

    A frame is a box of four straight lines, in one colour, each at least
    ``_FRAME_SIDE`` of the plot area's shorter side long and at most
    ``FRAME_WIDTH`` pixels thick, and differing from the background by at
    least ``_FRAME_CONTRAST``, as legends' light grey frames do. The lines
    along its top and bottom run from one side to the other, and a line down
    each side runs along ``_FRAME_LINED`` of the height between them or more.
    Its corners are square, or rounded by a few pixels at most. Lines can
    run on past its corners, as tick marks at an inset's corners do, and the
    lines from an inset to the part it magnifies, but the top and bottom
    each end at one side or the other, while grid lines run on past both.
    The flat and upright stretches of a curve, as a step plot draws, never
    close such a box, as a curve crosses each column once, and filled areas,
    such as bars and shaded spans, are thicker than lines, outlined or not.

    A frame hides what lies behind it, as an inset or a legend does, so no
    curve crosses its lines, as ``_crossed`` finds, while one runs through a
    box drawn around part of it, such as one marking the part an inset
    magnifies. Lines that cross a frame from side to side, such as an
    inset's grid lines, are not taken to cross it where they meet its tick
    marks.

    Args:
        plot: The plot area.
        ink: Its ink, as ``ink_mask`` finds it.
        background: Its background colour.
    """
    drawn = (np.abs(plot - background).max(axis=2) >= _FRAME_CONTRAST).astype(np.uint8)
    length = round(_FRAME_SIDE * min(drawn.shape)) | 1
    horizontal, vertical = (
        cv2.morphologyEx(
            drawn, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, shape)
        )
        for shape in ((length, 1), (1, length))
    )
    ends = [
        line for line in _pieces(horizontal) if line.bottom - line.top <= FRAME_WIDTH
    ]
    sides = [
        line for line in _pieces(vertical) if line.right - line.left <= FRAME_WIDTH
    ]
    # The ink that may cross the top and bottom, and the image and ink
    # turned over its diagonal, so the sides run across too.
    across = ink & (vertical == 0)
    turned, down = plot.transpose(1, 0, 2), (ink & (horizontal == 0)).T

    frames = []
    for top, bottom in itertools.permutations(ends, 2):
        # The columns that both lines run along, the sides at their ends.
        start, stop = max(top.left, bottom.left), min(top.right, bottom.right)
        if top.bottom >= bottom.top or stop - start < length:
            continue
        lined = _FRAME_LINED * (bottom.bottom - top.top)
        between = [
            side
            for side in sides
            if start - AXIS_MARGIN < side.right
            and side.left < stop + AXIS_MARGIN
            and min(side.bottom, bottom.bottom) - max(side.top, top.top) >= lined
        ]
        if len(between) <= 1:
            continue
        left = min(between, key=lambda side: side.left)
        right = max(between, key=lambda side: side.right)
        # Grid lines run on past both sides.
        if not all(
            left.left - AXIS_MARGIN <= line.left
            or line.right <= right.right + AXIS_MARGIN
            for line in (top, bottom)
        ):
            continue
        rows = slice(top.bottom, bottom.top)
        colours = [
            np.median(plot[line.top : line.bottom, line.left : line.right], axis=(0, 1))
            for line in (top, bottom, left, right)
        ]
        if not (
            np.ptp(colours, axis=0).max() >= _SAME_COLOUR
            or any(
                _crossed(plot, across, background, line, slice(left.right, right.left))
                for line in (top, bottom)
            )
            or any(
                _crossed(turned, down, background, line.turned(), rows)
                for line in (left, right)
            )
        ):
            frames.append(_Box(top.top, bottom.bottom, left.left, right.right))
    return frames


def _pieces(mask: npt.NDArray[np.integer | np.floating]) -> list[_Box]:
    """Return the boxes around the pieces of a mask: its connected pixels."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [
        _Box(int(top), int(top + rows), int(left), int(left + columns))
        for left, top, columns, rows, _ in stats[1:]
    ]


def _crossed(
    plot: Image,
    ink: npt.NDArray[np.bool_],
    background: npt.NDArray[np.float64],
    line: _Box,
    along: slice,
) -> bool:
    """Return whether a curve crosses a horizontal line, in the columns ``along``.

    One does where there is ink within ``AXIS_MARGIN`` above the line and
    within that below it, at most that far apart along the line, whose
    colours, those furthest from the background in each column, lie less
    than ``_SAME_COLOUR`` apart. So tick marks, in the colour of the line,
    do not cross it where a curve of another colour meets it from the other
    side.
    """
    places, colours = [], []
    for band in (
        slice(max(line.top - AXIS_MARGIN, 0), line.top),
        slice(line.bottom, line.bottom + AXIS_MARGIN),
    ):
        levels = plot[band, along].astype(np.float64)
        if not len(levels):  # the line lies along the edge of the plot
            return False
        inked = ink[band, along]
        strength = np.where(inked, np.linalg.norm(levels - background, axis=2), -1)
        (columns,) = np.nonzero(inked.any(axis=0))
        places.append(columns)
        colours.append(levels[strength.argmax(axis=0)[columns], columns])
    near = np.abs(places[0][:, np.newaxis] - places[1]) <= AXIS_MARGIN
    alike = (
        np.linalg.norm(colours[0][:, np.newaxis] - colours[1], axis=2) < _SAME_COLOUR
    )
    return bool(np.any(near & alike))


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
    background to their best match are labelled 0, as background, as are
    those more than ``_MATCH_DISTANCE`` from it, which show none of the
    colours, such as black text and arrows of annotations among coloured
    curves.
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
    distance = np.sqrt(np.maximum(squared, 0))
    cost = distance + _COVERAGE_COST * (1 - coverage) * lengths

    best = np.argmin(cost, axis=-1)[..., np.newaxis]
    shown = (np.take_along_axis(coverage, best, axis=-1) >= _MIN_COVERAGE) & (
        np.take_along_axis(distance, best, axis=-1) <= _MATCH_DISTANCE
    )
    return np.where(shown[..., 0], best[..., 0], 0)


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
    """Return a point at the middle of each branch of a curve in each column.

    In each column, the curve's pixels form runs, split by gaps of more than
    ``_RUN_GAP`` pixels. Runs in neighbouring columns that overlap, one to
    one, belong to one branch. A column gives a point at the median height
    of each of its runs whose branch spans at least ``_MIN_BRANCH`` of the
    width, so both sides of a loop give points, while short pieces such as
    the caps of error bars are ignored. A column with no such run, as where
    branches meet, gives one point at the median height of all its pixels.

    The curve's pixels are in a part of an image ``height`` pixels tall, from
    column ``left`` and row ``top``.
    """
    _, specks, stats, _ = cv2.connectedComponentsWithStats(in_colour.astype(np.uint8))
    areas = stats[:, cv2.CC_STAT_AREA]
    areas[0] = 0  # the pixels outside the curve
    in_curve = (areas >= _MIN_PIXELS)[specks]

    columns = np.flatnonzero(in_curve.any(axis=0))
    runs = [_runs(np.flatnonzero(in_curve[:, column])) for column in columns]
    branches = _branches(columns, runs)
    lengths = np.bincount(np.concatenate(branches))
    minimum = in_curve.shape[1] * _MIN_BRANCH

    points: list[tuple[float, float]] = []
    for column, column_runs, column_branches in zip(
        columns, runs, branches, strict=True
    ):
        major = [
            run
            for run, branch in zip(column_runs, column_branches, strict=True)
            if lengths[branch] >= minimum
        ]
        # The centre of the pixel in column c and row r is at
        # (c + 0.5, height - r - 0.5).
        points.extend(
            (column + left + 0.5, height - top - float(np.median(rows)) - 0.5)
            for rows in major or [np.concatenate(column_runs)]
        )
    data = np.array(points, dtype=np.float64)
    return data[np.lexsort((data[:, 1], data[:, 0]))]


def _runs(rows: npt.NDArray[np.intp]) -> list[npt.NDArray[np.intp]]:
    """Split the sorted rows of a column into runs with gaps of up to ``_RUN_GAP``."""
    return np.split(rows, np.flatnonzero(np.diff(rows) > _RUN_GAP + 1) + 1)


def _branches(
    columns: npt.NDArray[np.intp], runs: Sequence[Sequence[npt.NDArray[np.intp]]]
) -> list[npt.NDArray[np.intp]]:
    """Number the branch of each run, left to right.

    A run continues the branch of a run in the column before it if each
    overlaps the other, touching corners included, and nothing else there.
    """
    branches = [np.arange(len(runs[0]))]
    count = len(runs[0])
    for index in range(1, len(columns)):
        before, now = runs[index - 1], runs[index]
        touching = np.array(
            [[r[0] <= s[-1] + 1 and s[0] <= r[-1] + 1 for s in now] for r in before]
        ) & (columns[index] - columns[index - 1] == 1)
        numbers = np.arange(count, count + len(now))
        for j in range(len(now)):
            (partners,) = np.nonzero(touching[:, j])
            if len(partners) == 1 and np.count_nonzero(touching[partners[0]]) == 1:
                numbers[j] = branches[-1][partners[0]]
        branches.append(numbers)
        count += len(now)
    return branches


def _ink_levels(image: Image) -> npt.NDArray[np.float32]:
    """Return how far each pixel's grey level is from the median's, from 0 to 1."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return (np.abs(grey - np.median(grey)) / 255).astype(np.float32)


def _key_marker(
    levels: npt.NDArray[np.float32], key: Point
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Cut a legend key's marker out of an image, and mask out the key's line.

    The key is the piece of ink, differing from the background by at least
    ``_CONTRAST``, nearest its position, at most ``_KEY_REACH`` pixels away
    each way, so the middle of an open marker picks it. If the piece is
    ``_KEY_LINE`` times as wide as it is tall, or more, a line runs through the
    marker. The line's rows are those with ink across half the key or more,
    and the marker is the rest of the key, with the line's rows masked out,
    as lines cross the markers in the plot at other angles.

    Returns:
        The ink levels of the marker, with a pixel's margin, and the mask.

    Raises:
        ValueError: If there is no marker at the key's position.
    """
    height, width = levels.shape
    row, column = math.floor(height - key[1]), math.floor(key[0])
    if not (0 <= row < height and 0 <= column < width):
        msg = f"the marker position {key[0]:g},{key[1]:g} lies outside the image"
        raise ValueError(msg)
    top, left = max(row - _KEY_REACH, 0), max(column - _KEY_REACH, 0)
    window = levels[top : row + _KEY_REACH + 1, left : column + _KEY_REACH + 1]
    ink = window >= _CONTRAST / 255
    rows, columns = np.nonzero(ink)
    if not len(rows):
        msg = f"found no marker near {key[0]:g},{key[1]:g}"
        raise ValueError(msg)
    nearest = np.argmin((rows + top - row) ** 2 + (columns + left - column) ** 2)
    _, pieces = cv2.connectedComponents(ink.astype(np.uint8))
    piece = pieces == pieces[rows[nearest], columns[nearest]]
    rows, columns = np.nonzero(piece)

    line = np.zeros(len(window), dtype=np.bool_)  # the rows of the key's line
    if np.ptp(columns) + 1 >= _KEY_LINE * (np.ptp(rows) + 1):
        line = 2 * np.count_nonzero(piece, axis=1) >= np.ptp(columns) + 1
        marker = ~line[rows]
        if not np.any(marker):
            msg = f"found only a line, and no marker, at {key[0]:g},{key[1]:g}"
            raise ValueError(msg)
        rows = np.arange(rows[marker].min(), rows[marker].max() + 1)
        columns = columns[marker]

    box = (
        slice(max(rows.min() - 1, 0), rows.max() + 2),
        slice(max(columns.min() - 1, 0), columns.max() + 2),
    )
    fringe = cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    shape = np.where(fringe, window, 0)[box]
    mask = np.broadcast_to(~line[box[0], np.newaxis], shape.shape)
    return shape.astype(np.float32), mask.astype(np.float32)


def _matches(
    area: npt.NDArray[np.float32],
    shape: npt.NDArray[np.float32],
    mask: npt.NDArray[np.float32],
) -> list[tuple[float, float, float]]:
    """Return where a key's marker matches the image, and how well.

    A match is a local maximum of the correlation of the marker with the
    image, outside the mask, of at least ``_MARKER_SCORE``, where the image
    has at least ``_MARKER_INK`` of the marker's ink. Its centre is placed to
    a fraction of a pixel, at the top of a parabola through the correlation
    each way.

    Returns:
        The correlation, row and column of the centre of each match.
    """
    if area.shape[0] < shape.shape[0] or area.shape[1] < shape.shape[1]:
        return []
    scores = cv2.matchTemplate(area, shape, cv2.TM_CCOEFF_NORMED, mask=mask)
    scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)  # flat patches
    ink = ((shape >= _CONTRAST / 255) & (mask > 0)).astype(np.float32)
    shown = cv2.matchTemplate(area, ink, cv2.TM_CCORR) / float(np.sum(shape * ink))
    scores[shown < _MARKER_INK] = -1
    peaks = (scores >= _MARKER_SCORE) & (
        scores == cv2.dilate(scores, np.ones((3, 3), np.uint8))
    )
    padded = np.pad(scores, 1, constant_values=-1)
    middle_row, middle_column = (shape.shape[0] - 1) / 2, (shape.shape[1] - 1) / 2
    return [
        (
            float(scores[row, column]),
            row + middle_row + _vertex(padded[row : row + 3, column + 1]),
            column + middle_column + _vertex(padded[row + 1, column : column + 3]),
        )
        for row, column in np.argwhere(peaks)
    ]


def _vertex(values: npt.ArrayLike) -> float:
    """Return where the parabola through three values, a pixel apart, peaks.

    The values are those before, at and after a local maximum, and the peak
    is given from it, at most half a pixel away.
    """
    before, peak, after = np.asarray(values, dtype=np.float64)
    curvature = min(before - 2 * peak + after, -1e-9)  # flat at worst
    return float(np.clip((before - after) / (2 * curvature), -0.5, 0.5))


def _to_colour(levels: npt.NDArray[np.float64]) -> Colour:
    """Round red, green and blue levels to a colour."""
    red, green, blue = (int(level) for level in np.round(levels))
    return red, green, blue
