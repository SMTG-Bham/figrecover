"""Find the curves in an image of a plot."""

import itertools
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

from .calibration import Point

type Image = npt.NDArray[np.uint8]
type Colour = tuple[int, int, int]
type Style = Literal["solid", "dashed", "dotted", "dashdot", "markers"]
"""How a series is drawn: a line by the pattern of its dashes, or markers."""

STYLES: tuple[Style, ...] = ("solid", "dashed", "dotted", "dashdot", "markers")
"""Every style a series can be drawn in."""

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
_FIT = 6  # where a track is headed is fitted to this many of its last points
_TURN = 25  # the next dash of a curve turns from it by this many degrees or less
_GAP_WIDTHS = 12  # the dashes of a curve lie this many line widths apart or less
_FOLD = 90  # tracks heading this many degrees apart where they meet fold back
_SOLID = 0.9  # a curve inked along this fraction of its length or more is solid
_DOT = 2  # a dash at most this many line widths long is a dot
_DOTTED = 0.75  # a curve whose dashes are this fraction dots, or more, is dotted
_ONE_LINE = 1.5  # a run at most this many line widths tall holds one line
_DOT_GAP = 3  # the dots of a curve lie this many line widths apart or less
_MARKER_DEPTH = 1.5  # markers reach this many pixels deeper than half a line or more
_MARKER_SIZES = 2  # markers are within this factor of their median size
_MIN_MARKERS = 3  # a colour draws markers if it has this many or more
_THIN = 0.25  # this share of a curve's runs, or more, are no taller than its line
_FRAME_SIDE = 0.05  # frame lines span this fraction of the plot's shorter side
_FRAME_CONTRAST = 32  # its lines differ from the background by this much or more
_FRAME_LINED = 0.75  # its sides run along this fraction of its height or more
_KEY_REACH = 20  # a legend key's marker lies within this many pixels of its position
_KEY_LINE = 2  # a key this many times as wide as tall has a line through its marker
_MARKER_SCORE = 0.6  # a marker correlates with its key's this well or better
_MARKER_INK = 0.6  # a marker has at least this fraction of the ink of its key's
_MARKER_SPACING = 0.75  # markers are this many marker widths apart or more


@dataclass(frozen=True, slots=True, eq=False)
class Series:
    """A curve, or a series of markers, in a plot, and how it is drawn.

    Series that share a colour are told apart by how they are drawn, and
    curves of one colour and style by where they run.

    Attributes:
        colour: The colour it is drawn in, as ``(red, green, blue)``.
        style: How it is drawn: ``"solid"``, ``"dashed"``, ``"dotted"`` or
            ``"dashdot"`` for a line, by the pattern of its dashes, or
            ``"markers"``.
        width: The width of its line, or the size of its markers, in pixels
            in an image or in points in a PDF file.
        points: An ``(n, 2)`` array of the ``(x, y)`` positions of its
            points: those of pixel centres in an image, measured from the
            bottom-left corner, or in points in a PDF file.
    """

    colour: Colour
    style: Style
    width: float
    points: npt.NDArray[np.float64]

    @property
    def name(self) -> str:
        """Return its colour in hexadecimal and its style, such as 000000-dashed."""
        return f"{bytes(self.colour).hex()}-{self.style}"


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
    specks of fewer than four pixels are ignored. Each image column gives one
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
    masks, _, left, top = _masks(image, colours, origin)
    height = image.shape[0]
    return [_follow(in_colour, left, top, height) for in_colour in masks]


def trace_series(
    image: Image, colours: Sequence[Colour], origin: Point = (0, 0)
) -> list[list[Series]]:
    """Find the curves drawn in each of the given colours, told apart by style.

    The pixels of each colour are found as ``trace_curves`` finds them, and
    followed across the columns as tracks, by ``_tracks``: through crossings,
    each keeping on the way it was headed, and across gaps of up to
    ``_GAP_WIDTHS`` line widths, as between the dashes of a dashed curve.
    Tracks across at least ``_MIN_BRANCH`` of the width make the curves, by
    ``_curves``: the two sides of a loop are one curve, and so are the
    pieces of a curve hidden in places by another. Each curve is a series
    of its own, with its style, by ``_style``, and the width of its line, by
    ``_line_width``.

    A colour with one curve, or none, as markers alone draw, gives one
    series, of the points that ``trace_curves`` gives, in the style of its
    curve, or ``"markers"``.

    Args:
        image: Colour image of a plot.
        colours: The colours of the curves, such as ``find_colours`` returns.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        For each colour, its curves, most pixels first, each in the common
        colour that the colour picks, with a point for each column that it
        crosses, at the middle of its own pixels there, where curves cross
        too, or across a gap, on the straight line between its ends. The
        points are the pixel ``(x, y)`` coordinates of pixel centres,
        measured from the bottom-left corner and sorted by x, then y.

    Raises:
        ValueError: If no curve has one of the colours, two of them match the
            same curve, or one fills areas.
    """
    masks, found, left, top = _masks(image, colours, origin)
    height = image.shape[0]
    return [
        [series for series, _ in _styled(in_colour, colour, left, top, height)]
        for in_colour, colour in zip(masks, found, strict=True)
    ]


def find_series(image: Image, origin: Point = (0, 0)) -> list[Series]:
    """Find every curve in a plot, told apart by colour, style and where it runs.

    The colours are those that ``find_colours`` finds, and the curves of
    each those that ``trace_series`` finds.

    Args:
        image: Colour image of a plot.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        Every curve. Colours of one curve each come in the order that
        ``find_colours`` gives them, and the curves of colours of several,
        each colour's in the order ``trace_series`` gives them, come in among
        them by their pixels, counting those of a run of pixels that curves
        share a share each, the most first.
    """
    colours = _curve_colours(image, origin)
    masks, found, left, top = _masks(image, colours, origin)
    height = image.shape[0]
    styled = [
        _styled(in_colour, colour, left, top, height)
        for in_colour, colour in zip(masks, found, strict=True)
    ]
    queues = [[pairs[0] for pairs in styled if len(pairs) == 1]]
    queues += [pairs for pairs in styled if len(pairs) > 1]
    every: list[Series] = []
    while any(queues):
        first = max((queue for queue in queues if queue), key=lambda queue: queue[0][1])
        every.append(first.pop(0)[0])
    return every


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


def _masks(
    image: Image, colours: Sequence[Colour], origin: Point
) -> tuple[list[npt.NDArray[np.bool_]], list[Colour], int, int]:
    """Return the pixels of each colour in the plot area, as ``trace_curves`` has them.

    Returns:
        A mask of the pixels of each colour, in the plot area; the common
        colour that each colour picks; and the left and top edges of the
        plot area.

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
    masks = [labels == index for index in indices]
    return masks, [_to_colour(palette[index]) for index in indices], left, top


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
    ``_RUN_GAP`` pixels, and runs in neighbouring columns that overlap are
    followed as tracks, by ``_tracks``. A column gives a point at the median
    height of each of its runs that holds a track across at least
    ``_MIN_BRANCH`` of the width, so both sides of a loop give points, while
    short pieces such as the caps of error bars are ignored. A column with no
    such run gives one point at the median height of all its pixels.

    The curve's pixels are in a part of an image ``height`` pixels tall, from
    column ``left`` and row ``top``.
    """
    in_curve = _without_specks(in_colour)
    columns = np.flatnonzero(in_curve.any(axis=0))
    runs = [_runs(np.flatnonzero(in_curve[:, column])) for column in columns]
    minimum = in_curve.shape[1] * _MIN_BRANCH
    major = {
        (column, run)
        for track in _tracks(columns, runs, _width(runs), gaps=False)
        if track.span >= minimum
        for column, run in zip(track.columns, track.runs, strict=True)
    }

    points: list[tuple[float, float]] = []
    for column, column_runs in zip(columns.tolist(), runs, strict=True):
        kept = [
            run for index, run in enumerate(column_runs) if (column, index) in major
        ]
        # The centre of the pixel in column c and row r is at
        # (c + 0.5, height - r - 0.5).
        points.extend(
            (column + left + 0.5, height - top - float(np.median(rows)) - 0.5)
            for rows in kept or [np.concatenate(column_runs)]
        )
    data = np.array(points, dtype=np.float64).reshape(-1, 2)
    return data[np.lexsort((data[:, 1], data[:, 0]))]


def _without_specks(in_colour: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Return a colour's pixels less specks of fewer than ``_MIN_PIXELS`` pixels."""
    _, specks, stats, _ = cv2.connectedComponentsWithStats(in_colour.astype(np.uint8))
    areas = stats[:, cv2.CC_STAT_AREA]
    areas[0] = 0  # the pixels outside the curve
    return np.asarray((areas >= _MIN_PIXELS)[specks], dtype=np.bool_)


def _runs(rows: npt.NDArray[np.intp]) -> list[npt.NDArray[np.intp]]:
    """Split the sorted rows of a column into runs with gaps of up to ``_RUN_GAP``."""
    return np.split(rows, np.flatnonzero(np.diff(rows) > _RUN_GAP + 1) + 1)


def _width(runs: Sequence[Sequence[npt.NDArray[np.intp]]], share: float = 0.5) -> float:
    """Return the width of a curve's line, from the heights of its runs, or 1.

    That is the height that ``share`` of them reach no higher than: the
    median, or less where markers, which are taller, may be most of them.
    """
    heights = [run[-1] - run[0] + 1 for column in runs for run in column]
    return float(np.quantile(heights, share)) if heights else 1.0


@dataclass(slots=True, eq=False)
class _Track:
    """A curve followed across the columns of a plot, as ``_tracks`` follows it.

    For each column it crosses, ``columns`` holds the column, ``rows`` the
    height of its middle there, as a row from the top, ``runs`` the index of
    the run of pixels it lies in among those of the column, and ``shared``
    how many tracks lie in that run. ``clear`` holds the indices of the
    points where its line is clear of others, as ``_place`` finds them, as
    it is in a run alone. ``parent`` is the track it split
    from, if it started where a run parted in two, and ``pixels`` counts the
    pixels of its runs, those of a run that tracks share shared among them.
    """

    columns: list[int] = field(default_factory=list)
    rows: list[float] = field(default_factory=list)
    runs: list[int] = field(default_factory=list)
    shared: list[int] = field(default_factory=list)
    clear: list[int] = field(default_factory=list)
    parent: _Track | None = None
    pixels: float = 0

    @property
    def span(self) -> int:
        """Return how many columns it spans, from its first to its last."""
        return self.columns[-1] - self.columns[0] + 1

    def heading(self, column: int) -> tuple[float, float]:
        """Return the row where it is headed in a column, and its slope there.

        That is the line fitted to its last ``_FIT`` points where its line
        is clear of others, by least squares, so that tracks whose lines
        overlap, where curves cross, carry on as they came; or to its last
        points, if it has never been clear; or level through its only point.
        """
        recent = self.clear[-_FIT:] or range(len(self.columns))[-_FIT:]
        columns = [self.columns[index] for index in recent]
        rows = [self.rows[index] for index in recent]
        slope = _slope(columns, rows)
        return float(np.mean(rows) + slope * (column - np.mean(columns))), slope


def _tracks(
    columns: npt.NDArray[np.intp],
    runs: Sequence[Sequence[npt.NDArray[np.intp]]],
    width: float,
    *,
    gaps: bool,
) -> list[_Track]:
    """Follow the tracks of a curve across the runs of its pixels in each column.

    A run carries on the tracks of the runs in the column before that
    overlap it, touching corners included. Where a run overlaps several, as
    where curves cross, it carries on all their tracks; where several runs
    overlap one, as where curves part, its tracks carry on in those nearest
    where each is headed, one to a run while there are runs enough, first
    those in runs that overlap no other, so that crossing curves keep their
    tracks. A run left with no track starts one of its own, split from the
    tracks of the runs it overlaps.

    A track whose run overlaps none in the next column ends there, unless
    ``gaps`` is true: then it carries on in a run that no track has taken,
    nearest where it is headed, at most ``_GAP_WIDTHS`` times ``width``, the
    width of the curve's line, from its end, and either at most a pixel from
    where it is headed or in a direction from its end that turns by at most
    ``_TURN`` degrees from the track's, as the next dash of a dashed curve
    does. A track no longer than ``_DOT`` line widths, such as a dot of a
    dotted curve, heads no way yet: it carries on in the nearest such run at
    most ``_DOT_GAP`` line widths from its end, whichever way it lies. So
    does a track taken out of a run that it shares, by ``_released``, as a
    dashed curve is where it leaves another between dashes.
    """
    tracks: list[_Track] = []
    before: Sequence[npt.NDArray[np.intp]] = []
    held: list[list[_Track]] = []  # the tracks in each run of the column before
    waiting: list[_Track] = []  # tracks that lost their runs, until bridged
    last = -2
    for column, now in zip(columns.tolist(), runs, strict=True):
        hosts: list[list[_Track]] = [[] for _ in now]
        parents: list[list[_Track]] = [[] for _ in now]
        overlaps = [
            [
                index
                for index, run in enumerate(now)
                if column == last + 1
                and run[0] <= ahead[-1] + 1
                and ahead[0] <= run[-1] + 1
            ]
            for ahead in before
        ]
        lost = [
            track
            for near, carried in zip(overlaps, held, strict=True)
            if not near
            for track in carried
        ]
        for near, carried in sorted(
            zip(overlaps, held, strict=True), key=lambda pair: len(pair[0])
        ):
            for index in near:
                parents[index] += carried
            if near:
                _divide(carried, near, now, hosts, column)
        if gaps:
            lost += _released(hosts, now, column, width)
            waiting = [
                track
                for track in waiting + lost
                if column - track.columns[-1] <= _GAP_WIDTHS * width
            ]
            _bridge(waiting, now, hosts, column, width)
        for index, run in enumerate(now):
            if not hosts[index]:
                parent = parents[index][0] if parents[index] else None
                tracks.append(_Track(parent=parent))
                hosts[index].append(tracks[-1])
            _place(hosts[index], run, index, column, width)
        before, held, last = now, hosts, column
    return tracks


def _divide(
    tracks: Sequence[_Track],
    near: Sequence[int],
    runs: Sequence[npt.NDArray[np.intp]],
    hosts: list[list[_Track]],
    column: int,
) -> None:
    """Carry tracks on in the runs of a column near them, nearest where each is headed.

    ``near`` holds the indices of the runs, and ``hosts`` the tracks in each
    run so far, which this adds to. Each track takes a run of its own, that
    no track has yet, while there are runs enough; the rest join the run
    nearest where they are headed.
    """
    headed = [track.heading(column)[0] for track in tracks]
    costs = sorted(
        (_off(row, runs[index]), number, index)
        for number, row in enumerate(headed)
        for index in near
    )
    placed: set[int] = set()
    for _, number, index in costs:
        if number not in placed and not hosts[index]:
            hosts[index].append(tracks[number])
            placed.add(number)
    for _, number, index in costs:
        if number not in placed:
            hosts[index].append(tracks[number])
            placed.add(number)


def _released(
    hosts: list[list[_Track]],
    runs: Sequence[npt.NDArray[np.intp]],
    column: int,
    width: float,
) -> list[_Track]:
    """Take tracks out of the runs they share that hold only one line.

    ``hosts`` holds the tracks in each run of a column, and ``width`` is the
    width of the curve's line. The track headed nearest a run stays in it,
    and the others leave it if they are headed more than ``width`` beyond
    it, and the run is at most ``_ONE_LINE`` times as tall as a line along
    that track would be, so that curves that cross keep their runs.

    Returns:
        The tracks taken out.
    """
    released: list[_Track] = []
    for index, run in enumerate(runs):
        if len(hosts[index]) > 1:
            headings = [track.heading(column) for track in hosts[index]]
            offs = [_off(row, run) for row, _ in headings]
            nearest = int(np.argmin(offs))
            tall = _ONE_LINE * width * math.hypot(1, headings[nearest][1])
            if run[-1] - run[0] + 1 <= tall:
                kept = [
                    track
                    for number, (track, off) in enumerate(
                        zip(hosts[index], offs, strict=True)
                    )
                    if number == nearest or off <= width
                ]
                released += [track for track in hosts[index] if track not in kept]
                hosts[index][:] = kept
    return released


def _bridge(
    waiting: list[_Track],
    runs: Sequence[npt.NDArray[np.intp]],
    hosts: list[list[_Track]],
    column: int,
    width: float,
) -> None:
    """Carry tracks that lost their runs on across a gap, as ``_tracks`` says.

    ``hosts`` holds the tracks in each run of the column so far, which this
    adds to, and the tracks carried on are taken out of ``waiting``;
    ``width`` is the width of the curve's line. The nearest pairs of tracks
    and runs are joined first: those nearest where the tracks are headed, or
    a dot's nearest its end.
    """
    pairs = []
    for number, track in enumerate(waiting):
        row, slope = track.heading(column)
        across = column - track.columns[-1]
        for index, run in enumerate(runs):
            if hosts[index]:
                continue
            if track.span <= _DOT * width:  # a dot, which heads no way yet
                target = float(np.clip(track.rows[-1], run[0], run[-1]))
                distance = math.hypot(across, target - track.rows[-1])
                if distance <= _DOT_GAP * width:
                    pairs.append((distance, distance, number, index))
                continue
            target = float(np.clip(row, run[0], run[-1]))
            down = target - track.rows[-1]
            turn = abs(math.degrees(math.atan2(down, across) - math.atan(slope)))
            if math.hypot(across, down) <= _GAP_WIDTHS * width and (
                abs(target - row) <= 1 or turn <= _TURN
            ):
                pairs.append(
                    (abs(target - row), math.hypot(across, down), number, index)
                )
    carried: set[int] = set()
    for _, _, number, index in sorted(pairs):
        if number not in carried and not hosts[index]:
            hosts[index].append(waiting[number])
            carried.add(number)
    waiting[:] = [
        track for number, track in enumerate(waiting) if number not in carried
    ]


def _place(
    tracks: Sequence[_Track],
    run: npt.NDArray[np.intp],
    index: int,
    column: int,
    width: float,
) -> None:
    """Add a point in a run of a column to each of the tracks in it.

    A track alone in a run lies at its median row, unless the run is more
    than ``_ONE_LINE`` times as tall as its line, a line ``width`` wide being
    taller by the secant of its slope, as where a dash of another curve
    lies across it: then it lies half its line's height inside whichever end
    of the run is nearer where it is headed. Where tracks share a run,
    their lines are clear of one another, and keep the order in which they
    came, if its rows part into as many lumps as there are tracks, with gaps
    between them: each takes a lump, in turn, and lies at its median row. So
    are the lines of two tracks in a run more than ``_ONE_LINE`` times as
    tall as either line, as where curves touch: the upper lies half its
    line's height below the top of the run, and the lower as far above its
    bottom.
    Otherwise their lines overlap, as where curves cross, and the tracks
    share out the run's rows, each taking those nearest where it is headed,
    and lie at the median of theirs, or, with none, where they are headed,
    as far into the run as that lies.
    """
    clear = True
    if len(tracks) == 1:
        rows = [float(np.median(run))]
        if tracks[0].columns:
            row, slope = tracks[0].heading(column)
            line = width * math.hypot(1, slope)
            if run[-1] - run[0] + 1 > _ONE_LINE * line:  # with another line
                edges = [run[0] + (line - 1) / 2, run[-1] - (line - 1) / 2]
                rows = [min(edges, key=lambda edge: abs(edge - row))]
    else:
        headings = [track.heading(column) for track in tracks]
        tall = [width * math.hypot(1, slope) for _, slope in headings]
        order = np.argsort([track.rows[-1] for track in tracks], kind="stable")
        upper, *middle, lower = order
        lumps = np.split(run, np.flatnonzero(np.diff(run) > 1) + 1)
        rows = [0.0] * len(tracks)
        if len(lumps) == len(tracks):
            for lump, number in zip(lumps, order, strict=True):
                rows[number] = float(np.median(lump))
        elif not middle and run[-1] - run[0] + 1 > _ONE_LINE * max(tall):
            rows[upper] = run[0] + (tall[upper] - 1) / 2
            rows[lower] = run[-1] - (tall[lower] - 1) / 2
        else:
            clear = False
            headed = np.array([row for row, _ in headings])
            nearest = np.abs(run[:, np.newaxis] - headed).argmin(axis=1)
            rows = [
                float(np.median(run[nearest == number]))
                if np.any(nearest == number)
                else float(np.clip(row, run[0], run[-1]))
                for number, row in enumerate(headed)
            ]
    for track, row in zip(tracks, rows, strict=True):
        if clear:
            track.clear.append(len(track.columns))
        track.columns.append(column)
        track.rows.append(row)
        track.runs.append(index)
        track.shared.append(len(tracks))
        track.pixels += len(run) / len(tracks)


def _styled(
    in_colour: npt.NDArray[np.bool_], colour: Colour, left: int, top: int, height: int
) -> list[tuple[Series, float]]:
    """Return the curves of a colour as series, most pixels first, with their pixels.

    See ``trace_series`` for what the curves are. Markers, by ``_blobs``,
    taking the line to be as wide as ``_THIN`` of the runs of the colour's
    pixels are tall, are left out of the pixels that make the curves, and,
    if there are curves too, come first, as a series of their own. The
    curve's pixels are in a part of an image ``height`` pixels tall, from
    column ``left`` and row ``top``.
    """
    in_curve = _without_specks(in_colour)
    columns = np.flatnonzero(in_curve.any(axis=0))
    runs = [_runs(np.flatnonzero(in_curve[:, column])) for column in columns]
    marked, centres, size = _blobs(in_curve, _width(runs, _THIN))
    lines = in_curve & ~marked
    columns = np.flatnonzero(lines.any(axis=0))
    runs = [_runs(np.flatnonzero(lines[:, column])) for column in columns]
    at = dict(zip(columns.tolist(), runs, strict=True))
    minimum = in_curve.shape[1] * _MIN_BRANCH
    tracks = _tracks(columns, runs, _width(runs), gaps=True)
    curves = _curves([track for track in tracks if track.span >= minimum])
    described = [_described(curve, at, marked) for curve in curves]
    found = []
    if len(centres) and curves:
        # The centre of the pixel in column c and row r is at
        # (c + 0.5, height - r - 0.5).
        points = centres * (1, -1) + (left + 0.5, height - top - 0.5)
        series = Series(colour, "markers", size, points[np.argsort(points[:, 0])])
        found.append((series, float(np.count_nonzero(marked))))
    if len(found) + len(curves) > 1:
        return found + [
            (
                Series(colour, style, width, _points(curve, at, left, top, height)),
                sum(track.pixels for track in curve),
            )
            for curve, (style, width) in zip(curves, described, strict=True)
        ]
    alone: tuple[Style, float] = described[0] if described else ("markers", 0.0)
    series = Series(colour, *alone, _follow(in_colour, left, top, height))
    return [(series, float(np.count_nonzero(in_curve)))]


def _blobs(
    in_curve: npt.NDArray[np.bool_], width: float
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64], float]:
    """Find the markers among a colour's pixels, as blobs thicker than its line.

    A marker's core is its pixels deeper inside than half the width of the
    curve's line and ``_MARKER_DEPTH`` more, by the distance transform, and
    of at least ``_MIN_PIXELS`` pixels. Markers are those cores within a
    factor of ``_MARKER_SIZES`` of their median size, if there are at least
    ``_MIN_MARKERS`` of them, as filled markers draw; lines, dashes, open
    markers and crosses have none.

    Returns:
        The pixels of the markers, those as far from their cores as the
        cores lie deep, and a pixel more; the column and row of each core's
        middle; and their size across, twice their greatest depth. With no
        markers, none of the pixels, no middles and a size of 0.
    """
    depth = cv2.distanceTransform(in_curve.astype(np.uint8), cv2.DIST_L2, 5)
    deepest = width / 2 + _MARKER_DEPTH
    deep = (depth > deepest).astype(np.uint8)
    _, cores, stats, middles = cv2.connectedComponentsWithStats(deep)
    areas = stats[1:, cv2.CC_STAT_AREA]
    typical = float(np.median(areas)) if len(areas) else 0.0
    kept = (
        np.flatnonzero(
            (areas >= _MIN_PIXELS)
            & (areas * _MARKER_SIZES >= typical)
            & (areas <= typical * _MARKER_SIZES)
        )
        + 1
    )
    if len(kept) < _MIN_MARKERS:
        return np.zeros_like(in_curve), np.empty((0, 2)), 0.0
    inside = np.isin(cores, kept)
    reach = math.ceil(deepest) + 1
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)
    marked = cv2.dilate(inside.astype(np.uint8), disc) > 0
    size = 2 * float(depth[inside].max())
    return marked & in_curve, np.asarray(middles[kept], dtype=np.float64), size


def _curves(tracks: Sequence[_Track]) -> list[list[_Track]]:
    """Return the curves that tracks of one colour make, most pixels first.

    A track that split from another, or ended in the same run as another,
    makes one curve with it if they head at least ``_FOLD`` degrees apart
    there, as the two sides of a loop do where it turns back, by
    ``_folded``; tracks that part at a smaller angle, as curves drawn from
    one point do, are curves of their own. Then, nearest first, curves of
    which one starts where the other has ended, or at most ``_RUN_GAP``
    columns before, are one curve, as the pieces of a curve hidden in places
    by another are.
    """
    numbers = {id(track): number for number, track in enumerate(tracks)}
    owners = list(range(len(tracks)))  # the track that each joined, in turn

    def owner(number: int) -> int:
        while owners[number] != number:
            number = owners[number]
        return number

    ends: dict[tuple[int, int], list[_Track]] = {}
    for number, track in enumerate(tracks):
        parent = track.parent
        if parent is not None and id(parent) in numbers:
            start = track.columns[0]
            if _folded(track, 0, parent, start):
                owners[owner(number)] = owner(numbers[id(parent)])
        ends.setdefault((track.columns[-1], track.runs[-1]), []).append(track)
    for together in ends.values():
        for first, second in itertools.combinations(together, 2):
            if _folded(first, -1, second, first.columns[-1]):
                owners[owner(numbers[id(first)])] = owner(numbers[id(second)])

    joined: dict[int, list[_Track]] = {}
    for number, track in enumerate(tracks):
        joined.setdefault(owner(number), []).append(track)
    curves = list(joined.values())
    while True:
        pairs = [
            (_apart(first, second), number, other)
            for number, first in enumerate(curves)
            for other, second in enumerate(curves)
            if _first(second)[0] > _last(first)[0] - _RUN_GAP and number != other
        ]
        if not pairs:
            break
        _, number, other = min(pairs)
        curves[number] += curves.pop(other)
    return sorted(curves, key=lambda curve: -sum(track.pixels for track in curve))


def _folded(track: _Track, end: int, other: _Track, column: int) -> bool:
    """Return whether two tracks that meet head ``_FOLD`` degrees apart or more.

    ``end`` is 0 where the track starts, to compare the way it heads from
    its first ``_FIT`` points with the way the other heads from ``column``
    on, and -1 where it ends, to compare the ways both head over their last
    ``_FIT`` points.
    """
    if end == 0:
        mine = slice(None, _FIT)
        start = int(np.searchsorted(other.columns, column))
        theirs = slice(start, start + _FIT)
    else:
        mine = theirs = slice(-_FIT, None)
    slopes = [
        _slope(one.columns[part], one.rows[part])
        for one, part in ((track, mine), (other, theirs))
    ]
    turn = math.degrees(abs(math.atan(slopes[0]) - math.atan(slopes[1])))
    return turn >= _FOLD


def _first(curve: Sequence[_Track]) -> tuple[int, float]:
    """Return the column and row of a curve's first point."""
    track = min(curve, key=lambda one: one.columns[0])
    return track.columns[0], track.rows[0]


def _last(curve: Sequence[_Track]) -> tuple[int, float]:
    """Return the column and row of a curve's last point."""
    track = max(curve, key=lambda one: one.columns[-1])
    return track.columns[-1], track.rows[-1]


def _apart(first: Sequence[_Track], second: Sequence[_Track]) -> float:
    """Return how far one curve starts from where another ends, in pixels."""
    (end, row), (start, other) = _last(first), _first(second)
    return math.hypot(start - end, other - row)


def _slope(columns: Sequence[int], rows: Sequence[float]) -> float:
    """Return the slope of the line fitted to points by least squares, or 0."""
    if len(set(columns)) <= 1:
        return 0.0
    slope, _ = np.polyfit(columns, rows, 1)
    return float(slope)


def _points(
    curve: Sequence[_Track],
    at: dict[int, list[npt.NDArray[np.intp]]],
    left: int,
    top: int,
    height: int,
) -> npt.NDArray[np.float64]:
    """Return the points of a curve, as ``trace_series`` gives them.

    ``at`` holds the runs of each column. A run that two of the curve's
    tracks share gives one point, at its median row, and the columns that a
    track crosses a gap in, between dashes, give points on the straight line
    across the gap.
    """
    found: dict[tuple[int, int], list[float]] = {}
    for track in curve:
        for column, row, run in zip(track.columns, track.rows, track.runs, strict=True):
            found.setdefault((column, run), []).append(row)
    rows = [
        (column, mine[0] if len(mine) == 1 else float(np.median(at[column][run])))
        for (column, run), mine in found.items()
    ]
    for track in curve:
        gaps = np.flatnonzero(np.diff(track.columns) > 1)
        for start, end in zip(gaps, gaps + 1, strict=True):
            across = np.arange(track.columns[start] + 1, track.columns[end])
            heights = np.interp(
                across,
                [track.columns[start], track.columns[end]],
                [track.rows[start], track.rows[end]],
            )
            rows += zip(across.tolist(), heights.tolist(), strict=True)
    # The centre of the pixel in column c and row r is at (c + 0.5, height - r - 0.5).
    data = np.array(rows, dtype=np.float64) * (1, -1) + (left + 0.5, height - top - 0.5)
    return data[np.lexsort((data[:, 1], data[:, 0]))]


def _described(
    curve: Sequence[_Track],
    at: dict[int, list[npt.NDArray[np.intp]]],
    marked: npt.NDArray[np.bool_],
) -> tuple[Style, float]:
    """Return the style of a curve, by ``_style``, and its width, by ``_line_width``."""
    width = _line_width(curve, at)
    return _style(curve, at, width, marked), width


def _line_width(
    curve: Sequence[_Track], at: dict[int, list[npt.NDArray[np.intp]]]
) -> float:
    """Return the width of a curve's line, in pixels.

    That is the median height of the runs it has alone, across the way it
    heads: the height times the cosine of its slope there.
    """
    widths = []
    for track in curve:
        slopes = np.gradient(track.rows, track.columns) if track.span > 1 else [0.0]
        widths += [
            (at[column][run][-1] - at[column][run][0] + 1) / math.hypot(1, slope)
            for column, run, shared, slope in zip(
                track.columns, track.runs, track.shared, slopes, strict=False
            )
            if shared == 1
        ]
    return float(np.median(widths)) if widths else 1.0


def _style(
    curve: Sequence[_Track],
    at: dict[int, list[npt.NDArray[np.intp]]],
    width: float,
    marked: npt.NDArray[np.bool_],
) -> Style:
    """Return the style of a curve's line, from the dashes along its tracks.

    A dash is a stretch of neighbouring columns, and its length the diagonal
    of the box around its runs. A curve inked along at least ``_SOLID`` of
    its length, counting the gaps between its dashes, but not those whose
    straight line across lies mostly within a pixel of ``marked`` pixels,
    where markers were taken out, is solid. Otherwise a
    dash is a dot if its length, less the width of the line, which the box
    takes in across the dash, is at most ``_DOT`` times the width, and the
    curve is dotted if at least ``_DOTTED`` of its dashes are dots, dashed if
    at least that fraction are not, and dashdot otherwise.
    """
    lengths: list[float] = []
    gaps: list[float] = []
    for track in curve:
        breaks = np.flatnonzero(np.diff(track.columns) > 1) + 1
        pieces = np.split(np.arange(len(track.columns)), breaks)
        for piece in pieces:
            spans = [at[track.columns[i]][track.runs[i]] for i in piece]
            extent = max(run[-1] for run in spans) - min(run[0] for run in spans) + 1
            lengths.append(math.hypot(len(piece), extent))
        for after in breaks:
            ends = [track.columns[after - 1], track.columns[after]]
            across = np.arange(ends[0] + 1, ends[1])
            rows = np.interp(across, ends, track.rows[after - 1 : after + 1])
            near = np.round(rows).astype(np.intp) + np.array([[-1], [0], [1]])
            hidden = marked[np.clip(near, 0, len(marked) - 1), across].any(axis=0)
            if 2 * np.count_nonzero(hidden) <= len(across):
                down = track.rows[after] - track.rows[after - 1]
                gaps.append(math.hypot(ends[1] - ends[0], down) - 1)
    if sum(lengths) >= _SOLID * (sum(lengths) + sum(gaps)):
        return "solid"
    dots = float(np.mean([length - width <= _DOT * width for length in lengths]))
    if dots >= _DOTTED:
        return "dotted"
    return "dashed" if dots <= 1 - _DOTTED else "dashdot"


def _off(row: float, run: npt.NDArray[np.intp]) -> float:
    """Return how far a row lies outside a run of rows, or 0 if within it."""
    return float(max(run[0] - row, row - run[-1], 0))


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
