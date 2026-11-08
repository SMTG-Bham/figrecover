"""Find the curves in an image of a plot."""

import dataclasses
import itertools
import math
import os
import warnings
from collections.abc import Iterable, Sequence
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
_FAINT = 16  # a pixel this far from the background shows some ink
_SAME_INK = 6  # inks whose directions from the background are this many degrees apart
_INK_BIN_WIDTH = 8  # directions are grouped in bins this many levels wide, about 4°
_MIXED = 12  # groups of a colour's pixels this many degrees apart are several inks
_MIXED_SHARE = 0.25  # such groups each hold this fraction of the colour's pixels
_PART = 0.1  # an overlap of translucent colours shows this much of each, or more
_SURROUNDED = 0.5  # and this fraction of the pixels around it show one or the other
_MIN_CURVE_FRACTION = 0.75  # a curve has at least this many pixels per column
_MIN_INK_FRACTION = 0.5  # or a thin one this much ink per column, at full strength
_CONTRAST = 64  # a curve differs from the background by this much in some channel
_AREA_DEPTH = 4  # a colour whose pixels lie this deep inside its shapes fills areas
_LINE_SPAN = 0.6  # straight lines across this fraction of the plot are grid lines
_LINE_GAP = 13  # and their gaps, such as dashes or curves over them, are this short
_BACKGROUND_SHARE = 1 / 3  # a background has this many pixels of a fill's, or more
_COVERAGE_COST = 0.25  # how much a pixel's distance from full coverage counts
_MIN_COVERAGE = 0.5  # less covered pixels belong to the background
_MIN_PIXELS = 4  # smaller specks are noise, such as where two other curves cross
_RUN_GAP = 6  # a curve's pixels in a column this far apart or less are one run
_MIN_BRANCH = 0.05  # branches across less of the plot's width than this are minor
_SNAP_REACH = 12  # an axis line this many pixels inside the axes given is theirs
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
_GREY_SPREAD = 8  # a pixel's levels this far apart or less are grey
_GREY_SHARE = 0.005  # a greyscale image has this fraction of other pixels at most
_KEY_REACH = 20  # a legend key's marker lies within this many pixels of its position
_KEY_WIDE = 1.5  # a key this many times as wide as tall may have a line through it
_KEY_LINE = 3  # the line runs on this many pixels or more beyond the key's marker
_KEY_LINE_LEVEL = 0.25  # its pixels this far from the line's level are the marker's
_KEY_HUES = 10  # key colours this many degrees apart or less are one colour
_INK_ANGLE = 20  # ink lies this many degrees or less from its colour
_MARKER_SCORE = 0.6  # a marker correlates with its key's this well or better
_MARKER_INK = 0.6  # a marker has this fraction of the ink of each half of its key's
_MARKER_SPACING = 0.75  # markers are this many marker widths apart or more
_MARKER_MARGIN = 0.05  # keys scoring this much less than the best near it are rivals
_REACH = 2  # rivals' markers, and fits to a place, lie this many pixels away or less
_AMBIGUOUS = 0.3  # a fit by less than this fraction better than a rival's is ambiguous
_ALIKE = 0.05  # keys' markers differing less than this look alike
_HIDDEN_SHOWN = 0.2  # a partly hidden marker shows this fraction of its ink or more
_HIDDEN_SCORE = 0.75  # and correlates with its key's marker this well or better
_HIDDEN_GAIN = 0.6  # and explains the image under it this fraction better
_MOST_HIDDEN = 2  # a marker hides parts of this many others at most
_ALIGNED = 2  # markers this many pixels apart or less across share an x position
_GAP = 1.5  # a series' markers this many times their usual spacing apart leave a gap
_FLAT = 1e-6  # parts of an image that vary less than this are flat


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


def _greyscale(image: Image) -> bool:
    """Return whether an image has no colour, as greyscale and photocopied ones have.

    Its red, green and blue levels lie within ``_GREY_SPREAD`` of each other
    in all but ``_GREY_SHARE`` of its pixels.
    """
    spread = np.ptp(image.astype(np.int16), axis=2)
    return bool(np.mean(spread > _GREY_SPREAD) <= _GREY_SHARE)


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
    the plot, as grid lines are, nor show where translucent colours overlap.

    Args:
        image: Colour image of a plot.
        count: How many curves to find.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        The colours of the ``count`` curves with the most pixels, most common
        first.

    Raises:
        ValueError: If there are fewer than ``count`` curves. The message
            names the curves found, and the colours left out and why, with
            how many pixels show each.
    """
    candidates = _candidates(image, origin)
    curves = [colour for colour, _, reason in candidates if reason is None]
    if len(curves) < count:
        found = ", ".join(
            f"#{bytes(colour).hex()} ({pixels} pixels)"
            for colour, pixels, reason in candidates
            if reason is None
        )
        msg = f"found only {len(curves)} of {count} curves, {found}"
        msg = msg if curves else "no curve found"
        raise ValueError(msg + _reasons(candidates))
    return curves[:count]


def _left_out(image: Image, origin: Point) -> str:
    """Say which colours that could be curves' were left out, and why.

    See ``_reasons`` for what it says.
    """
    return _reasons(_candidates(image, origin))


def _reasons(candidates: Iterable[tuple[Colour, int, str | None]]) -> str:
    """Describe the colours left out, as ``_candidates`` gives them.

    That is "; left out", then each colour, how many pixels show it, and
    why it is not a curve's; or nothing if none were left out.
    """
    left_out = "; ".join(
        f"#{bytes(colour).hex()} ({pixels} pixels), as it {reason}"
        for colour, pixels, reason in candidates
        if reason is not None
    )
    return f"; left out {left_out}" if left_out else ""


def _curve_colours(image: Image, origin: Point) -> list[Colour]:
    """Return the colours of all the curves in a plot, most pixels first.

    See ``find_colours`` for which colours are curves.
    """
    candidates = _candidates(image, origin)
    return [colour for colour, _, reason in candidates if reason is None]


def _candidates(image: Image, origin: Point) -> list[tuple[Colour, int, str | None]]:
    """Return the colours in a plot that could be curves', most pixels first.

    Each comes with how many pixels show it, and why it is not a curve's,
    or None if it is. See ``find_colours`` for which colours are curves.
    """
    plot = _plot_area(image, origin)[0]
    palette = _palette(plot)
    labels = _labelled(plot, palette)
    counts = np.bincount(labels.ravel(), minlength=len(palette))
    candidates = []
    for index, colour in enumerate(palette[1:], start=1):
        own = labels == index
        contrast = int(np.abs(colour - palette[0]).max())
        reason = None
        if contrast < _CONTRAST:
            reason = f"is too pale, {contrast} levels from the background"
        elif _fills_areas(own):
            reason = "fills areas"
        elif _draws_grid(own):
            reason = "is drawn in lines across the plot, as grid lines are"
        elif _composite(palette, labels, index):
            reason = "is where two translucent colours overlap"
        candidates.append((_to_colour(colour), int(counts[index]), reason))
    return candidates


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

    Warns:
        UserWarning: If a curve crosses most columns more than once, as
            several curves of one colour do, or a loop.

    Raises:
        ValueError: If no curve has one of the colours, two of them match the
            same curve, or one fills areas.
    """
    masks, _, left, top = _masks(image, colours, origin)
    height = image.shape[0]
    curves = [_follow(in_colour, left, top, height) for in_colour in masks]
    for colour, points in zip(colours, curves, strict=True):
        _, crossings = np.unique(points[:, 0], return_counts=True)
        times = int(np.median(crossings))
        if times > 1:
            message = (
                f"the curve of colour #{bytes(colour).hex()} crosses most columns "
                f"{times} times: it may be {times} curves of that colour, or loop"
            )
            warnings.warn(message, stacklevel=2)
    return curves


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


@dataclass(frozen=True, slots=True)
class MarkerSeries:
    """The markers of one series in a plot, as ``find_marker_series`` finds them.

    Attributes:
        points: An ``(n, 2)`` array of the pixel ``(x, y)`` coordinates of the
            centres of the markers found, measured from the bottom-left
            corner and sorted by x, then y.
        ambiguous: An ``(n,)`` array saying whether each marker found might as
            well be another series', whose marker fits it almost as well.
        hidden: An ``(m, 2)`` array of the pixel coordinates where markers of
            the series are probably hidden under others', sorted by x.
    """

    points: npt.NDArray[np.float64]
    ambiguous: npt.NDArray[np.bool_]
    hidden: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _Key:
    """A legend key's marker, to look for in a plot area.

    Attributes:
        shape: The ink levels of the marker, with a pixel's margin, as
            ``_key_marker`` cuts it out.
        mask: Where the shape counts: everywhere but on the key's line.
        colour: The index of the marker's colour in the palette.
        size: The most pixels the marker spans either way.
        row: The row of the key's position, measured in the plot area.
        column: The column of the key's position.
    """

    shape: npt.NDArray[np.float32]
    mask: npt.NDArray[np.float32]
    colour: int
    size: int
    row: float
    column: float


@dataclass(frozen=True, slots=True)
class _Marker:
    """A marker found in a plot area.

    Attributes:
        key: The index of its key.
        row: The row of its centre, to a fraction of a pixel.
        column: The column of its centre.
        top: The top row of ``shape`` in the plot area.
        left: The left column of ``shape``.
        shape: Its key's marker, shifted by a fraction of a pixel to fit it.
        mask: Its key's mask.
        ambiguous: Whether another key's marker fits it almost as well.
    """

    key: int
    row: float
    column: float
    top: int
    left: int
    shape: npt.NDArray[np.float32]
    mask: npt.NDArray[np.float32]
    ambiguous: bool = False


def find_markers(
    image: Image, keys: Sequence[Point], origin: Point = (0, 0)
) -> list[npt.NDArray[np.float64]]:
    """Find the markers of each series in a plot, by their shapes and colours.

    See ``find_marker_series``, which also tells which markers might be
    another series', and where markers are probably hidden.

    Args:
        image: Colour image of a plot.
        keys: The pixel position of the marker in each series' legend key,
            measured from the bottom-left corner. Each key's marker must have
            a shape or a colour of its own.
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
    return [series.points for series in find_marker_series(image, keys, origin)]


def find_marker_series(
    image: Image, keys: Sequence[Point], origin: Point = (0, 0)
) -> list[MarkerSeries]:
    """Find the markers of each series in a plot, by their shapes and colours.

    Each key is the position of a series' marker in the legend. Its colour is
    the direction, seen from the background, of the piece of ink nearest it,
    and keys whose colours lie at most ``_KEY_HUES`` degrees apart share
    one. Each pixel of the image shows the key colour nearest its own, if at
    most ``_INK_ANGLE`` degrees away, as ``_ink_levels`` measures it, so
    series that share a shape and differ in colour are told apart. Each
    key's marker is cut out of the image in its colour, leaving out a line
    through it, as legends of line plots draw, and looked for in the plot
    area, the part of the image above and right of ``origin``, less the
    frames inside it, such as those of insets and legends, with all they
    hold, as ``_frames`` finds them. Keys whose markers look alike at this
    size, as ``_warn_alike`` measures them, give a warning.

    A marker lies where a key's marker correlates with the plot by
    ``_MARKER_SCORE`` or more, and each half of the key's ink shows, by
    ``_MARKER_INK`` or more, as ``_scores`` finds it, unless a better match
    lies closer than ``_MARKER_SPACING`` of a marker's width. Where other
    keys' markers match almost as well, as ``_assign`` finds them, each is
    fitted to the plot, with the markers it partly hides or that partly
    hide it, and the one that explains the plot best wins, as ``_resolve``
    chooses it; the marker is ambiguous if a rival explains it almost as
    well. Markers partly hidden under those found, or over them, are then
    looked for in what those leave uncovered, as ``_uncover`` finds them.
    Series usually share x positions, so where a series has no marker at
    one that others share, under another series' marker, its marker is
    looked for again there, as ``_rethink`` does, and is otherwise probably
    hidden, as ``_missing`` finds it. The keys themselves are not markers,
    if the legend lies in the plot area.

    Args:
        image: Colour image of a plot.
        keys: The pixel position of the marker in each series' legend key,
            measured from the bottom-left corner. Each key's marker must have
            a shape or a colour of its own.
        origin: Pixel position where the axes cross, measured from the
            bottom-left corner.

    Returns:
        The markers of each key's series, with positions measured as for
        ``find_markers``.

    Raises:
        ValueError: If there is no marker at a key, or there is no plot area.
    """
    offsets = _offsets(image)
    palette, colours = _key_palette([_key_colour(offsets, key) for key in keys])
    levels = _ink_levels(offsets, palette)
    shapes = [
        _key_marker(levels[colour], key)
        for key, colour in zip(keys, colours, strict=True)
    ]
    plot, left, top = _plot_area(image, origin)
    height = image.shape[0]
    found = [
        _Key(
            shape,
            mask,
            colour,
            max(shape.shape) - 2,
            height - top - y - 0.5,
            x - left - 0.5,
        )
        for (shape, mask), colour, (x, y) in zip(shapes, colours, keys, strict=True)
    ]
    _warn_alike(keys, found)
    area = _ink_levels(_offsets(plot), palette)
    markers = [
        marker
        for colour in range(len(palette))
        for marker in _assign(area[colour], found, colour)
    ]
    markers += _uncover(area, found, markers)
    markers = _rethink(area, found, markers)
    missing = _missing(found, markers)

    # The centre of the pixel in column c and row r is at
    # (c + 0.5, height - r - 0.5).
    def placed(row: float, column: float) -> Point:
        return column + left + 0.5, height - top - row - 0.5

    series = []
    for key in range(len(keys)):
        own = sorted(
            (*placed(marker.row, marker.column), marker.ambiguous)
            for marker in markers
            if marker.key == key
        )
        series.append(
            MarkerSeries(
                np.array([(x, y) for x, y, _ in own], dtype=np.float64).reshape(-1, 2),
                np.array([flag for _, _, flag in own], dtype=np.bool_),
                np.array(
                    sorted(placed(row, column) for row, column in missing[key]),
                    dtype=np.float64,
                ).reshape(-1, 2),
            )
        )
    return series


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
    labels = _labelled(plot, palette)
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

    An axis line that lies a few pixels inside the axes given, as one does
    when a reference point is taken a little below or left of it, is left
    out too, as ``_axis_line`` finds it, so the part is the same as if the
    point had been taken on the line.

    Raises:
        ValueError: If the axes leave no plot area.
    """
    msg = "the axes leave no plot area to trace"
    height = image.shape[0]
    left = max(0, int(origin[0]) + AXIS_MARGIN)
    bottom = max(0, height - int(origin[1]) - AXIS_MARGIN)
    if image[:bottom, left:].size == 0:
        raise ValueError(msg)
    ink = ink_mask(image) > 0
    bottom = _axis_line(ink[:, left:], bottom)
    # Turned so that the y-axis line runs along the bottom, from the x-axis.
    width = ink.shape[1]
    left = width - _axis_line(ink[:bottom].T[::-1, ::-1], width - left)
    ink = ink[:bottom, left:]
    top, right = _inside_frame(ink)
    if top >= bottom or right == 0:
        raise ValueError(msg)
    plot = image[top:bottom, left : left + right]
    background = np.median(plot.reshape(-1, 3), axis=0)
    frames = _frames(plot, ink[top:, :right], background)
    return _Box(top, bottom, left, left + right), frames


def _axis_line(ink: npt.NDArray[np.bool_], bottom: int) -> int:
    """Return the row below a plot area, past any axis line along its bottom.

    The plot area ends at row ``bottom`` of the ink, ``AXIS_MARGIN`` above
    the axis given. An axis line is a row of ink within ``_SNAP_REACH`` rows
    above that, or the margin below it, that runs unbroken from within
    ``_SNAP_REACH`` columns of the left edge, where the other axis lies,
    across at least ``_LINE_SPAN`` of the width. If there is one, the area
    ends as it does when the axis given lies at the middle of the line's top
    row, as a click on it gives.
    """
    height, width = ink.shape
    for row in range(max(bottom - _SNAP_REACH, 0), min(bottom + AXIS_MARGIN, height)):
        starts = np.flatnonzero(ink[row, :_SNAP_REACH])
        if len(starts):
            along = ink[row, starts[0] :]
            run = len(along) if along.all() else int(np.argmin(along))
            if run >= width * _LINE_SPAN:
                return max(row + 1 - AXIS_MARGIN, 0)
    return bottom


def _inside_frame(ink: npt.NDArray[np.bool_]) -> tuple[int, int]:
    """Find the frame closing the axes in the part of an image beyond them.

    The frame's top is the topmost line that runs right from the left edge,
    next to the y-axis, across at least ``FRAME_REACH`` of the width, and is
    at most ``FRAME_WIDTH`` pixels thick. It lies in the upper half, so an
    open plot's x-axis line, below its middle, is not taken for one. Its
    right side runs down from the end of the top, along ``FRAME_REACH`` of
    the height. Both are dropped with ``AXIS_MARGIN`` beyond them.

    Returns:
        The first row below the frame and the number of columns left of it,
        or 0 and the width if there is no frame.
    """
    height, width = ink.shape
    rows = ink.all(axis=1)
    runs = np.where(rows, width, np.argmin(ink, axis=1))  # ink from the left edge
    long = np.flatnonzero(runs >= width * FRAME_REACH)
    if not len(long) or long[0] >= height / 2:
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
        if run >= height * FRAME_REACH:
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
    background. So is a colour that mixes the pale blends of several inks,
    as thin lines of several colours give, as ``_mixed`` finds. The inks of
    thin lines, which never reach their full colour and spread over many
    shades, are then found as ``_faint_inks`` finds them.

    Each colour but the background is given as the ink at the strength its
    most covered pixels show, as ``_at_full_strength`` finds it, and the
    colours are ranked by how many pixels show them, most first.
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
    directions, lengths = _inks(plot.astype(np.float64) - colours[0])
    groups = np.full(labels.shape, -1)
    shown = lengths >= _FAINT
    groups[shown] = _ink_groups(directions[shown])
    strength = np.linalg.norm(colours - colours[0], axis=1)
    keep = [0]
    for index in range(1, len(colours)):
        own = labels == index
        stronger = np.isin(labels, np.flatnonzero(strength > strength[index]))
        touching = cv2.dilate(stronger.astype(np.uint8), np.ones((3, 3), np.uint8))
        blend = 2 * np.count_nonzero(own & (touching > 0)) >= np.count_nonzero(own)
        if not blend and not _mixed(groups, own):
            keep.append(index)
    colours = colours[keep]
    labels = _classify(plot, colours)
    colours = np.concatenate(
        [colours, _faint_inks(colours, labels, groups, directions, lengths)]
    )
    return _at_full_strength(plot, colours, directions, lengths)


def _inks(
    offsets: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the direction and length of each offset of a colour from the background.

    A colour drawn over the background at any coverage, by antialiasing or
    transparency, lies on the ray from the background through it, so its
    direction tells its ink whatever its coverage. The direction of no
    offset at all is zero.
    """
    lengths = np.linalg.norm(offsets, axis=-1)
    directions = np.divide(
        offsets,
        lengths[..., np.newaxis],
        out=np.zeros_like(offsets),
        where=lengths[..., np.newaxis] > 0,
    )
    return directions, lengths


def _ink_groups(directions: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Number directions from the background, alike for those of one ink.

    Directions less than ``_SAME_INK`` degrees apart are one ink's, and the
    inks are numbered from the one with the most directions.
    """
    if not len(directions):
        return np.zeros(0, dtype=np.intp)
    # Directions as levels: a step of 2/255 is about a quarter of a degree.
    levels = np.round((directions + 1) * 127.5).astype(np.uint8)
    apart = 2 * math.sin(math.radians(_SAME_INK) / 2) * 127.5
    return _cluster(levels, _INK_BIN_WIDTH, apart)[2]


def _mixed(groups: npt.NDArray[np.intp], own: npt.NDArray[np.bool_]) -> bool:
    """Return whether the pixels of a colour show several inks rather than one.

    The pale blends of thin lines of several colours can be counted as one
    colour. The inks of its pixels, as ``_ink_groups`` numbers them, -1 for
    those less than ``_FAINT`` from the background, then lie along separate
    lines: ``_MIXED_SHARE`` of its pixels or more show other inks than the
    most common, away from it. Compression can shift the directions of the
    pixels of one line apart, but they stay together.
    """
    inked = own & (groups >= 0)
    common = int(np.bincount(groups[inked], minlength=1).argmax())
    near = cv2.dilate((inked & (groups == common)).astype(np.uint8), np.ones((5, 5)))
    apart = inked & (groups != common) & (near == 0)
    return bool(np.count_nonzero(apart) > _MIXED_SHARE * np.count_nonzero(inked))


def _faint_inks(
    colours: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp],
    groups: npt.NDArray[np.intp],
    directions: npt.NDArray[np.float64],
    lengths: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Find the inks of thin curves that the common colours leave unexplained.

    A thin antialiased line never reaches its full colour: its pixels spread
    over many shades, each too rare to be a common colour, but all lie on the
    ray from the background towards its ink. So the pixels that show some
    ink, at least ``_FAINT`` from the background, but that no colour claims,
    being labelled with none or with one more than ``_MIXED`` degrees off
    their own direction, are taken by their inks, as ``_ink_groups`` numbers
    them, but for the pale edges of a colour's pixels, within 2 pixels of
    them and ``_MIXED`` degrees of their direction.

    An ink is a curve's if its pixels, each weighted by its strength as a
    fraction of the strongest's, add up to ``_MIN_INK_FRACTION`` of the
    plot's width or more, as a line half a pixel wide across it does, and it
    stands out from the background by ``_CONTRAST``; but not if most of its
    pixels lie within 2 pixels of those of a stronger ink, a colour's or
    another such, as the fringes that compression gives a line do. Its colour
    is the background plus its direction at the strength that only a
    fiftieth of its pixels exceed.

    Args:
        colours: The background and the common colours.
        labels: The colour that each pixel shows, as ``_classify`` finds it.
        groups: The ink of each pixel, or -1 where it shows none.
        directions: Each pixel's direction from the background.
        lengths: How far each pixel lies from the background.

    Returns:
        The colours of the inks found, one per row.
    """
    background = colours[0]
    inks, strengths = _inks(colours - background)
    alike = math.cos(math.radians(_MIXED))
    claimed = (labels > 0) & (np.sum(directions * inks[labels], axis=-1) >= alike)
    free = ~claimed & (groups >= 0)
    # Each ink: its strength, its pixels and its colour, or None for a colour's.
    found: list[
        tuple[float, npt.NDArray[np.bool_], npt.NDArray[np.float64] | None]
    ] = []
    for index in range(1, len(colours)):
        own = claimed & (labels == index)
        near = cv2.dilate(own.astype(np.uint8), np.ones((5, 5)))
        free &= ~((near > 0) & (directions @ inks[index] >= alike))
        found.append((float(strengths[index]), own, None))
    width = labels.shape[1]
    for group in np.unique(groups[free]):
        own = free & (groups == group)
        strength = lengths[own]
        full = float(np.percentile(strength, 98))
        if strength.sum() / full >= width * _MIN_INK_FRACTION:
            (direction,), _ = _inks(
                (directions[own] * strength[:, np.newaxis]).sum(axis=0, keepdims=True)
            )
            colour = np.clip(background + full * direction, 0, 255)
            if np.abs(colour - background).max() >= _CONTRAST:
                found.append((full, own, colour))
    kept: list[npt.NDArray[np.float64]] = []
    for full, own, colour in found:
        stronger = np.zeros(own.shape, dtype=np.uint8)
        for other, pixels, _ in found:
            if other > full:
                stronger[pixels] = 1
        beside = cv2.dilate(stronger, np.ones((5, 5)))
        if colour is not None and (
            2 * np.count_nonzero(own & (beside > 0)) < np.count_nonzero(own)
        ):
            kept.append(colour)
    return np.array(kept).reshape(-1, 3)


def _at_full_strength(
    plot: Image,
    colours: npt.NDArray[np.float64],
    directions: npt.NDArray[np.float64],
    lengths: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the colours at the strength of their inks, most pixels first.

    Each colour but the background, ``colours[0]``, becomes the background
    plus the colour's direction from it, at the strength that only a fiftieth
    of the pixels that show it exceed along that direction: the full colour
    of a line a pixel wide or more, or the strongest that a thinner line
    shows. Only pixels within ``_MIXED`` degrees of the direction count, so
    not those where a curve of a darker ink crosses; a colour without such
    pixels stays as it is. The colours are then ranked by the pixels that
    show them, most first. ``directions`` and
    ``lengths`` give each pixel's direction from the background and how far
    it lies from it.
    """
    background = colours[0]
    labels = _classify(plot, colours)
    inks, strengths = _inks(colours - background)
    on_ray = math.cos(math.radians(_MIXED))
    strong = colours.copy()
    for index in range(1, len(colours)):
        along = lengths[(labels == index) & (directions @ inks[index] >= on_ray)]
        full = np.percentile(along, 98) if len(along) else strengths[index]
        strong[index] = np.clip(background + full * inks[index], 0, 255)
    counts = np.bincount(labels.ravel(), minlength=len(colours))
    return strong[[0, *(np.argsort(-counts[1:], kind="stable") + 1)]]


def _fills_areas(in_colour: npt.NDArray[np.bool_]) -> bool:
    """Return whether a colour's pixels fill areas rather than draw lines.

    Most pixels of a line lie close to its edges, and most of an area's lie
    more than ``_AREA_DEPTH`` pixels inside it.
    """
    depth = cv2.distanceTransform(in_colour.astype(np.uint8), cv2.DIST_L2, 3)
    return bool(np.median(depth[in_colour]) > _AREA_DEPTH)


def _composite(
    colours: npt.NDArray[np.float64], labels: npt.NDArray[np.intp], index: int
) -> bool:
    """Return whether a colour is where two others overlap, rather than a curve's.

    Translucent colours drawn over each other, such as two shaded bands,
    give a colour whose offset from the background, ``colours[0]``, is a
    blend of theirs: at least ``_PART`` of each, within ``_FINE_SAME_COLOUR``.
    The overlap lies among them, so ``_SURROUNDED`` of the pixels around its
    own, within 2 pixels, show one of the two, while those around a curve
    show mostly the background.
    """
    own = labels == index
    ring = (cv2.dilate(own.astype(np.uint8), np.ones((5, 5))) > 0) & ~own
    around = np.bincount(labels[ring], minlength=len(colours))
    offsets = colours - colours[0]
    others = (other for other in range(1, len(colours)) if other != index)
    for first, second in itertools.combinations(others, 2):
        if around[first] + around[second] < _SURROUNDED * np.count_nonzero(ring):
            continue
        blend = offsets[[first, second]].T
        parts = np.linalg.lstsq(blend, offsets[index])[0]
        if np.all(parts >= _PART) and (
            np.linalg.norm(blend @ parts - offsets[index]) <= _FINE_SAME_COLOUR
        ):
            return True
    return False


def _draws_grid(in_colour: npt.NDArray[np.bool_]) -> bool:
    """Return whether most of a colour's pixels lie on lines across the plot.

    The lines are horizontal or vertical, and run across at least
    ``_LINE_SPAN`` of the plot area, as grid lines do, with gaps of up to
    ``_LINE_GAP`` pixels, where curves drawn over them break them or they
    are dashed.
    """
    height, width = in_colour.shape
    mask = in_colour.astype(np.uint8)
    lines = []
    for gap, span in (
        ((_LINE_GAP, 1), (round(width * _LINE_SPAN), 1)),
        ((1, _LINE_GAP), (1, round(height * _LINE_SPAN))),
    ):
        closed = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, gap)
        )
        lines.append(
            cv2.morphologyEx(
                closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, span)
            )
        )
    on_lines = np.count_nonzero(in_colour & (cv2.bitwise_or(*lines) > 0))
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


def _labelled(plot: Image, colours: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Label each pixel with the index of its colour, leaving out darker ones' edges.

    Pixels are labelled as ``_classify`` labels them. Antialiasing blends
    the edges of a curve with the background, so those of a dark curve show
    lighter colours on its ray, such as a lighter grey curve's. Pixels of a
    colour that touch those of a stronger colour, within ``_MIXED`` degrees
    of its direction from the background, are taken for that colour's edges,
    and labelled 0.
    """
    labels = _classify(plot, colours)
    inks, strengths = _inks(colours - colours[0])
    alike = inks @ inks.T >= math.cos(math.radians(_MIXED))
    edges = np.zeros(labels.shape, dtype=np.bool_)
    for index in range(1, len(colours)):
        darker = np.flatnonzero(alike[index] & (strengths > strengths[index]))
        touching = cv2.dilate(np.isin(labels, darker).astype(np.uint8), np.ones((3, 3)))
        edges |= (labels == index) & (touching > 0)
    return np.where(edges, 0, labels)


def _match(palette: npt.NDArray[np.float64], colour: Colour) -> int:
    """Return the index of the curve colour in the palette that ``colour`` picks.

    That is the nearest within ``_MATCH_DISTANCE``, or failing that, the
    nearest of those whose ink lies on the colour's ray from the background,
    within ``_MIXED`` degrees, no further out than the colour: a line of the
    colour that is thinner than a pixel, or drawn translucent, shows only
    part of it.
    """
    distances = np.linalg.norm(palette[1:] - colour, axis=1)
    near = distances <= _MATCH_DISTANCE
    if not np.any(near):
        directions, lengths = _inks(palette[1:] - palette[0])
        (wanted,), (length,) = _inks(np.array([colour]) - palette[0])
        cosines = np.clip(directions @ wanted, -1, 1)
        near = (np.degrees(np.arccos(cosines)) <= _MIXED) & (lengths <= length + _FAINT)
    if not np.any(near):
        msg = f"no curve of colour #{bytes(colour).hex()} found"
        raise ValueError(msg)
    return int(np.argmin(np.where(near, distances, np.inf))) + 1


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


def _offsets(image: Image) -> npt.NDArray[np.float32]:
    """Return how far each pixel's colour lies from the background, the median."""
    background = np.median(image.reshape(-1, 3), axis=0)
    return np.asarray(image - background, dtype=np.float32)


def _key_piece(
    ink: npt.NDArray[np.bool_], key: Point
) -> tuple[npt.NDArray[np.bool_], int, int]:
    """Return the piece of ink nearest a key's position, and where it was cut out.

    The piece is cut out of the part of the image at most ``_KEY_REACH``
    pixels from the key each way, whose top row and left column are returned
    with it, so the middle of an open marker picks it. Specks of fewer than
    ``_MIN_PIXELS`` pixels, such as compression leaves, are left out.

    Raises:
        ValueError: If the key lies outside the image, or there is no ink near
            it.
    """
    height, width = ink.shape
    row, column = math.floor(height - key[1]), math.floor(key[0])
    if not (0 <= row < height and 0 <= column < width):
        msg = f"the marker position {key[0]:g},{key[1]:g} lies outside the image"
        raise ValueError(msg)
    top, left = max(row - _KEY_REACH, 0), max(column - _KEY_REACH, 0)
    window = ink[top : row + _KEY_REACH + 1, left : column + _KEY_REACH + 1]
    _, pieces, stats, _ = cv2.connectedComponentsWithStats(window.astype(np.uint8))
    sizable = stats[:, cv2.CC_STAT_AREA] >= _MIN_PIXELS
    sizable[0] = False  # the background
    rows, columns = np.nonzero(sizable[pieces])
    if not len(rows):
        msg = f"found no marker near {key[0]:g},{key[1]:g}"
        raise ValueError(msg)
    nearest = np.argmin((rows + top - row) ** 2 + (columns + left - column) ** 2)
    return pieces == pieces[rows[nearest], columns[nearest]], top, left


def _key_colour(
    offsets: npt.NDArray[np.float32], key: Point
) -> npt.NDArray[np.float64]:
    """Return the colour of a key's marker, less the background's.

    The marker is the piece of ink, differing from the background by at least
    ``_CONTRAST`` in some channel, nearest the key's position, as
    ``_key_piece`` finds it. Its colour lies in the direction of its
    strongest pixels, those at least half as far from the background in some
    channel as the furthest, and as far as any of its pixels reaches that
    way.
    """
    strength = np.abs(offsets).max(axis=2)
    piece, top, left = _key_piece(strength >= _CONTRAST, key)
    rows, columns = np.nonzero(piece)
    colours = offsets[rows + top, columns + left].astype(np.float64)
    strengths = strength[rows + top, columns + left]
    direction = colours[strengths >= strengths.max() / 2].sum(axis=0)
    direction /= np.linalg.norm(direction)
    return np.asarray(direction * float((colours @ direction).max()))


def _key_palette(
    colours: Sequence[npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Gather the keys' colours, less the background's, into the markers' colours.

    Colours lying at most ``_KEY_HUES`` degrees apart, seen from the
    background, are one colour, as far from the background as the furthest
    of them.

    Returns:
        The colours, less the background's, and the index of each key's.
    """
    palette: list[npt.NDArray[np.float64]] = []
    indices = []
    for colour in colours:
        for index, other in enumerate(palette):
            if _angle(colour, other) <= _KEY_HUES:
                if np.linalg.norm(colour) > np.linalg.norm(other):
                    palette[index] = colour
                indices.append(index)
                break
        else:
            indices.append(len(palette))
            palette.append(colour)
    return np.array(palette), indices


def _angle(first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]) -> float:
    """Return the angle between two colours less the background's, in degrees."""
    cosine = first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
    return math.degrees(math.acos(min(float(cosine), 1)))


def _ink_levels(
    offsets: npt.NDArray[np.float32], palette: npt.NDArray[np.float64]
) -> list[npt.NDArray[np.float32]]:
    """Return how much of each colour of the palette each pixel shows.

    A pixel shows the colour whose direction from the background lies
    nearest its own, if at most ``_INK_ANGLE`` degrees away, and none of the
    others, so ink of other colours is not taken for a marker's. Its level
    is how far it lies towards the colour, as a fraction of the colour's
    distance from the background, at most 1, times the most the colour
    differs from the background in a channel, out of 255: so a level is how
    much the pixel differs from the background in that channel, from 0 to 1.

    Args:
        offsets: Each pixel's colour less the background's, as ``_offsets``
            gives them.
        palette: The colours, less the background's.
    """
    lengths = np.linalg.norm(palette, axis=1)
    along = offsets @ (palette / lengths[:, np.newaxis]).T.astype(np.float32)
    cosines = along / np.maximum(np.linalg.norm(offsets, axis=2), 1)[..., np.newaxis]
    nearest = np.argmax(cosines, axis=2)
    shown = np.max(cosines, axis=2) >= math.cos(math.radians(_INK_ANGLE))
    levels = np.clip(along / lengths, 0, 1) * (np.abs(palette).max(axis=1) / 255)
    return [
        np.where(shown & (nearest == index), levels[..., index], 0).astype(np.float32)
        for index in range(len(palette))
    ]


def _key_marker(
    levels: npt.NDArray[np.float32], key: Point
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Cut a legend key's marker out of an image, and mask out the key's line.

    The key is the piece of ink, at least ``_CONTRAST`` levels from the
    background, nearest its position, as ``_key_piece`` finds it. A key at
    least ``_KEY_WIDE`` times as wide as tall may have a line through its
    marker: the rows with ink within ``_KEY_LINE`` pixels of either end. It
    does if they are fewer than the rows of the marker, the other rows'
    ink, leaving out faint ink, less than half as strong as the strongest,
    such as compression leaves beside the line. The line is masked out of
    the marker, as lines cross the markers in the plot at other angles,
    except where the marker hides it: where the pixels of its rows, across
    the marker, differ from the line's level beside it by more than
    ``_KEY_LINE_LEVEL``, as an open marker's face does. Under the mask, the
    marker is bridged by what lies above and below.

    Returns:
        The ink levels of the marker, with a pixel's margin, and the mask.

    Raises:
        ValueError: If there is no marker at the key's position.
    """
    piece, top, left = _key_piece(levels >= _CONTRAST / 255, key)
    window = levels[top : top + piece.shape[0], left : left + piece.shape[1]]
    rows, columns = np.nonzero(piece)
    hidden = np.zeros(window.shape, dtype=np.bool_)  # the key's line
    if np.ptp(columns) + 1 >= _KEY_WIDE * (np.ptp(rows) + 1):
        line = np.zeros(len(window), dtype=np.bool_)  # its rows
        line[
            rows[
                (columns < columns.min() + _KEY_LINE)
                | (columns > columns.max() - _KEY_LINE)
            ]
        ] = True
        marker = ~line[rows] & (window[rows, columns] >= window[piece].max() / 2)
        if not np.any(marker):
            msg = f"found only a line, and no marker, at {key[0]:g},{key[1]:g}"
            raise ValueError(msg)
        if np.count_nonzero(line) < len(np.unique(rows[marker])):
            first, last = columns[marker].min(), columns[marker].max()
            beside = np.ones(window.shape[1], dtype=np.bool_)
            beside[first : last + 1] = False
            for row in np.flatnonzero(line):
                hidden[row] = beside
                level = np.median(window[row, beside & piece[row]])
                hidden[row, first : last + 1] = (
                    np.abs(window[row, first : last + 1] - level) <= _KEY_LINE_LEVEL
                )
            rows = np.arange(rows[marker].min(), rows[marker].max() + 1)
            columns = columns[marker]

    box = (
        slice(max(rows.min() - 1, 0), rows.max() + 2),
        slice(max(columns.min() - 1, 0), columns.max() + 2),
    )
    fringe = cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    shape = np.where(fringe, window, 0).astype(np.float32)
    for column in np.flatnonzero(hidden.any(axis=0)):
        under = hidden[:, column]
        shape[under, column] = np.interp(
            np.flatnonzero(under), np.flatnonzero(~under), shape[~under, column]
        )
    return shape[box], (~hidden[box]).astype(np.float32)


def _scores(
    area: npt.NDArray[np.float32],
    shape: npt.NDArray[np.float32],
    mask: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Return how well a key's marker correlates with the image at each place.

    A place is where the marker's top-left corner lies. The correlation is
    left out of the mask, and is -1 where the image is flat, or has less
    than ``_MARKER_INK`` of the ink of the top, bottom, left or right half of
    the marker, so a marker with thin strokes, such as a cross, is not found
    along a line.
    """
    if area.shape[0] < shape.shape[0] or area.shape[1] < shape.shape[1]:
        return np.empty((0, 0), dtype=np.float32)
    scores = np.nan_to_num(  # -1 at flat patches
        cv2.matchTemplate(area, shape, cv2.TM_CCOEFF_NORMED, mask=mask),
        nan=-1,
        posinf=-1,
        neginf=-1,
    ).astype(np.float32)
    ink = ((shape >= _CONTRAST / 255) & (mask > 0)).astype(np.float32)
    rows, columns = shape.shape
    for half in (
        np.s_[: (rows + 1) // 2],
        np.s_[rows // 2 :],
        np.s_[:, : (columns + 1) // 2],
        np.s_[:, columns // 2 :],
    ):
        part = np.zeros_like(ink)
        part[half] = ink[half]
        shown = cv2.matchTemplate(area, part, cv2.TM_CCORR)
        scores[shown < _MARKER_INK * float(np.sum(shape * part))] = -1
    return scores


def _peaks(
    scores: npt.NDArray[np.float32], size: tuple[int, int]
) -> list[tuple[float, float, float]]:
    """Return where a key's marker matches the image, and how well.

    A match is a local maximum of the scores, as ``_scores`` gives them, of at
    least ``_MARKER_SCORE``. Its centre is placed to a fraction of a pixel,
    at the top of a parabola through the scores each way.

    Args:
        scores: The scores of the marker.
        size: The rows and columns of the marker.

    Returns:
        The score, row and column of the centre of each match.
    """
    if not scores.size:  # the marker is bigger than the plot area
        return []
    peaks = (scores >= _MARKER_SCORE) & (
        scores == cv2.dilate(scores, np.ones((3, 3), np.uint8))
    )
    padded = np.pad(scores, 1, constant_values=-1)
    middle_row, middle_column = (size[0] - 1) / 2, (size[1] - 1) / 2
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


def _near(
    scores: npt.NDArray[np.float32], size: tuple[int, int], row: float, column: float
) -> float:
    """Return a key's best score with its marker's centre near a place.

    Its centre lies within ``_REACH`` pixels of the place each way.
    """
    top = round(row - (size[0] - 1) / 2)
    left = round(column - (size[1] - 1) / 2)
    window = scores[
        max(top - _REACH, 0) : top + _REACH + 1,
        max(left - _REACH, 0) : left + _REACH + 1,
    ]
    return float(window.max()) if window.size else -1.0


def _at_key(found: Sequence[_Key], row: float, column: float) -> bool:
    """Return whether a place lies on a key, closer than its marker's size."""
    return any(
        math.dist((row, column), (key.row, key.column)) < key.size for key in found
    )


def _assign(
    area: npt.NDArray[np.float32], found: Sequence[_Key], colour: int
) -> list[_Marker]:
    """Find the markers of the keys of one colour in its ink levels.

    Matches, as ``_peaks`` finds them, are taken best first, leaving out those
    on keys and those closer than ``_MARKER_SPACING`` of the smallest
    marker's size to a marker already found. A match's rivals are the other
    keys whose markers score within ``_MARKER_MARGIN`` of it near it, as
    ``_near`` finds them; without rivals, its key's marker is fitted to the
    image there, as ``_fit`` fits it, and with them, ``_resolve`` chooses.
    """
    members = [index for index, key in enumerate(found) if key.colour == colour]
    scores = {
        index: _scores(area, found[index].shape, found[index].mask) for index in members
    }
    candidates = sorted(
        (
            (score, row, column, index)
            for index in members
            for score, row, column in _peaks(scores[index], found[index].shape.shape)
        ),
        reverse=True,
    )
    spacing = _MARKER_SPACING * min(found[index].size for index in members)
    placed: list[_Marker] = []
    for score, row, column, index in candidates:
        if _at_key(found, row, column) or any(
            math.dist((row, column), (marker.row, marker.column)) < spacing
            for marker in placed
        ):
            continue
        rivals = [
            other
            for other in members
            if other != index
            and _near(scores[other], found[other].shape.shape, row, column)
            >= score - _MARKER_MARGIN
        ]
        if rivals:
            placed.append(
                _resolve(area, found, [index, *rivals], (row, column), placed)
            )
        else:
            placed.append(_fit(area, found[index], index, row, column)[1])
    return placed


def _resolve(
    area: npt.NDArray[np.float32],
    found: Sequence[_Key],
    rivals: Sequence[int],
    place: tuple[float, float],
    known: Sequence[_Marker],
) -> _Marker:
    """Choose which of rival keys' markers best explains the image near a place.

    Each rival's marker is fitted to the image there, as ``_fit`` fits it,
    and with it the markers it partly hides, or that partly hide it, as
    ``_story`` finds them, given the markers already known. The marker whose
    story differs least from the image, as ``_misfits`` measures it, wins.
    It is ambiguous if the story of a rival whose key is not in the
    winner's differs from the image by less than ``_AMBIGUOUS`` more, as a
    fraction of the winner's difference.
    """
    stories = [
        _story(area, found, _fit(area, found[index], index, *place)[1], known)
        for index in rivals
    ]
    misfits = _misfits(area, stories)
    best = int(np.argmin(misfits))
    told = {marker.key for marker in stories[best]}
    others = [
        misfit
        for misfit, story in zip(misfits, stories, strict=True)
        if story[0].key not in told
    ]
    return dataclasses.replace(
        stories[best][0],
        ambiguous=bool(others)
        and min(others) - misfits[best] < _AMBIGUOUS * misfits[best],
    )


def _fit(
    area: npt.NDArray[np.float32], key: _Key, index: int, row: float, column: float
) -> tuple[float, _Marker]:
    """Fit a key's marker to the image near a place, to a fraction of a pixel.

    The marker is shifted by the fraction of a pixel that its centre lies
    from the place, and placed up to ``_REACH`` pixels from there each way,
    where it differs least from the image, as ``TM_SQDIFF_NORMED`` measures
    it: the sum of the squared differences, outside the mask, over the
    square root of the product of the sums of the squares. Unlike a
    correlation, this tells markers of much the same shape and different
    sizes apart, such as circles and octagons.

    Returns:
        The difference, and the marker.
    """
    rows, columns = key.shape.shape
    top = round(row - (rows - 1) / 2)
    left = round(column - (columns - 1) / 2)
    down, across = row - (rows - 1) / 2 - top, column - (columns - 1) / 2 - left
    moved = np.asarray(
        cv2.warpAffine(
            key.shape,
            np.array([[1, 0, across], [0, 1, down]], dtype=np.float32),
            (columns, rows),
            flags=cv2.INTER_CUBIC,
        ),
        dtype=np.float32,
    )
    patch = _window(
        area, top - _REACH, left - _REACH, rows + 2 * _REACH, columns + 2 * _REACH
    )
    misfits = cv2.matchTemplate(patch, moved, cv2.TM_SQDIFF_NORMED, mask=key.mask)
    misfits = np.nan_to_num(misfits, nan=math.inf)
    at_row, at_column = np.unravel_index(np.argmin(misfits), misfits.shape)
    return float(misfits[at_row, at_column]), _Marker(
        index,
        row + int(at_row) - _REACH,
        column + int(at_column) - _REACH,
        top + int(at_row) - _REACH,
        left + int(at_column) - _REACH,
        moved,
        key.mask,
    )


def _warn_alike(keys: Sequence[Point], found: Sequence[_Key]) -> None:
    """Warn of keys of one colour whose markers look alike at this size.

    They do if each, fitted to the other as ``_fit`` fits it to a plot,
    differs from it by less than ``_ALIKE``.
    """
    for (first, one), (second, other) in itertools.combinations(
        zip(keys, found, strict=True), 2
    ):
        if (
            one.colour == other.colour
            and max(_difference(one, other), _difference(other, one)) < _ALIKE
        ):
            (x, y), (u, v) = first, second
            warnings.warn(
                f"the markers of the keys at {x:g},{y:g} and {u:g},{v:g} look alike "
                "at this size, so some may be taken for each other",
                stacklevel=3,
            )


def _difference(one: _Key, other: _Key) -> float:
    """Return how much a key's marker differs from another's, fitted to it."""
    rows, columns = other.shape.shape
    return _fit(
        np.pad(other.shape, _REACH),
        one,
        0,
        (rows - 1) / 2 + _REACH,
        (columns - 1) / 2 + _REACH,
    )[0]


def _story(
    area: npt.NDArray[np.float32],
    found: Sequence[_Key],
    fit: _Marker,
    known: Sequence[_Marker],
) -> list[_Marker]:
    """Tell the best story of the image around a marker fitted to it.

    The marker may be alone, or drawn over or under others of its colour,
    partly hidden. Then up to ``_MOST_HIDDEN`` others are found, one by one,
    around those found, in what they and the markers already known leave
    uncovered, as ``_hidden`` finds them, if they explain the image, as
    ``_explains`` judges; and each time, the marker is moved to fit what the
    others leave uncovered. The story is whichever of the marker alone and
    the marker with the others differs less from the image, as ``_misfits``
    measures them.

    Returns:
        The story's markers, the fitted one first.
    """
    story = [fit]
    while len(story) <= _MOST_HIDDEN and (hidden := _hiding(area, found, story, known)):
        story.append(max(hidden, key=lambda pair: pair[0])[1])
        covered = _footprints(area.shape, [*story[1:], *known])
        moved = [
            pair
            for around in story[1:]
            if (pair := _hidden(area, covered, found, fit.key, around)) is not None
        ]
        story[0] = max(moved, key=lambda pair: pair[0], default=(0, story[0]))[1]
    if len(story) == 1:
        return story
    alone, together = _misfits(area, [[fit], story])
    return story if together < alone else [fit]


def _hiding(
    area: npt.NDArray[np.float32],
    found: Sequence[_Key],
    story: Sequence[_Marker],
    known: Sequence[_Marker],
) -> list[tuple[float, _Marker]]:
    """Return markers partly hidden under or over a story's, that explain the image.

    Each key's marker of the story's colour is looked for around each of the
    story's markers, in what they and the markers already known leave
    uncovered, as ``_hidden`` finds it, and kept if it explains the image,
    as ``_explains`` judges, with its correlation.
    """
    covered = _footprints(area.shape, [*story, *known])
    return [
        pair
        for around in story
        for index, key in enumerate(found)
        if key.colour == found[around.key].colour
        and (pair := _hidden(area, covered, found, index, around)) is not None
        and _explains(area, *pair, [*story, *known])
    ]


def _hidden(
    area: npt.NDArray[np.float32],
    covered: npt.NDArray[np.bool_],
    found: Sequence[_Key],
    index: int,
    around: _Marker,
) -> tuple[float, _Marker] | None:
    """Find a key's marker, partly hidden, around another, by what shows of it.

    Its centre lies closer to the other's than half the sum of their sizes,
    and what shows of it is the part of its shape, outside its mask, that
    the ``covered`` pixels leave uncovered. That part must hold at least
    ``_HIDDEN_SHOWN`` of its ink, and the image at least ``_MARKER_INK`` of
    the ink of that part. It is where that part correlates best with the
    image, its centre placed to a fraction of a pixel, at the top of a
    parabola through the correlations each way.

    Returns:
        The correlation and the marker, or None if no place shows enough of
        it.
    """
    key = found[index]
    rows, columns = key.shape.shape
    reach = (key.size + found[around.key].size) / 2
    span = 2 * math.ceil(reach) + 1
    top = round(around.row - (rows - 1) / 2) - math.ceil(reach)
    left = round(around.column - (columns - 1) / 2) - math.ceil(reach)
    image = _window(area, top, left, rows + span - 1, columns + span - 1)
    hidden = _window(
        covered.astype(np.float32), top, left, rows + span - 1, columns + span - 1
    )
    ones = np.ones_like(image)
    mask = key.mask
    ink = key.shape * (key.shape >= _CONTRAST / 255) * mask
    showing = _sums(ones, hidden, ink)
    present = _sums(image, hidden, (ink > 0) * mask) / np.maximum(showing, 1e-9)
    count = np.maximum(_sums(ones, hidden, mask), 1)
    shape_sum = _sums(ones, hidden, key.shape * mask)
    image_sum = _sums(image, hidden, mask)
    covariance = _sums(image, hidden, key.shape * mask) - shape_sum * image_sum / count
    spread = np.sqrt(
        np.maximum(_sums(ones, hidden, key.shape**2 * mask) - shape_sum**2 / count, 0)
        * np.maximum(_sums(image**2, hidden, mask) - image_sum**2 / count, 0)
    )
    scores = covariance / np.maximum(spread, 1e-9)
    offsets = np.arange(span) - math.ceil(reach)
    shows = (
        (showing >= _HIDDEN_SHOWN * float(np.sum(ink)))
        & (present >= _MARKER_INK)
        & (spread > _FLAT)
        & (np.hypot(*np.meshgrid(offsets, offsets, indexing="ij")) <= reach)
    )
    if not np.any(shows):
        return None
    scores = np.where(shows, scores, -1)
    at_row, at_column = np.unravel_index(
        np.argmax(np.where(shows, scores, -np.inf)), scores.shape
    )
    padded = np.pad(scores, 1, constant_values=-1)
    return float(scores[at_row, at_column]), _Marker(
        index,
        top
        + at_row
        + (rows - 1) / 2
        + _vertex(padded[at_row : at_row + 3, at_column + 1]),
        left
        + at_column
        + (columns - 1) / 2
        + _vertex(padded[at_row + 1, at_column : at_column + 3]),
        top + int(at_row),
        left + int(at_column),
        key.shape,
        key.mask,
    )


def _sums(
    levels: npt.NDArray[np.float32],
    covered: npt.NDArray[np.float32],
    kernel: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Return the sums of levels times a kernel's, at each place, where uncovered.

    A place is where the kernel's top-left corner lies, and ``covered`` is 1
    on covered pixels and 0 elsewhere. The sums are those of correlations
    with all the levels, less those with the covered ones.
    """
    kernel = kernel.astype(np.float32)
    return np.asarray(
        cv2.matchTemplate(levels, kernel, cv2.TM_CCORR)
        - cv2.matchTemplate(levels * covered, kernel, cv2.TM_CCORR),
        dtype=np.float32,
    )


def _explains(
    area: npt.NDArray[np.float32],
    score: float,
    marker: _Marker,
    known: Sequence[_Marker],
) -> bool:
    """Return whether a partly hidden marker explains the image.

    It does if what shows of it correlates with the image by
    ``_HIDDEN_SCORE`` or more, as ``_hidden`` finds it, and drawn with the
    known markers it differs from the image, under it, at least
    ``_HIDDEN_GAIN`` less than they do alone, as ``_misfit`` measures it.
    """
    if score < _HIDDEN_SCORE:
        return False
    rows, columns = marker.shape.shape
    window = _window(area, marker.top, marker.left, rows, columns)
    where = _counted([*known, marker], marker.top, marker.left, (rows, columns))
    without, joined = (
        _misfit(
            _render(markers, marker.top, marker.left, (rows, columns)), window, where
        )
        for markers in (known, [*known, marker])
    )
    return joined <= (1 - _HIDDEN_GAIN) * without


def _uncover(
    area: Sequence[npt.NDArray[np.float32]],
    found: Sequence[_Key],
    placed: Sequence[_Marker],
) -> list[_Marker]:
    """Find markers partly hidden under those already found, or over them.

    Round by round, each key's marker is looked for around each marker
    found in the round before, in what all the markers found leave
    uncovered, as ``_hidden`` finds it; ink of the other colours of the
    palette covers it too. It is a marker if it explains the image, as
    ``_explains`` judges, and lies on no key. Of those closer than
    ``_MARKER_SPACING`` of a marker's size, of one colour, the best
    correlated wins.

    Returns:
        The markers found.
    """
    added: list[_Marker] = []
    around = list(placed)
    inked = [levels >= _CONTRAST / 255 for levels in area]
    while around:
        known = [*placed, *added]
        hiding = []  # what covers markers of each colour
        for colour in range(len(area)):
            covered = _footprints(area[0].shape, known)
            for other, ink in enumerate(inked):
                if other != colour:
                    covered |= ink
            hiding.append(covered)
        candidates = []
        for marker, (index, key) in itertools.product(around, enumerate(found)):
            pair = _hidden(area[key.colour], hiding[key.colour], found, index, marker)
            if (
                pair is not None
                and not _at_key(found, pair[1].row, pair[1].column)
                and _explains(
                    area[key.colour],
                    *pair,
                    [other for other in known if found[other.key].colour == key.colour],
                )
            ):
                candidates.append(pair)
        around = []
        for _, candidate in sorted(candidates, key=lambda pair: -pair[0]):
            colour = found[candidate.key].colour
            if not any(
                found[other.key].colour == colour
                and math.dist(
                    (candidate.row, candidate.column), (other.row, other.column)
                )
                < _MARKER_SPACING * found[candidate.key].size
                for other in around
            ):
                around.append(candidate)
        added += around
    return added


def _rethink(
    area: Sequence[npt.NDArray[np.float32]],
    found: Sequence[_Key],
    markers: Sequence[_Marker],
) -> list[_Marker]:
    """Look again for markers probably hidden altogether, where series share x.

    Where ``_missing`` finds a series' marker probably hidden, under the
    nearest marker of another key of its colour, its key's marker is fitted
    there, as ``_fit`` fits it, and the story around it told, as ``_story``
    tells it, given the other markers. If the story holds a marker of the
    other's key too, and differs less from the image than the other alone,
    as ``_misfits`` measures them, the story replaces the other, as when the
    other was taken to lie where the hidden marker does.

    Returns:
        The markers, with those replaced.
    """
    markers = list(markers)
    for index, places in enumerate(_missing(found, markers)):
        colour = found[index].colour
        for row, column in places:
            nearest = sorted(
                (
                    marker
                    for marker in markers
                    if marker.key != index
                    and found[marker.key].colour == colour
                    and math.dist((row, column), (marker.row, marker.column))
                    < (found[index].size + found[marker.key].size) / 2
                ),
                key=lambda marker: math.dist(
                    (row, column), (marker.row, marker.column)
                ),
            )
            for other in nearest[:1]:  # the marker it lies under
                known = [
                    marker
                    for marker in markers
                    if marker is not other and found[marker.key].colour == colour
                ]
                fit = _fit(area[colour], found[index], index, row, column)[1]
                story = _story(area[colour], found, fit, known)
                before, after = _misfits(area[colour], [[other], story])
                if after < before and any(
                    marker.key == other.key for marker in story[1:]
                ):
                    markers = [
                        *known,
                        *story,
                        *(
                            marker
                            for marker in markers
                            if found[marker.key].colour != colour
                        ),
                    ]
    return markers


def _missing(
    found: Sequence[_Key], markers: Sequence[_Marker]
) -> list[list[tuple[float, float]]]:
    """Return where each key's markers are probably hidden altogether.

    Series usually share x positions. Where other series have markers in a
    gap between a series' markers, further apart than ``_GAP`` times its
    markers' usual spacing, more than ``_ALIGNED`` pixels from them, its
    marker would lie on the straight line between them. It is probably
    hidden there if another series' marker lies closer than half the sum of
    their sizes.

    Returns:
        For each key, the rows and columns of the places.
    """
    columns = np.sort([marker.column for marker in markers])
    groups = np.split(columns, np.flatnonzero(np.diff(columns) > _ALIGNED) + 1)
    shared = [float(group.mean()) for group in groups if len(group)]
    places: list[list[tuple[float, float]]] = []
    for index, key in enumerate(found):
        own = sorted(
            (marker.column, marker.row) for marker in markers if marker.key == index
        )
        lost = []
        if len(own) > 1:
            xs, ys = np.array(own).T
            gaps = np.diff(xs)
            for column in shared:
                after = int(np.searchsorted(xs, column))
                row = float(np.interp(column, xs, ys))
                if (
                    0 < after < len(xs)
                    and gaps[after - 1] > _GAP * np.median(gaps)
                    and min(column - xs[after - 1], xs[after] - column) > _ALIGNED
                    and any(
                        marker.key != index
                        and math.dist((row, column), (marker.row, marker.column))
                        < (key.size + found[marker.key].size) / 2
                        for marker in markers
                    )
                ):
                    lost.append((row, column))
        places.append(lost)
    return places


def _window(
    area: npt.NDArray[np.float32], top: int, left: int, rows: int, columns: int
) -> npt.NDArray[np.float32]:
    """Return part of an image, from its top-left corner, with zeros beyond it."""
    window = np.zeros((rows, columns), dtype=np.float32)
    _stamp(window, area, -top, -left)
    return window


def _stamp(
    canvas: npt.NDArray[np.float32], piece: npt.NDArray[np.float32], top: int, left: int
) -> None:
    """Draw a piece on a canvas from its top-left corner, keeping the higher level."""
    first, last = max(top, 0), min(top + piece.shape[0], canvas.shape[0])
    start, stop = max(left, 0), min(left + piece.shape[1], canvas.shape[1])
    if first < last and start < stop:
        canvas[first:last, start:stop] = np.maximum(
            canvas[first:last, start:stop],
            piece[first - top : last - top, start - left : stop - left],
        )


def _render(
    markers: Iterable[_Marker], top: int, left: int, size: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Draw markers' shapes in part of a plot area, from its top-left corner."""
    canvas = np.zeros(size, dtype=np.float32)
    for marker in markers:
        _stamp(canvas, marker.shape, marker.top - top, marker.left - left)
    return canvas


def _counted(
    markers: Iterable[_Marker], top: int, left: int, size: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """Return where markers' shapes count in part of a plot area, outside masks."""
    canvas = np.zeros(size, dtype=np.float32)
    for marker in markers:
        _stamp(canvas, marker.mask, marker.top - top, marker.left - left)
    return canvas > 0


def _footprints(
    size: tuple[int, ...], markers: Iterable[_Marker]
) -> npt.NDArray[np.bool_]:
    """Return the pixels that markers' ink covers in a plot area, and a pixel around."""
    canvas = np.zeros(size[:2], dtype=np.float32)
    for marker in markers:
        _stamp(
            canvas,
            (marker.shape >= _CONTRAST / 255).astype(np.float32),
            marker.top,
            marker.left,
        )
    return cv2.dilate(canvas, np.ones((3, 3), np.uint8)) > 0


def _misfit(
    drawn: npt.NDArray[np.float32],
    image: npt.NDArray[np.float32],
    where: npt.NDArray[np.bool_],
) -> float:
    """Return how much a drawing differs from an image where it counts.

    The difference is measured as ``_fit`` measures it, and is infinite if
    either is blank there.
    """
    drawn, image = drawn[where], image[where]
    scale = math.sqrt(float(np.sum(drawn**2)) * float(np.sum(image**2)))
    return float(np.sum((drawn - image) ** 2)) / scale if scale else math.inf


def _misfits(
    area: npt.NDArray[np.float32], stories: Sequence[Sequence[_Marker]]
) -> list[float]:
    """Return how much each story's markers, drawn together, differ from the image.

    Each is drawn in the part of the plot area that all the stories' markers
    span, and differs from it where any of their shapes count, as
    ``_counted`` finds them, as ``_misfit`` measures it.
    """
    every = [marker for story in stories for marker in story]
    top = min(marker.top for marker in every)
    left = min(marker.left for marker in every)
    bottom = max(marker.top + marker.shape.shape[0] for marker in every)
    right = max(marker.left + marker.shape.shape[1] for marker in every)
    window = _window(area, top, left, bottom - top, right - left)
    where = _counted(every, top, left, window.shape)
    return [
        _misfit(_render(story, top, left, window.shape), window, where)
        for story in stories
    ]


def _to_colour(levels: npt.NDArray[np.float64]) -> Colour:
    """Round red, green and blue levels to a colour."""
    red, green, blue = (int(level) for level in np.round(levels))
    return red, green, blue
