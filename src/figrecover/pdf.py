"""Read plots from PDF files exactly, from their vector drawing."""

import ctypes
import functools
import itertools
import math
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from .axes import (
    _LINE_COVER,
    _SCALE,
    _SNAP_REACH,
    Axes,
    _beyond,
    _check_one_y_axis,
    _decimal_comma,
    _factor,
    _in_line,
    _leftmost_column,
    _lines,
    _lowest_row,
    _normal,
    _number,
    _reading,
    _scales,
    _Text,
    _title,
    _transposed,
    _with_decimal_points,
    _without_second_lines,
    _x_title,
)
from .calibration import Calibration, Point
from .digitize import (
    _MATCH_DISTANCE,
    AXIS_MARGIN,
    Colour,
    Image,
    _curve_colours,
    remove_grid,
    trace_curves,
)
from .errorbars import ErrorBars

type Box = tuple[float, float, float, float]
"""A rectangle's left, bottom, right and top edges, in points."""

PREVIEW_SCALE = 2
"""Pixels per point in a page drawn as an image, as for a preview."""

_FLAT = 1e-3  # a line whose ends are this many points apart across it is straight
_TURNED = 1  # characters turned this many degrees apart or more are on separate lines
_GAP = 0.6  # characters this many font sizes apart or more are in separate texts
_SPACE = 0.2  # characters this many font sizes apart or more have a space between
_BASELINE = 0.6  # characters this many font sizes above or below a line are not on it
_SMALL = 0.85  # characters this fraction of a line's font size or less, and
_RISE = 0.15  # this many font sizes above or below it, are superscripts or subscripts
_MARK_REACH = 0.1  # a tick mark reaches to within this many label heights of its axis
_CENTRED = 0.5  # labels this many label heights or less from their marks centre on them
_SLACK = 0.01  # a point this many points beyond the axes or a clip lies on them
_MARKER_SIZE = 36  # a marker is at most this many points across
_SAME_SHAPE = 0.01  # outlines whose points lie this many points apart are the same
_LABEL_REACH = 4  # a legend's label lies within this many of its heights of its key
_KEY_REACH = 10  # a key lies within this many points of the marker it picks
_ON_BAR = 0.01  # a point this many points or less from a straight line lies on it
_WHITE: Colour = (255, 255, 255)
_SUPERSCRIPTS = str.maketrans(
    "0123456789+-\u2212=()ni",
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
    "\u207a\u207b\u207b\u207c\u207d\u207e\u207f\u2071",
)
_SUBSCRIPTS = str.maketrans(
    "0123456789+-\u2212=()aehklmnopstx",
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    "\u208a\u208b\u208b\u208c\u208d\u208e"
    "\u2090\u2091\u2095\u2096\u2097\u2098\u2099\u2092\u209a\u209b\u209c\u2093",
)


@dataclass(frozen=True, slots=True)
class _Path:
    """A path drawn on a page, in points from the bottom-left corner of the page.

    ``subpaths`` holds the points that each subpath passes through, without
    the control points of its curved pieces, and ``closed`` whether each is
    closed. ``segments`` holds the ends of each straight piece, as rows of x
    and y at one end and x and y at the other. ``stroke`` and ``fill`` are
    the colours of the path's outline and inside, or None if it has none,
    and ``clip`` is the box beyond which it is hidden. ``anchor`` is where
    the origin of the form that holds the path lies, or None if the page
    holds it; a form drawn as a marker is placed with its origin at the
    point it marks.
    """

    subpaths: tuple[npt.NDArray[np.float64], ...]
    closed: tuple[bool, ...]
    segments: npt.NDArray[np.float64]
    stroke: Colour | None
    fill: Colour | None
    clip: Box
    anchor: Point | None


@dataclass(frozen=True, slots=True)
class _Marker:
    """A marker drawn on a page, in points from the bottom-left corner of the page.

    ``centre`` is the point it marks, and ``outline`` the points its paths
    pass through, and the middles of their straight pieces, less the middle
    of their box, the same for markers of one shape wherever they lie.
    ``stroke``, ``fill`` and ``clip`` are those of its paths, as ``_Path``
    gives them.
    """

    centre: Point
    outline: npt.NDArray[np.float64]
    stroke: Colour | None
    fill: Colour | None
    clip: Box

    @property
    def colour(self) -> Colour:
        """Return the colour that tells its series apart.

        That is its fill, or its outline's colour if it has no fill or a
        white one, as open markers have.
        """
        shown = [c for c in (self.fill, self.stroke) if c is not None and c != _WHITE]
        return shown[0] if shown else _WHITE


@dataclass(frozen=True, slots=True)
class _Picture:
    """An image drawn on a page, on white, the box it fills, and its clip, in points."""

    image: Image
    box: Box
    clip: Box


@dataclass(frozen=True, slots=True)
class Drawing:
    """What is drawn on a page of a PDF, as ``load_pdf`` reads it.

    Positions on the page are in points, 72 to the inch, from the bottom-left
    corner of the page, as PDF files give them.

    Attributes:
        path: The PDF file.
        page: The number of the page, from 1.
        box: The left, bottom, right and top edges of the part of the page
            read, such as one figure on it.
        texts: The lines of text on the page, for figrecover's own use.
        paths: The paths drawn on the page, for figrecover's own use.
        pictures: The images drawn on the page, for figrecover's own use.
    """

    path: str
    page: int
    box: Box
    texts: tuple[_Text, ...] = field(repr=False)
    paths: tuple[_Path, ...] = field(repr=False)
    pictures: tuple[_Picture, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Char:
    """A character on a page, in points from the bottom-left corner.

    ``box`` runs along the text over the character's advance, the space it
    takes, and across the text over its ink. ``advance`` is how far along the
    text the next character starts, or None if the text is slanted, so that
    the box cannot show it. ``angle`` is the angle in degrees, anticlockwise
    from across, at which the text runs, and ``size`` its font size.
    """

    text: str
    box: Box
    origin: Point
    size: float
    angle: float
    advance: float | None


@dataclass(frozen=True, slots=True)
class _Series:
    """A curve in a plot: its colour, its points, and its length, in points."""

    colour: Colour
    points: npt.NDArray[np.float64]
    length: float


@functools.cache
def _pdfium() -> Any:
    """Return the PDF library, pypdfium2, which the ``pdf`` extra installs."""
    try:
        # Imported here, as it is an optional dependency.
        import pypdfium2  # noqa: PLC0415
        import pypdfium2.raw  # noqa: PLC0415
    except ModuleNotFoundError as error:
        msg = "reading PDF files needs the pdf extra: pip install 'figrecover[pdf]'"
        raise ModuleNotFoundError(msg) from error
    return pypdfium2


def load_pdf(
    path: str | os.PathLike[str], page: int = 1, box: Box | None = None
) -> Drawing:
    """Read what is drawn on a page of a PDF: its text, paths and images.

    Text is read character by character, and characters next to one another
    along a line, less than ``_GAP`` font sizes apart, make a text, with a
    space where they lie ``_SPACE`` font sizes apart or more. Characters at
    most ``_SMALL`` the size of the line's largest, and ``_RISE`` of its size
    or more above or below it, are written as superscripts or subscripts, so
    that 10 with a small 2 raised beside it reads as 10 squared. Paths are
    read with the points they pass through, leaving out the control points
    of curved pieces, and images on white, with any transparency, as they
    are shown.

    The PDF is read with pypdfium2, which the ``pdf`` extra installs.

    Args:
        path: The PDF file.
        page: The number of the page to read, from 1.
        box: The left, bottom, right and top edges of the part of the page
            to read, in points from its bottom-left corner, such as one figure
            on it. Text whose middle lies outside it is left out, and so are
            paths and images wholly outside it; the rest is hidden beyond it,
            as though clipped. Defaults to the whole page.

    Returns:
        What is drawn on the page, within the box.

    Raises:
        ModuleNotFoundError: If the ``pdf`` extra is not installed.
        ValueError: If the file cannot be read as a PDF, has no such page, or
            the box lies off the page.
    """
    pdfium = _pdfium()
    try:
        document = pdfium.PdfDocument(os.fspath(path))
    except pdfium.PdfiumError as error:
        msg = f"cannot read PDF {os.fspath(path)!r}: {error}"
        raise ValueError(msg) from None
    if not 1 <= page <= len(document):
        msg = f"{os.fspath(path)!r} has no page {page}, only {len(document)}"
        raise ValueError(msg)
    sheet = document[page - 1]
    bounds = _edges(sheet.get_bbox())
    area = bounds if box is None else _overlap(bounds, box)
    if area is None:
        msg = f"the box {_written(box)} lies off the page, {_written(bounds)}"
        raise ValueError(msg)
    texts = [text for text in _texts(sheet) if _holds(area, (text.x, text.y))]
    paths: list[_Path] = []
    pictures: list[_Picture] = []
    for drawn, matrix, clip, anchor in _objects(sheet, area):
        if drawn.type == pdfium.raw.FPDF_PAGEOBJ_PATH:
            paths += _path(drawn.raw, matrix, clip, anchor)
        elif drawn.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE:
            pictures += _picture(drawn, matrix, clip)
    return Drawing(
        os.fspath(path), page, area, tuple(texts), tuple(paths), tuple(pictures)
    )


def render_pdf(drawing: Drawing, scale: float = PREVIEW_SCALE) -> Image:
    """Draw the part of a page that was read as an image, as for a preview.

    Args:
        drawing: What is drawn on the page, as ``load_pdf`` returns it.
        scale: Pixels per point.

    Returns:
        A ``(height, width, 3)`` ``uint8`` array of red, green and blue
        levels, whose bottom-left corner is that of the drawing's box.
    """
    sheet = _pdfium().PdfDocument(drawing.path)[drawing.page - 1]
    left, bottom, right, top = _edges(sheet.get_bbox())
    edges = drawing.box
    crop = (edges[0] - left, edges[1] - bottom, right - edges[2], top - edges[3])
    bitmap = sheet.render(scale=scale, crop=crop, rev_byteorder=True)
    return np.array(bitmap.to_numpy()[..., :3], dtype=np.uint8)


def remove_pdf_grid(drawing: Drawing) -> Drawing:
    """Paint over long horizontal and vertical lines in the images on a page.

    Grid lines drawn as paths are left out of the curves anyway. Those in
    images are painted over as ``remove_grid`` paints them over.

    Args:
        drawing: What is drawn on the page, as ``load_pdf`` returns it.

    Returns:
        The drawing with grid lines painted over in its images.
    """
    pictures = tuple(
        replace(picture, image=remove_grid(picture.image))
        for picture in drawing.pictures
    )
    return replace(drawing, pictures=pictures)


def read_pdf_axes(drawing: Drawing) -> Axes:
    """Read the ticks and titles of a plot's axes from the text and lines on a page.

    The tick labels are found and read as ``read_axes`` finds and reads them
    in an image, but from the text that the PDF holds, so they need no OCR,
    and superscripts are known from the sizes and heights of the characters.
    The x-axis line is the lowest straight horizontal line above the x-axis
    tick labels that runs along ``_LINE_COVER`` of their span, and the y-axis
    line the leftmost vertical line right of the y-axis tick labels that runs
    along theirs. Without one, its place is taken to be just beyond the tick
    labels and their tick marks, as in an image.

    The ticks lie at the tick marks, which are straight lines across an axis
    that reach to within ``_MARK_REACH`` label heights of it, such as grid
    lines, so they are exact. Each label is aligned with its tick by its
    middle, or, if the ends at one side of the labels lie nearer the marks,
    as those of rotated labels do, by that end, and its tick is the mark
    nearest that point, within ``_SNAP_REACH`` of the tick spacing; a label
    without a mark that near keeps its tick at its middle.

    The titles are found as ``read_axes`` finds them, from the text beyond
    the tick labels. A y-axis title written upwards is read along its lines,
    as written.

    Args:
        drawing: What is drawn on the page, as ``load_pdf`` returns it.

    Returns:
        The axes, with a calibration fitted to their ticks, whose positions
        are in points from the bottom-left corner of the page.

    Raises:
        ValueError: If an axis has too few tick labels in line, an offset to
            add to its tick values, or the plot has a second y-axis, on the
            right, with other values.
    """
    texts = list(drawing.texts)
    if _decimal_comma(texts):
        texts = [_with_decimal_points(text) for text in texts]
    numbers = [text for text in texts if _number(_reading(text))]
    x_row = _lowest_row(numbers)
    above = [text for text in numbers if text.y > max(t.top for t in x_row)]
    y_column = _leftmost_column(above)
    height = float(np.median([text.top - text.bottom for text in y_column]))
    x_labels, x_values, x_log = _in_line(x_row, lambda t: t.x, "x", None, height)
    y_labels, y_values, y_log = _in_line(y_column, lambda t: t.y, "y", None, height)

    lines = np.vstack(
        [np.empty((0, 4)), *(path.segments for path in drawing.paths if path.stroke)]
    )
    across = lines[:, [1, 0, 3, 2]]  # x and y swapped, so the y-axis runs along x
    x_axis = _axis_line(lines, [(t.x, t.top) for t in x_labels])
    y_axis = _axis_line(across, [(t.y, t.right) for t in y_labels])
    origin = (
        y_axis[0]
        if y_axis
        else _beyond([t.right for t in y_labels], x_labels[0].x, height),
        x_axis[0]
        if x_axis
        else _beyond([t.top for t in x_labels], y_labels[0].y, height),
    )
    x_ends = [
        [t.x for t in x_labels],
        [t.left for t in x_labels],
        [t.right for t in x_labels],
    ]
    y_ends = [
        [t.y for t in y_labels],
        [t.bottom for t in y_labels],
        [t.top for t in y_labels],
    ]
    x_positions = _ticks(lines, origin[1], x_ends, height)
    y_positions = _ticks(across, origin[0], y_ends, height)

    words = _without_second_lines(
        [text for text in texts if text not in {*x_labels, *y_labels}], x_labels
    )
    x_scale, y_scale = _scales(words, origin, x_labels, y_labels)
    words = [word for word in words if not _SCALE.fullmatch(word.plain)]
    end = x_axis[1] if x_axis else max(t.right for t in x_labels)
    top = y_axis[1] if y_axis else max(t.top for t in y_labels)
    x_span = (origin[0], max(end, x_labels[-1].right))
    y_span = (origin[1], max(top, y_labels[-1].top))
    x_title, x_factor = _factor(_x_title(words, x_labels, x_span))
    y_title, y_factor = _factor(_y_title(words, y_labels, y_span))
    x_ticks = tuple(
        (float(position), value * x_scale * x_factor)
        for position, value in zip(x_positions, x_values, strict=True)
    )
    y_ticks = tuple(
        (float(position), value * y_scale * y_factor)
        for position, value in zip(y_positions, y_values, strict=True)
    )
    calibration = Calibration.from_ticks(
        x_ticks, y_ticks, origin, x_log=x_log, y_log=y_log
    )
    _check_one_y_axis(numbers, end, calibration, None, height)
    return Axes(calibration, x_ticks, y_ticks, x_title, y_title)


def find_pdf_colours(
    drawing: Drawing, count: int = 1, origin: Point = (0, 0)
) -> list[Colour]:
    """Find the colours of the curves in a plot on a page of a PDF.

    The curves, and series of markers, are told apart by colour; see
    ``trace_pdf_curves`` for which paths, markers and images make them.

    Args:
        drawing: What is drawn on the page, as ``load_pdf`` returns it.
        count: How many curves to find.
        origin: Position where the axes cross, in points.

    Returns:
        The colours of the ``count`` longest curves, longest first, by the
        length of the lines through their points.

    Raises:
        ValueError: If there are fewer than ``count`` curves.
    """
    series = _series(drawing, origin)
    if len(series) < count:
        msg = f"found only {len(series)} of {count} curves"
        raise ValueError(msg if series else "no curve found")
    return [curve.colour for curve in series[:count]]


def trace_pdf_curves(
    drawing: Drawing, colours: Sequence[Colour], origin: Point = (0, 0)
) -> list[npt.NDArray[np.float64]]:
    """Find the curves of the given colours in a plot on a page of a PDF.

    The curve of a colour is the points that the paths stroked in it, and
    not filled, pass through, in the order they are drawn, without the
    control points of curved pieces. A subpath whose points all lie along
    one straight horizontal or vertical line, such as an axis line, a tick
    mark, a grid line or the line in a legend's key, is left out, and so is
    a closed one, such as a frame. Points left of or below the axes, or
    hidden by a clip, as those beyond the limits of the axes are, are left
    out too. Curves drawn in images, as plots of many points often are, are
    traced as ``trace_curves`` traces them, from where the axes cross, if
    they cross the image, and placed on the page where the image is shown.

    A colour drawn as markers, as in a scatter plot or a line with markers,
    gives the points that its markers mark instead, in the order they are
    drawn, and its lines are left out, as a line through markers passes
    through the same points. A marker is a shape at most ``_MARKER_SIZE``
    points across, other than a straight horizontal or vertical line, such
    as the cap of an error bar, drawn at least twice; nor is an arrowhead
    lying wholly beyond one end of a straight line whose other end lies at a
    marker, as one-sided error bars have. A form holding it, as
    matplotlib draws markers, marks the point where the form's origin lies,
    if within the shape; otherwise the shape must be filled or closed, as a
    piece of a curve is not, and marks the middle of its box. Paths drawn
    one after another that mark the same point, such as the inside and the
    outline of a marker drawn apart, make one marker. Its colour is its
    fill, or its outline's colour if it has no fill or a white one, as open
    markers have. Markers in a legend are left out; see
    ``find_pdf_markers``.

    Each colour picks the curve whose colour is nearest, up to 96 levels
    away.

    Args:
        drawing: What is drawn on the page, as ``load_pdf`` returns it.
        colours: The colour of each curve, such as ``find_pdf_colours``
            returns.
        origin: Position where the axes cross, in points.

    Returns:
        One ``(n, 2)`` array per colour, of the ``(x, y)`` positions in points
        of the points the curve passes through, or that the markers mark.
        Those of curves traced in images are those of pixel centres, sorted
        by x, then y.

    Raises:
        ValueError: If no curve has one of the colours, or two of them match
            the same curve.
    """
    series = _series(drawing, origin)
    indices = [_nearest(series, colour) for colour in colours]
    for (first, index), (second, other) in itertools.combinations(
        zip(colours, indices, strict=True), 2
    ):
        if index == other:
            msg = (
                f"colours #{bytes(first).hex()} and #{bytes(second).hex()} both "
                f"match the curve of colour #{bytes(series[index].colour).hex()}"
            )
            raise ValueError(msg)
    return [series[index].points for index in indices]


def find_pdf_markers(
    drawing: Drawing, keys: Sequence[Point], origin: Point = (0, 0)
) -> list[npt.NDArray[np.float64]]:
    """Find the markers of each series in a plot on a page of a PDF, by their shapes.

    Each key is the position of a series' marker in the legend, or of any of
    its markers. The marker nearest it, within ``_KEY_REACH`` points, picks
    the series: the markers of the same shape, whose outlines lie within
    ``_SAME_SHAPE`` of one another, and the same colours inside and out. See
    ``trace_pdf_curves`` for which shapes are markers and what they mark.

    Markers in a legend are left out: those with a label beside them, a
    text right of and above the axes whose middle lies within half its
    height of the marker's centre heightwise, and whose nearer end lies
    within ``_LABEL_REACH`` of its heights of it. So are markers left of or
    below the axes, or hidden by a clip.

    Args:
        drawing: What is drawn on the page, as ``load_pdf`` returns it.
        keys: The position of a marker of each series, such as that in its
            legend key, in points.
        origin: Position where the axes cross, in points.

    Returns:
        One ``(n, 2)`` array per key, of the ``(x, y)`` positions in points
        of the points that the markers mark, in the order they are drawn. A
        series whose markers are all hidden gives an empty array.

    Raises:
        ValueError: If there is no marker near a key, or two keys pick
            markers of the same series.
    """
    markers = _markers(drawing)
    picked = [_picked(markers, key) for key in keys]
    for (first, marker), (second, other) in itertools.combinations(
        zip(keys, picked, strict=True), 2
    ):
        if _same_style(marker, other):
            msg = (
                f"the markers near {first[0]:g},{first[1]:g} and "
                f"{second[0]:g},{second[1]:g} belong to the same series"
            )
            raise ValueError(msg)
    plotted = [
        marker for marker in markers if not _in_legend(marker, drawing.texts, origin)
    ]
    return [
        _dots(
            key.colour,
            [marker for marker in plotted if _same_style(marker, key)],
            origin,
        ).points
        for key in picked
    ]


def find_pdf_error_bars(
    drawing: Drawing, points: Sequence[npt.NDArray[np.float64]]
) -> list[ErrorBars]:
    """Find the error bars through the points of each series in a plot on a page.

    Error bars are straight vertical or horizontal lines, stroked but not
    filled, that pass through a point, to within ``_ON_BAR`` points across
    and between their ends; matplotlib draws each as a subpath of its own.
    Their ends are the bounds, exactly as drawn, even beyond the limits of
    the axes. A one-sided bar, as matplotlib draws with ``lolims`` or
    ``uplims``, ends at its point. Caps across the ends of bars, and the
    arrowheads of one-sided bars, add nothing.

    Bars are matched to points by where they lie, not by colour, so bars
    drawn in another colour than their markers are found too. Colour only
    settles which point a bar belongs to where it is in doubt, as where
    series share an x: a bar through several points belongs to those with a
    marker drawn in its colour, if any, and a point with several bars one
    way takes the one through it alone, or else the one in its marker's
    colour; if that leaves more than one, the point has no bar that way.
    Lines that run from one edge of their clip to the other, as grid lines
    and reference lines do across the axes, are not bars.

    Args:
        drawing: What is drawn on the page, as ``load_pdf`` returns it.
        points: The points of each series, in points, such as
            ``trace_pdf_curves`` or ``find_pdf_markers`` returns.

    Returns:
        The error bars of each series, through the points given, with the
        ends of their bars in points, or NaN where a point has no bar that
        way.
    """
    every = np.vstack([np.empty((0, 2)), *points])
    low, high = np.full_like(every, np.nan), np.full_like(every, np.nan)
    markers = _markers(drawing)
    centres = np.array([marker.centre for marker in markers]).reshape(-1, 2)
    near = np.abs(every[:, np.newaxis] - centres).max(axis=2) <= _ON_BAR
    colours = [
        markers[int(np.argmax(row))].colour if row.any() else None for row in near
    ]
    for along, (lines, line_colours) in enumerate(_straight_lines(drawing)):
        chosen = _bars_through(every, colours, lines, line_colours, along)
        low[chosen >= 0, along] = lines[chosen[chosen >= 0], 1]
        high[chosen >= 0, along] = lines[chosen[chosen >= 0], 2]
    ends = np.cumsum([0, *(len(series) for series in points)])
    return [
        ErrorBars(every[start:end], low[start:end], high[start:end])
        for start, end in itertools.pairwise(ends)
    ]


def _texts(sheet: Any) -> list[_Text]:
    """Return the lines of text on a page, joining characters next to one another.

    Characters run on a line in the order they are drawn, as the text of a
    label or title is, and those that PDFium makes up, such as line breaks
    between characters placed one by one, are left out.
    """
    raw = _pdfium().raw
    textpage = sheet.get_textpage()
    chars = [
        _char(textpage.raw, index)
        for index in range(textpage.count_chars())
        if not raw.FPDFText_IsGenerated(textpage.raw, index)
    ]
    texts = [_joined(run) for run in _runs(chars)]
    return [text for text in texts if text.text]


def _char(textpage: Any, index: int) -> _Char:
    """Return a character of a page's text, as ``_Char`` describes it.

    The angle and font size are those of the character's matrix, which takes
    in any scaling and turning of its text.
    """
    raw = _pdfium().raw
    matrix = raw.FS_MATRIX()
    raw.FPDFText_GetMatrix(textpage, index, matrix)
    angle = math.degrees(math.atan2(matrix.b, matrix.a))
    scale = math.sqrt(abs(matrix.a * matrix.d - matrix.b * matrix.c))
    x, y = ctypes.c_double(), ctypes.c_double()
    raw.FPDFText_GetCharOrigin(textpage, index, x, y)
    left, right, bottom, top = (ctypes.c_double() for _ in range(4))
    raw.FPDFText_GetCharBox(textpage, index, left, right, bottom, top)
    ink = (left.value, bottom.value, right.value, top.value)
    loose = raw.FS_RECTF()  # the advance along the text, and the font's height
    raw.FPDFText_GetLooseCharBox(textpage, index, loose)
    box, advance = ink, None
    if abs(math.sin(math.radians(angle))) < _FLAT:  # across
        box = (loose.left, ink[1], loose.right, ink[3])
        advance = loose.right - loose.left
    elif abs(math.cos(math.radians(angle))) < _FLAT:  # upwards or downwards
        box = (ink[0], loose.bottom, ink[2], loose.top)
        advance = loose.top - loose.bottom
    return _Char(
        chr(raw.FPDFText_GetUnicode(textpage, index)),
        box,
        (x.value, y.value),
        raw.FPDFText_GetFontSize(textpage, index) * scale,
        angle,
        advance,
    )


def _runs(chars: Sequence[_Char]) -> Iterator[list[_Char]]:
    """Split characters into runs of those next to one another, by ``_next_to``."""
    run: list[_Char] = []
    for char in chars:
        if run and not _next_to(run, char):
            yield run
            run = []
        run.append(char)
    if run:
        yield run


def _next_to(run: Sequence[_Char], char: _Char) -> bool:
    """Return whether a character carries on a run of characters along its line.

    It does if it runs the same way as the last, starts less than ``_GAP``
    font sizes from where the last ends, or, in slanted text, after its start
    and within that of a font size beyond it, and lies within ``_BASELINE``
    of the line's font size of its line, which superscripts do.
    """
    last, base = run[-1], _base(run)
    if abs((char.angle - last.angle + 180) % 360 - 180) >= _TURNED:
        return False
    along, up = _directions(base.angle)
    size = max(char.size, last.size)
    step = float(np.dot(char.origin, along) - np.dot(last.origin, along))
    if last.advance is None:
        near = 0 < step < (1 + _GAP) * size
    else:
        near = abs(step - last.advance) < _GAP * size
    rise = float(np.dot(char.origin, up) - np.dot(base.origin, up))
    return near and abs(rise) < _BASELINE * base.size


def _joined(run: Sequence[_Char]) -> _Text:
    """Return a run of characters as a text, with its box and slant.

    A space parts characters ``_SPACE`` font sizes apart or more, unless one
    is a space already, and small characters raised or lowered from the
    line, by ``_SMALL`` and ``_RISE``, are written as superscripts or
    subscripts, where Unicode has them.
    """
    base = _base(run)
    along, up = _directions(base.angle)
    parts = [run[0].text]
    for last, char in itertools.pairwise(run):
        gap = np.dot(char.origin, along) - np.dot(last.origin, along)
        if (
            last.advance is not None
            and gap - last.advance >= _SPACE * max(char.size, last.size)
            and not (last.text.isspace() or char.text.isspace())
        ):
            parts.append(" ")
        rise = float(np.dot(char.origin, up) - np.dot(base.origin, up))
        text = char.text
        if char.size <= _SMALL * base.size and rise >= _RISE * base.size:
            text = text.translate(_SUPERSCRIPTS)
        elif char.size <= _SMALL * base.size and rise <= -_RISE * base.size:
            text = text.translate(_SUBSCRIPTS)
        parts.append(text)
    inked = [char.box for char in run if not char.text.isspace()] or [run[0].box]
    return _Text(
        _normal("".join(parts)),
        min(box[0] for box in inked),
        min(box[1] for box in inked),
        max(box[2] for box in inked),
        max(box[3] for box in inked),
        slant=base.angle,
    )


def _base(run: Sequence[_Char]) -> _Char:
    """Return the first of the largest characters of a run, which sets its line."""
    return max(run, key=lambda char: char.size)


def _directions(
    angle: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return unit vectors along and up from text that runs at an angle, in degrees."""
    turn = math.radians(angle)
    return (
        np.array([math.cos(turn), math.sin(turn)]),
        np.array([-math.sin(turn), math.cos(turn)]),
    )


def _objects(
    sheet: Any, area: Box
) -> Iterator[tuple[Any, npt.NDArray[np.float64], Box | None, Point | None]]:
    """Yield each object drawn on a page, its matrix, its box, and its form's origin.

    The matrix maps from the object's own space to the page's. The box is
    where ``area`` and the boxes around the object's clip and those of the
    forms that hold it overlap, or None if the object is hidden. The origin
    is where that of the form that holds the object lies on the page, or
    None if the page holds it. Forms hold the figures that documents take in
    from other PDF files, and markers drawn many times, and PDFium gives an
    object's clip in the space of the form that holds it.
    """
    forms: list[tuple[npt.NDArray[np.float64], Box | None]] = [(np.eye(3), area)]
    for drawn in sheet.get_objects():
        del forms[drawn.level + 1 :]  # the page, and the forms that hold it
        outer, shown = forms[-1]
        a, b, c, d, e, f = drawn.get_matrix().get()
        matrix = outer @ np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        clip = None if shown is None else _clip(drawn.raw, outer, shown)
        yield drawn, matrix, clip, _mapped(outer, (0, 0)) if drawn.level else None
        forms.append((matrix, clip))  # a form's objects come next, a level deeper


def _path(
    handle: Any,
    matrix: npt.NDArray[np.float64],
    clip: Box | None,
    anchor: Point | None,
) -> list[_Path]:
    """Return a path drawn on a page, or none if it is not shown.

    ``matrix`` maps from the path's own space to the page's, ``clip`` is the
    box it is shown in, or None if it is hidden, and ``anchor`` where the
    origin of the form that holds it lies, as ``_Path`` has it.
    """
    raw = _pdfium().raw
    fill, stroke = ctypes.c_int(), ctypes.c_int()
    raw.FPDFPath_GetDrawMode(handle, fill, stroke)
    stroke_colour = (
        _colour(raw.FPDFPageObj_GetStrokeColor, handle) if stroke.value else None
    )
    fill_colour = _colour(raw.FPDFPageObj_GetFillColor, handle) if fill.value else None
    subpaths, closed, segments = _outline(handle, matrix)
    points = np.vstack(subpaths)
    if (
        (stroke_colour is None and fill_colour is None)
        or clip is None
        or _overlap(clip, (*points.min(axis=0), *points.max(axis=0))) is None
    ):
        return []
    return [_Path(subpaths, closed, segments, stroke_colour, fill_colour, clip, anchor)]


def _outline(
    handle: Any, matrix: npt.NDArray[np.float64]
) -> tuple[
    tuple[npt.NDArray[np.float64], ...], tuple[bool, ...], npt.NDArray[np.float64]
]:
    """Return the subpaths of a path, whether each is closed, and its straight pieces.

    The subpaths, pieces and closed flags are those of ``_Path``. PDFium
    gives the points in the path's own space, which ``matrix`` maps to the
    page's, and each curved piece as three points: two control points and
    its end. Each subpath starts with a move.
    """
    raw = _pdfium().raw
    subpaths: list[list[Point]] = []
    closed: list[bool] = []
    segments: list[tuple[float, float, float, float]] = []
    x, y = ctypes.c_float(), ctypes.c_float()
    controls = 0
    for index in range(raw.FPDFPath_CountSegments(handle)):
        segment = raw.FPDFPath_GetPathSegment(handle, index)
        raw.FPDFPathSegment_GetPoint(segment, x, y)
        point = _mapped(matrix, (x.value, y.value))
        kind = raw.FPDFPathSegment_GetType(segment)
        if kind == raw.FPDF_SEGMENT_MOVETO:
            subpaths.append([point])
            closed.append(False)
        elif kind == raw.FPDF_SEGMENT_BEZIERTO:
            controls = (controls + 1) % 3
            if not controls:
                subpaths[-1].append(point)
        else:
            segments.append((*subpaths[-1][-1], *point))
            subpaths[-1].append(point)
        if raw.FPDFPathSegment_GetClose(segment):
            closed[-1] = True
            segments.append((*point, *subpaths[-1][0]))
    return (
        tuple(np.array(subpath, dtype=np.float64) for subpath in subpaths),
        tuple(closed),
        np.array(segments, dtype=np.float64).reshape(-1, 4),
    )


def _picture(
    drawn: Any, matrix: npt.NDArray[np.float64], clip: Box | None
) -> list[_Picture]:
    """Return an image drawn on a page, on white, or none if it is not shown.

    An image fills the square of side 1 in its own space, which ``matrix``
    maps to the page's, and ``clip`` is the box it is shown in, or None if it
    is hidden. PDFium draws it as it is shown, with any mask, at about as
    many pixels as it has.
    """
    corners = matrix @ np.array([[0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]])
    box = (*corners[:2].min(axis=1), *corners[:2].max(axis=1))
    if clip is None or _overlap(box, clip) is None:
        return []
    shown = drawn.get_bitmap(render=True).to_numpy().astype(np.float64)
    alpha = shown[..., 3:] / 255  # PDFium draws blue, green, red and alpha
    on_white = shown[..., 2::-1] * alpha + 255 * (1 - alpha)
    return [_Picture(np.round(on_white).astype(np.uint8), box, clip)]


def _colour(getter: Callable[..., int], handle: Any) -> Colour | None:
    """Return an object's stroke or fill colour, or None if it is transparent."""
    red, green, blue, alpha = (ctypes.c_uint() for _ in range(4))
    getter(handle, red, green, blue, alpha)
    if not alpha.value:
        return None
    return red.value, green.value, blue.value


def _clip(handle: Any, outer: npt.NDArray[np.float64], area: Box) -> Box | None:
    """Return the box that an object is shown in, or None if it is hidden.

    That is where ``area`` and the box around each path of the object's clip
    overlap. PDFium gives the clip in the space of the form that holds the
    object, which ``outer`` maps to the page's, and leaves out a clip that
    the object lies within.
    """
    raw = _pdfium().raw
    clip = raw.FPDFPageObj_GetClipPath(handle)
    x, y = ctypes.c_float(), ctypes.c_float()
    boxes = [area]
    for index in range(raw.FPDFClipPath_CountPaths(clip)):
        points = []
        for number in range(raw.FPDFClipPath_CountPathSegments(clip, index)):
            segment = raw.FPDFClipPath_GetPathSegment(clip, index, number)
            raw.FPDFPathSegment_GetPoint(segment, x, y)
            points.append(_mapped(outer, (x.value, y.value)))
        edges = np.array(points)
        boxes.append((*edges.min(axis=0), *edges.max(axis=0)))
    return _overlap(*boxes)


def _axis_line(lines: npt.NDArray[np.float64], labels: Sequence[Point]) -> Point | None:
    """Return where an axis line lies across its axis, and where it ends along it.

    With the axis along x, as the x-axis is and the y-axis is once x and y
    are swapped, the line is the lowest straight horizontal one at or above
    the median inner edge of the labels, which are given as their positions
    along the axis and their inner edges, that runs along ``_LINE_COVER`` of
    the labels' span. It ends at its right. An axis without such a line gives
    None.
    """
    first, last = labels[0][0], labels[-1][0]
    edge = float(np.median([inner for _, inner in labels]))
    flat = lines[np.abs(lines[:, 1] - lines[:, 3]) <= _FLAT]
    starts = np.minimum(flat[:, 0], flat[:, 2])
    ends = np.maximum(flat[:, 0], flat[:, 2])
    covered = np.minimum(ends, last) - np.maximum(starts, first)
    found = np.flatnonzero(
        (covered >= _LINE_COVER * (last - first)) & (flat[:, 1] >= edge)
    )
    if not len(found):
        return None
    lowest = found[np.argmin(flat[found, 1])]
    return float(flat[lowest, 1]), float(ends[lowest])


def _ticks(
    lines: npt.NDArray[np.float64],
    axis: float,
    anchors: Sequence[Sequence[float]],
    height: float,
) -> npt.NDArray[np.float64]:
    """Return where the tick of each label lies along its axis.

    With the axis along x, as for ``_axis_line``, the tick marks are the
    straight vertical lines that reach to within ``_MARK_REACH`` of a label's
    ``height`` of the axis, at ``axis``. The anchors are the points along the
    axis by which each label may be aligned with its tick: first its middle,
    then its ends. The labels centre on their ticks if their middles lie
    within ``_CENTRED`` label heights of the marks, by their median distance,
    as a label's box can stretch to take in a superscript, and a log axis
    has minor tick marks that an end may lie nearer. Otherwise they are
    aligned by the ends that lie nearest the marks, as those of rotated
    labels do. Each tick lies at the mark nearest its anchor, if within
    ``_SNAP_REACH`` of the tick spacing, or else at the label's middle.
    """
    middles, *ends = (np.array(points, dtype=np.float64) for points in anchors)
    upright = lines[np.abs(lines[:, 0] - lines[:, 2]) <= _FLAT]
    reach = _MARK_REACH * height
    low = np.minimum(upright[:, 1], upright[:, 3]) - reach
    high = np.maximum(upright[:, 1], upright[:, 3]) + reach
    marks = np.unique(upright[(low <= axis) & (axis <= high), 0])
    if not len(marks):
        return middles

    def offsets(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        distances = np.abs(points[:, np.newaxis] - marks).min(axis=1)
        return np.asarray(distances, dtype=np.float64)

    best = middles
    if np.median(offsets(middles)) > _CENTRED * height:
        best = min(ends, key=lambda points: float(np.median(offsets(points))))
    nearest = marks[np.abs(best[:, np.newaxis] - marks).argmin(axis=1)]
    spacing = float(np.median(np.diff(middles)))
    return np.where(offsets(best) <= _SNAP_REACH * spacing, nearest, middles)


def _y_title(
    words: Sequence[_Text], y_labels: Sequence[_Text], span: tuple[float, float]
) -> str | None:
    """Return the lines of text nearest left of the y-axis tick labels.

    The title is centred within ``span``, along the axis. If the text nearest
    the labels is written upwards, the title is the lines written upwards,
    read from the leftmost, as they are written; if not, it is the lines
    written across, in line with the nearest.
    """
    left = min(label.left for label in y_labels)
    beside = [word for word in words if word.x < left and span[0] <= word.y <= span[1]]
    if not beside:
        return None
    nearest = max(beside, key=lambda word: word.right)
    if abs(nearest.slant - 90) < _TURNED:
        upwards = [word for word in beside if abs(word.slant - 90) < _TURNED]
        # Transposed, the lines of a title written upwards run from the last up.
        return _title(_lines(map(_transposed, upwards))[::-1])
    column = [
        word for word in beside if abs(word.x - nearest.x) < nearest.right - nearest.x
    ]
    return _title(_lines(column))


def _series(drawing: Drawing, origin: Point) -> list[_Series]:
    """Return the curves in a plot on a page, longest first, by colour.

    See ``trace_pdf_curves`` for which paths, markers and points make the
    curves. Curves in a colour with no points shown are left out.
    """
    marked: dict[Colour, list[_Marker]] = {}
    for marker in _markers(drawing):
        if not _in_legend(marker, drawing.texts, origin):
            marked.setdefault(marker.colour, []).append(marker)
    parts: dict[Colour, list[tuple[npt.NDArray[np.float64], float]]] = {}
    for path in drawing.paths:
        if path.stroke is None or path.fill is not None or path.stroke in marked:
            continue
        for points, closed in zip(path.subpaths, path.closed, strict=True):
            if closed or np.ptp(points, axis=0).min() <= _FLAT:
                continue
            shown = _shown(points, path.clip, origin)
            parts.setdefault(path.stroke, []).append(
                (points[shown], _length(points, shown))
            )
    series = [
        _Series(
            colour,
            np.vstack([points for points, _ in pieces]),
            sum(length for _, length in pieces),
        )
        for colour, pieces in parts.items()
    ]
    series += [_dots(colour, markers, origin) for colour, markers in marked.items()]
    series += _traced(drawing, origin)
    return sorted(
        (curve for curve in series if len(curve.points)),
        key=lambda curve: -curve.length,
    )


def _traced(drawing: Drawing, origin: Point) -> list[_Series]:
    """Return the curves traced in the images on a page, by colour.

    Each image that reaches beyond where the axes cross is traced as
    ``trace_curves`` traces an image of a plot, from where the axes cross in
    it, or from its edges if they cross outside it or within a pixel of its
    edges, as axes drawn as lines around an image of curves do. Pixel
    positions are placed on the page where the image is shown.
    """
    found = []
    for picture in drawing.pictures:
        left, bottom, right, top = picture.box
        if right <= origin[0] or top <= origin[1]:
            continue
        height, width = picture.image.shape[:2]
        scale = np.array([(right - left) / width, (top - bottom) / height])
        corner = np.array([left, bottom])
        inner = (np.array(origin) - corner) / scale
        inner[inner < 1] = -AXIS_MARGIN  # so no pixels are left out for the axes
        start = (float(inner[0]), float(inner[1]))
        colours = _curve_colours(picture.image, start)
        for colour, curve in zip(
            colours, trace_curves(picture.image, colours, start), strict=True
        ):
            points = corner + curve * scale
            shown = _shown(points, picture.clip, origin)
            found.append(_Series(colour, points[shown], _length(points, shown)))
    return found


def _markers(drawing: Drawing) -> list[_Marker]:
    """Return the markers drawn on a page, in the order they are drawn.

    See ``trace_pdf_curves`` for which shapes are markers and which points
    they mark.
    """
    markers: list[_Marker] = []
    last = -1  # the index of the path that gave the last marker
    lines = _straight_lines(drawing)
    centres = [m.centre for m in map(_marker, drawing.paths) if m is not None]
    for index, path in enumerate(drawing.paths):
        marker = _marker(path)
        if marker is None or _arrowhead(path, lines, centres):
            continue
        if last == index - 1 and markers[-1].centre == marker.centre:
            first = markers.pop()  # one marker, drawn in two paths
            marker = _Marker(
                marker.centre,
                np.vstack([first.outline, marker.outline]),
                first.stroke or marker.stroke,
                first.fill or marker.fill,
                first.clip,
            )
        markers.append(marker)
        last = index
    shapes: list[list[int]] = []  # the indices of the markers of each shape
    for index, marker in enumerate(markers):
        for shape in shapes:
            if _same_shape(markers[shape[0]].outline, marker.outline):
                shape.append(index)
                break
        else:
            shapes.append([index])
    repeated = sorted(index for shape in shapes if len(shape) > 1 for index in shape)
    return [markers[index] for index in repeated]


def _marker(path: _Path) -> _Marker | None:
    """Return the marker that a path draws, or None if it is not one.

    See ``trace_pdf_curves`` for which shapes are markers, leaving aside
    whether they are drawn more than once, and which points they mark.
    """
    points = np.vstack(path.subpaths)
    low, high = points.min(axis=0), points.max(axis=0)
    if (high - low).max() > _MARKER_SIZE or (high - low).min() <= _FLAT:
        return None
    middle = (low + high) / 2
    if path.anchor is not None and _holds((*low, *high), path.anchor):
        centre = path.anchor
    elif path.fill is not None or any(path.closed):
        centre = (float(middle[0]), float(middle[1]))
    else:
        return None
    # With the middles of its straight pieces, a square's outline is not a cross's.
    outline = np.vstack([points, (path.segments[:, :2] + path.segments[:, 2:]) / 2])
    return _Marker(centre, outline - middle, path.stroke, path.fill, path.clip)


def _straight_lines(
    drawing: Drawing,
) -> list[tuple[npt.NDArray[np.float64], list[Colour]]]:
    """Return the straight lines on a page that may be error bars, and their colours.

    The lines run along x, then along y: those level, then those upright.
    Each way, a row of each line holds where it lies across that way, and
    where it starts and ends along it. See ``find_pdf_error_bars`` for which
    lines may be error bars.
    """
    found: list[list[tuple[float, float, float]]] = [[], []]
    colours: list[list[Colour]] = [[], []]
    for path in drawing.paths:
        if path.stroke is None or path.fill is not None:
            continue
        for points, closed in zip(path.subpaths, path.closed, strict=True):
            spans = np.ptp(points, axis=0)
            if closed or np.count_nonzero(spans <= _FLAT) != 1:
                continue
            along = int(np.argmax(spans))
            start, end = points[:, along].min(), points[:, along].max()
            clip = path.clip[along], path.clip[along + 2]
            if start <= clip[0] + _SLACK and end >= clip[1] - _SLACK:
                continue  # across the axes, as a grid line
            found[along].append((float(points[0, 1 - along]), start, end))
            colours[along].append(path.stroke)
    return [
        (np.array(lines, dtype=np.float64).reshape(-1, 3), way)
        for lines, way in zip(found, colours, strict=True)
    ]


def _bars_through(
    points: npt.NDArray[np.float64],
    colours: Sequence[Colour | None],
    lines: npt.NDArray[np.float64],
    line_colours: Sequence[Colour],
    along: int,
) -> npt.NDArray[np.intp]:
    """Return the index of the error bar through each point one way, or -1 if none.

    ``lines`` are those along x, if ``along`` is 0, or y, as
    ``_straight_lines`` gives them, and ``colours`` those of the markers at
    the points, or None. See ``find_pdf_error_bars`` for which bar a point
    takes.
    """
    if not len(lines):
        return np.full(len(points), -1, dtype=np.intp)
    on = (
        (np.abs(points[:, 1 - along, np.newaxis] - lines[:, 0]) <= _ON_BAR)
        & (lines[:, 1] - _ON_BAR <= points[:, along, np.newaxis])
        & (points[:, along, np.newaxis] <= lines[:, 2] + _ON_BAR)
    )
    same = np.array(
        [[colour == other for other in line_colours] for colour in colours],
        dtype=np.bool_,
    ).reshape(on.shape)
    # A bar through several points keeps those of its colour, if any.
    on = np.where((on & same).any(axis=0), on & same, on)
    # A point with several bars keeps those through it alone, or of its colour.
    alone = on & (np.count_nonzero(on, axis=0) == 1)
    on = np.where(alone.any(axis=1, keepdims=True), alone, on)
    on = np.where((on & same).any(axis=1, keepdims=True), on & same, on)
    return np.where(np.count_nonzero(on, axis=1) == 1, on.argmax(axis=1), -1)


def _arrowhead(
    path: _Path,
    lines: Sequence[tuple[npt.NDArray[np.float64], list[Colour]]],
    centres: Sequence[Point],
) -> bool:
    """Return whether a path is the arrowhead of a one-sided error bar.

    That is a shape in a form whose origin lies at an end of a straight line
    that may be an error bar, as ``_straight_lines`` gives them, and which
    lies wholly beyond that end, while the line's other end lies at the
    centre of a marker, one of ``centres``: matplotlib draws the arrowheads
    of bars with ``lolims`` or ``uplims`` so, from the points they bound.
    """
    if path.anchor is None:
        return False
    points = np.vstack(path.subpaths)
    anchor = np.array(path.anchor)
    marked = np.array(centres, dtype=np.float64).reshape(-1, 2)
    for along, (found, _) in enumerate(lines):
        at = found[np.abs(found[:, 0] - anchor[1 - along]) <= _ON_BAR]
        start, end = points[:, along].min(), points[:, along].max()
        past_end = (np.abs(at[:, 2] - anchor[along]) <= _ON_BAR) & (
            start >= at[:, 2] - _ON_BAR
        )
        past_start = (np.abs(at[:, 1] - anchor[along]) <= _ON_BAR) & (
            end <= at[:, 1] + _ON_BAR
        )
        tails = np.empty((len(at), 2))
        tails[:, 1 - along] = anchor[1 - along]
        tails[:, along] = np.where(past_end, at[:, 1], at[:, 2])
        tails = tails[past_end | past_start]
        if np.any(np.abs(tails[:, np.newaxis] - marked).max(axis=2) <= _ON_BAR):
            return True
    return False


def _same_shape(
    first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]
) -> bool:
    """Return whether each point of two outlines lies near one of the other's.

    Near is within ``_SAME_SHAPE`` points across and along. Outlines whose
    widths or heights differ by more than that twice over are not compared
    point by point, which is slower.
    """
    if np.abs(np.ptp(first, axis=0) - np.ptp(second, axis=0)).max() > 2 * _SAME_SHAPE:
        return False
    apart = np.abs(first[:, np.newaxis] - second).max(axis=2)
    return bool(max(apart.min(axis=0).max(), apart.min(axis=1).max()) <= _SAME_SHAPE)


def _same_style(first: _Marker, second: _Marker) -> bool:
    """Return whether two markers have the same shape and colours."""
    return (
        first.fill == second.fill
        and first.stroke == second.stroke
        and _same_shape(first.outline, second.outline)
    )


def _in_legend(marker: _Marker, texts: Sequence[_Text], origin: Point) -> bool:
    """Return whether a marker is a legend's key, with its label beside it.

    See ``find_pdf_markers`` for where the label lies.
    """
    x, y = marker.centre
    for text in texts:
        height = text.top - text.bottom
        gap = text.left - x if text.x > x else x - text.right
        if (
            text.x >= origin[0]
            and text.y >= origin[1]
            and abs(text.y - y) <= height / 2
            and 0 <= gap <= _LABEL_REACH * height
        ):
            return True
    return False


def _picked(markers: Sequence[_Marker], key: Point) -> _Marker:
    """Return the marker nearest a key, within ``_KEY_REACH`` points."""
    distances = [math.dist(marker.centre, key) for marker in markers]
    if not distances or min(distances) > _KEY_REACH:
        msg = f"found no marker near {key[0]:g},{key[1]:g}"
        raise ValueError(msg)
    return markers[int(np.argmin(distances))]


def _dots(colour: Colour, markers: Sequence[_Marker], origin: Point) -> _Series:
    """Return the points that markers mark, those shown, as a series of a colour."""
    points = np.array([marker.centre for marker in markers], dtype=np.float64)
    clips = np.array([marker.clip for marker in markers], dtype=np.float64)
    points, clips = points.reshape(-1, 2), clips.reshape(-1, 4)
    shown = _shown(points, clips, origin)
    return _Series(colour, points[shown], _length(points, shown))


def _shown(
    points: npt.NDArray[np.float64],
    clip: Box | npt.NDArray[np.float64],
    origin: Point,
) -> npt.NDArray[np.bool_]:
    """Return which points are shown within a clip, and right of and above the axes.

    ``clip`` is one box for all the points, or a row of edges for each.
    """
    edges = np.asarray(clip, dtype=np.float64)
    low = np.maximum(origin, edges[..., :2]) - _SLACK
    high = edges[..., 2:] + _SLACK
    return np.all((low <= points) & (points <= high), axis=1)


def _length(points: npt.NDArray[np.float64], shown: npt.NDArray[np.bool_]) -> float:
    """Return the length of the lines between neighbouring points that are shown."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(steps[shown[1:] & shown[:-1]].sum())


def _nearest(series: Sequence[_Series], colour: Colour) -> int:
    """Return the index of the curve whose colour is nearest to ``colour``."""
    distances = [math.dist(colour, curve.colour) for curve in series]
    if not distances or min(distances) > _MATCH_DISTANCE:
        msg = f"no curve of colour #{bytes(colour).hex()} found"
        raise ValueError(msg)
    return int(np.argmin(distances))


def _mapped(matrix: npt.NDArray[np.float64], point: Point) -> Point:
    """Return a point mapped by a matrix that maps from one space to another."""
    x, y, _ = matrix @ (point[0], point[1], 1)
    return float(x), float(y)


def _edges(box: Sequence[float]) -> Box:
    """Return the left, bottom, right and top edges of a box, as floats."""
    left, bottom, right, top = (float(edge) for edge in box)
    return left, bottom, right, top


def _overlap(*boxes: Box) -> Box | None:
    """Return the box where boxes overlap, or None if they do not."""
    left, bottom = max(box[0] for box in boxes), max(box[1] for box in boxes)
    right, top = min(box[2] for box in boxes), min(box[3] for box in boxes)
    return (left, bottom, right, top) if left <= right and bottom <= top else None


def _holds(box: Box, point: Point) -> bool:
    """Return whether a point lies in a box."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _written(box: Box | None) -> str:
    """Return a box's edges as they are written for --box."""
    return ",".join(f"{edge:g}" for edge in box or ())
