"""Read the axes of a plot from the text in its image."""

import functools
import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from .calibration import Calibration, Point, Tick
from .digitize import Image, ink_mask

MIN_TICKS = 3
"""Fewest labelled ticks read on each axis, so that a misread label shows."""

_TICK_TOLERANCE = 0.1  # a tick label this fraction of the tick spacing off is misread
_AXIS_REACH = 3  # an axis line may lie this many pixels beyond the first tick
_TICK_GAP = 0.6  # tick marks and their padding take this many label heights
_LINE_COVER = 0.9  # an axis line has ink along this fraction of its labels' span
_MARK_LENGTH = 2  # a tick mark has ink this many pixels out from its axis line
_SNAP_REACH = 0.25  # a label lies within this fraction of the tick spacing of its tick
_SNAP_SLACK = 2  # labels centred within this many pixels of the tick marks stay put
_RAISED = 0.25  # a superscript sits this fraction of its label's height up or more
_RAISED_TICK = 1.4  # a label this many times taller than usual has a tick on top
_WRITTEN = 1.5  # words this much wider than tall run across
_LINE_SPACING = 1  # a title's lines lie this many line heights apart or less
_NUMBER = re.compile(  # 1, -2.5, 1,000, 2.5e-3 or 10%
    r"([+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)%?",
    re.IGNORECASE,
)
_TRAILING_DASH = re.compile(r"(.*\d%?)-+")  # a tick mark after a y-axis label
_POWER = re.compile(  # 10^2, 10^{2}, 10-2, 2x10^2 or 2^5: a power, as OCR reads it
    r"(?:(\d+(?:\.\d*)?|\.\d+)[x\u00d7])?"
    r"(?:(\d+(?:\.\d*)?|\.\d+)\^\{?|10(?=[+-]))([+-]?\d+)\}?"
)
_DIGITS = re.compile(r"\d+(?:\.\d+)?")  # 25, or 2^5 read without its superscript
_SCALE = re.compile(r"(?:1e|[x\u00d7]10\^?)([+-]?\d+)")  # 1e-5 or 10 to the -5
_OFFSET = re.compile(r"[+-].*\d(?:e|[x\u00d7]10).*")  # +1e3, added to the ticks
# Superscript signs and digits, and the degree and ordinal signs that OCR reads
# for a superscript zero.
_SUPERSCRIPT = re.compile(
    r"\^?([\u207a\u207b\u2070\u00b9\u00b2\u00b3\u2074-\u2079\u00b0\u00ba]+)"
)
_PLAIN = str.maketrans(  # minus signs, en dashes and superscripts
    "\u2212\u2013\u207b\u207a\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u00b0\u00ba",
    "---+012345678900",
)


@dataclass(frozen=True, slots=True)
class Axes:
    """The axes of a plot, read from the text in its image.

    Attributes:
        calibration: The map from pixel to data coordinates, fitted to the
            ticks, whose origin is where the axis lines cross, and which is
            logarithmic along each logarithmic axis.
        x_ticks: The pixel x-coordinate and value of each x-axis tick read,
            left to right.
        y_ticks: The pixel y-coordinate and value of each y-axis tick read,
            bottom to top.
        x_title: The title of the x-axis, or None if there is none.
        y_title: The title of the y-axis, or None if there is none.
    """

    calibration: Calibration
    x_ticks: tuple[Tick, ...]
    y_ticks: tuple[Tick, ...]
    x_title: str | None
    y_title: str | None


@dataclass(frozen=True, slots=True)
class _Text:
    """A line of text and its box, in pixels from the bottom-left corner."""

    text: str
    left: float
    bottom: float
    right: float
    top: float

    @property
    def x(self) -> float:
        """Return the middle of the box from left to right."""
        return (self.left + self.right) / 2

    @property
    def y(self) -> float:
        """Return the middle of the box from bottom to top."""
        return (self.bottom + self.top) / 2

    @property
    def plain(self) -> str:
        """Return the text without spaces or typographic minus signs.

        Superscripts become plain characters after a caret, so 10 to the
        power of -2 reads as 10^-2.
        """
        text = _SUPERSCRIPT.sub(r"^\1", self.text.replace(" ", ""))
        return text.translate(_PLAIN)


def read_axes(image: Image) -> Axes:
    """Read the ticks and titles of a plot's axes from the text in its image.

    The x-axis tick labels are the lowest row of at least ``MIN_TICKS``
    numbers, and the y-axis tick labels the leftmost column of at least
    ``MIN_TICKS`` numbers above them, aligned on the right. Labels whose values
    do not fall in line with the others, such as misread ones, are dropped.
    Each tick lies at the middle of its label. A scale factor printed where
    plotting libraries put one, such as ``1e-5`` above the y-axis or below
    the x-axis, multiplies the values of its axis's ticks.

    Numbers may be written as powers, such as 10^2, 2x10^3 or 2^5 with
    superscript exponents. OCR often misses superscripts, reading 2^5 as 25,
    so a label that reads as digits is a power if its last digit is drawn
    raised: the digits drawn full size are its base, the rest its exponent,
    and a raised minus sign before them makes the exponent negative. The base
    of an axis is the one most of its powers have, and a label that reads as
    that base followed by digits is a power of it if its last digit is drawn
    raised, or if most of the other labels on its axis that read as powers of
    it or as it followed by digits show that they are.
    An axis is logarithmic if more of its labels fall in line on a logarithmic
    scale than on a linear one.

    The axes cross where the axis lines run beside the tick labels. Without
    an axis line, its place is taken to be just beyond the tick labels and
    their tick marks, but no more than halfway to the first tick of the other
    axis, so a curve along that tick is kept. Where the labels are not
    centred on tick marks drawn outside an axis line, but end at them, as
    rotated labels do, each tick lies at the mark nearest the end of its
    label.

    The titles are the nearest lines of text beyond the tick labels, centred
    along their axes, from where the axes cross to the far end of the axis
    line or labels, so a title may be aligned with either end. A title runs
    on over the lines that follow, each within a line's height of the one
    before and aligned with it. A y-axis title is read as written upwards, as
    most are, unless its words are much wider than tall.

    The text is read with RapidOCR, which the ``ocr`` extra installs.

    Args:
        image: Colour image of a plot.

    Returns:
        The axes, with a calibration fitted to their ticks.

    Raises:
        ModuleNotFoundError: If the ``ocr`` extra is not installed.
        ValueError: If an axis has too few tick labels in line, an offset to
            add to its tick values, or the plot has a second y-axis, on the
            right, with other values.
    """
    texts = _read_text(image)
    ink = ink_mask(image)
    numbers = [
        text
        for text in texts
        if _NUMBER.fullmatch(_reading(text)) or _POWER.fullmatch(_reading(text))
    ]
    x_row = _lowest_row(numbers)
    above = [text for text in numbers if text.y > max(t.top for t in x_row)]
    y_column = _leftmost_column(above)
    # Tick marks run beside the y-axis labels, so they keep the usual height.
    height = float(np.median([text.top - text.bottom for text in y_column]))
    x_labels, x_values, x_log = _in_line(x_row, lambda t: t.x, "x", ink, height)
    y_labels, y_values, y_log = _in_line(y_column, lambda t: t.y, "y", ink, height)
    x_line, y_line = _axis_lines(ink > 0, x_labels, y_labels)
    x_pixels = np.array([label.x for label in x_labels])
    y_pixels = np.array([label.y for label in y_labels])
    if y_line is not None:  # the x-axis line, with the x-axis ticks
        marks = _x_marks(ink > 0, y_line)
        ends = [[label.left for label in x_labels], [label.right for label in x_labels]]
        x_pixels = _snap([x_pixels, *map(np.array, ends)], marks)
    if x_line is not None:
        marks = _y_marks(ink > 0, x_line)
        ends = [[label.bottom for label in y_labels], [label.top for label in y_labels]]
        y_pixels = _snap([y_pixels, *map(np.array, ends)], marks)
    origin = (
        x_line
        if x_line is not None
        else _beyond([t.right for t in y_labels], x_pixels[0], height),
        y_line
        if y_line is not None
        else _beyond([t.top for t in x_labels], y_pixels[0], height),
    )

    words = [text for text in texts if text not in {*x_labels, *y_labels}]
    x_scale, y_scale = _scales(words, origin, y_labels)
    words = [word for word in words if not _SCALE.fullmatch(word.plain)]
    x_ticks = tuple(
        (float(pixel), value * x_scale)
        for pixel, value in zip(x_pixels, x_values, strict=True)
    )
    y_ticks = tuple(
        (float(pixel), value * y_scale)
        for pixel, value in zip(y_pixels, y_values, strict=True)
    )
    calibration = Calibration.from_ticks(
        x_ticks, y_ticks, origin, x_log=x_log, y_log=y_log
    )
    if y_line is None:
        end = max(label.right for label in x_labels)
    else:
        row = round(ink.shape[0] - y_line - 0.5)
        end = _line_end(ink[row] > 0, origin[0])
    _check_one_y_axis(numbers, end, calibration, ink, height)
    top = max(label.top for label in y_labels)
    if x_line is not None:
        top = max(top, _line_end(ink[::-1, round(x_line - 0.5)] > 0, origin[1]))
    x_span = (origin[0], max(end, x_labels[-1].right))
    y_span = (origin[1], top)
    return Axes(
        calibration,
        x_ticks,
        y_ticks,
        _x_title(words, x_labels, x_span),
        _y_title(words, y_labels, y_span),
    )


@functools.cache
def _engine() -> Any:
    """Load the OCR engine, once, as it is slow to load."""
    try:
        # Imported here, as it is an optional dependency.
        from rapidocr import RapidOCR  # noqa: PLC0415
    except ModuleNotFoundError as error:
        msg = "reading the axes needs the ocr extra: pip install 'figrecover[ocr]'"
        raise ModuleNotFoundError(msg) from error
    return RapidOCR(params={"Global.log_level": "critical"})


def _read_text(image: Image) -> list[_Text]:
    """Find the lines of text in an image."""
    found = _engine()(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if found.boxes is None:
        return []
    height = image.shape[0]
    return [
        _Text(
            text.strip(),
            float(box[:, 0].min()),
            height - float(box[:, 1].max()),
            float(box[:, 0].max()),
            height - float(box[:, 1].min()),
        )
        for box, text in zip(found.boxes, found.txts, strict=True)
    ]


def _transposed(text: _Text) -> _Text:
    """Return a text with x and y swapped, as though the image were transposed.

    Transposed, text written upwards runs across from left to right, and the
    text nearest the y-axis tick labels on their left lies nearest them below.
    """
    return _Text(text.text, text.bottom, text.left, text.top, text.right)


def _lowest_row(numbers: Sequence[_Text]) -> list[_Text]:
    """Return the lowest row of at least ``MIN_TICKS`` numbers."""
    rows = [
        row for row in _groups(numbers, lambda text: text.y) if len(row) >= MIN_TICKS
    ]
    if not rows:
        msg = f"found no row of {MIN_TICKS} or more numbers to read as x-axis ticks"
        raise ValueError(msg)
    return rows[0]


def _leftmost_column(numbers: Sequence[_Text]) -> list[_Text]:
    """Return the leftmost column of at least ``MIN_TICKS`` right-aligned numbers."""
    columns = [
        column
        for column in _groups(numbers, lambda text: text.right)
        if len(column) >= MIN_TICKS
    ]
    if not columns:
        msg = f"found no column of {MIN_TICKS} or more numbers to read as y-axis ticks"
        raise ValueError(msg)
    return columns[0]


def _groups(texts: Sequence[_Text], key: Callable[[_Text], float]) -> list[list[_Text]]:
    """Split texts into groups whose keys are within half a line height, by key."""
    if not texts:
        return []
    tolerance = np.median([text.top - text.bottom for text in texts]) / 2
    groups: list[list[_Text]] = []
    previous = -math.inf
    for text in sorted(texts, key=key):
        if key(text) - previous > tolerance:
            groups.append([])
        groups[-1].append(text)
        previous = key(text)
    return groups


def _in_line(
    labels: Sequence[_Text],
    position: Callable[[_Text], float],
    name: str,
    ink: npt.NDArray[np.uint8],
    height: float,
) -> tuple[list[_Text], list[float], bool]:
    """Return the most tick labels whose values are in line with their positions.

    The values are taken on a linear scale and on a logarithmic one, and the
    scale on which more labels are in line wins, the linear one in a tie. The
    ink shows which labels have superscripts, and ``height`` is the usual
    height of a label.

    Returns:
        The labels in line, by position, their values, and whether the scale
        is logarithmic.
    """
    labels = sorted(labels, key=position)
    pixels = np.array([position(label) for label in labels])
    values = _values(labels, ink, height)
    tolerance = _TICK_TOLERANCE * np.median(np.diff(pixels))
    linear = _most_in_line(pixels, values, tolerance)
    exponents = np.log10(values, out=np.full_like(values, np.nan), where=values > 0)
    logarithmic = _most_in_line(pixels, exponents, tolerance)
    log = bool(np.count_nonzero(logarithmic) > np.count_nonzero(linear))
    best = logarithmic if log else linear
    if np.count_nonzero(best) < MIN_TICKS:
        msg = (
            f"only {np.count_nonzero(best)} of the {len(labels)} {name}-axis tick "
            "labels are in line, on a linear or a logarithmic scale"
        )
        raise ValueError(msg)
    return list(itertools.compress(labels, best)), values[best].tolist(), log


def _values(
    labels: Sequence[_Text], ink: npt.NDArray[np.uint8], height: float
) -> npt.NDArray[np.float64]:
    """Return the values of tick labels, which are numbers or powers.

    OCR often misses superscripts, reading 2^5 as 25. A label that reads as
    digits shows that it is a power if its last digit is drawn raised, and
    its digits drawn full size are then its base. The base of the axis is the
    one most labels shown as powers have, by their text or raised digits, or
    10 if none are. A label that reads as that base followed by digits is a
    power of it if its last digit is raised, or if most of the other labels
    that read as powers of the base or as it followed by digits show that
    they are powers. One misread label, such as 100 read with a degree sign
    for its last 0, so does not make an axis of 100, 101 and 102 logarithmic.
    """
    readings = [_reading(label, height) for label in labels]
    written = [_POWER.fullmatch(reading) for reading in readings]
    raised = [
        _exponent(ink, label) if _DIGITS.fullmatch(reading) else None
        for label, reading in zip(labels, readings, strict=True)
    ]
    bases = Counter(
        [match[2] or "10" for match in written if match]
        + [
            base
            for reading, exponent in zip(readings, raised, strict=True)
            if exponent and (base := _base(reading, exponent[0]))
        ]
    )
    base = bases.most_common(1)[0][0] if bases else "10"
    together = [
        re.fullmatch(rf"{re.escape(base)}(\d+)", reading) for reading in readings
    ]
    of_base = [bool(match and (match[2] or "10") == base) for match in written]
    shown = [
        power or bool(exponent and match)
        for power, exponent, match in zip(of_base, raised, together, strict=True)
    ]
    candidates = [
        index for index, match in enumerate(together) if of_base[index] or match
    ]
    values = []
    for index, reading in enumerate(readings):
        others = [shown[other] for other in candidates if other != index]
        mostly = 2 * sum(others) > len(others)
        exponent = raised[index]
        if match := written[index]:
            power = float(match[2] or 10) ** int(match[3])
            values.append(float(match[1] or 1) * power)
        elif (match := together[index]) and (exponent or mostly):
            sign = exponent[1] if exponent else 1
            values.append(float(base) ** (sign * int(match[1])))
        else:  # a number, less any % and thousands separators
            values.append(float(_NUMBER.sub(r"\1", reading).replace(",", "")))
    return np.array(values)


def _base(reading: str, length: int) -> str | None:
    """Return the first ``length`` digits of a reading, with any point among them.

    Returns None unless they make a number and more digits follow them.
    """
    match = re.match(rf"(?:\d\.?){{{length}}}(?=\d)", reading)
    return match[0] if match and _DIGITS.fullmatch(match[0]) else None


def _reading(label: _Text, height: float = math.inf) -> str:
    """Return a label's plain text without dashes that are tick marks.

    Where no axis line joins them, OCR can read a tick mark as a dash: after
    a y-axis label, or before an x-axis label, whose box the tick then makes
    ``_RAISED_TICK`` times taller than the usual ``height`` or more.
    """
    text = label.plain
    if match := _TRAILING_DASH.fullmatch(text):
        text = match[1]
    if text.startswith("-") and label.top - label.bottom >= _RAISED_TICK * height:
        text = text[1:]
    return text


def _exponent(ink: npt.NDArray[np.uint8], label: _Text) -> tuple[int, int] | None:
    """Return how long a label's base is and the sign of its exponent, if any.

    The characters are the pieces of ink in the label's box at least a third as
    tall as the tallest, which leaves out points, minus signs and flat tick
    marks. A character is raised if its bottom is ``_RAISED`` of the
    characters' height or more above the lowest bottom. The last character
    is an exponent if it is raised, and the base is then the characters that
    are not, leaving out tick marks above the label. The exponent is negative
    if a flat piece, at least twice as wide as tall, lies level with the
    exponent and before it, after the middle of the base's last character.

    Returns:
        The number of characters in the base and the sign of the exponent, or
        None if the label has no exponent.
    """
    height = ink.shape[0]
    box = ink[
        max(height - math.ceil(label.top), 0) : height - math.floor(label.bottom),
        max(math.floor(label.left), 0) : math.ceil(label.right),
    ]
    stats = cv2.connectedComponentsWithStats(box)[2][1:]  # the first is background
    if not len(stats):
        return None
    left, top, width, tall = stats[:, :4].T
    characters = np.flatnonzero(tall >= tall.max() / 3)
    bottoms = top + tall
    lowest = bottoms[characters].max()  # rows count downwards
    raised = lowest - bottoms >= _RAISED * (lowest - top[characters].min())
    last = characters[np.argmax(left[characters] + width[characters])]
    if not raised[last]:
        return None
    base = characters[~raised[characters]]
    middles = top + tall / 2
    centres = left + width / 2
    minus = (
        (tall < tall.max() / 3)
        & (width >= 2 * tall)
        & (top[last] <= middles)
        & (middles <= bottoms[last])
        & (centres[base].max() < centres)
        & (centres < left[last])
    )
    return len(base), -1 if minus.any() else 1


def _most_in_line(
    pixels: npt.NDArray[np.float64], values: npt.NDArray[np.float64], tolerance: float
) -> npt.NDArray[np.bool_]:
    """Return which values lie on the line that the most of them lie close to.

    Each pair of values sets a line, so a misread value cannot skew it. Values
    that are not a number lie on no line.
    """
    best = np.zeros(len(values), dtype=np.bool_)
    for first, second in itertools.combinations(np.flatnonzero(~np.isnan(values)), 2):
        if values[first] == values[second]:
            continue
        per_unit = (pixels[second] - pixels[first]) / (values[second] - values[first])
        expected = pixels[first] + (values - values[first]) * per_unit
        in_line = np.abs(pixels - expected) <= tolerance
        if np.count_nonzero(in_line) > np.count_nonzero(best):
            best = in_line
    return best


def _axis_lines(
    ink: npt.NDArray[np.bool_], x_labels: Sequence[_Text], y_labels: Sequence[_Text]
) -> tuple[float | None, float | None]:
    """Find the x-coordinate of the y-axis line and the y-coordinate of the x-axis's.

    The y-axis line is the leftmost column, from the right of the y-axis tick
    labels to just past the first x-axis tick, with ink along ``_LINE_COVER``
    of the span of the y-axis ticks, as a flat stretch of a curve on the first
    tick seldom has. The x-axis line is found the same way. An axis without
    such a line gives None.
    """
    height = ink.shape[0]
    first_x, last_x = x_labels[0].x, x_labels[-1].x
    first_y, last_y = y_labels[0].y, y_labels[-1].y

    # Columns run from the labels to the plot; so do rows, read bottom to top.
    start = math.ceil(max(label.right for label in y_labels))
    rows = slice(height - math.floor(last_y), height - math.ceil(first_y) + 1)
    columns = ink[rows, start : math.floor(first_x) + _AXIS_REACH + 1]
    x = _first_line(columns.sum(axis=0), last_y - first_y, start + 0.5)

    start = math.ceil(max(label.top for label in x_labels))
    rows = slice(height - math.floor(first_y) - _AXIS_REACH, height - start + 1)
    lines = ink[rows, math.ceil(first_x) : math.floor(last_x) + 1][::-1]
    y = _first_line(lines.sum(axis=1), last_x - first_x, start - 0.5)
    return x, y


def _beyond(edges: Sequence[float], first: float, height: float) -> float:
    """Return where a missing axis line lies, beyond its labels' inner edges.

    Tick marks and their padding take ``_TICK_GAP`` of a label's ``height``
    beyond the labels, but the line lies no more than halfway from them to
    ``first``, the first tick of the other axis.
    """
    edge = float(np.median(edges))
    return min(edge + _TICK_GAP * height, (edge + float(first)) / 2)


def _first_line(
    counts: npt.NDArray[np.intp], span: float, start: float
) -> float | None:
    """Return the position of the first count of ``_LINE_COVER`` of the span, if any.

    ``start`` is the position of the centre of the first pixel counted.
    """
    long = np.flatnonzero(counts >= span * _LINE_COVER)
    return float(start + long[0]) if len(long) else None


def _line_end(pixels: npt.NDArray[np.bool_], start: float) -> float:
    """Return where a line ends, running on from ``start``.

    ``pixels`` marks which pixels along the line, from position 0, have ink.
    """
    run = pixels[math.ceil(start) :]
    return math.ceil(start) + (len(run) if run.all() else int(np.argmin(run)))


def _check_one_y_axis(
    numbers: Sequence[_Text],
    end: float,
    calibration: Calibration,
    ink: npt.NDArray[np.uint8],
    height: float,
) -> None:
    """Refuse a second y-axis, on the right, whose values differ from the left's.

    Its tick labels would be a column of numbers beyond ``end``, the right end
    of the x-axis, aligned on the left, with values in line. A copy of the
    left y-axis on the right, with the same values, is fine.

    Raises:
        ValueError: If there is such an axis.
    """
    beyond = [number for number in numbers if number.left > end]
    columns = [
        column
        for column in _groups(beyond, lambda text: text.left)
        if len(column) >= MIN_TICKS
    ]
    if not columns:
        return
    try:
        labels, values, _ = _in_line(columns[0], lambda t: t.y, "right", ink, height)
    except ValueError:
        return  # not an axis
    heights = np.array([[0.0, label.y] for label in labels])
    left = calibration.to_data(heights)[:, 1]
    if not np.allclose(left, values, atol=_TICK_TOLERANCE * np.ptp(values)):
        msg = (
            "the plot has a second y-axis, on the right, with other values than "
            "the left's; crop it out of the image"
        )
        raise ValueError(msg)


def _x_marks(ink: npt.NDArray[np.bool_], line: float) -> npt.NDArray[np.float64]:
    """Return the x-coordinates of the tick marks below the x-axis line.

    ``line`` is the line's lowest row, as ``_axis_lines`` finds it, and a tick
    mark has ink in the ``_MARK_LENGTH`` rows below it.
    """
    row = round(ink.shape[0] - line - 0.5)
    return _middles(ink[row + 1 : row + 1 + _MARK_LENGTH].all(axis=0))


def _y_marks(ink: npt.NDArray[np.bool_], line: float) -> npt.NDArray[np.float64]:
    """Return the y-coordinates of the tick marks left of the y-axis line.

    ``line`` is the line's leftmost column, as ``_axis_lines`` finds it, and a
    tick mark has ink in the ``_MARK_LENGTH`` columns left of it.
    """
    column = round(line - 0.5)
    marked = ink[:, max(column - _MARK_LENGTH, 0) : column].all(axis=1)
    return np.asarray(ink.shape[0] - _middles(marked), dtype=np.float64)


def _middles(marked: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """Return the middles of the runs of marked pixels, as pixel coordinates."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], marked.astype(np.int8), [0]))))
    return (edges[::2] + edges[1::2]) / 2


def _snap(
    anchors: Sequence[npt.NDArray[np.float64]], marks: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return where the tick of each label lies, from anchors and tick marks.

    The anchors are points of each label by which it may be aligned with its
    tick: first its middle, then its ends. Ticks lie at the labels' middles,
    unless those lie more than ``_SNAP_SLACK`` from the marks, by their median
    distance, and the ends at one side lie at most half as far, and within
    ``_SNAP_REACH`` of the tick spacing: labels end at their ticks, as rotated
    labels do. Each label then moves to the mark nearest that end, if it is
    as close. Tick marks are not always exactly where the ticks are, as
    plotting libraries may move them to whole pixels, so centred labels stay.
    """
    middles, *ends = anchors
    spacing = float(np.median(np.diff(np.sort(middles))))
    if len(marks) < MIN_TICKS:
        return middles

    def offsets(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.abs(points[:, np.newaxis] - marks).min(axis=1)

    def offset(points: npt.NDArray[np.float64]) -> float:
        return float(np.median(offsets(points)))

    best = min(ends, key=offset)
    if (
        offset(middles) <= _SNAP_SLACK
        or offset(best) > _SNAP_REACH * spacing
        or 2 * offset(best) > offset(middles)
    ):
        return middles
    nearest = marks[np.abs(best[:, np.newaxis] - marks).argmin(axis=1)]
    return np.where(offsets(best) <= _SNAP_REACH * spacing, nearest, best)


def _scales(
    words: Sequence[_Text], origin: Point, y_labels: Sequence[_Text]
) -> tuple[float, float]:
    """Return the scale factors printed beside the x and y axes, or 1.

    Plotting libraries print the y-axis's scale factor or offset above its
    tick labels, starting at the axis, and the x-axis's below the axis. Text
    elsewhere that looks like one, such as a legend entry of 1e-3, is not one.
    """
    top = max(label.top for label in y_labels)
    reach = origin[0] + np.median([label.top - label.bottom for label in y_labels])
    scales = [1.0, 1.0]
    for word in words:
        if word.y < origin[1]:
            axis = 0
        elif word.bottom > top and word.left <= reach:
            axis = 1
        else:
            continue
        if match := _SCALE.fullmatch(word.plain):
            scales[axis] = 10.0 ** int(match[1])
        elif _OFFSET.fullmatch(word.plain):
            msg = f"cannot read an axis whose ticks are offset by {word.text}"
            raise ValueError(msg)
    return scales[0], scales[1]


def _x_title(
    words: Sequence[_Text], x_labels: Sequence[_Text], span: tuple[float, float]
) -> str | None:
    """Return the lines of text nearest below the x-axis tick labels.

    The title is centred within ``span``, along the axis.
    """
    bottom = min(label.bottom for label in x_labels)
    below = [word for word in words if word.y < bottom and span[0] <= word.x <= span[1]]
    return _title(_lines(below))


def _y_title(
    words: Sequence[_Text],
    y_labels: Sequence[_Text],
    span: tuple[float, float],
) -> str | None:
    """Return the lines of text nearest left of the y-axis tick labels.

    The title is centred within ``span``, along the axis. It is written
    upwards unless the words nearest the labels are ``_WRITTEN`` times wider
    than tall, and so run across.
    """
    left = min(label.left for label in y_labels)
    beside = [word for word in words if word.x < left and span[0] <= word.y <= span[1]]
    if not beside:
        return None
    nearest = max(beside, key=lambda word: word.right)
    column = [
        word for word in beside if abs(word.x - nearest.x) < nearest.right - nearest.x
    ]
    width = sum(word.right - word.left for word in column)
    height = sum(word.top - word.bottom for word in column)
    if width >= _WRITTEN * height:
        return _title(_lines(column))
    # Transposed, the lines of a title written upwards run from the last up.
    return _title(_lines(map(_transposed, beside))[::-1])


def _lines(texts: Iterable[_Text]) -> list[_Text]:
    """Return the lines of a title, the topmost first, each as one text.

    The first line is the topmost text and the text level with it. Each line
    after it lies below the one before, within ``_LINE_SPACING`` of its height,
    centre to centre, and aligned with it, the middle of one lying within the
    other, so text further on, such as a caption, is left out.
    """
    remaining = list(texts)
    lines: list[_Text] = []
    while remaining:
        nearest = max(remaining, key=lambda text: text.top)
        half = nearest.top - nearest.y
        line = _line([text for text in remaining if abs(text.y - nearest.y) < half])
        if lines and not _follows(line, lines[-1]):
            break
        lines.append(line)
        remaining = [text for text in remaining if text.y <= nearest.y - half]
    return lines


def _line(texts: Sequence[_Text]) -> _Text:
    """Return the texts of a line, read from left to right, as one text."""
    return _Text(
        " ".join(text.text for text in sorted(texts, key=lambda text: text.x)),
        min(text.left for text in texts),
        min(text.bottom for text in texts),
        max(text.right for text in texts),
        max(text.top for text in texts),
    )


def _follows(line: _Text, before: _Text) -> bool:
    """Return whether a line carries on a title from the line before it."""
    return before.y - line.y <= _LINE_SPACING * (before.top - before.bottom) and (
        before.left <= line.x <= before.right or line.left <= before.x <= line.right
    )


def _title(lines: Sequence[_Text]) -> str | None:
    """Return the lines of a title as one line, or None if there are none."""
    return " ".join(line.text for line in lines) or None
