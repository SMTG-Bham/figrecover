"""Read the axes of a plot from the text in its image."""

import functools
import itertools
import math
import re
from collections.abc import Callable, Sequence
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
_RAISED = 0.25  # a superscript sits this fraction of its label's height up or more
_RAISED_TICK = 1.4  # a label this many times taller than usual has a tick on top
_NUMBER = re.compile(  # 1, -2.5, 1,000, 2.5e-3 or 10%
    r"([+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)%?",
    re.IGNORECASE,
)
_TRAILING_DASH = re.compile(r"(.*\d%?)-+")  # a tick mark after a y-axis label
_POWER = re.compile(  # 10^2, 10^{2}, 10-2 or 2x10^2: a power of ten, as OCR reads it
    r"(?:(\d+(?:\.\d*)?|\.\d+)[x\u00d7])?10(?:\^\{?|(?=[+-]))([+-]?\d+)\}?"
)
_RUN_TOGETHER = re.compile(r"10(\d+)")  # 102: 10^2 read without its superscript
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

    Numbers may be written as powers of ten, such as 10^2 or 2x10^3 with
    superscript exponents. OCR often misses superscripts, reading 10^2 as 102,
    so a label that reads as 10 followed by digits is a power of ten if its
    last digit is drawn raised, a raised minus sign before it making the
    exponent negative, or if most of the other labels on its axis that read
    as 10 followed by digits or as powers of ten show that they are.
    An axis is logarithmic if more of its labels fall in line on a logarithmic
    scale than on a linear one.

    The axes cross where the axis lines run beside the tick labels, or failing
    that at the first tick on each axis. The titles are the nearest text
    beyond the tick labels.

    The text is read with RapidOCR, which the ``ocr`` extra installs.

    Args:
        image: Colour image of a plot.

    Returns:
        The axes, with a calibration fitted to their ticks.

    Raises:
        ModuleNotFoundError: If the ``ocr`` extra is not installed.
        ValueError: If an axis has too few tick labels in line, or an offset
            to add to its tick values.
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
    origin = _cross_axes(ink > 0, x_labels, y_labels)

    words = [text for text in texts if text not in {*x_labels, *y_labels}]
    x_scale, y_scale = _scales(words, origin, y_labels)
    words = [word for word in words if not _SCALE.fullmatch(word.plain)]
    x_ticks = tuple(
        (label.x, value * x_scale)
        for label, value in zip(x_labels, x_values, strict=True)
    )
    y_ticks = tuple(
        (label.y, value * y_scale)
        for label, value in zip(y_labels, y_values, strict=True)
    )
    return Axes(
        Calibration.from_ticks(x_ticks, y_ticks, origin, x_log=x_log, y_log=y_log),
        x_ticks,
        y_ticks,
        _x_title(words, x_labels),
        _y_title(words, y_labels),
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
    """Return the values of tick labels, which are numbers or powers of ten.

    OCR often misses superscripts, reading 10^2 as 102. A label that reads as
    10 followed by digits is a power of ten if its last digit is drawn raised,
    or if most of the other labels that read as powers of ten or as 10 and
    digits show that they are powers, by their text or a raised digit. One
    misread label, such as 100 read with a degree sign for its last 0, so
    does not make an axis of 100, 101 and 102 logarithmic.
    """
    readings = [_reading(label, height) for label in labels]
    signs = [
        _exponent_sign(ink, label) if _RUN_TOGETHER.fullmatch(reading) else None
        for label, reading in zip(labels, readings, strict=True)
    ]
    shown = [
        bool(_POWER.fullmatch(reading)) or sign is not None
        for reading, sign in zip(readings, signs, strict=True)
    ]
    candidates = [
        index
        for index, reading in enumerate(readings)
        if _POWER.fullmatch(reading) or _RUN_TOGETHER.fullmatch(reading)
    ]
    values = []
    for index, (reading, sign) in enumerate(zip(readings, signs, strict=True)):
        others = [shown[other] for other in candidates if other != index]
        mostly = 2 * sum(others) > len(others)
        if (match := _RUN_TOGETHER.fullmatch(reading)) and (sign or mostly):
            values.append(10.0 ** ((sign or 1) * int(match[1])))
        elif match := _POWER.fullmatch(reading):
            values.append(float(match[1] or 1) * 10.0 ** int(match[2]))
        else:  # a number, less any % and thousands separators
            values.append(float(_NUMBER.sub(r"\1", reading).replace(",", "")))
    return np.array(values)


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


def _exponent_sign(ink: npt.NDArray[np.uint8], label: _Text) -> int | None:
    """Return the sign of a label's exponent, or None if it has none.

    The characters are the pieces of ink in the label's box at least a third as
    tall as the tallest, which leaves out points, minus signs and tick marks.
    The last is an exponent if its bottom is ``_RAISED`` of the characters'
    height or more above the lowest bottom. The exponent is negative if a
    flat piece, at least twice as wide as tall, lies level with the exponent
    and before it, after the middle of the second character, the 0 of 10.
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
    last = characters[np.argmax(left[characters] + width[characters])]
    lift = bottoms[characters].max() - bottoms[last]  # rows count downwards
    if lift < _RAISED * (bottoms[characters].max() - top[characters].min()):
        return None
    after = np.sort(left[characters] + width[characters] / 2)[
        min(1, len(characters) - 1)
    ]
    middles = top + tall / 2
    centres = left + width / 2
    minus = (
        (tall < tall.max() / 3)
        & (width >= 2 * tall)
        & (top[last] <= middles)
        & (middles <= bottoms[last])
        & (after < centres)
        & (centres < left[last])
    )
    return -1 if minus.any() else 1


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


def _cross_axes(
    ink: npt.NDArray[np.bool_], x_labels: Sequence[_Text], y_labels: Sequence[_Text]
) -> Point:
    """Find where the axis lines cross, or failing that the first ticks.

    The y-axis line is the leftmost column, from the right of the y-axis tick
    labels to just past the first x-axis tick, with ink along at least half of
    the span of the y-axis ticks. The x-axis line is found the same way.
    """
    height = ink.shape[0]
    first_x, last_x = x_labels[0].x, x_labels[-1].x
    first_y, last_y = y_labels[0].y, y_labels[-1].y

    # Columns run from the labels to the plot; so do rows, read bottom to top.
    start = math.ceil(max(label.right for label in y_labels))
    rows = slice(height - math.floor(last_y), height - math.ceil(first_y) + 1)
    columns = ink[rows, start : math.floor(first_x) + _AXIS_REACH + 1]
    x = _first_line(columns.sum(axis=0), last_y - first_y, start + 0.5, first_x)

    start = math.ceil(max(label.top for label in x_labels))
    rows = slice(height - math.floor(first_y) - _AXIS_REACH, height - start + 1)
    lines = ink[rows, math.ceil(first_x) : math.floor(last_x) + 1][::-1]
    y = _first_line(lines.sum(axis=1), last_x - first_x, start - 0.5, first_y)
    return x, y


def _first_line(
    counts: npt.NDArray[np.intp], span: float, start: float, default: float
) -> float:
    """Return the position of the first count of at least half the span.

    ``start`` is the position of the centre of the first pixel counted.
    """
    long = np.flatnonzero(counts >= span / 2)
    return float(start + long[0]) if len(long) else default


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


def _x_title(words: Sequence[_Text], x_labels: Sequence[_Text]) -> str | None:
    """Return the line of text nearest below the x-axis tick labels."""
    bottom = min(label.bottom for label in x_labels)
    first, last = x_labels[0].x, x_labels[-1].x
    below = [word for word in words if word.y < bottom and first <= word.x <= last]
    if not below:
        return None
    nearest = max(below, key=lambda word: word.top)
    line = [word for word in below if abs(word.y - nearest.y) < nearest.top - nearest.y]
    return " ".join(word.text for word in sorted(line, key=lambda word: word.x))


def _y_title(words: Sequence[_Text], y_labels: Sequence[_Text]) -> str | None:
    """Return the column of text nearest left of the y-axis tick labels."""
    left = min(label.left for label in y_labels)
    first, last = y_labels[0].y, y_labels[-1].y
    beside = [word for word in words if word.x < left and first <= word.y <= last]
    if not beside:
        return None
    nearest = max(beside, key=lambda word: word.right)
    column = [
        word for word in beside if abs(word.x - nearest.x) < nearest.right - nearest.x
    ]
    return " ".join(word.text for word in sorted(column, key=lambda word: word.y))
