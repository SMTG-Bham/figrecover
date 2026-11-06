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
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
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
    Each tick lies at the middle of its label. A scale factor printed beside
    an axis, such as ``1e-5``, multiplies the values of its ticks.

    Numbers may be written as powers of ten, such as 10^2 or 2x10^3 with
    superscript exponents. OCR often misses superscripts, reading 10^2 as 102,
    so if any label on an axis is a power of ten, so are those that read as 10
    followed by digits.
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
    numbers = [
        text
        for text in texts
        if _NUMBER.fullmatch(text.plain) or _POWER.fullmatch(text.plain)
    ]
    x_labels, x_values, x_log = _in_line(_lowest_row(numbers), lambda t: t.x, "x")
    above = [text for text in numbers if text.y > max(t.top for t in x_labels)]
    y_labels, y_values, y_log = _in_line(_leftmost_column(above), lambda t: t.y, "y")
    origin = _cross_axes(image, x_labels, y_labels)

    words = [text for text in texts if text not in numbers]
    x_scale, y_scale = _scales(words, origin)
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
    labels: Sequence[_Text], position: Callable[[_Text], float], name: str
) -> tuple[list[_Text], list[float], bool]:
    """Return the most tick labels whose values are in line with their positions.

    The values are taken on a linear scale and on a logarithmic one, and the
    scale on which more labels are in line wins, the linear one in a tie.

    Returns:
        The labels in line, by position, their values, and whether the scale
        is logarithmic.
    """
    labels = sorted(labels, key=position)
    pixels = np.array([position(label) for label in labels])
    values = _values(labels)
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


def _values(labels: Sequence[_Text]) -> npt.NDArray[np.float64]:
    """Return the values of tick labels, which are numbers or powers of ten.

    If any label is a power of ten, so are those that read as 10 followed by
    digits, such as 102 for 10^2, as OCR often misses superscripts.
    """
    powers = any(_POWER.fullmatch(label.plain) for label in labels)
    values = []
    for label in labels:
        if powers and (match := _RUN_TOGETHER.fullmatch(label.plain)):
            values.append(10.0 ** int(match[1]))
        elif match := _POWER.fullmatch(label.plain):
            values.append(float(match[1] or 1) * 10.0 ** int(match[2]))
        else:
            values.append(float(label.plain))
    return np.array(values)


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
    image: Image, x_labels: Sequence[_Text], y_labels: Sequence[_Text]
) -> Point:
    """Find where the axis lines cross, or failing that the first ticks.

    The y-axis line is the leftmost column, from the right of the y-axis tick
    labels to just past the first x-axis tick, with ink along at least half of
    the span of the y-axis ticks. The x-axis line is found the same way.
    """
    ink = ink_mask(image) > 0
    height = image.shape[0]
    first_x, last_x = x_labels[0].x, x_labels[-1].x
    first_y, last_y = y_labels[0].y, y_labels[-1].y

    # Columns run from the labels to the plot; so do rows, read bottom to top.
    start = math.ceil(max(label.right for label in y_labels))
    rows = slice(height - math.floor(last_y), height - math.ceil(first_y) + 1)
    columns = ink[rows, start : math.floor(first_x) + _AXIS_REACH + 1]
    x = _first_line(columns.sum(axis=0), last_y - first_y, start, first_x)

    start = math.ceil(max(label.top for label in x_labels))
    rows = slice(height - math.floor(first_y) - _AXIS_REACH, height - start + 1)
    lines = ink[rows, math.ceil(first_x) : math.floor(last_x) + 1][::-1]
    y = _first_line(lines.sum(axis=1), last_x - first_x, start, first_y)
    return x, y


def _first_line(
    counts: npt.NDArray[np.intp], span: float, start: int, default: float
) -> float:
    """Return the position of the first count of at least half the span."""
    long = np.flatnonzero(counts >= span / 2)
    return float(start + long[0]) if len(long) else default


def _scales(words: Sequence[_Text], origin: Point) -> tuple[float, float]:
    """Return the scale factors printed beside the x and y axes, or 1."""
    scales = [1.0, 1.0]
    for word in words:
        if match := _SCALE.fullmatch(word.plain):
            axis = 1 if word.y > origin[1] else 0  # above the x-axis: the y-axis's
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
