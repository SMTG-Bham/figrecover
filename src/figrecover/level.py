"""Level images of plots whose axes are slightly turned, as scans often are."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from .calibration import Point
from .digitize import FRAME_WIDTH, Image, ink_mask

MAX_SKEW = 10
"""Most degrees from horizontal or vertical that ``find_skew`` finds lines turned."""

LEVEL_THRESHOLD = 0.1
"""Fewest degrees that ``level_image`` turns an image by, when it finds the angle."""

_LINE_LENGTH = 0.2  # lines span this fraction of the image's shorter side or more
_SHORTEST = 10  # but at least this many pixels, in the smallest images
_LINE_GAP = 2  # pixels missing from a line, such as where it fades, that it bridges
_HOUGH_STEP = 0.25  # degrees between the angles that lines are first looked for at
_BAND = 4  # pixels either side of a line in which its middle is found
_PASSES = 3  # times a line's middle is found again, about the line found before
_TYPICAL = 0.5  # a line's ink across it lies within this fraction of its median
_OUTLIER = 0.5  # pixels off the line through the rest that a line's middle may lie
_KEPT = 0.75  # a line has its middle on it along this fraction of its length or more
_AGREE = 0.05  # lines turned within this many degrees of each other agree


@dataclass(frozen=True, slots=True, eq=False)
class Levelling:
    """How ``level_image`` turned an image so that its axes are level.

    Attributes:
        angle: The angle in degrees by which the image was turned clockwise,
            or anticlockwise if negative, which is the angle by which its
            axes were turned anticlockwise; 0 if it was left as it is.
        matrix: The affine map from pixel positions in the image as given to
            those in the levelled image, both measured from the bottom-left
            corner, as a ``(2, 3)`` array.
    """

    angle: float
    matrix: npt.NDArray[np.float64]

    def points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return where pixel positions in the image as given lie in the levelled one.

        Args:
            points: Pixel ``(x, y)`` positions, measured from the bottom-left
                corner, such as reference points clicked on the image.

        Returns:
            An ``(n, 2)`` array of the positions in the levelled image: the
            same positions if the image was left as it is.
        """
        return _mapped(points, self.matrix)

    def back(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return where pixel positions in the levelled image lie in the image as given.

        Args:
            points: Pixel ``(x, y)`` positions, measured from the bottom-left
                corner, such as the points of a curve traced in the levelled
                image.

        Returns:
            An ``(n, 2)`` array of the positions in the image as given: the
            same positions if the image was left as it is.
        """
        inverse = cv2.invertAffineTransform(self.matrix)
        return _mapped(points, np.asarray(inverse, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class _Line:
    """A long straight line in an image, close to horizontal or vertical.

    Attributes:
        angle: The angle in degrees by which it is turned anticlockwise from
            horizontal or vertical.
        length: The number of pixels along it in which it was measured.
        upright: Whether it is close to vertical.
        middle: The pixel position of its middle, measured from the
            bottom-left corner.
    """

    angle: float
    length: float
    upright: bool
    middle: Point


def find_skew(image: Image) -> float:
    """Find the angle by which the axes of a plot are turned, as in a scan.

    The angle is that of the long straight lines in the image, such as the
    axis lines, a frame and grid lines, which are horizontal and vertical in
    a plot drawn level. Lines at least ``_LINE_LENGTH`` of the image's
    shorter side long, at most ``FRAME_WIDTH`` pixels thick, and within
    ``MAX_SKEW`` degrees of horizontal or vertical, are found with a Hough
    transform, and each line's angle is measured to a small fraction of a
    pixel along its length, from the middle of its ink across it, as
    ``_tilt`` measures it. The straight stretches of curves are lines too,
    so the angle is the one that most of the lines' length agrees on, within
    ``_AGREE`` degrees, and only if both horizontal and vertical lines agree
    on it, as the axes of a plot do and the stretches of its curves seldom
    do.

    Args:
        image: Colour image of a plot.

    Returns:
        The angle in degrees by which the axes are turned anticlockwise, or
        clockwise if negative, or 0 if no horizontal and vertical lines
        agree on one.
    """
    return _skew(_lines(image))


def level_image(image: Image, angle: float | None = None) -> tuple[Image, Levelling]:
    """Turn an image of a plot whose axes are turned, so that they are level.

    The image is turned about its middle, with Lanczos interpolation, onto a
    larger image that holds all of it, whose corners are filled in its
    background colour, the median. It is also moved by up to half a pixel
    each way, so that the long straight lines that agree on the angle, as
    ``find_skew`` finds them, run along the middles of rows and columns, as
    they are drawn in a plot drawn level, rather than blurred over two.

    Args:
        image: Colour image of a plot.
        angle: The angle in degrees by which the axes are turned
            anticlockwise, or clockwise if negative, to turn the image back
            by; or None to find it with ``find_skew``, turning the image only
            if it is ``LEVEL_THRESHOLD`` or more.

    Returns:
        The levelled image, or the image as it is if it is not turned, at an
        angle of 0; and how it was turned, to find where positions in the
        image lie in the levelled image, and back.
    """
    lines = [] if angle == 0 else _lines(image)
    if angle is None:
        angle = _skew(lines)
        if abs(angle) < LEVEL_THRESHOLD:
            angle = 0.0
    if not angle:
        return image, Levelling(0.0, np.eye(2, 3))
    height, width = image.shape[:2]
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    # Less a hair, so that an image turned by a hair keeps its size.
    size = (
        math.ceil(width * abs(cos) + height * abs(sin) - 1e-6),
        math.ceil(width * abs(sin) + height * abs(cos) - 1e-6),
    )
    # Turned clockwise about the middle of the image onto the middle of the
    # levelled one, with y counting up, and moved onto the pixels' middles.
    matrix = np.array(
        [
            [cos, sin, size[0] / 2 - cos * width / 2 - sin * height / 2],
            [-sin, cos, size[1] / 2 + sin * width / 2 - cos * height / 2],
        ]
    )
    agreeing = [line for line in lines if abs(line.angle - angle) <= _AGREE]
    matrix[:, 2] += _snap(matrix, agreeing)
    # OpenCV counts rows down from the middle of the top pixel, and columns
    # from the middle of the left one.
    to_opencv = [
        np.array([[1, 0, -0.5], [0, -1, extent - 0.5], [0, 0, 1]])
        for extent in (height, size[1])
    ]
    opencv = to_opencv[1] @ np.vstack((matrix, (0, 0, 1))) @ np.linalg.inv(to_opencv[0])
    background = np.median(image.reshape(-1, 3), axis=0)
    levelled = cv2.warpAffine(
        image,
        opencv[:2],
        size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(float(level) for level in background),
    )
    return np.asarray(levelled, dtype=np.uint8), Levelling(float(angle), matrix)


def _mapped(
    points: npt.ArrayLike, matrix: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return pixel positions mapped by an affine map, given as a ``(2, 3)`` array."""
    positions = np.array(points, dtype=np.float64).reshape(-1, 2)
    return positions @ matrix[:, :2].T + matrix[:, 2]


def _snap(
    matrix: npt.NDArray[np.float64], lines: Sequence[_Line]
) -> npt.NDArray[np.float64]:
    """Return how far to move a levelled image so its lines run along pixels' middles.

    Each horizontal line's middle lies a fraction of a pixel above or below
    the middle of a row once levelled by ``matrix``, and each vertical
    line's likewise beside the middle of a column. The image is moved by the
    mean of those fractions, weighted by the lines' lengths, taken round a
    circle, as a line half a pixel above the middle of a row lies half a
    pixel below the middle of the next.
    """
    shift = np.zeros(2)
    for axis in range(2):  # x from the vertical lines, y from the horizontal ones
        chosen = [line for line in lines if line.upright == (axis == 0)]
        middles = _mapped([line.middle for line in chosen], matrix)[:, axis]
        lengths = np.array([line.length for line in chosen])
        phases = 2 * np.pi * (middles - 0.5)  # 0 at the middles of pixels
        mean = math.atan2(lengths @ np.sin(phases), lengths @ np.cos(phases))
        shift[axis] = -mean / (2 * np.pi)
    return shift


def _skew(lines: Sequence[_Line]) -> float:
    """Return the angle that most of the length of lines agrees on, as ``find_skew``."""
    if not lines:
        return 0.0
    angles = np.array([line.angle for line in lines])
    lengths = np.array([line.length for line in lines])
    upright = np.array([line.upright for line in lines])
    agree = np.abs(angles[:, np.newaxis] - angles) <= _AGREE
    best = agree[int(np.argmax(agree @ lengths))]
    if upright[best].all() or not upright[best].any():
        return 0.0
    order = np.argsort(angles[best])
    total = np.cumsum(lengths[best][order])
    return float(angles[best][order][np.searchsorted(total, total[-1] / 2)])


def _lines(image: Image) -> list[_Line]:
    """Return the long straight lines in an image that ``find_skew`` looks for.

    The ink of thicker shapes, such as filled areas, is left out, and the
    rest made thicker by a pixel each way, so that the faint stretches of a
    turned line, where it lies between two rows, are not taken for gaps.
    """
    ink = ink_mask(image)
    # The ink less what a square of FRAME_WIDTH + 1 pixels fits inside.
    square = np.ones((FRAME_WIDTH + 1, FRAME_WIDTH + 1), np.uint8)
    thin = cv2.morphologyEx(ink, cv2.MORPH_TOPHAT, square)
    thin = cv2.dilate(thin, np.ones((3, 3), np.uint8))
    length = max(round(_LINE_LENGTH * min(ink.shape)), _SHORTEST)
    segments = cv2.HoughLinesP(
        thin,
        1,
        math.radians(_HOUGH_STEP),
        threshold=length // 2,
        minLineLength=length,
        maxLineGap=_LINE_GAP,
    )
    found = np.zeros((0, 4), np.intp) if segments is None else segments.reshape(-1, 4)
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    levels = np.abs(grey - np.median(grey)) / 255
    height = image.shape[0]
    lines = []
    for left, top, right, bottom in found.astype(np.intp).tolist():
        direction = math.degrees(math.atan2(bottom - top, right - left))
        if abs(math.remainder(direction, 180)) <= MAX_SKEW:
            tilt = _tilt(levels, (left, top), (right, bottom))
            if tilt is not None:  # rows count down, so a rising line slopes down
                angle, measured, (column, row) = tilt
                middle = (column + 0.5, height - row - 0.5)
                lines.append(_Line(-angle, measured, False, middle))
        elif abs(math.remainder(direction - 90, 180)) <= MAX_SKEW:
            tilt = _tilt(levels.T, (top, left), (bottom, right))
            if tilt is not None:  # a line turned anticlockwise leans left at the top
                angle, measured, (row, column) = tilt
                middle = (column + 0.5, height - row - 0.5)
                lines.append(_Line(angle, measured, True, middle))
    return lines


def _tilt(
    levels: npt.NDArray[np.float64], start: tuple[int, int], end: tuple[int, int]
) -> tuple[float, float, Point] | None:
    """Measure the angle of a line close to horizontal, found between two pixels.

    In each column, the line's middle is the mean row of the ink within
    ``_BAND`` pixels of it, weighted by how far each pixel is from the
    background, and its angle is that of the straight line fitted to its
    middles by least squares. Columns that the line does not cross alone,
    as where a curve, a tick mark or a label meets it, are left out: those
    whose ink across it differs from the median by more than ``_TYPICAL`` of
    it, and those whose middle lies more than ``_OUTLIER`` pixels off the
    line fitted to the rest. The line is found again ``_PASSES`` times about
    the line fitted before, as the line between the two pixels can stray
    from its middle.

    Args:
        levels: How far each pixel is from the background, from 0 to 1.
        start: The column and row of a pixel at one end of the line.
        end: The column and row of a pixel at the other end.

    Returns:
        The angle in degrees by which the line runs down to the right, the
        number of columns it was measured in, and the column and row of its
        middle; or None if it was measured in fewer than ``_KEPT`` of its
        columns.
    """
    (first, first_row), (last, last_row) = sorted([start, end])
    columns = np.arange(first, last + 1)
    middles = np.interp(columns, [first, last], [first_row, last_row])
    across = np.arange(-_BAND, _BAND + 1)
    kept = np.ones(len(columns), dtype=np.bool_)
    fit = np.zeros(2)
    for _ in range(_PASSES):
        rows = np.round(middles).astype(np.intp)[:, np.newaxis] + across
        inside = (rows >= 0) & (rows < len(levels))
        weights = np.where(
            inside, levels[rows.clip(0, len(levels) - 1), columns[:, np.newaxis]], 0
        )
        totals = weights.sum(axis=1)
        typical = np.median(totals[kept])
        kept = (totals > 0) & (np.abs(totals - typical) <= _TYPICAL * typical)
        if np.count_nonzero(kept) < _KEPT * len(columns):
            return None
        centres = (weights * rows).sum(axis=1) / np.maximum(totals, 1e-12)
        fit = np.polyfit(columns[kept], centres[kept], 1)
        kept &= np.abs(centres - np.polyval(fit, columns)) <= _OUTLIER
        if np.count_nonzero(kept) < _KEPT * len(columns):
            return None
        fit = np.polyfit(columns[kept], centres[kept], 1)
        middles = np.polyval(fit, columns).astype(np.float64)
    middle = (first + last) / 2
    return (
        math.degrees(math.atan(fit[0])),
        float(np.count_nonzero(kept)),
        (middle, float(np.polyval(fit, middle))),
    )
