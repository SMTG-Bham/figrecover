"""Read error bars: the bounds drawn through the points of a plot."""

import math
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

_MARKER_DEPTH = 1  # a filled marker lies this many pixels deeper than its lines
_MIN_BAR = 5  # a bar without a cap runs at least this many pixels beyond its marker
_CROSSING = 6  # a bar runs on across ink of other colours this many pixels wide
_CAP_SHIFT = 1  # a cap's middle lies within this many pixels of its bar's
_LEAVING = 2  # a line leaving a bar's path lies this many pixels beside it
_STRAIGHT = 1  # the middles of a bar's rows lie this many pixels apart or less


@dataclass(frozen=True, slots=True)
class ErrorBars:
    """The error bars drawn through the points of a series.

    Positions are those of the points: pixels of an image, or points on a
    PDF's page, from its bottom-left corner. ``calibration.to_data`` turns
    each array into data; on a reversed axis, the lower or left end is the
    larger value.

    Attributes:
        points: An ``(n, 2)`` array of the ``(x, y)`` position of each point.
        low: An ``(n, 2)`` array of the x of the left end of each point's
            horizontal bar and the y of the lower end of its vertical bar,
            NaN where it has no such bar or the end cannot be seen.
        high: The same for the right and upper ends.
    """

    points: npt.NDArray[np.float64]
    low: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _Bars:
    """The error bars of markers in a part of an image, in rows and columns.

    A row or column coordinate is measured from the top or left edge of the
    part, so the middle of the pixel in row r and column c lies at r + 0.5
    and c + 0.5. ``middles`` holds the row and column of the middle of each
    marker, ``low`` and ``high`` the top and left ends of its bars, and their
    bottom and right ends, as rows and columns, NaN where it has none or the
    end cannot be seen. ``drawn`` is where the bars and their caps lie.
    """

    middles: npt.NDArray[np.float64]
    low: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]
    drawn: npt.NDArray[np.bool_]


def _line_depth(in_colour: npt.NDArray[np.bool_]) -> float:
    """Return how deep the middles of a colour's lines lie, in pixels.

    That is the median depth of its ridges: the pixels deepest inside the
    colour among their neighbours. The lines outnumber the markers on them.
    """
    depth = cv2.distanceTransform(in_colour.astype(np.uint8), cv2.DIST_L2, 3)
    ridges = in_colour & (depth >= cv2.dilate(depth, np.ones((3, 3), np.uint8)))
    return float(np.median(depth[ridges])) if ridges.any() else 0.0


def _filled_markers(in_colour: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    """Return the boxes of the filled markers drawn in a colour.

    A filled marker's middle lies ``_MARKER_DEPTH`` pixels or more deeper
    inside the colour than the middles of its lines and bars, as
    ``_line_depth`` finds them, and its box is that of its middle, widened by
    how deep its middle lies. A bar or line drawn through a marker deepens it
    a little where they meet, so the middle's height is taken where it is
    wider than a bar, and its width where it is taller than one; a middle
    no wider or taller than a bar, as where lines cross, is not a marker's.

    Returns:
        A row of the top, bottom, left and right edges of each marker's box,
        the bottom and right one past its last row and column.
    """
    depth = cv2.distanceTransform(in_colour.astype(np.uint8), cv2.DIST_L2, 3)
    line = _line_depth(in_colour)
    deep = line + _MARKER_DEPTH
    middles = (depth >= deep).astype(np.uint8)
    count, pieces, stats, _ = cv2.connectedComponentsWithStats(middles)
    wide = 2 * max(math.ceil(line), 1) + 1
    wider = cv2.erode(middles, np.ones((1, wide), np.uint8)) > 0
    taller = cv2.erode(middles, np.ones((wide, 1), np.uint8)) > 0
    reach = round(deep)
    boxes = []
    for index in range(1, count):
        left, top, width, height = stats[index, :4]
        part = (slice(top, top + height), slice(left, left + width))
        piece = pieces[part] == index
        rows = np.flatnonzero((piece & wider[part]).any(axis=1))
        columns = np.flatnonzero((piece & taller[part]).any(axis=0))
        if not (len(rows) and len(columns)):
            continue  # where lines cross, not a marker
        boxes.append(
            (
                top + rows[0] - reach,
                top + rows[-1] + 1 + reach,
                left + columns[0] - reach,
                left + columns[-1] + 1 + reach,
            )
        )
    return np.array(boxes, dtype=np.intp).reshape(-1, 4)


def _read_bars(
    in_colour: npt.NDArray[np.bool_],
    hidden: npt.NDArray[np.bool_],
    boxes: npt.NDArray[np.intp],
) -> _Bars:
    """Follow the error bars of markers, up and down and across.

    Each marker's bars run along its middle row and column, from the edges
    of its box, as ``_follow_bar`` follows them. ``in_colour`` is where the
    colour of the bars lies, and ``hidden`` where ink of other colours, that
    may hide them, lies.
    """
    half = max(math.ceil(_line_depth(in_colour)), 1)
    middles = np.column_stack(
        [(boxes[:, 0] + boxes[:, 1]) / 2, (boxes[:, 2] + boxes[:, 3]) / 2]
    ).reshape(-1, 2)
    low, high = np.full_like(middles, np.nan), np.full_like(middles, np.nan)
    drawn = np.zeros_like(in_colour)
    ways = [
        (in_colour, hidden, drawn, (0, 1)),
        (in_colour.T, hidden.T, drawn.T, (2, 3)),
    ]
    for axis, (colour, other, marked, (near, far)) in enumerate(ways):
        for index, box in enumerate(boxes):
            column = int(middles[index, 1 - axis])  # that of the bar, or its row
            for ends, start, step in ((low, box[near] - 1, -1), (high, box[far], 1)):
                ends[index, axis], runs = _follow_bar(
                    colour, other, (int(start), column), step, half
                )
                for row, first, last in runs:
                    marked[row, first : last + 1] = True
    return _Bars(middles, low, high, drawn)


def _follow_bar(
    in_colour: npt.NDArray[np.bool_],
    hidden: npt.NDArray[np.bool_],
    start: tuple[int, int],
    step: int,
    half: int,
) -> tuple[float, list[tuple[int, int, int]]]:
    """Follow a bar along its column from a marker's edge, one way, to its end.

    ``start`` is the row just beyond the marker and the bar's column, and
    ``step`` is -1 to follow the bar up, or 1 down; the bar lies within
    ``half`` columns of its column. It runs on while its colour lies there,
    across ink of other colours up to ``_CROSSING`` pixels wide, and most of
    its rows are no wider than it, as a marker's are. A cap, a short line
    across its end, wider than the bar and centred on it within
    ``_CAP_SHIFT`` pixels, places the end at its middle, and the bar needs
    only a row before it. Without a cap, the bar must run at least
    ``_MIN_BAR`` pixels, so the tip of a marker is not taken for one, and be
    straight, the middles of its rows no more than ``_STRAIGHT`` pixels
    apart, as a line across its path is not; and a line of its colour that
    leaves its path at its end, within ``_LEAVING`` pixels of it, as the
    marker's own line does, is not a bar.

    Returns:
        The row coordinate of the end of the bar, or NaN if there is none, or
        its end is hidden, by ink of another colour or the edge of the image;
        and the bar's pixels, and its cap's, as the row and first and last
        columns of each run of them: the bar's own columns, where most of its
        rows lie, and the whole of its cap.
    """
    length = in_colour.shape[0]
    columns = slice(max(start[1] - half, 0), start[1] + half + 1)
    band = in_colour[:, columns].any(axis=1)
    covered = hidden[:, columns].any(axis=1)
    last, row = _walk(band, covered, start[0], step)
    if last is None:
        return math.nan, []
    rows = [index for index in range(start[0], last + step, step) if band[index]]
    runs = [_run(in_colour[index], columns) for index in rows]
    capped = _capped(runs, start[1], half)
    narrow = np.array([run for run in runs if run[1] - run[0] <= 2 * half + 1])
    middles = narrow.reshape(-1, 2).sum(axis=1) / 2
    beside = slice(max(start[1] - half - _LEAVING, 0), start[1] + half + _LEAVING + 1)
    if 2 * len(narrow) <= len(runs) - capped or (
        not capped
        and (
            abs(last - start[0]) + 1 < _MIN_BAR
            or (0 <= row < length and in_colour[row, beside].any())
            or np.ptp(middles) > _STRAIGHT
        )
    ):
        return math.nan, []  # no bar, or a line leaving the bar's path or across it
    first, end = np.median(narrow, axis=0).astype(int) if len(narrow) else runs[0]
    pixels = [(index, first, end) for index in rows[: len(rows) - capped]]
    pixels += [(index, *run) for index, run in zip(rows, runs, strict=True)][
        len(rows) - capped :
    ]
    if not 0 <= row < length or covered[row]:
        return math.nan, pixels
    if capped:
        return (min(rows[-capped:]) + max(rows[-capped:]) + 1) / 2, pixels
    return last + (step > 0), pixels


def _walk(
    band: npt.NDArray[np.bool_], covered: npt.NDArray[np.bool_], row: int, step: int
) -> tuple[int | None, int]:
    """Walk along a bar's path from a row, one way, while its colour lies there.

    ``band`` is where the bar's colour lies along its path, and ``covered``
    where ink of other colours does, which the walk crosses if no more than
    ``_CROSSING`` pixels wide and the bar's colour lies beyond.

    Returns:
        The last row of the bar's colour, or None if there is none, and the
        row after it, where the walk stopped.
    """
    last = None
    while 0 <= row < len(band):
        if band[row]:
            last, row = row, row + step
            continue
        ahead = row
        while (
            0 <= ahead < len(band) and covered[ahead] and abs(ahead - row) < _CROSSING
        ):
            ahead += step
        if ahead == row or not (0 <= ahead < len(band) and band[ahead]):
            break
        row = ahead
    return last, row


def _run(pixels: npt.NDArray[np.bool_], columns: slice) -> tuple[int, int]:
    """Return the first and last column of the run of pixels through some columns."""
    inside = np.flatnonzero(pixels[columns]) + columns.start
    gaps = np.flatnonzero(~pixels)
    before, after = gaps[gaps < inside.min()], gaps[gaps > inside.max()]
    return (
        int(before.max()) + 1 if len(before) else 0,
        int(after.min()) - 1 if len(after) else len(pixels) - 1,
    )


def _capped(runs: list[tuple[int, int]], column: int, half: int) -> int:
    """Return how many of the runs at the end of a bar make its cap.

    Those are the last runs wider than the bar, centred on its column within
    ``_CAP_SHIFT``, so long as there are no more of them than the bar is
    wide; otherwise there is no cap.
    """
    count = 0
    for first, end in reversed(runs):
        if end - first <= 2 * half + 1 or abs((first + end) / 2 - column) > _CAP_SHIFT:
            break
        count += 1
    return count if count <= 2 * half + 1 else 0
