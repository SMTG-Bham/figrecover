"""Recognise aligned runs of legend keys and text in figure coordinates."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .axes import _read_text, _Text
from .calibration import Point
from .digitize import Colour, Image
from .masks import Rectangle

_MIN_KEY = 5
_MAX_KEY = 72
_INK = 50
_MIN_SCORE = 0.65


@dataclass(frozen=True)
class Entry:
    """Printed legend label, sampled colour and full row bounds."""

    name: str | None
    colour: Colour
    key: Point
    box: Rectangle
    width: float
    height: float
    broken: bool | None = None


def _aligned(a: Entry, b: Entry) -> bool:
    """Legend entries share a row or a column and similar sample dimensions."""
    dx, dy = abs(a.key[0] - b.key[0]), abs(a.key[1] - b.key[1])
    em = max(a.box[3] - a.box[1], b.box[3] - b.box[1])
    return (dx <= em and dy <= 2.2 * em) or (dy <= em / 3 and dx <= 14 * em)


def grouped(entries: list[Entry]) -> list[Entry]:
    """Retain aligned entries in reading order."""
    kept = [
        a for a in entries if any(a.key != b.key and _aligned(a, b) for b in entries)
    ]
    return sorted(kept, key=lambda a: (-round(a.key[1] / 4), a.key[0]))


def _shapes(image: Image) -> list[tuple[Rectangle, Colour]]:
    paper = np.median(image.reshape(-1, 3)[::16], axis=0)
    ink = (np.abs(image.astype(float) - paper).max(axis=2) > _INK).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    shapes = []
    height = image.shape[0]
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not (
            _MIN_KEY <= w <= _MAX_KEY
            and 0 < h < _MAX_KEY / 2
            and w >= 0.65 * h
            and area >= w
        ):
            continue
        pixels = image[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i]
        distance = ((pixels - paper) ** 2).sum(axis=1)
        r, g, b = pixels[distance.argmax()]
        shapes.append(
            (
                (float(x), float(height - y - h), float(x + w), float(height - y)),
                (int(r), int(g), int(b)),
            )
        )
    return shapes


def _words(image: Image) -> list[_Text]:
    """Locate unrecognised words when OCR is unavailable."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ink = (abs(grey.astype(float) - np.median(grey)) > _INK).astype(np.uint8)
    joined = cv2.dilate(ink, np.ones((1, 5), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    height = image.shape[0]
    return [
        _Text(
            "", float(x + 2), float(height - y - h), float(x + w - 2), float(height - y)
        )
        for x, y, w, h, _ in stats[1:]
        if _MIN_KEY < h < _MAX_KEY / 2 and w > 1.8 * h
    ]


def raster_entries(image: Image, texts: Sequence[_Text] | None = None) -> list[Entry]:
    """Match a compact sample to the left of each printed word."""
    if texts is None:
        candidates = _matched_entries(image, _words(image))
        if not candidates:
            return []
        provisional = grouped(candidates)
        try:
            named = raster_entries(image, _read_text(image))
        except ModuleNotFoundError:
            return provisional
        # OCR can merge the printed sample with the word. Match the word's
        # right edge to the geometric row, whose key is independent of OCR.
        result = []
        for entry in named:
            aligned = [
                p
                for p in provisional
                if abs(p.box[2] - entry.box[2]) < entry.box[3] - entry.box[1]
                and abs(p.key[1] - entry.key[1]) < (entry.box[3] - entry.box[1]) / 3
            ]
            result.append(
                replace(aligned[0], name=entry.name) if len(aligned) == 1 else entry
            )
        accounted = {e.key for e in result}
        return grouped(
            result
            + [
                p
                for p in provisional
                if p.key not in accounted and any(_aligned(p, n) for n in result)
            ]
        )
    return grouped(_matched_entries(image, texts))


def _matched_entries(image: Image, texts: Sequence[_Text]) -> list[Entry]:
    """Pair individual samples with adjacent words before testing alignment."""
    shapes = _shapes(image)
    entries = []
    for text in texts:
        if text.score < _MIN_SCORE or (
            text.text and not any(c.isalpha() for c in text.text)
        ):
            continue
        em = text.top - text.bottom
        options = [
            (box, colour)
            for box, colour in shapes
            if 0 <= text.left - box[2] <= 2 * em
            and abs((box[1] + box[3]) / 2 - text.y) < 0.43 * em
            and box[2] - box[0] >= 0.35 * em
            and box[3] - box[1] <= 1.2 * em
        ]
        if not options:
            continue
        box, colour = max(options, key=lambda p: p[0][2])
        x0, y0, x1, y1 = box
        entries.append(
            Entry(
                text.text or None,
                colour,
                ((x0 + x1) / 2, (y0 + y1) / 2),
                (
                    x0 - 2,
                    min(y0, text.bottom) - 2,
                    text.right + 2,
                    max(y1, text.top) + 2,
                ),
                x1 - x0,
                y1 - y0,
            )
        )
    return entries


def line_patterns(image: Image, entries: Sequence[Entry]) -> list[Entry]:
    """Distinguish continuous and broken samples when two labels share an ink."""
    result = []
    for entry in entries:
        peers = [
            e for e in entries if e.colour == entry.colour and e.width >= 2 * e.height
        ]
        width = max((e.width for e in peers), default=0)
        if len({e.name for e in peers}) <= 1 or entry.width < 2 * entry.height:
            result.append(entry)
            continue
        right = round(entry.key[0] + entry.width / 2)
        x = max(0, round(right - width) - 1)
        y = round(image.shape[0] - entry.key[1])
        strip = image[max(0, y - 2) : max(0, y + 3), x : max(0, right + 1)]
        close = np.abs(strip.astype(np.int16) - entry.colour).max(axis=2) <= _INK
        positions = np.flatnonzero(close.any(axis=0))
        measured = entry
        if len(positions) and np.ptp(positions) >= 0.7 * width:
            first, last = float(x + positions[0]), float(x + positions[-1] + 1)
            measured = replace(
                entry,
                broken=bool(np.any(np.diff(positions) > 1)),
                key=((first + last) / 2, entry.key[1]),
                width=last - first,
                box=(min(entry.box[0], first - 2), *entry.box[1:]),
            )
        result.append(measured)
    return result


def closest(colour: Colour, entries: Sequence[Entry]) -> int:
    """Return the nearest key when the ink matches; unmatched curves sort last."""
    if not entries:
        return 0
    delta = np.asarray([e.colour for e in entries]) - colour
    distances = np.linalg.norm(delta, axis=1)
    index = int(np.argmin(distances))
    # A shared colour by itself cannot associate a name with a specific marker shape.
    competing = {
        e.name
        for e, d in zip(entries, distances, strict=True)
        if d <= distances[index] + 2
    }
    return index if distances[index] <= _INK and len(competing) == 1 else len(entries)


def marker_entry(key: Point, entries: Sequence[Entry]) -> Entry | None:
    """Locate the entry for a user-selected marker in full-figure coordinates."""
    return next(
        (
            e
            for e in entries
            if e.box[0] <= key[0] <= e.box[2] and e.box[1] <= key[1] <= e.box[3]
        ),
        None,
    )


def scoped(entries: Sequence[Entry], bounds: Rectangle) -> list[Entry]:
    """Prefer a local legend where panels give conflicting names to the same ink."""

    def inside(entry: Entry) -> bool:
        return (
            bounds[0] <= entry.key[0] <= bounds[2]
            and bounds[1] <= entry.key[1] <= bounds[3]
        )

    preferred = [e for e in entries if inside(e)]
    return [
        e
        for e in entries
        if inside(e)
        or not any(p.colour == e.colour and p.name != e.name for p in preferred)
    ]


def excluding(entries: Sequence[Entry], offset: Point) -> list[Rectangle]:
    """Translate legend rows to panel coordinates."""
    x, y = offset
    return [
        (left - x, bottom - y, right - x, top - y)
        for left, bottom, right, top in (e.box for e in entries)
    ]
