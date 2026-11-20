"""Classify visible plot objects in source pixel coordinates."""

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from . import annotations, legends
from .axes import _Text
from .calibration import Calibration, Point
from .digitize import (
    _AREA_DEPTH,
    Colour,
    Image,
    Series,
    _bands,
    _depth,
    _framed_area,
    _plot_area,
    _stroke_inks,
    find_error_bars,
    find_series,
    ink_mask,
)
from .masks import Rectangle, painted
from .plot_objects import PlotObject, references, without_objects

_MIN_GRID = 3
_SPAN = 0.85
_MIN_VISIBLE_INK = 0.1
_MAX_COLOUR_RESIDUAL = 48
_ROLES = (
    (re.compile(r"\b(mean|average)\b", re.IGNORECASE), "mean curve"),
    (
        re.compile(r"\b(fit(?:ted)?|model|theor(?:y|etical))\b", re.IGNORECASE),
        "fitted or theoretical curve",
    ),
    (re.compile(r"\b(replicate|rep)\s*\d*\b", re.IGNORECASE), "replicate trace"),
)


def reference_objects(
    image: Image, origin: Point, calibration: Calibration | None
) -> list[PlotObject]:
    """Find solid and dashed references and regularly spaced parallel gridlines."""
    plot, left, top = _plot_area(image, origin)
    objects = references(plot, left, top, image.shape[0])
    result = []
    for obj in objects:
        axis = int(obj.axis == "y")
        position = cast("float", obj.coordinate)
        value = (
            float(calibration.to_data([(position, position)])[0, axis])
            if calibration
            else position
        )
        peers = sorted(
            cast("float", o.coordinate)
            for o in objects
            if o.axis == obj.axis and o.colour == obj.colour
        )
        gaps = np.diff(peers)
        is_grid = len(peers) >= _MIN_GRID and np.ptp(gaps) <= max(
            2, 0.08 * np.median(gaps)
        )
        result.append(
            replace(obj, value=value, kind="gridline" if is_grid else obj.kind)
        )
    return result


def _object(
    kind: str, colour: Colour, mask: npt.NDArray[np.bool_], left: int = 0, top: int = 0
) -> list[PlotObject]:
    """Crop a nonempty object to its occupied pixels."""
    rows, cols = np.nonzero(mask)
    if not rows.size:
        return []
    x0, x1 = int(cols.min()), int(cols.max() + 1)
    y0, y1 = int(rows.min()), int(rows.max() + 1)
    return [PlotObject(kind, colour, left + x0, top + y0, mask[y0:y1, x0:x1].copy())]


def _ink_object(
    kind: str, image: Image, mask: npt.NDArray[np.bool_]
) -> list[PlotObject]:
    """Measure the strongest visible ink of an axis, word or arrow."""
    pixels = image[mask]
    if not len(pixels):
        return []
    paper = np.median(image.reshape(-1, 3)[::16], axis=0)
    r, g, b = pixels[np.argmax(np.linalg.norm(pixels - paper, axis=1))]
    return _object(kind, (int(r), int(g), int(b)), mask)


def _rectangle_mask(image: Image, boxes: Sequence[Rectangle]) -> npt.NDArray[np.bool_]:
    return np.any(painted(image, boxes) != image, axis=2)


def _axis_object(image: Image, origin: Point) -> list[PlotObject]:
    area, _ = _framed_area(image, origin)
    h, w = image.shape[:2]
    x, y = round(origin[0]), h - round(origin[1])
    mask = np.zeros((h, w), dtype=bool)
    ink = ink_mask(image) > 0

    def mark(strip: npt.NDArray[np.bool_], target: npt.NDArray[np.bool_]) -> None:
        # A few data pixels near an open side do not establish an axis.
        if np.any(strip.sum(axis=1) > strip.shape[1] * 0.6):
            target[:] |= strip

    for row in (y, area.top - 3):
        rows = slice(max(0, row - 5), min(h, row + 6))
        cols = slice(max(0, x - 1), min(w, area.right + 4))
        mark(ink[rows, cols], mask[rows, cols])
    for col in (x, area.right + 2):
        cols = slice(max(0, col - 5), min(w, col + 6))
        rows = slice(max(0, area.top - 4), min(h, y + 1))
        mark(ink[rows, cols].T, mask[rows, cols].T)
    return _ink_object("axes and ticks", image, mask)


def _fills(image: Image, origin: Point) -> list[PlotObject]:
    plot, left, top = _plot_area(image, origin)
    labels, palette, measured = _stroke_inks(plot)
    depths = [_depth(labels == i) for i in range(1, len(palette))]
    bands = _bands(labels, palette, depths)
    result = []
    for i, depth in enumerate(depths, 1):
        if depth <= _AREA_DEPTH and i not in bands:
            continue
        mask = labels == i
        # A full-height, broad filled column marks a stage. A strip following
        # a thinner stroke is a band. Other fill lacks either kind of evidence.
        spans = np.any(mask.mean(axis=0) > _SPAN)
        kind = (
            "background span"
            if spans
            else "shaded band"
            if i in bands
            else "filled area"
        )
        r, g, b = measured[i]
        result.extend(_object(kind, (int(r), int(g), int(b)), mask, left, top))
    return result


def _marks(image: Image, series: Series, kind: str) -> list[PlotObject]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    radius = max(1, round(series.width / 2))
    for x, y in series.points:
        cv2.circle(mask, (int(x), image.shape[0] - 1 - int(y)), radius, 1, -1)
    return _object(kind, series.colour, mask.astype(bool))


def _connects(series: Series, centres: npt.NDArray[np.float64]) -> bool:
    """Require the line to pass through at least two detected marker centres."""
    if len(centres) <= 1:
        return False
    radius = max(2, series.width)
    return sum(
        bool(np.min(np.linalg.norm(series.points - p, axis=1)) <= radius)
        for p in centres
    ) >= 0.75 * len(centres)


def _data_objects(
    image: Image, origin: Point, entries: Sequence[legends.Entry]
) -> list[PlotObject]:
    series = find_series(image, origin)
    colours = list(dict.fromkeys(s.colour for s in series))
    result = []
    bars = find_error_bars(image, colours, origin)
    centres = {colour: bar.points for colour, bar in zip(colours, bars, strict=True)}
    marker_inks = {s.colour for s in series if s.style == "markers"}
    for colour, found in zip(colours, bars, strict=True):
        if len(found.points) and colour not in marker_inks:
            result.extend(
                _marks(
                    image,
                    Series(colour, "markers", 3, found.points),
                    "measurement markers",
                )
            )
        ink = np.zeros(image.shape[:2], np.uint8)
        for centre, low, high in zip(found.points, found.low, found.high, strict=True):
            for axis in range(2):
                ends = [centre.copy(), centre.copy()]
                for point, value in zip(ends, (low[axis], high[axis]), strict=True):
                    if np.isfinite(value):
                        point[axis] = value
                if ends[0][axis] != ends[1][axis]:
                    a, b = [(int(x), image.shape[0] - 1 - int(y)) for x, y in ends]
                    cv2.line(ink, a, b, 1, 2)
        result.extend(_object("error bars", colour, ink.astype(bool)))
    for stroke in series:
        if stroke.style == "markers":
            kind = "measurement markers"
        else:
            keys = [
                e
                for e in entries
                if e.broken is None or e.broken == (stroke.style != "solid")
            ]
            index = legends.closest(stroke.colour, keys)
            name = keys[index].name if index < len(keys) else ""
            kind = next(
                (role for pattern, role in _ROLES if pattern.search(name or "")),
                "data line",
            )
            if kind == "data line" and _connects(stroke, centres[stroke.colour]):
                kind = "connecting line"
        result.extend(_marks(image, stroke, kind))
    # Tracing interpolates across dashes and occlusions. Object masks show
    # only the portion of that path painted in the measured series ink.
    paper = np.median(image.reshape(-1, 3)[::16], axis=0)
    visible = []
    for obj in result:
        left, top, right, bottom = obj.box
        pixels = image[top:bottom, left:right] - paper
        ray = np.asarray(obj.colour) - paper
        fraction = np.clip((pixels @ ray) / max(1, ray @ ray), 0, 1)
        residual = np.max(np.abs(pixels - fraction[..., None] * ray), axis=2)
        mask = (
            obj.mask & (fraction > _MIN_VISIBLE_INK) & (residual < _MAX_COLOUR_RESIDUAL)
        )
        visible.extend(_object(obj.kind, obj.colour, mask, left, top))
    return visible


def legend_objects(image: Image, entries: Sequence[legends.Entry]) -> list[PlotObject]:
    """Return the visible ink inside each identified legend row."""
    return [
        o
        for entry in entries
        for o in _object("legend", entry.colour, _rectangle_mask(image, [entry.box]))
    ]


def classify(
    image: Image,
    origin: Point = (0, 0),
    *,
    calibration: Calibration | None = None,
    entries: Sequence[legends.Entry] | None = None,
    texts: Sequence[_Text] | None = None,
) -> list[PlotObject]:
    """Return observable plot classes, with local masks and calibrated reference values.

    Roles such as fitted, mean or replicate require a matching printed label.
    Text classification uses the optional OCR extra. Coordinates of class masks
    are measured from the top left of the image. Reference values use the
    pixel axes unless a calibration is supplied.
    """
    entries = legends.raster_entries(image, texts) if entries is None else entries
    entries = legends.line_patterns(image, entries)
    result = _axis_object(image, origin)
    rectangles = [e.box for e in entries]
    result.extend(legend_objects(image, entries))
    clean = painted(image, rectangles)
    cleaned, boxes, _ = annotations.excluded(clean, origin, texts, None)
    for box in boxes:
        result.extend(_ink_object("annotation", clean, _rectangle_mask(clean, [box])))
    after_text = painted(clean, boxes)
    arrows = np.any(after_text != cleaned, axis=2)
    count, labels = cv2.connectedComponents(arrows.astype(np.uint8))
    for index in range(1, count):
        result.extend(_ink_object("arrow or bracket", after_text, labels == index))
    refs = reference_objects(cleaned, origin, calibration)
    result.extend(refs)
    cleaned = without_objects(cleaned, refs)
    result.extend(_fills(cleaned, origin))
    result.extend(_data_objects(cleaned, origin, entries))
    return result
