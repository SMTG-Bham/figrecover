"""Find text annotations and their arrow or bracket shafts inside raster axes."""

import itertools
from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from .axes import _read_text, _Text
from .calibration import Point
from .digitize import Image, _Box, _framed_area
from .masks import Rectangle, painted

_CONTRAST = 26
_MIN_CONFIDENCE = 0.7
_MIN_LETTERS = 2
_MIN_WORD_ASPECT = 0.65  # short horizontal labels; vertical caps mimic letters
_MIN_GLYPH = 5
_MIN_SHAFT = 20
_MAX_SHAFT_RADIUS = 2.5
_END_SHARE = 0.12
_CAP_WIDTH = 3
_HEAD_REACH = 12


def _ink(image: Image) -> npt.NDArray[np.uint8]:
    """Contrast with the dominant paper colour, including on dark backgrounds."""
    paper = np.median(image, axis=(0, 1))
    return (np.max(np.abs(image - paper), axis=2) >= _CONTRAST).astype(np.uint8)


def _has_letters(ink: npt.NDArray[np.uint8]) -> bool:
    """Look for adjacent character-sized components before running optional OCR."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    glyphs = sorted(
        (x, y, w, h) for x, y, w, h, _ in stats[1:] if h >= _MIN_GLYPH and w <= 2 * h
    )
    return any(
        0 < x2 - x <= w + h and abs(y2 - y) <= h and h / 2 <= h2 <= 2 * h
        for (x, y, w, h), (x2, y2, _, h2) in itertools.pairwise(glyphs)
    )


def _rectangle(text: _Text) -> Rectangle:
    return (
        text.left - 2,
        text.bottom - 2,
        text.right + 2,
        text.top + 2,
    )


def _shaft(
    ys: npt.NDArray[np.intp], xs: npt.NDArray[np.intp], *, labelled: bool
) -> bool:
    """A thin straight shaft with a wider head or cap confined to an end."""
    points = np.column_stack((xs, ys)).astype(np.float32)
    vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L1, 0, 0.01, 0.01).ravel()
    shifted = points - (cx, cy)
    along = shifted @ np.array((vx, vy))
    signed = shifted @ np.array((-vy, vx))
    across = abs(signed)
    fraction = (along - along.min()) / np.ptp(along)
    middle = (fraction > _END_SHARE) & (fraction < 1 - _END_SHARE)
    radius = np.percentile(across[middle], 99)
    if radius > _MAX_SHAFT_RADIUS or np.max(across[~middle]) <= radius * 2 + 0.7:
        return False
    if labelled:
        return True
    # An isolated bracket has two transverse caps. An isolated arrow has
    # two wings that widen as they retreat from a narrow terminal point.
    caps = []
    for distance in (along - along.min(), along.max() - along):
        tip = signed[distance <= _CAP_WIDTH]
        rear = signed[(distance > _CAP_WIDTH) & (distance <= _HEAD_REACH)]
        tip_width = min(float(tip.max()), float(-tip.min()))
        rear_width = min(float(rear.max()), float(-rear.min()))
        if rear_width > max(3, 2 * radius) and rear_width > 1.75 * max(1, tip_width):
            return True
        caps.append(tip_width > max(4, 2.5 * radius) and tip_width > 2 * rear_width)
    return all(caps)


def _text_boxes(
    image: Image, area: _Box, frames: Sequence[_Box], texts: Sequence[_Text] | None
) -> list[Rectangle]:
    """Keep credible words wholly inside this plot and outside framed objects."""
    height = image.shape[0]
    roi = image[area.top : area.bottom, area.left : area.right]
    ink = _ink(roi)
    if texts is None:
        if not _has_letters(ink):
            return []
        try:
            texts = _read_text(image)
        except ModuleNotFoundError:
            return []
    boxes = []
    for text in texts:
        if (
            text.right - text.left < _MIN_WORD_ASPECT * (text.top - text.bottom)
            or text.score < _MIN_CONFIDENCE
            or (
                sum(c.isalpha() for c in text.text) < _MIN_LETTERS
                and not (
                    any(c.isalpha() for c in text.text)
                    and any(c.isdigit() for c in text.text)
                )
            )
        ):
            continue
        box = _rectangle(text)
        left, bottom, right, top = box
        local = _Box(
            round(height - top), round(height - bottom), round(left), round(right)
        )
        region = local.within(area)
        if (
            area.holds(local)
            and not any(frame.holds(region) for frame in frames)
            and np.any(ink[region.top : region.bottom, region.left : region.right])
        ):
            boxes.append(box)
    return boxes


def excluded(
    image: Image,
    origin: Point,
    texts: Sequence[_Text] | None,
    saved: Sequence[Rectangle] | None,
) -> tuple[Image, list[Rectangle], int]:
    """Mask plot text, adjacent headed shafts and isolated arrows or brackets."""
    area, frames = _framed_area(image, origin)
    roi = image[area.top : area.bottom, area.left : area.right]
    height = image.shape[0]
    boxes = _text_boxes(image, area, frames, texts) if saved is None else list(saved)
    result = painted(image, boxes)
    plot = result[area.top : area.bottom, area.left : area.right]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(_ink(plot))
    shafts = np.zeros(labels.shape, dtype=np.uint8)
    count = 0
    for index, (x, y, w, h, _) in enumerate(stats[1:], 1):
        if max(w, h) < _MIN_SHAFT:
            continue
        left, top = x + area.left, height - (y + area.top)
        near = any(
            max(x0 - left - w, left - x1, y0 - top, top - h - y1) < y1 - y0
            for x0, y0, x1, y1 in boxes
        )
        if near or 0.13 * max(plot.shape[:2]) <= max(w, h) <= 0.75 * max(
            plot.shape[:2]
        ):
            ys, xs = np.nonzero(labels[y : y + h, x : x + w] == index)
            if _shaft(ys, xs, labelled=near):
                shafts[y + ys, x + xs] = 1
                count += 1
    if count:
        edges = cv2.dilate(shafts, np.ones((3, 3), np.uint8))
        result = result.copy()
        result[area.top : area.bottom, area.left : area.right][edges > 0] = np.median(
            roi, axis=(0, 1)
        )
    return result, boxes, count
