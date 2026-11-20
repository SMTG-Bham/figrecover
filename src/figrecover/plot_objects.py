"""Objects recognised inside a plot, using bottom-left figure coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .digitize import Colour, Image


@dataclass(frozen=True, slots=True, eq=False)
class PlotObject:
    """One graphic with a local mask and its array origin in the source image.

    The optional coordinate is the bottom-left pixel coordinate of a reference
    line along its axis. The value is calibrated when a calibration is supplied.
    """

    kind: str
    colour: tuple[int, int, int]
    left: int
    top: int
    mask: npt.NDArray[np.bool_]
    axis: str | None = None
    coordinate: float | None = None
    value: float | None = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Left, top, right and bottom of the local mask in image array coordinates."""
        h, w = self.mask.shape
        return self.left, self.top, self.left + w, self.top + h


def without_objects(image: Image, objects: Sequence[PlotObject]) -> Image:
    """Paint the pixels of non-data objects with the background colour."""
    if not objects:
        return image
    cleaned = image.copy()
    paper = np.median(image.reshape(-1, 3)[::16], axis=0)
    for obj in objects:
        if obj.kind in {
            "reference line",
            "gridline",
            "annotation",
            "arrow or bracket",
            "legend",
        }:
            left, top, right, bottom = obj.box
            cleaned[top:bottom, left:right][obj.mask] = paper
    return cleaned


_EDGE_ALPHA = 0.05
_MIN_COVER = 0.13  # permits dotted patterns
_MIN_REACH = 0.83  # nearly spans the axes (an open frame includes a margin)
_MAX_THICKNESS = 8
_MIN_CONTRAST = 32
_INK_DISTANCE = 36
_SECTIONS = 12
_SMALL_GAP = 3
_MIN_REPEATS = 3


def _sections(pixels: npt.NDArray[np.bool_], ink: npt.NDArray[np.bool_]) -> bool:
    """Require a continuous or repeating rule, allowing occlusion by other ink."""
    pos = np.flatnonzero(pixels)
    if not len(pos) or np.ptp(pos) < len(pixels) * _MIN_REACH:
        return False
    span = pixels[pos[0] : pos[-1] + 1]
    if not all(np.any(part) for part in np.array_split(span, _SECTIONS)):
        return False
    gaps = ~ink[pos[0] : pos[-1] + 1]
    edges = np.flatnonzero(np.diff(np.r_[False, gaps, False])).reshape(-1, 2)
    lengths = np.diff(edges, axis=1).ravel()
    # A flat baseline interrupted by a peak is data. A dashed or dotted
    # rule has many similar gaps; a solid one continues behind other ink.
    return bool(
        not len(lengths)
        or lengths.max() <= _SMALL_GAP
        or (
            len(lengths) >= _MIN_REPEATS and lengths.max() <= 3 * np.median(lengths) + 2
        )
    )


def _rows(
    plot: Image, paper: npt.NDArray[np.float64]
) -> list[tuple[int, int, Colour, npt.NDArray[np.bool_]]]:
    """Group thin neighbouring rows which repeat one ink across the plot."""
    delta = plot.astype(np.int16) - paper
    contrast = np.abs(delta).max(axis=2)
    ink = contrast >= _MIN_CONTRAST
    relevant = np.flatnonzero(ink.mean(axis=1) >= _MIN_COVER)
    if not len(relevant):
        return []
    groups = np.split(relevant, np.flatnonzero(np.diff(relevant) > 1) + 1)
    rows = []
    for group in groups:
        if len(group) > _MAX_THICKNESS:
            continue
        # Strongest common colour in the most heavily inked row.
        row = int(group[np.argmax(contrast[group].sum(axis=1))])
        colours, counts = np.unique(plot[row][ink[row]], axis=0, return_counts=True)
        colour = colours[counts.argmax()]
        own = np.abs(plot[group].astype(np.int16) - colour).max(axis=2) <= _INK_DISTANCE
        if own.any(axis=0).mean() < _MIN_COVER or not _sections(
            own.any(axis=0), ink[group].any(axis=0)
        ):
            continue
        start, end = max(0, int(group[0]) - 1), min(len(plot), int(group[-1]) + 2)
        # Include adjacent antialiasing in the observed columns. Preserve
        # different-colour ink where a curve is drawn over the reference.
        ray = colour - paper
        patch = delta[start:end]
        fraction = np.clip((patch @ ray) / (ray @ ray), 0, 1)
        residual = abs(patch - fraction[..., None] * ray).max(axis=2)
        mask = (residual <= _INK_DISTANCE) & (fraction > _EDGE_ALPHA) & own.any(axis=0)
        rows.append(
            (start, end, (int(colour[0]), int(colour[1]), int(colour[2])), mask)
        )
    return rows


def references(
    plot: Image, left: int, top: int, height: int, *, include_flat_data: bool = False
) -> list[PlotObject]:
    """Find horizontal and vertical reference strokes, preserving a lone data line."""
    paper = np.median(plot.reshape(-1, 3)[::16], axis=0)
    result = []
    for axis, view in (("y", plot), ("x", plot.transpose(1, 0, 2))):
        for start, _, colour, mask in _rows(view, paper):
            positions, _ = np.nonzero(mask)
            centre = start + float(np.mean(positions)) + 0.5
            if axis == "y":
                result.append(
                    PlotObject(
                        "reference line",
                        colour,
                        left,
                        top + start,
                        mask,
                        axis,
                        height - top - centre,
                    )
                )
            else:
                result.append(
                    PlotObject(
                        "reference line",
                        colour,
                        left + start,
                        top,
                        mask.T,
                        axis,
                        left + centre,
                    )
                )
    if result:
        clean = without_objects(
            plot,
            [
                PlotObject(o.kind, o.colour, o.left - left, o.top - top, o.mask)
                for o in result
            ],
        )
        ink = (np.abs(clean.astype(float) - paper).max(axis=2) > _MIN_CONTRAST).astype(
            np.uint8
        )
        # Require independent data with appreciable extent after removing
        # the references; a lone constant series can itself fill an axis.
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
        if not any(
            max(w, h) > min(plot.shape[:2]) * 0.08 for _, _, w, h, _ in stats[1:]
        ):
            return (
                [
                    PlotObject("data line", o.colour, o.left, o.top, o.mask)
                    for o in result
                ]
                if include_flat_data
                else []
            )
    return result
