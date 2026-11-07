"""Find the plots in an image of several, such as the panels of a figure."""

import math
import re
import string
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from .axes import Axes, _read_axes, _read_text, _Text
from .calibration import Point
from .digitize import AXIS_MARGIN, Image, ink_mask

_LINE_LENGTH = 0.1  # axis lines span this fraction of the image's shorter side or more
_LABEL_REACH = 0.25  # a label lies this fraction of its axes' size from their corner
# A label, such as (a), a), a. or a, or a character or none in brackets, as
# OCR reads some labels, such as (c) as ().
_LABEL = re.compile(r"[(\[]?([a-z])[)\].]?|[(\[]\S?[)\]]", re.IGNORECASE)
_MISREAD = {"b": "p", "d": "p"}  # OCR reads the labels (b) and (d) as (p)


@dataclass(frozen=True, slots=True, eq=False)
class Panel:
    """A plot in an image of one or more, such as a panel of a figure.

    Pixel positions in a panel are measured from the bottom-left corner of
    its image, as in any image, so add ``offset`` to place them in the whole
    image.

    Attributes:
        label: The letter of the panel's label, such as ``a`` for (a), or
            None if it has none that could be read.
        image: The panel's part of the whole image, in which to trace its
            curves: the whole image if it is the only panel, or else a part
            that runs up to the top and right of the panel's axes, as the
            image beyond them belongs to other panels. Its label is painted
            over in the background colour, inside its axes, as it is usually
            drawn in a curve's colour.
        offset: The pixel position, in the whole image, of the bottom-left
            corner of ``image``.
        axes: The panel's axes, in pixels of ``image``.
    """

    label: str | None
    image: Image
    offset: Point
    axes: Axes


@dataclass(frozen=True, slots=True)
class _Box:
    """A rectangle of an image, in rows from the top and columns from the left.

    ``bottom`` and ``right`` are one past its last row and column.
    """

    top: int
    bottom: int
    left: int
    right: int

    def holds(self, other: _Box) -> bool:
        """Return whether another box lies within this one."""
        return (
            self.top <= other.top
            and other.bottom <= self.bottom
            and self.left <= other.left
            and other.right <= self.right
        )

    def within(self, other: _Box) -> _Box:
        """Return this box measured from the top-left corner of another."""
        return _Box(
            self.top - other.top,
            self.bottom - other.top,
            self.left - other.left,
            self.right - other.left,
        )


def read_panels(image: Image) -> list[Panel]:
    """Find the plots in an image, such as the panels of a figure, and read their axes.

    The plots are found by their axes, as ``_boxes`` finds them: a horizontal
    and a vertical axis line that meet, as frames do too. An image with one,
    or none, such as a plot without axis lines, is one plot, whose axes are
    read from the whole image, as ``read_axes`` reads them.

    In an image of several, each plot's axes are read from its part of the
    image, as ``_parts`` finds it, which holds its tick labels and titles
    but no other plot's axes. Each plot's label, such as (a), is the text
    nearest the top-left corner of its axes written as a letter, as
    ``_label`` finds it. OCR misreads labels, such as (b) as (p), or misses
    them, but plots are labelled in reading order, so the letters read are
    taken for the alphabet in that order if most of them are, as
    ``_in_sequence`` finds. A plot whose axes cannot be read, such as a
    colour bar, or a plot whose ticks are labelled only on a plot beside it,
    is left out with a warning.

    Each plot's label is painted over in the background colour, inside its
    axes, in the image of the plot in which its curves are traced, so it is
    not traced as part of a curve of the same colour.

    The text is read with RapidOCR, which the ``ocr`` extra installs.

    Args:
        image: Colour image of one or more plots.

    Returns:
        The plots, in reading order: in rows from the top, each from the
        left.

    Raises:
        ModuleNotFoundError: If the ``ocr`` extra is not installed.
        ValueError: If the image is one plot whose axes cannot be read, as
            for ``read_axes``, or the axes of none of its plots can be read.
    """
    ink = ink_mask(image)
    boxes = _in_order(_boxes(ink))
    if len(boxes) <= 1:
        texts = _read_text(image)
        axes = _read_axes(image, texts)
        if not boxes:
            return [Panel(None, image, (0, 0), axes)]
        label = _label(texts, boxes[0], image.shape[0])
        return [Panel(_letter(label), _painted(image, label, boxes[0]), (0, 0), axes)]

    height = image.shape[0]
    read: list[tuple[_Text | None, Image, Point, Axes]] = []
    for box, part in zip(boxes, _parts(ink > 0, boxes), strict=True):
        piece = image[part.top : part.bottom, part.left : part.right]
        inside = box.within(part)
        texts = _read_text(piece)
        label = _label(texts, inside, piece.shape[0])
        try:
            axes = _read_axes(piece, texts)
        except ValueError as error:
            named = f"({letter}) " if (letter := _letter(label)) else ""
            msg = (
                f"left out the panel {named}at {box.left},{height - box.bottom}, "
                f"as its axes cannot be read: {error}"
            )
            warnings.warn(msg, stacklevel=2)
            continue
        traced = _painted(piece, label, inside)[inside.top :, : inside.right]
        read.append((label, traced, (part.left, height - part.bottom), axes))
    if not read:
        msg = f"cannot read the axes of any of the {len(boxes)} panels found"
        raise ValueError(msg)
    letters = _in_sequence([_letter(label) for label, *_ in read])
    return [
        Panel(letter, traced, offset, axes)
        for letter, (_, traced, offset, axes) in zip(letters, read, strict=True)
    ]


def _boxes(ink: npt.NDArray[np.uint8]) -> list[_Box]:
    """Find the boxes around the axes of the plots in an image, by their lines.

    A plot's axes are a horizontal line and a vertical one, each at least
    ``_LINE_LENGTH`` of the image's shorter side long, that meet, with the
    other lines that meet them, such as the rest of a frame, or grid lines.
    The box around them takes in the tick marks that run on from the lines
    at the corners. Boxes within another's, such as an inset's, or a
    legend's with square corners, are left out.
    """
    # An odd length, as OpenCV shifts what an even one keeps by a pixel.
    length = round(_LINE_LENGTH * min(ink.shape)) | 1
    horizontal, vertical = (
        cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, shape)
        )
        for shape in ((length, 1), (1, length))
    )
    _, pieces, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_or(horizontal, vertical)
    )
    both = set(np.unique(pieces[horizontal > 0])) & set(np.unique(pieces[vertical > 0]))
    boxes = [
        _Box(int(top), int(top + rows), int(left), int(left + columns))
        for left, top, columns, rows, _ in stats[sorted(both)]
    ]
    return [
        box
        for box in boxes
        if not any(other is not box and other.holds(box) for other in boxes)
    ]


def _in_order(boxes: Sequence[_Box]) -> list[_Box]:
    """Return boxes in reading order: in rows from the top, each from the left.

    A box is in the row of the topmost box above it if its top lies above
    that box's middle, so a tall box beside two short ones comes before both.
    """
    rows: list[list[_Box]] = []
    for box in sorted(boxes, key=lambda box: box.top):
        if rows and 2 * box.top < rows[-1][0].top + rows[-1][0].bottom:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [box for row in rows for box in sorted(row, key=lambda box: box.left)]


def _parts(ink: npt.NDArray[np.bool_], boxes: Sequence[_Box]) -> list[_Box]:
    """Return the part of the image in which to read the axes of each plot.

    A plot's part runs to the edges of the image, but no further than
    ``AXIS_MARGIN`` short of the boxes of the plots above, below and left of
    it, so it holds all its tick labels and titles, including the last
    x-axis tick label, which can stick out right of its axes, and text above
    them, such as a scale factor. It also holds text of the plots around it,
    such as the x-axis tick labels and title of the plot above, which lie
    neither below its x-axis tick labels nor left of its y-axis ones, where
    its titles are read. On the right, it stops at the text of the plot on
    the right, as ``_beside`` finds it, so that plot's y-axis tick labels
    are not read as a second y-axis.
    """
    height, width = ink.shape
    parts = []
    for box in boxes:
        top, bottom, left, right = 0, height, 0, width
        for other in boxes:
            across = other.left < box.right and box.left < other.right
            level = other.top < box.bottom and box.top < other.bottom
            if across and other.bottom <= box.top:
                top = max(top, min(other.bottom + AXIS_MARGIN, box.top))
            elif across and box.bottom <= other.top:
                bottom = min(bottom, max(other.top - AXIS_MARGIN, box.bottom))
            elif level and other.right <= box.left:
                left = max(left, min(other.right + AXIS_MARGIN, box.left))
            elif level and box.right <= other.left:
                right = min(right, _beside(ink, box, other))
        parts.append(_Box(top, bottom, left, right))
    return parts


def _beside(ink: npt.NDArray[np.bool_], box: _Box, other: _Box) -> int:
    """Return the first column of the text of a plot on the right of another.

    That is the first column with ink between the two boxes, in the rows
    that both span, past any ink that runs on from the box on the left,
    such as tick marks on the right of its axes, or the left of the other
    box, if there is none.
    """
    rows = slice(max(box.top, other.top), min(box.bottom, other.bottom))
    inked = ink[rows, box.right : other.left].any(axis=0)
    start = len(inked) if inked.all() else int(np.argmin(inked))
    ahead = np.flatnonzero(inked[start:])
    return box.right + start + (int(ahead[0]) if len(ahead) else len(inked) - start)


def _label(texts: Sequence[_Text], box: _Box, height: int) -> _Text | None:
    """Return the label of a plot, or None if it has none.

    The label is the text nearest the top-left corner of the plot's box, by
    its middle, written as a label, as ``_LABEL`` matches, and lying within
    ``_LABEL_REACH`` of the box's width and height of that corner, as labels
    lie inside the axes, above them or left of them. ``height`` is that of
    the image, to measure the box in pixels from the bottom.
    """
    corner = (box.left, height - box.top)
    reach = (
        _LABEL_REACH * (box.right - box.left),
        _LABEL_REACH * (box.bottom - box.top),
    )
    labels = [
        text
        for text in texts
        if _LABEL.fullmatch(text.plain)
        and abs(text.x - corner[0]) <= reach[0]
        and abs(text.y - corner[1]) <= reach[1]
    ]
    return min(
        labels, key=lambda label: math.dist((label.x, label.y), corner), default=None
    )


def _letter(label: _Text | None) -> str | None:
    """Return the letter of a label, or None if it has none that can be read."""
    match = _LABEL.fullmatch(label.plain) if label else None
    return match[1] if match else None


def _in_sequence(letters: Sequence[str | None]) -> list[str | None]:
    """Return the letters of plots' labels, or of the alphabet if most are those.

    The letters, of the plots in reading order, are None where no label was
    read. Plots are labelled in reading order, so if more than half the
    letters read are those of the alphabet in that order, ignoring case, or
    letters that OCR misreads them as, each plot takes its letter of the
    alphabet, as a capital if most of those read are. Otherwise a letter
    read for more than one plot is dropped.
    """
    alphabet = string.ascii_lowercase
    if len(letters) <= len(alphabet):
        expected = alphabet[: len(letters)]
        matched = [
            letter
            for letter, wanted in zip(letters, expected, strict=True)
            if letter is not None
            and letter.lower() in {wanted, _MISREAD.get(wanted, wanted)}
        ]
        if 2 * len(matched) > len(letters) - letters.count(None):
            capitals = 2 * sum(letter.isupper() for letter in matched) > len(matched)
            return list(expected.upper() if capitals else expected)
    counts = Counter(letters)
    return [letter if counts[letter] == 1 else None for letter in letters]


def _painted(image: Image, text: _Text | None, box: _Box) -> Image:
    """Return an image with a text's box painted in its median colour, if any.

    Only the part of the text's box inside a plot's box, ``AXIS_MARGIN``
    clear of its edges, is painted, as the box of a label inside the axes
    can take in their lines, which mark where the plot area ends.
    """
    if text is None:
        return image
    height = image.shape[0]
    top = max(height - math.ceil(text.top), box.top + AXIS_MARGIN)
    bottom = min(height - math.floor(text.bottom), box.bottom - AXIS_MARGIN)
    left = max(math.floor(text.left), box.left + AXIS_MARGIN)
    right = min(math.ceil(text.right), box.right - AXIS_MARGIN)
    painted = image.copy()
    painted[top : max(bottom, top), left : max(right, left)] = np.median(
        image.reshape(-1, 3), axis=0
    )
    return painted
