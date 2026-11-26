"""Locate captioned figures on a PDF page from text and drawing bounds."""

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .axes import _Text

type Box = tuple[float, float, float, float]

_CAPTION = re.compile(
    r"^(?:(?:supplementary|suppl\.?)\s+)?(?:figure\s+|fig\.\s*|fig\s+)"
    r"(S?\d+[A-Za-z]?)\b(.*)$",
    re.IGNORECASE,
)
_LOWERCASE = re.compile(r"^[a-z]+\b")
_CAPTION_WORDS = ("continued", "in vitro", "in vivo", "in situ", "in silico")
_GROUP_GAP = 6  # points between pieces of a flat vector drawing
_MIN_SIZE = 16  # points across and down, to leave out rules and small ornaments
_LABEL_REACH = 24  # points from a flat drawing to its tick labels or axis titles
_PROSE_LENGTH = 35  # characters in one typeset line of an article


@dataclass(frozen=True)
class Graphic:
    """A graphic's page bounds, including any labels if it is a form or image."""

    box: Box
    self_contained: bool


def _label(text: str) -> str | None:
    """Read a caption label, excluding mentions of a figure in prose."""
    match = _CAPTION.match(text.strip())
    if not match:
        return None
    label, tail = match.groups()
    tail = tail.strip()
    if tail == "." or tail.startswith((",", ";", ")")):
        return None
    if _LOWERCASE.match(tail) and not tail.startswith(_CAPTION_WORDS):
        return None
    return label.upper()


def _union(a: Box, b: Box) -> Box:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _gap(a: Box, b: Box) -> tuple[float, float]:
    return max(0, a[0] - b[2], b[0] - a[2]), max(0, a[1] - b[3], b[1] - a[3])


def _grouped(graphics: Iterable[Graphic], page: Box) -> list[Graphic]:
    """Join nearby strokes into figures and ignore page backgrounds."""
    groups: list[Graphic] = []
    for graphic in graphics:
        box = graphic.box
        self_contained = graphic.self_contained
        if (
            box[0] <= page[0] + 1
            and box[1] <= page[1] + 1
            and box[2] >= page[2] - 1
            and box[3] >= page[3] - 1
        ):
            continue
        others = []
        for group in groups:
            if max(_gap(box, group.box)) <= _GROUP_GAP:
                box = _union(box, group.box)
                self_contained = self_contained and group.self_contained
            else:
                others.append(group)
        groups = [*others, Graphic(box, self_contained)]
    return [
        g for g in groups if min(g.box[2] - g.box[0], g.box[3] - g.box[1]) >= _MIN_SIZE
    ]


def _bounds(text: _Text) -> Box:
    return text.left, text.bottom, text.right, text.top


def _body(text: _Text) -> bool:
    return len(text.text) >= _PROSE_LENGTH and sum(
        c.isalpha() for c in text.text
    ) >= 0.7 * len(text.text)


def _distance(caption: _Text, box: Box, body: Sequence[_Text]) -> float:
    """Measure the gap to a figure in the caption's column, stopping at body text."""
    dx, dy = _gap(_bounds(caption), box)
    if dx > _LABEL_REACH:
        return math.inf
    bottom, top = sorted((caption.y, (box[1] + box[3]) / 2))
    if any(
        bottom < line.y < top
        and _gap(_bounds(line), box)[0] == 0
        and not box[1] <= line.y <= box[3]
        for line in body
    ):
        return math.inf
    return dy + dx + 0.05 * abs(caption.left - box[0])


def select_figure(
    texts: Sequence[_Text],
    graphics: Iterable[Graphic],
    page: Box,
    number: int,
    wanted: str | None,
) -> Box | None:
    """Return a selected figure's box, or list choices for an ambiguous page."""
    captions = [(label, text) for text in texts if (label := _label(text.text))]
    if wanted is None and len(captions) <= 1:
        return None
    groups = _grouped(graphics, page)
    body = [text for text in texts if _body(text) and not _label(text.text)]
    matches = sorted(
        (distance, i, j)
        for i, (_, caption) in enumerate(captions)
        for j, group in enumerate(groups)
        if math.isfinite(distance := _distance(caption, group.box, body))
    )
    selected: dict[str, Box] = {}
    used: set[int] = set()
    for _, i, j in matches:
        label = captions[i][0]
        if label in selected or j in used:
            continue
        group = groups[j]
        box = group.box
        if not group.self_contained:
            for line in texts:
                if (
                    not _label(line.text)
                    and not _body(line)
                    and max(_gap(group.box, _bounds(line))) < _LABEL_REACH
                ):
                    box = _union(box, _bounds(line))
        selected[label] = (
            max(page[0], math.floor(box[0]) - 1),
            max(page[1], math.floor(box[1]) - 1),
            min(page[2], math.ceil(box[2]) + 1),
            min(page[3], math.ceil(box[3]) + 1),
        )
        used.add(j)
    if wanted is not None and wanted.upper() in selected:
        return selected[wanted.upper()]
    if wanted is None and len(selected) <= 1:
        return None
    choices = ", ".join(
        f"figure {label}: --box {','.join(f'{x:g}' for x in box)}"
        for label, box in sorted(
            selected.items(), key=lambda item: (-item[1][3], item[1][0])
        )
    )
    issue = f"no figure {wanted}" if wanted else "choose a figure with --figure"
    msg = f"page {number}: {issue}" + (f" ({choices})" if choices else "")
    raise ValueError(msg)
