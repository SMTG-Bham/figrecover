"""Draw recognised objects in class colours and label the colours on the image."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .digitize import Colour, Image
from .plot_objects import PlotObject

CLASS_COLOURS: dict[str, Colour] = {
    "background span": (204, 186, 128),
    "filled area": (168, 177, 182),
    "shaded band": (131, 192, 224),
    "axes and ticks": (86, 86, 86),
    "gridline": (127, 118, 137),
    "reference line": (195, 75, 44),
    "data line": (42, 101, 161),
    "connecting line": (30, 141, 151),
    "replicate trace": (74, 142, 102),
    "mean curve": (17, 98, 63),
    "fitted or theoretical curve": (120, 79, 156),
    "measurement markers": (0, 77, 109),
    "error bars": (205, 125, 0),
    "annotation": (176, 64, 128),
    "arrow or bracket": (146, 64, 30),
    "legend": (69, 63, 143),
}


def compose(image: Image, objects: Sequence[PlotObject]) -> tuple[Image, Counter[str]]:
    """Paint local masks onto the original coordinate system, with a readable key."""
    base = cv2.addWeighted(image, 0.33, np.full_like(image, 255), 0.67, 0).astype(
        np.uint8
    )
    counts = Counter(o.kind for o in objects)
    names = [name for name in CLASS_COLOURS if counts[name]]
    for name in names:
        for obj in objects:
            if obj.kind == name:
                left, top, right, bottom = obj.box
                x0, y0 = max(left, 0), max(top, 0)
                x1, y1 = min(right, base.shape[1]), min(bottom, base.shape[0])
                if x1 > x0 and y1 > y0:
                    mask = obj.mask[y0 - top : y1 - top, x0 - left : x1 - left]
                    base[y0:y1, x0:x1][mask] = CLASS_COLOURS[name]
    line_height = 23
    font = cv2.FONT_HERSHEY_SIMPLEX
    keywidth = (
        max((cv2.getTextSize(name, font, 0.48, 1)[0][0] for name in names), default=60)
        + 55
    )
    canvas = np.full(
        (max(len(base), line_height * (len(names) + 1)), base.shape[1] + keywidth, 3),
        255,
        np.uint8,
    )
    canvas[: len(base), : base.shape[1]] = base
    for i, name in enumerate(names, 1):
        x, y = base.shape[1] + 11, i * line_height
        cv2.rectangle(canvas, (x, y - 9), (x + 13, y + 1), CLASS_COLOURS[name], -1)
        cv2.putText(
            canvas, name, (x + 20, y + 1), font, 0.48, (24, 24, 24), 1, cv2.LINE_AA
        )
    return canvas, counts


def save(path: Path, image: Image, objects: Sequence[PlotObject]) -> Counter[str]:
    """Write the class view to a separate image."""
    canvas, counts = compose(image, objects)
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write classes image {path}")
    return counts
