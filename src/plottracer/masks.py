"""Rectangular exclusions in bottom-left image or page coordinates."""

import math
from collections.abc import Sequence

import numpy as np

from .digitize import Image

type Rectangle = tuple[float, float, float, float]


def painted(image: Image, boxes: Sequence[Rectangle]) -> Image:
    """Fill excluded pixel centres with the image's background colour."""
    if not boxes:
        return image
    result = image.copy()
    height, width = image.shape[:2]
    background = np.median(image, axis=(0, 1))
    for left, bottom, right, top in boxes:
        x0, x1 = max(0, math.ceil(left - 0.5)), min(width, math.floor(right - 0.5) + 1)
        y0 = max(0, math.ceil(height - top - 0.5))
        y1 = min(height, math.floor(height - bottom - 0.5) + 1)
        if x0 < x1 and y0 < y1:
            result[y0:y1, x0:x1] = background
    return result
