"""Greyscale curves in clean figures and simulated printed scans."""

import io

import cv2
import numpy as np
import pytest
from matplotlib.figure import Figure

from plottracer import Image, find_series
from plottracer.halftone import recover_greys


def render(fig: Figure) -> Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", backend="agg")
    bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None
    return np.asarray(bgr[..., ::-1], dtype=np.uint8).copy()


def print_scan(image: Image) -> Image:
    """Expose a 45-degree, 85 lpi dot screen at 600 dpi, then scan at 300 dpi."""
    grey = image[..., 0]
    rows, cols = np.indices(grey.shape, dtype=np.float32)
    scale = 2 * np.pi * (85 / 600) / np.sqrt(2)
    screen = (np.cos(scale * (cols + rows)) + np.cos(scale * (cols - rows)) + 2) / 4
    counts, bins = np.histogram(screen, bins=256, range=(0, 1))
    threshold = np.interp(screen, bins[1:], counts.cumsum() / screen.size)
    dots = (grey / 255 >= threshold).astype(np.uint8) * 255
    scan = cv2.resize(dots, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return np.repeat(scan[..., None], 3, axis=2).astype(np.uint8)


@pytest.mark.parametrize("phase", [0.0, 1.4, 2.6])
@pytest.mark.parametrize("width", [0.9, 1.5])
def test_printed_scan_preserves_two_crossing_greys(phase: float, width: float) -> None:
    fig = Figure(figsize=(3.2, 2.1), dpi=600)
    ax = fig.subplots()
    x = np.linspace(0, 10, 1000)
    truths = [1.6 + 0.6 * np.sin(x + phase), 1.2 + 0.7 * np.cos(x * 0.8)]
    for grey, truth in zip(["0.18", "0.6"], truths, strict=True):
        ax.plot(x, truth, color=grey, linewidth=width)
    ax.set(xlim=(0, 10), ylim=(0, 3))
    scan = print_scan(render(fig))
    origin = ax.transData.transform((0, 0)) / 2
    curves = sorted(
        find_series(scan, (origin[0], origin[1]))[:2], key=lambda s: s.colour
    )
    for series, truth in zip(curves, truths, strict=True):
        xx, yy = ax.transData.inverted().transform(series.points * 2).T
        assert len(np.unique(xx)) >= 0.9 * ax.bbox.width / 2
        assert max(abs(yy - np.interp(xx, x, truth))) / 3 <= 0.01


@pytest.mark.parametrize("reverse", [False, True])
def test_clean_four_greys_preserve_stroke_centres(reverse: bool) -> None:
    fig = Figure(figsize=(6.4, 4), dpi=100)
    ax = fig.subplots()
    x = np.linspace(0, 10, 2000)
    truths = [1.4 + 0.75 * np.sin(x * 0.7 + i) for i in range(4)]
    greys = ["black", "0.3", "0.5", "0.7"]
    for i in reversed(range(4)) if reverse else range(4):
        ax.plot(x, truths[i], color=greys[i], linewidth=1.5)
    ax.set(xlim=(0, 10), ylim=(0, 3))
    image = render(fig)
    assert recover_greys(image) is image
    curves = sorted(find_series(image, (80, 44)), key=lambda s: s.colour)
    for series, truth in zip(curves, truths, strict=True):
        xx, yy = ax.transData.inverted().transform(series.points).T
        assert len(np.unique(xx)) > 0.9 * ax.bbox.width
        assert max(abs(yy - np.interp(xx, x, truth))) / 3 <= 0.01
