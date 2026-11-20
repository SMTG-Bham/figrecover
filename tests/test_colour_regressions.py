"""Regression plots for automatic series selection, drawn at publication size."""

import io

import cv2
import numpy as np
import pytest
from matplotlib.figure import Figure

from figrecover import Image, find_series, trace_curves

X = np.linspace(0, 10, 2000)
ORIGIN = (80, 44)  # default axes on a 6.4 by 4 inch figure at 100 dpi


def image_of(figure: Figure) -> Image:
    """Render a plot to the RGB pixels that the CLI reads from a PNG."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", backend="agg")
    bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None
    return np.asarray(bgr[..., ::-1], dtype=np.uint8).copy()


@pytest.mark.parametrize("width", [2, 2.5, 3])
def test_thick_crossing_colours_have_one_trace_each(width: float) -> None:
    # Use the six known waves from the colour benchmark.
    rng = np.random.default_rng(2026)
    phases = rng.uniform(0, 2 * np.pi, 6)
    amplitudes = rng.uniform(0.6, 1.4, 6)
    frequencies = rng.uniform(0.4, 1.2, 6)
    figure = Figure(figsize=(6.4, 4), dpi=100)
    axes = figure.subplots()
    colours = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]
    for index, (amp, freq, phase) in enumerate(
        zip(amplitudes, frequencies, phases, strict=True)
    ):
        axes.plot(
            X, 1.75 + amp * np.sin(freq * X + phase), color=f"C{index}", linewidth=width
        )
    axes.set(xlim=(0, 10), ylim=(0, 3.5))
    image = image_of(figure)

    found = find_series(image, ORIGIN)
    assert len(found) == len({s.colour for s in found}) == len(colours)
    for ink, plain, amp, freq, phase in zip(
        colours,
        trace_curves(image, colours, ORIGIN),
        amplitudes,
        frequencies,
        phases,
        strict=True,
    ):
        series = min(found, key=lambda s: np.linalg.norm(np.subtract(s.colour, ink)))
        assert series.style == "solid"
        np.testing.assert_array_equal(series.points, plain)
        x, y = axes.transData.inverted().transform(series.points).T
        error = np.abs(y - (1.75 + amp * np.sin(freq * x + phase))) / 3.5
        assert round(float(np.percentile(error, 95)), 3) <= 0.007


@pytest.mark.parametrize("over", [True, False], ids=["over", "under"])
@pytest.mark.parametrize("half_width", [0.15, 0.25, 0.4])
@pytest.mark.parametrize("layout", ["near", "parallel", "crossing"])
def test_find_series_ranks_the_curves_ahead_of_their_bands(
    layout: str, half_width: float, over: bool
) -> None:
    if layout == "near":
        curves = (1.4 + 0.3 * np.sin(X), 1.6 + 0.3 * np.sin(X + 1))
    elif layout == "parallel":
        curves = (1.3 + 0.4 * np.sin(X), 1.6 + 0.4 * np.sin(X))
    else:
        curves = (1.5 + 0.5 * np.sin(X), 0.6 + 0.2 * X)
    figure = Figure(figsize=(6.4, 4), dpi=100)
    axes = figure.subplots()
    for index, y in enumerate(curves):
        axes.plot(X, y, color=f"C{index}", zorder=2)
        axes.fill_between(
            X,
            y - half_width,
            y + half_width,
            color=f"C{index}",
            alpha=0.3,
            linewidth=0,
            zorder=3 if over else 1,
        )
    axes.set(xlim=(0, 10), ylim=(0, 3))
    found = find_series(image_of(figure), ORIGIN)[:2]

    matched = set()
    for series in found:
        x, y = axes.transData.inverted().transform(series.points).T
        errors = [np.abs(y - np.interp(x, X, truth)) / 3 for truth in curves]
        nearest = int(np.argmin([np.median(error) for error in errors]))
        assert np.percentile(errors[nearest], 95) <= 0.005
        matched.add(nearest)
    assert matched == {0, 1}
