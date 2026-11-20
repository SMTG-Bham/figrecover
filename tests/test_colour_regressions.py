"""Regression plots for automatic series selection, drawn at publication size."""

import io
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from matplotlib.figure import Figure

from figrecover import Image, find_series, trace_curves
from figrecover.cli import main
from figrecover.digitize import _stroke_ranges

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


def test_trace_ranges_enclose_the_known_three_curve_probe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    figure = Figure(figsize=(6.4, 4), dpi=100)
    axes = figure.subplots()
    x = np.linspace(0, 10, 4000)
    curves: list[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = [
        lambda x: np.sin(4 * x),
        lambda x: 0.25 * x - 1.25,
        lambda x: 1.2 * np.exp(-(((x - 5) / 0.08) ** 2)) - 1.3,
    ]
    for index, curve in enumerate(curves):
        axes.plot(x, curve(x), color=f"C{index}", linewidth=1.5)
    axes.set(xlim=(0, 10), ylim=(-1.5, 1.5))
    axes.spines[["top", "right"]].set_visible(False)
    path = tmp_path / "probe.png"
    figure.savefig(path, format="png", backend="agg")
    args = [str(path)]
    for corner in [(0, -1.5), (10, -1.5), (0, 1.5)]:
        a, b = axes.transData.transform(corner)
        args.extend(("-p", f"{corner[0]},{corner[1]}", "-l", f"{a},{b}"))
    for c in ("1f77b4", "ff7f0e", "2ca02c"):
        args.extend(("-c", c))
    assert main([*args, "--trace-range"]) == 0
    first = capsys.readouterr().out
    assert first.splitlines()[0] == "series,x,y,trace_low,trace_high"
    rows = np.loadtxt(io.StringIO(first), delimiter=",", skiprows=1)
    pixel = 3 / (
        axes.transData.transform((0, 1.5))[1] - axes.transData.transform((0, -1.5))[1]
    )
    for i, curve in enumerate(curves, 1):
        px, py, low, high = rows[rows[:, 0] == i, 1:].T
        truth = curve(px)
        assert len(px) == 493
        assert np.all(low <= py)
        assert np.all(py <= high)
        assert np.all(low - pixel / 2 <= truth)
        assert np.all(truth <= high + pixel / 2)
        large = np.abs(truth - py) / 3 > 0.005
        assert np.all((high - low)[large] > 1.5 * (1.5 * 100 / 72) * pixel)
    assert main(args) == 0
    # The additional columns leave the original CSV values byte-identical.
    assert capsys.readouterr().out == "".join(
        ",".join(row.split(",")[:-2]) + "\n" for row in first.splitlines()
    )


def test_interpolated_points_without_stroke_pixels_have_empty_ranges() -> None:
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    ink = (31, 119, 180)
    image[50:52, 10:53] = ink
    image[50:52, 57:107] = ink
    # One point in a horizontal gap, one between tracks, one on the stroke.
    curve = np.array([(54.5, 49), (60.5, 72), (61.5, 49)])
    (spans,) = _stroke_ranges(image, [ink], [curve], (0, 0))
    assert np.isnan(spans[:2]).all()
    np.testing.assert_array_equal(spans[2], (48, 50))
