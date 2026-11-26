"""Reference line positions and exclusion from same-ink curves."""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from matplotlib.figure import Figure

from plottracer import Calibration, PlotObject, classify, find_series, trace_curves
from plottracer.cli import main
from plottracer.plot_objects import without_objects


@pytest.mark.parametrize("colour", ["black", "tab:red"])
@pytest.mark.parametrize("pattern", ["-", "--", ":", "-."])
def test_reference_lines_leave_only_data(colour: str, pattern: str) -> None:
    fig = Figure(figsize=(6.4, 4), dpi=100)
    ax = fig.subplots()
    x = np.linspace(0, 10, 2000)
    ax.plot(x, 1.2 + np.exp(-x / 4) * np.cos(2 * x), color="black", lw=1.5)
    ax.axhline(2.5, color=colour, linestyle=pattern, linewidth=1)
    ax.axvline(4.5, color=colour, linestyle=pattern, linewidth=1)
    ax.set(xlim=(0, 10), ylim=(0, 3))
    data = io.BytesIO()
    fig.savefig(data, format="png", backend="agg")
    bgr = cv2.imdecode(np.frombuffer(data.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None
    image = np.asarray(bgr[..., ::-1], dtype=np.uint8).copy()
    fit = Calibration.from_points(
        [(0, 0), (10, 0), (0, 3)], [(80, 44), (576, 44), (80, 352)]
    )
    objects = classify(image, fit.origin, calibration=fit)
    references = [s for s in objects if s.kind == "reference line"]
    assert {s.axis for s in references} == {"x", "y"}
    for obj in references:
        expected, step = (4.5, fit.x_scale) if obj.axis == "x" else (2.5, fit.y_scale)
        assert obj.value is not None
        assert abs(expected - obj.value) <= abs(step)
    for obj in classify(image, fit.origin):
        if obj.axis:
            assert obj.value == obj.coordinate
    (found,) = find_series(without_objects(image, objects), fit.origin)
    (selected,) = trace_curves(image, [(0, 0, 0)], fit.origin, objects=objects)
    for traced in (found.points, selected):
        xx, yy = fit.to_data(traced).T
        e = abs(yy - 1.2 - np.exp(-xx / 4) * np.cos(2 * xx)) / 3
        assert len(np.unique(xx)) > 0.9 * ax.bbox.width
        assert np.quantile(e, 0.95) <= 0.01


def test_standalone_flat_line_retains_its_data_role() -> None:
    image = np.full((160, 240, 3), 255, np.uint8)
    image[70:72, 4:239] = 0
    assert not any(o.kind == "reference line" for o in classify(image, (0, 0)))


def test_pdf_reference_lines_report_data_coordinates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    figure = Figure(figsize=(6.4, 4), dpi=100)
    axes = figure.subplots()
    axes.plot([0, 5, 10], [0.4, 1.5, 1.1], color="black")
    axes.axhline(2.5, color="black", linestyle="--")
    axes.axvline(4.5, color="tab:red", linestyle=":")
    axes.set(xlim=(0, 10), ylim=(0, 3))
    path = tmp_path / "references.pdf"
    figure.savefig(path)
    assert main([str(path), "-n", "1"]) == 0
    output = capsys.readouterr()
    assert "reference line at y = 2.5" in output.err
    assert "reference line at x = 4.5" in output.err
    assert len(output.out.splitlines()) == 4


def test_supplied_data_objects_remain_visible() -> None:
    image = np.full((100, 120, 3), 255, np.uint8)
    image[30:34, 10:80] = 0
    line = PlotObject("data line", (0, 0, 0), 10, 30, np.ones((4, 70), dtype=bool))
    np.testing.assert_array_equal(without_objects(image, [line]), image)
