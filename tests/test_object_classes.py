"""Inspect the object classes and the data that survives each exclusion."""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from plottracer import (
    Calibration,
    Image,
    PlotObject,
    class_view,
    classify,
    find_series,
    load_image,
)
from plottracer.cli import main
from plottracer.digitize import _band_edges
from plottracer.legends import Entry, line_patterns
from plottracer.plot_objects import without_objects

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FIT = Calibration.from_points(
    [(0, 0), (10, 0), (0, 3)], [(80, 44), (576, 44), (80, 352)]
)


def variable_plot(path: Path, kind: str, colour: str) -> Image:
    figure = Figure(figsize=(6.4, 4), dpi=100)
    ax = figure.subplots()
    x = np.linspace(0, 10, 500)
    y = 1.4 + 0.5 * np.sin(x)
    ax.plot(x, y, color="tab:blue", lw=1.5)
    if kind == "gridline":
        ax.grid(color=colour, linestyle="--", linewidth=0.8)
    elif kind == "shaded band":
        ax.fill_between(x, y - 0.2, y + 0.2, color=colour, alpha=0.35)
    elif kind == "background span":
        ax.axvspan(2, 4, color=colour, alpha=0.2)
    elif kind == "filled area":
        ax.fill_between([4, 6], [2.3, 2.3], [2.8, 2.8], color=colour, alpha=0.4)
    elif kind in {"arrow", "bracket"}:
        ax.add_patch(
            FancyArrowPatch(
                (2, 2.35),
                (6, 2.65),
                arrowstyle="->" if kind == "arrow" else "|-|",
                mutation_scale=14,
                color=colour,
            )
        )
    elif kind == "annotation":
        ax.annotate(
            "peak here",
            xy=(2, 2.15),
            xytext=(4, 2.7),
            color=colour,
            arrowprops={"arrowstyle": "->", "color": colour},
        )
    ax.set(xlim=(0, 10), ylim=(0, 3))
    figure.savefig(path, backend="agg")
    return load_image(path)


@pytest.mark.parametrize(
    "kind",
    [
        "gridline",
        "shaded band",
        "background span",
        "filled area",
        "annotation",
        "arrow",
        "bracket",
    ],
)
@pytest.mark.parametrize("colour", ["tab:blue", "tab:orange"])
def test_class_exclusion_preserves_only_the_data(
    tmp_path: Path, kind: str, colour: str
) -> None:
    image = variable_plot(tmp_path / "classes.png", kind, colour)
    objects = classify(image, FIT.origin, calibration=FIT)
    kinds = Counter(o.kind for o in objects)
    expected = "arrow or bracket" if kind in {"arrow", "bracket"} else kind
    assert kinds[expected] >= 1
    assert kinds["axes and ticks"] == kinds["data line"] == 1
    if kind == "annotation":
        assert "arrow or bracket" in kinds
    (series,) = find_series(without_objects(image, objects), FIT.origin)
    x, y = FIT.to_data(series.points).T
    assert len(x) > 450
    assert np.quantile(abs(y - 1.4 - 0.5 * np.sin(x)) / 3, 0.95) <= 0.01


def test_legend_roles_require_visible_semantic_evidence(tmp_path: Path) -> None:
    fig = Figure(figsize=(6.4, 4), dpi=100)
    ax = fig.subplots()
    x = np.linspace(0, 10, 500)
    for i, label in enumerate(("Mean", "Replicate 1", "Theory")):
        ax.plot(x, 0.5 + i + 0.2 * np.sin(x + i), label=label)
    ax.legend(frameon=False)
    ax.set(xlim=(0, 10), ylim=(0, 3))
    path = tmp_path / "roles.png"
    fig.savefig(path, backend="agg")
    image = load_image(path)
    objs = classify(image, FIT.origin)
    assert Counter(o.kind for o in objs) == {
        "axes and ticks": 1,
        "legend": 3,
        "mean curve": 1,
        "replicate trace": 1,
        "fitted or theoretical curve": 1,
    }
    for line in find_series(without_objects(image, objs), FIT.origin):
        i = [(31, 119, 180), (255, 127, 14), (44, 160, 44)].index(line.colour)
        xx, yy = FIT.to_data(line.points).T
        assert np.quantile(abs(yy - (0.5 + i + 0.2 * np.sin(xx + i))) / 3, 0.95) <= 0.01


def test_connected_markers_and_ambiguous_dashes(tmp_path: Path) -> None:
    fig = Figure(figsize=(6.4, 4), dpi=100)
    ax = fig.subplots()
    x = np.linspace(0, 10, 20)
    ax.plot(x, 1 + 0.4 * np.sin(x), color="tab:blue", marker="o", ms=7)
    ax.plot(x, 2 + 0.2 * np.sin(x), color="tab:orange", linestyle="--")
    ax.set(xlim=(0, 10), ylim=(0, 3))
    path = tmp_path / "markers.png"
    fig.savefig(path, backend="agg")
    items = classify(load_image(path), FIT.origin)
    kinds = Counter(o.kind for o in items)
    assert (
        kinds["connecting line"]
        == kinds["measurement markers"]
        == kinds["data line"]
        == 1
    )
    assert not kinds["fitted or theoretical curve"]


def test_error_bar_classes_include_visible_groups() -> None:
    items = classify(load_image(EXAMPLES / "errorbars.png"), FIT.origin)
    kinds = Counter(o.kind for o in items)
    assert kinds["measurement markers"] == kinds["error bars"] == 4


@pytest.mark.parametrize("extension", ["png", "pdf"])
def test_class_view_leaves_csv_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extension: str
) -> None:
    path = EXAMPLES / f"labelled.{extension}"
    output = tmp_path / "classes.png"
    assert main([str(path)]) == 0
    data = capsys.readouterr().out
    assert main([str(path), "--classes", str(output)]) == 0
    run = capsys.readouterr()
    assert data == run.out
    assert "classes:" in run.err
    assert "data line (1)" in run.err
    saved = cv2.imread(str(output))
    assert saved is not None
    image = cv2.cvtColor(saved, cv2.COLOR_BGR2RGB)
    rgb = class_view.CLASS_COLOURS["data line"]
    assert (image[:, :400] == rgb).all(axis=2).sum() > 50


def test_batch_class_views_protect_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = EXAMPLES / "labelled.pdf", EXAMPLES / "labelled.png"
    directory = tmp_path / "classes"
    assert main([str(first), str(second), "--classes", str(directory)]) == 0
    assert sorted(p.name for p in directory.iterdir()) == [
        "labelled.2.classes.png",
        "labelled.classes.png",
    ]
    with pytest.raises(SystemExit):
        main([str(first), "--classes", str(first)])
    assert "would overwrite the image" in capsys.readouterr().err


def test_clipped_and_off_page_class_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.full((20, 30, 3), 255, np.uint8)
    mask = np.ones((4, 5), dtype=bool)
    objects = [PlotObject("data line", (0, 0, 0), x, 18, mask) for x in (-2, 40)]
    canvas, counts = class_view.compose(image, objects)
    assert counts["data line"] == 2
    assert (canvas[18:20, :3] == class_view.CLASS_COLOURS["data line"]).all()
    monkeypatch.setattr("cv2.imwrite", lambda *_: False)
    with pytest.raises(OSError, match="could not write classes image"):
        class_view.save(tmp_path / "output.png", image, [])


def test_single_flat_measurement_and_grid_only() -> None:
    image = np.full((160, 240, 3), 255, np.uint8)
    image[70:72, 4:239] = 0
    assert [o.kind for o in classify(image, entries=[], texts=[])] == ["data line"]
    (line,) = find_series(image)
    assert np.ptp(line.points[:, 1]) == 0
    image[50:52, 4:239] = 0
    image[90:92, 4:239] = 0
    assert find_series(image) == []


def test_visible_error_bars_without_a_separate_marker_series(tmp_path: Path) -> None:
    fig = Figure(figsize=(6.4, 4), dpi=100)
    ax = fig.subplots()
    x = np.linspace(0, 10, 300)
    y = 1 + 0.5 * np.sin(x)
    ax.plot(x, y, color="tab:blue")
    ax.errorbar(x[150], y[150], yerr=0.3, color="tab:blue", marker="o", ms=9, capsize=4)
    ax.set(xlim=(0, 10), ylim=(0, 3))
    path = tmp_path / "single-error.png"
    fig.savefig(path, backend="agg")
    kinds = {o.kind for o in classify(load_image(path), FIT.origin)}
    assert {"measurement markers", "error bars", "data line"} <= kinds


@pytest.mark.parametrize("second_style", ["--", "-"])
def test_shared_ink_needs_distinct_legend_evidence(
    tmp_path: Path, second_style: str
) -> None:
    fig = Figure(figsize=(6.4, 4), dpi=100)
    ax = fig.subplots()
    x = np.linspace(0, 10, 500)
    for i, (label, style) in enumerate((("Mean", "-"), ("Fitted", second_style))):
        ax.plot(
            x, 1 + i + 0.2 * np.sin(x), label=label, linestyle=style, color="tab:blue"
        )
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.legend(loc="upper right", frameon=False)
    path = tmp_path / "one-ink.png"
    fig.savefig(path, backend="agg")
    image = load_image(path)
    objects = classify(image, FIT.origin)
    kinds = Counter(o.kind for o in objects)
    if second_style == "--":
        assert kinds["mean curve"] == kinds["fitted or theoretical curve"] == 1
        (fitted,) = (o for o in objects if o.kind == "fitted or theoretical curve")
        left, top, right, bottom = fitted.box
        pixels = image[top:bottom, left:right][fitted.mask]
        assert np.all(pixels.min(axis=1) < 240)
        assert np.count_nonzero(~fitted.mask.any(axis=0)) > 20
    else:
        assert kinds["data line"] == 2
    series = find_series(without_objects(image, objects), FIT.origin)
    assert len(series) == 2
    for stroke in series:
        xx, yy = FIT.to_data(stroke.points).T
        baseline = round(float(np.mean(yy)))
        assert baseline in {1, 2}
        assert np.quantile(abs(yy - baseline - 0.2 * np.sin(xx)) / 3, 0.95) <= 0.01


def test_ink_along_only_one_side_of_a_fill_remains_data() -> None:
    labels = np.zeros((80, 100), np.intp)
    labels[30:60, 10:91] = 1
    labels[27:30, 10:91] = 2
    assert _band_edges(labels, {1}) == set()


def test_off_panel_keys_preserve_only_observed_patterns() -> None:
    image = np.full((100, 140, 3), 255, np.uint8)
    entries = [
        Entry(label, (0, 0, 0), (150, y), (140, y - 2, 190, y + 2), 20, 1)
        for label, y in (("Mean", 50), ("Fit", 30))
    ]
    assert line_patterns(image, entries) == entries


def test_class_view_of_lone_stroke_has_no_exclusions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = np.full((160, 240, 3), 255, np.uint8)
    image[70:72, 4:239] = 0
    path = tmp_path / "level.png"
    assert cv2.imwrite(str(path), image)
    assert (
        main(
            [
                str(path),
                "-p",
                "0,0",
                "-p",
                "1,0",
                "-p",
                "0,1",
                "-l",
                "0,0",
                "-l",
                "160,0",
                "-l",
                "0,80",
                "--no-annotations",
                "--classes",
                str(tmp_path / "classes.png"),
            ]
        )
        == 0
    )
    report = capsys.readouterr().err
    assert "classes: data line (1)" in report
    assert "excluded from line traces" not in report
