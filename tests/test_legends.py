"""Recover named curves from shared vector and raster legend samples."""

import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from figrecover.axes import _Text
from figrecover.cli import main
from figrecover.legends import Entry, _joined_sample, raster_entries


def draw(path: Path, location: str) -> None:
    figure = Figure(figsize=(9, 4), dpi=100, layout="constrained")
    panels = figure.subplots(1, 2)
    x = np.linspace(0, 10, 201)
    xs = x[::16] if location == "markers" else x
    for index, panel in enumerate(panels):
        for i, name in enumerate(("Control", "Treated")):
            panel.plot(
                xs,
                1 + i * 0.5 + 0.5 * np.sin((0.6 if index == 0 else 1.4) * xs + 1.4 * i),
                label=name,
                color=("tab:blue", "tab:orange")[i],
                marker=("o", "^")[i] if location == "markers" else None,
                linestyle="none" if location == "markers" else "-",
            )
        panel.set(xlim=(0, 10), ylim=(0, 3), xlabel="Time (s)", ylabel="Signal")
        panel.set_title(f"({'ab'[index]})", loc="left")
    if location in {"outside", "markers"}:
        figure.legend(
            *panels[0].get_legend_handles_labels(), loc="outside upper center", ncols=2
        )
    else:
        panels[0].legend(loc="upper right", frameon=location != "unframed")
    figure.savefig(path, backend="pdf" if path.suffix == ".pdf" else "agg")


@pytest.mark.parametrize("extension", ["png", "pdf"])
@pytest.mark.parametrize("location", ["outside", "inside", "unframed", "markers"])
def test_shared_legend_names_masks_and_orders_both_panels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extension: str, location: str
) -> None:
    path = tmp_path / f"panels.{extension}"
    draw(path, location)
    keys = ("359,381", "478,381") if extension == "png" else ("259,275", "343,275")
    options = ["-m", keys[0], "-m", keys[1]] if location == "markers" else ["-n", "2"]
    assert main([str(path), *options]) == 0
    captured = capsys.readouterr()
    assert "Control = -c 1f77b4" in captured.err
    assert "Treated = -c ff7f0e" in captured.err
    header, *rows = csv.reader(io.StringIO(captured.out))
    assert header == ["panel", "series", "Time (s)", "Signal"]
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for panel, name, x, y in rows:
        groups[panel, name].append((float(x), float(y)))
    assert [name for _, name in groups] == ["Control", "Treated"] * 2
    for index, ((_, name), points) in enumerate(groups.items()):
        x, y = np.array(points).T
        i = int(name == "Treated")
        truth = 1 + i * 0.5 + 0.5 * np.sin((0.6 if index < 2 else 1.4) * x + 1.4 * i)
        assert x.max() - x.min() > 8
        assert np.quantile(abs(y - truth) / 3, 0.95) < 0.01


@pytest.mark.usefixtures("without_ocr")
def test_unframed_legend_is_masked_without_ocr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    figure = Figure(figsize=(6.4, 4), dpi=100)
    panel = figure.subplots()
    x = np.linspace(0, 10, 1000)
    panel.plot(x, 1 + 0.5 * np.sin(x), color="tab:blue", label="Control")
    panel.plot(x, 1.6 + 0.5 * np.cos(x), color="tab:orange", label="Treated")
    panel.legend(frameon=False)
    panel.set(xlim=(0, 10), ylim=(0, 3))
    path = tmp_path / "manual.png"
    figure.savefig(path, backend="agg")
    assert (
        main(
            [
                str(path),
                "-p",
                "0,0",
                "-p",
                "10,0",
                "-p",
                "0,3",
                "-l",
                "80,44",
                "-l",
                "576,44",
                "-l",
                "80,352",
                "-n",
                "2",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "legend" in captured.err
    _, *rows = csv.reader(io.StringIO(captured.out))
    assert {r[0] for r in rows} == {"1", "2"}
    for n in ("1", "2"):
        xx, yy = np.asarray([r[1:] for r in rows if r[0] == n], dtype=float).T
        truth = 1 + 0.5 * np.sin(xx) if n == "1" else 1.6 + 0.5 * np.cos(xx)
        assert np.quantile(abs(yy - truth) / 3, 0.95) < 0.01


@pytest.mark.parametrize("extension", ["png", "pdf"])
def test_shared_legend_appears_once_in_class_view(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extension: str
) -> None:
    source = tmp_path / f"outside.{extension}"
    view = tmp_path / "view.png"
    draw(source, "outside")
    assert main([str(source), "-n", "2", "--classes", str(view)]) == 0
    report = capsys.readouterr().err
    assert "legend (2)" in report
    assert "data line (4)" in report
    assert view.exists()


def test_legend_key_touching_the_data_is_measured_between_rows() -> None:
    image = np.full((100, 180, 3), 255, np.uint8)
    colours = [(31, 119, 180), (255, 127, 14), (44, 160, 44)]
    texts = []
    for i, (name, colour) in enumerate(
        zip(("Mean", "Replicate 1", "Theory"), colours, strict=True)
    ):
        y = 30 + 20 * i
        image[y : y + 2, 74:104] = colour
        texts.append(_Text(name, 110, 92 - y, 165, 106 - y))
    image[48:50, :76] = colours[0]
    found = raster_entries(image, texts)
    assert [(e.name, e.colour) for e in found] == list(
        zip(("Mean", "Replicate 1", "Theory"), colours, strict=True)
    )
    assert found[1].width == 30


def test_aligned_labels_alone_do_not_establish_a_legend_key() -> None:
    image = np.full((100, 180, 3), 255, np.uint8)
    word = _Text("Threshold", 100, 40, 155, 60)
    top = Entry("Mean", (0, 0, 0), (68, 72), (60, 67, 140, 79), 20, 2)
    for x in (68, 88):
        bottom = Entry("Theory", (0, 0, 0), (x, 30), (x - 10, 24, 145, 39), 20, 2)
        assert _joined_sample(image, word, [top, bottom]) is None
