"""Regression plots for exclusions in images and page coordinates."""

import csv
import io
import json
import shlex
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from figrecover import annotations
from figrecover.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("vector", [False, True])
def test_repeatable_rectangles_exclude_intervals(
    vector: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factor = 0.72 if vector else 1
    image = EXAMPLES / ("series.pdf" if vector else "series.png")
    args = [str(image), "-p", "0,0", "-p", "10,0", "-p", "0,1", "-c", "1f77b4"]
    for x, y in [(80, 44), (576, 44), (80, 147)]:
        args.extend(("-l", f"{x * factor},{y * factor}"))
    for left, right in [(200, 230), (300, 350), (800, 830)]:
        args.extend(("--mask", f"{left * factor},0,{right * factor},{400 * factor}"))
    sidecar = tmp_path / "masked.json"
    assert main([*args, "--metadata", str(sidecar)]) == 0
    original = capsys.readouterr().out
    points = np.array(list(csv.reader(io.StringIO(original)))[1:], dtype=float)
    pixels = points[:, 0] * 49.6 + 80
    assert len(points) > 20
    for left, right in [(200, 230), (300, 350)]:
        assert not ((pixels > left + 0.1) & (pixels < right - 0.1)).any()
    replay = shlex.split(json.loads(sidecar.read_text())["reproduce"])
    assert main(replay[1:]) == 0
    assert capsys.readouterr().out == original


def annotated_figure(path: Path, shape: str) -> None:
    figure = Figure(figsize=(6.4, 4), dpi=100)
    ax = figure.subplots()
    x = np.linspace(0, 10, 2000)
    ax.plot(x, 1.2 + np.exp(-x / 4) * np.cos(2 * x), color="black", linewidth=1.5)
    if shape == "arrow":
        ax.annotate(
            "first peak",
            xy=(0, 2.2),
            xytext=(2, 2.7),
            arrowprops={"arrowstyle": "->", "color": "black"},
        )
        ax.text(6, 2.5, "T = 300 K\nn = 12", fontsize=10)
    if shape == "bracket":
        ax.add_patch(
            FancyArrowPatch(
                (2, 2.4), (6, 2.4), arrowstyle="|-|", mutation_scale=10, color="black"
            )
        )
        ax.text(3.7, 2.47, "ΔT")
    if shape == "watermark":
        ax.text(
            5,
            1.7,
            "DRAFT",
            rotation=22,
            ha="center",
            fontsize=38,
            color="black",
            alpha=0.75,
        )
    if shape == "gradient":
        ax.imshow(
            np.repeat(np.linspace(0.5, 1, 200)[:, None], 200, axis=1),
            extent=(0, 10, 0, 3),
            aspect="auto",
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="lower",
        )
    ax.set(xlim=(0, 10), ylim=(0, 3), xlabel="Time (s)", ylabel="Voltage (mV)")
    ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, backend="agg")


@pytest.mark.parametrize("shape", ["arrow", "bracket", "watermark", "gradient"])
def test_plot_annotations_preserve_the_curve(
    shape: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "annotated.png"
    annotated_figure(image, shape)
    assert main([str(image), "-c", "000000"]) == 0
    captured = capsys.readouterr()
    header, *points = csv.reader(io.StringIO(captured.out))
    assert header == ["Time (s)", "Voltage (mV)"]
    x, y = np.asarray(points, dtype=float).T
    assert len(x) > 440
    assert np.quantile(abs(y - 1.2 - np.exp(-x / 4) * np.cos(2 * x)) / 3, 0.95) < 0.01
    if shape != "gradient":
        assert "in the plot area" in captured.err


def test_manual_annotation_replay_without_ocr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "bracket.png"
    annotated_figure(image, "bracket")
    points = [
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
    ]
    sidecar = tmp_path / "run.json"
    assert main([str(image), *points, "--metadata", str(sidecar)]) == 0
    original = capsys.readouterr()
    assert "in the plot area" in original.err
    replay = shlex.split(json.loads(sidecar.read_text())["reproduce"])
    monkeypatch.setattr(annotations, "_read_text", unavailable)
    assert main(replay[1:]) == 0
    assert capsys.readouterr().out == original.out
    assert main([str(image), *points]) == 0
    assert "in the plot area" not in capsys.readouterr().err
    assert (
        main([str(image), *points, "--no-annotations", "--metadata", str(sidecar)]) == 0
    )
    assert "--no-annotations" in json.loads(sidecar.read_text())["reproduce"]


def unavailable(*_args: object, **_kwargs: object) -> list[object]:
    raise ModuleNotFoundError("OCR is unavailable")


def test_short_data_mark_next_to_text_is_preserved() -> None:
    image = np.full((130, 180, 3), 255, dtype=np.uint8)
    image[50:55, 90:97] = 0
    image[68:72, 80:117] = 0
    masked, _, count = annotations.excluded(image, (0, 0), None, [(78, 55, 119, 65)])
    assert np.array_equal(masked[50:55, 90:97], image[50:55, 90:97])
    assert count == 0
