"""Tests for the command-line interface, using the example plots."""

import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from figrecover import Image, Point, __version__, interactive
from figrecover.cli import main

type Curve = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

EXAMPLES = Path(__file__).parents[1] / "examples"
POINTS = ["-p", "0,0", "-p", "10,0", "-p", "0,1"]
# The pixel positions of POINTS that examples/make_examples.py prints.
LOCATIONS = ["-l", "80,44", "-l", "576,44", "-l", "80,147"]
GRID_LOCATIONS = ["-l", "80,44", "-l", "576,44", "-l", "80,167"]
SPECTRUM = [str(EXAMPLES / "spectrum.png"), *POINTS]
GRID = [str(EXAMPLES / "grid.png"), *POINTS]


def spectrum(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in spectrum.png."""
    return 0.5 + 2 * np.exp(-2 * (x - 3) ** 2) + np.exp(-((x - 7) ** 2) / 2)


def damped(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in grid.png."""
    return 1.2 + np.exp(-x / 4) * np.cos(2 * x)


def read_csv(text: str) -> npt.NDArray[np.float64]:
    """Parse CSV output, checking its header."""
    header, _, body = text.partition("\n")
    assert header == "x,y"
    return np.loadtxt(body.splitlines(), delimiter=",", ndmin=2)


def assert_recovers(text: str, curve: Curve) -> None:
    """Check that CSV output follows the true curve across the whole x-axis."""
    data = read_csv(text)
    assert data[:, 0].min() < 0.1
    assert data[:, 0].max() > 9.9
    np.testing.assert_allclose(data[:, 1], curve(data[:, 0]), atol=0.04)


def test_writes_csv_to_standard_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*SPECTRUM, *LOCATIONS]) == 0
    assert_recovers(capsys.readouterr().out, spectrum)


def test_writes_csv_and_preview_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output, preview = tmp_path / "data.csv", tmp_path / "preview.png"

    arguments = [*SPECTRUM, *LOCATIONS, "-o", str(output), "--preview", str(preview)]
    assert main(arguments) == 0

    assert capsys.readouterr().out == ""
    assert_recovers(output.read_text(), spectrum)
    image = cv2.imread(str(preview))
    assert image is not None
    assert image.shape == (400, 640, 3)


def test_removes_grids(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*GRID, *GRID_LOCATIONS, "--remove-grid"]) == 0
    assert_recovers(capsys.readouterr().out, damped)


def test_handles_dark_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dark = tmp_path / "dark.png"
    image = cv2.imread(str(EXAMPLES / "spectrum.png"))
    assert image is not None
    cv2.imwrite(str(dark), cv2.bitwise_not(image))

    assert main([str(dark), *POINTS, *LOCATIONS]) == 0
    assert_recovers(capsys.readouterr().out, spectrum)


def test_asks_for_missing_locations_and_prints_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[Sequence[str]] = []

    def pick_points(_image: Image, labels: Sequence[str]) -> list[Point]:
        asked.append(labels)
        return [(80, 44), (576, 44), (80, 147)]

    monkeypatch.setattr(interactive, "pick_points", pick_points)

    assert main(SPECTRUM) == 0
    assert asked == [["0,0", "10,0", "0,1"]]
    captured = capsys.readouterr()
    assert captured.err == f"figrecover: locations: {' '.join(LOCATIONS)}\n"
    assert_recovers(captured.out, spectrum)


def test_can_be_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    def pick_points(_image: Image, _labels: Sequence[str]) -> list[Point]:
        raise KeyboardInterrupt

    monkeypatch.setattr(interactive, "pick_points", pick_points)

    assert main(SPECTRUM) == 130


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["missing.png", *POINTS, *LOCATIONS], "cannot read image"),
        ([*SPECTRUM, "-l", "80,44"], "got 3 data points but 1 pixel locations"),
        ([*SPECTRUM, *LOCATIONS, "--preview", "no/such/dir.png"], "cannot write"),
    ],
)
def test_reports_errors(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 1
    assert capsys.readouterr().err.startswith(f"figrecover: error: {message}")


def test_accepts_points_with_spaces_and_exponents(
    capsys: pytest.CaptureFixture[str],
) -> None:
    points = ["-p", " 0 , 0 ", "-p", "1e1,0", "-p", "0,1.0"]

    assert main([str(EXAMPLES / "spectrum.png"), *points, *LOCATIONS]) == 0
    assert_recovers(capsys.readouterr().out, spectrum)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["plot.png", "-p", "0,0", "-p", "1,0"], "at least 3 reference points"),
        (["plot.png", "-p", "0"], "expected a point written as x,y, got '0'"),
        (["plot.png", "-p", "1,2,3"], "expected a point written as x,y"),
        (["plot.png", "-p", "a,b"], "expected a point written as x,y"),
        (["plot.png"], "required: -p/--point"),
        ([], "required: image"),
    ],
)
def test_usage_errors_exit_with_status_2(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"{__version__}\n"


def test_module_runs_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["figrecover", "--version"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("figrecover", run_name="__main__")

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"{__version__}\n"
