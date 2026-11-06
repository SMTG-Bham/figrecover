"""Tests for the command-line interface, using the example plots."""

import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from figrecover import Axes, Calibration, Image, Point, __version__, cli, interactive
from figrecover.cli import main

type Curve = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

EXAMPLES = Path(__file__).parents[1] / "examples"
POINTS = ["-p", "0,0", "-p", "10,0", "-p", "0,1"]
# The pixel positions of POINTS that examples/make_examples.py prints.
LOCATIONS = ["-l", "80,44", "-l", "576,44", "-l", "80,147"]
GRID_LOCATIONS = ["-l", "80,44", "-l", "576,44", "-l", "80,167"]
SPECTRUM = [str(EXAMPLES / "spectrum.png"), *POINTS]
GRID = [str(EXAMPLES / "grid.png"), *POINTS]
SERIES = [str(EXAMPLES / "series.png"), *POINTS, *LOCATIONS]
LABELLED = str(EXAMPLES / "labelled.png")


def spectrum(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in spectrum.png."""
    return 0.5 + 2 * np.exp(-2 * (x - 3) ** 2) + np.exp(-((x - 7) ** 2) / 2)


def damped(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in grid.png."""
    return 1.2 + np.exp(-x / 4) * np.cos(2 * x)


def decay(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the blue curve that examples/make_examples.py draws in series.png."""
    return 2.5 * np.exp(-x / 3)


def growth(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the orange curve that examples/make_examples.py draws in series.png."""
    return 2.5 * (1 - np.exp(-x / 3))


def wave(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the green curve that examples/make_examples.py draws in series.png."""
    return 1.25 + np.sin(x) / 2


def ringing(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in labelled.png."""
    return np.exp(-x / 4) * np.cos(2 * x)


def trend(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the black line that examples/make_examples.py draws in series.png."""
    return 0.4 + 0.2 * x


def read_csv(text: str, header: str = "x,y") -> npt.NDArray[np.float64]:
    """Parse CSV output, checking its header."""
    first, _, body = text.partition("\n")
    assert first == header
    return np.loadtxt(body.splitlines(), delimiter=",", ndmin=2)


def assert_follows(data: npt.NDArray[np.float64], curve: Curve) -> None:
    """Check that x and y columns follow the true curve across the whole x-axis."""
    assert data[:, 0].min() < 0.1
    assert data[:, 0].max() > 9.9
    np.testing.assert_allclose(data[:, 1], curve(data[:, 0]), atol=0.04)


def assert_recovers(text: str, curve: Curve, header: str = "x,y") -> None:
    """Check that CSV output follows the true curve."""
    assert_follows(read_csv(text, header), curve)


def assert_recovers_series(text: str, curves: Sequence[Curve]) -> None:
    """Check that each series in CSV output follows its true curve, in order."""
    data = read_csv(text, header="series,x,y")
    assert set(data[:, 0]) == set(range(1, len(curves) + 1))
    for series, curve in enumerate(curves, start=1):
        assert_follows(data[data[:, 0] == series, 1:], curve)


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


def test_traces_several_series_and_prints_their_colours(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*SERIES, "-n", "4"]) == 0

    captured = capsys.readouterr()
    colours = "-c 2ea12e -c ff7f0e -c 1f77b4 -c 040404"  # the longest first
    assert captured.err == f"figrecover: colours: {colours}\n"
    assert_recovers_series(captured.out, [wave, growth, decay, trend])


def test_traces_the_given_colours(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preview = tmp_path / "preview.png"

    arguments = [*SERIES, "-c", "1f77b4", "-c", "#FF7F0E", "--preview", str(preview)]
    assert main(arguments) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert_recovers_series(captured.out, [decay, growth])
    image = cv2.imread(str(preview))
    assert image is not None
    for blue, green, red in [(180, 119, 31), (14, 127, 255)]:  # each curve's colour
        assert np.any(np.all(image == (blue, green, red), axis=-1))


def test_writes_one_series_without_a_series_column(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*SERIES, "-c", "2ca02c"]) == 0
    assert_recovers(capsys.readouterr().out, wave)


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


def test_reads_the_axes_without_reference_points(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preview = tmp_path / "preview.png"

    assert main([LABELLED, "--preview", str(preview)]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        'figrecover: x-axis "Time (s)" with ticks at 0, 2, 4, 6, 8, 10\n'
        'figrecover: y-axis "Voltage (mV)" with ticks at '
        "-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1\n"
    )
    assert_recovers(captured.out, ringing, header="Time (s),Voltage (mV)")
    image = cv2.imread(str(preview))
    assert image is not None
    assert np.any(np.all(image == (255, 0, 0), axis=-1))  # the ticks, in blue


def test_quotes_titles_in_the_csv_header(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calibration = Calibration.from_points(
        [(0, 0), (10, 0), (0, 1)], [(80, 44), (576, 44), (80, 147)]
    )
    axes = Axes(calibration, ((80, 0), (576, 10)), ((44, 0), (147, 1)), "E, eV", None)
    monkeypatch.setattr(cli, "read_axes", lambda _image: axes)

    assert main([str(EXAMPLES / "spectrum.png")]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        'figrecover: x-axis "E, eV" with ticks at 0, 10\n'
        "figrecover: y-axis with ticks at 0, 1\n"
    )
    assert_recovers(captured.out, spectrum, header='"E, eV",y')


@pytest.mark.usefixtures("without_ocr")
def test_reports_a_missing_ocr_extra(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([LABELLED]) == 1
    assert "needs the ocr extra" in capsys.readouterr().err


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
        ([*SERIES, "-n", "5"], "found only 4 of 5 curves"),
        ([*SERIES, "-c", "00ffff"], "no curve of colour #00ffff found"),
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
        ([*SERIES, "-n", "0"], "expected a positive whole number, got '0'"),
        ([*SERIES, "-n", "two"], "expected a positive whole number"),
        ([*SERIES, "-c", "red"], "expected a colour written as RRGGBB, got 'red'"),
        ([*SERIES, "-c", "#12345"], "expected a colour written as RRGGBB"),
        ([*SERIES, "-n", "2", "-c", "000000"], "not allowed with argument"),
        (["plot.png", "-l", "80,44"], "give the reference points of the -l"),
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
