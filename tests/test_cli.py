"""Tests for the command-line interface, using the example plots."""

import csv
import dataclasses
import gzip
import hashlib
import io
import json
import runpy
import shlex
import sys
import warnings
from collections.abc import Callable, Sequence
from importlib import metadata as packages
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from plottracer import (
    Axes,
    Calibration,
    Image,
    MarkerSeries,
    Panel,
    Point,
    Series,
    __version__,
    cli,
    interactive,
    load_image,
    metadata,
)
from plottracer.cli import _names, _options, main

type Curve = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

EXAMPLES = Path(__file__).parents[1] / "examples"
POINTS = ["-p", "0,0", "-p", "10,0", "-p", "0,1"]
# The pixel positions of POINTS that examples/make_examples.py prints.
LOCATIONS = ["-l", "80,44", "-l", "576,44", "-l", "80,147"]
GRID_LOCATIONS = ["-l", "80,44", "-l", "576,44", "-l", "80,167"]
SPECTRUM = [str(EXAMPLES / "spectrum.png"), *POINTS]
GRID = [str(EXAMPLES / "grid.png"), *POINTS]
SERIES = [str(EXAMPLES / "series.png"), *POINTS, *LOCATIONS]
STYLES = [str(EXAMPLES / "styles.png"), *POINTS, *LOCATIONS]
LABELLED = str(EXAMPLES / "labelled.png")
# The pixel positions of POINTS, and of the legend's markers, that
# examples/make_examples.py prints.
MARKER_LOCATIONS = ["-l", "55,83", "-l", "627,83", "-l", "55,185"]
MARKER_KEYS = ["-m", "148,18", "-m", "260,18", "-m", "377,18", "-m", "483,18"]
MARKERS = [str(EXAMPLES / "markers.png"), *POINTS, *MARKER_LOCATIONS]
# The positions of the legend's markers in similar.png, overlapping.png and
# coloured.png, which examples/make_examples.py prints with LOCATIONS.
KEYED = ["-m", "98,339", "-m", "184,339", "-m", "270,339", "-m", "355,339"]
LOGARITHMIC = str(EXAMPLES / "logarithmic.png")
BINARY = str(EXAMPLES / "binary.png")
PANELS = str(EXAMPLES / "panels.png")
INSET = str(EXAMPLES / "inset.png")
LABELLED_PDF = str(EXAMPLES / "labelled.pdf")
# The positions of POINTS in series.pdf, in points, that
# examples/make_examples.py prints.
PDF_LOCATIONS = ["-l", "57.6,31.68", "-l", "414.72,31.68", "-l", "57.6,105.6"]
SERIES_PDF = [str(EXAMPLES / "series.pdf"), *POINTS, *PDF_LOCATIONS]
MARKERS_PDF = str(EXAMPLES / "markers.pdf")
STYLES_PDF = str(EXAMPLES / "styles.pdf")
# styles.pdf has the axes of series.pdf.
STYLES_AT_POINTS = [STYLES_PDF, *POINTS, *PDF_LOCATIONS]
# The positions of POINTS, and of the legend's markers, in markers.pdf, in
# points, that examples/make_examples.py prints.
MARKER_PDF_LOCATIONS = ["-l", "39.89,59.29", "-l", "451.44,59.29", "-l", "39.89,133.26"]
LABELLED_SVG = str(EXAMPLES / "labelled.svg")
MARKER_PDF_KEYS = [
    *("-m", "108.45,12.91", "-m", "187.85,12.91"),
    *("-m", "272.06,12.91", "-m", "347.51,12.91"),
]
# The colours of the series with error bars in errorbars.png and .pdf, from
# the top, and the pixel positions of POINTS in the image, and their
# positions in the PDF, in points, that examples/make_examples.py prints.
ERROR_COLOURS = ["-c", "1f77b4", "-c", "ff7f0e", "-c", "2ca02c", "-c", "d62728"]
ERROR_LOCATIONS = ["-l", "80,44", "-l", "576,44", "-l", "80,121"]
ERRORS = [str(EXAMPLES / "errorbars.png"), *ERROR_COLOURS, "--error-bars"]
ERRORS_PDF = [str(EXAMPLES / "errorbars.pdf"), *ERROR_COLOURS, "--error-bars"]
BOUNDED = "series,x,y,y_low,y_high,x_low,x_high"


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


def lowpass(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in logarithmic.png."""
    return 1e3 / (1 + (x / 100) ** 2)


def quadratic(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in binary.png."""
    return x**2 / 4


def trend(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the black line that examples/make_examples.py draws in series.png."""
    return 0.4 + 0.2 * x


def ripple(level: int) -> Curve:
    """Return a curve that examples/make_examples.py draws in similar.png."""
    return lambda x: 0.3 + 0.66 * level + 0.1 * np.sin(x + level)


def rising(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a line that examples/make_examples.py draws in overlapping.png."""
    return 1.5 + 0.15 * (x - 5)


def falling(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a line that examples/make_examples.py draws in overlapping.png."""
    return 1.5 - 0.15 * (x - 5)


def shallow(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a line that examples/make_examples.py draws in overlapping.png."""
    return 1.5 + 0.05 * (x - 5)


def above(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a line that examples/make_examples.py draws in overlapping.png."""
    return rising(x) + 0.05


def read_csv(text: str, header: str = "x,y") -> npt.NDArray[np.float64]:
    """Parse CSV output, checking its header."""
    first, _, body = text.partition("\n")
    assert first == header
    return np.loadtxt(body.splitlines(), delimiter=",", ndmin=2)


def read_bounds(text: str, header: str = BOUNDED) -> npt.NDArray[np.float64]:
    """Parse CSV output with the bounds of error bars, empty as NaN, and its header."""
    first, _, body = text.partition("\n")
    assert first == header
    return np.genfromtxt(body.splitlines(), delimiter=",", ndmin=2)


def error_bounds(series: int) -> tuple[npt.NDArray[np.float64], ...]:
    """Return the points and bounds of a series that make_examples.py draws.

    The series are those of errorbars.png, numbered from 1, from the top:
    each marker's x and y, and its ``y_low``, ``y_high``, ``x_low`` and
    ``x_high``, NaN where it has none.
    """
    curves: list[tuple[Curve, float, float, float]] = [
        (lambda x: 3.4 + np.sin(x) / 4, 0.2, 0.2, np.nan),
        (lambda x: 2.4 + np.cos(x) / 4, 0.1, 0.3, np.nan),
        (lambda x: 1.5 + np.sin(x / 2) / 5, 0.15, 0.15, 0.3),
        (lambda x: 0.6 + 0.15 * np.cos(x / 2), 0.2, 0.2, np.nan),
    ]
    curve, below, above, across = curves[series - 1]
    x = np.arange(0.5, 10, 1.0)
    y = curve(x)
    return x, y, y - below, y + above, x - across, x + across


def assert_follows(data: npt.NDArray[np.float64], curve: Curve) -> None:
    """Check that x and y columns follow the true curve across the whole x-axis."""
    assert data[:, 0].min() < 0.1
    assert data[:, 0].max() > 9.9
    np.testing.assert_allclose(data[:, 1], curve(data[:, 0]), atol=0.04)


def assert_recovers(text: str, curve: Curve, header: str = "x,y") -> None:
    """Check that CSV output follows the true curve."""
    assert_follows(read_csv(text, header), curve)


def assert_follows_lowpass(data: npt.NDArray[np.float64]) -> None:
    """Check that x and y columns follow lowpass on logarithmic axes, x from 1 to 10^4.

    The y values must be within 0.03 decades, or 7 percent, of the true ones.
    """
    assert data[:, 0].min() < 1.1
    assert data[:, 0].max() > 9000
    np.testing.assert_allclose(
        np.log10(data[:, 1]), np.log10(lowpass(data[:, 0])), atol=0.03
    )


def assert_recovers_series(text: str, curves: Sequence[Curve]) -> None:
    """Check that each series in CSV output follows its true curve, in order."""
    data = read_csv(text, header="series,x,y")
    assert set(data[:, 0]) == set(range(1, len(curves) + 1))
    for series, curve in enumerate(curves, start=1):
        assert_follows(data[data[:, 0] == series, 1:], curve)


def test_writes_csv_to_standard_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*SPECTRUM, *LOCATIONS]) == 0
    assert_recovers(capsys.readouterr().out, spectrum)


def test_reads_reference_points_taken_a_little_below_the_axis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 5 pixels below the x-axis line, where tracing once crashed (#32).
    assert main([*SPECTRUM, "-l", "80,39", "-l", "576,39", "-l", "80,147"]) == 0
    data = read_csv(capsys.readouterr().out)
    # The points are those traced from the axis line, though the reference
    # points calibrate them as given, 5 pixels low.
    np.testing.assert_allclose(data[:, 1], spectrum(data[:, 0]), atol=0.1)


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
    colours = "-c 2ca02c -c 1f77b4 -c ff7f0e -c 000000"  # the most pixels first
    assert captured.err == f"plottracer: colours: {colours}\n"
    assert_recovers_series(captured.out, [wave, decay, growth, trend])


def test_says_when_the_image_is_greyscale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The blue and green curves of series.png become the same grey.
    image = tmp_path / "grey.png"
    grey = cv2.imread(str(EXAMPLES / "series.png"), cv2.IMREAD_GRAYSCALE)
    assert grey is not None
    cv2.imwrite(str(image), grey)

    assert main([str(image), *POINTS, *LOCATIONS, "-n", "4"]) == 0

    # The two curves of one grey are told apart by where they run.
    captured = capsys.readouterr()
    assert captured.err == (
        "plottracer: the image is greyscale, so curves of different colours may "
        "share a grey\n"
        "plottracer: colours: -c 919191 -c 000000 -c 5c5c5c:solid -c 5c5c5c:solid2\n"
    )
    assert_recovers_series(captured.out, [growth, trend, decay, wave])

    assert main([str(image), *POINTS, *LOCATIONS, "-c", "5c5c5c"]) == 0
    warning = capsys.readouterr().err
    assert "2 curves" in warning
    assert "5c5c5c" in warning


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


def test_finds_series_by_their_markers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preview = tmp_path / "preview.png"

    assert main([*MARKERS, *MARKER_KEYS, "--preview", str(preview)]) == 0

    captured = capsys.readouterr()
    data = marker_csv(captured.out, ["Decay", "Growth", "Wave", "Trend"])
    counts = [np.count_nonzero(data[:, 0] == series) for series in range(1, 5)]
    found = ", ".join(map(str, counts))
    assert captured.err.startswith(
        f"plottracer: markers found in each series: {found}\n"
    )
    # The markers lie every half unit from 0.5 to 9.5, even where the series
    # cross.
    assert_marks(data, [decay, growth, wave, trend], [19, 19, 19, 19])
    image = cv2.imread(str(preview))
    assert image is not None
    for red, green, blue in cli.SERIES_COLOURS[:4]:  # each series' colour
        assert np.any(np.all(image == (blue, green, red), axis=-1))


def marker_csv(
    text: str, names: Sequence[str], header: str = "series,x,y"
) -> npt.NDArray[np.float64]:
    """Map the printed names to their requested positions for numeric checks."""
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == header.split(",")
    return np.asarray(
        [
            [
                int(row[0]) if row[0].isdigit() else names.index(row[0]) + 1,
                *map(float, row[1:]),
            ]
            for row in rows[1:]
        ]
    )


def assert_marks(
    data: npt.NDArray[np.float64], curves: Sequence[Curve], counts: Sequence[int]
) -> None:
    """Check how many of each series' markers lie on its true curve.

    The markers lie every half unit from 0.5 to 9.5, and each series gives
    ``counts`` of them, within 0.03 of its curve, and no others.
    """
    for series, (curve, count) in enumerate(zip(curves, counts, strict=True), 1):
        points = data[data[:, 0] == series, 1:]
        on_curve = (np.abs(points[:, 0] - np.round(2 * points[:, 0]) / 2) < 0.03) & (
            np.abs(points[:, 1] - curve(points[:, 0])) < 0.03
        )
        assert np.count_nonzero(on_curve) == count
        assert len(points) == count


def test_tells_apart_markers_alike_in_shape(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([str(EXAMPLES / "similar.png"), *POINTS, *LOCATIONS, *KEYED]) == 0

    captured = capsys.readouterr()
    # Circles look like hexagons and octagons at 6 points, but are told apart.
    assert captured.err == (
        "plottracer: the markers of the keys at 98,339 and 270,339 look alike at "
        "this size, so some may be taken for each other\n"
        "plottracer: the markers of the keys at 98,339 and 355,339 look alike at "
        "this size, so some may be taken for each other\n"
        "plottracer: markers found in each series: 19, 19, 19, 19\n"
        "plottracer: legend (in this plot): Series 1 = -m 99,338.5, "
        "Series 2 = -m 184.5,339, Series 3 = -m 270.5,338.5, Series 4 = -m 356,338.5\n"
    )
    data = marker_csv(captured.out, [f"Series {i}" for i in range(1, 5)])
    assert_marks(data, [ripple(level) for level in range(4)], [19] * 4)


def test_tells_apart_markers_of_one_shape_by_colour(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([str(EXAMPLES / "coloured.png"), *POINTS, *LOCATIONS, *KEYED]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        "plottracer: markers found in each series: 19, 19, 19, 19\n"
        "plottracer: legend (in this plot): Series 1 = -c 1f77b4, "
        "Series 2 = -c ff7f0e, Series 3 = -c 2ca02c, Series 4 = -c d62728\n"
    )
    data = marker_csv(captured.out, [f"Series {i}" for i in range(1, 5)])
    assert_marks(data, [ripple(level) for level in range(4)], [19] * 4)


def test_finds_markers_partly_hidden_and_reports_those_hidden(
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = str(EXAMPLES / "overlapping.png")

    assert main([image, *POINTS, *LOCATIONS, *KEYED]) == 0

    # Diamonds cover the tops of the circles; the series cross at x = 5, where
    # a square hides a circle altogether, and two markers hide the circle at
    # 5.5 and at 4, but for the lower edge.
    captured = capsys.readouterr()
    assert captured.err.startswith(
        "plottracer: markers found in each series: 17, 20, 19, 18\n"
        "plottracer: series 1: 17 found, 2 probably hidden near x = 4, 5\n"
        "plottracer: series 4: 18 found, 1 probably hidden near x = 5\n"
    )
    data = marker_csv(captured.out, ["Series 1", "Series 2", "Series 3", "Series 4"])
    on_curves = [17, 19, 18, 17]
    for series, (curve, count) in enumerate(
        zip([rising, falling, shallow, above], on_curves, strict=True), start=1
    ):
        points = data[data[:, 0] == series, 1:]
        assert (
            np.count_nonzero(np.abs(points[:, 1] - curve(points[:, 0])) < 0.03) == count
        )


@pytest.mark.parametrize("quality", [60, 25])
def test_finds_markers_in_compressed_images(
    quality: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    compressed = tmp_path / "markers.jpg"
    image = cv2.imread(MARKERS[0])
    assert image is not None
    cv2.imwrite(str(compressed), image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    assert main([str(compressed), *MARKERS[1:], *MARKER_KEYS]) == 0

    data = marker_csv(capsys.readouterr().out, ["Decay", "Growth", "Wave", "Trend"])
    assert_marks(data, [decay, growth, wave, trend], [19, 19, 19, 19])


def test_reports_markers_ambiguous_hidden_or_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # On markers.png, x runs from 0 at pixel 55 to 10 at 627.
    found = [
        MarkerSeries(
            np.array([[112.2, 100], [140.8, 100]]),
            np.array([True, False]),
            np.empty((0, 2)),
        ),
        MarkerSeries(
            np.array([[112.2, 120], [198, 120]]),
            np.array([False, True]),
            np.array([[140.8, 120], [169.6, 120]]),
        ),
        MarkerSeries(np.empty((0, 2)), np.empty(0, dtype=np.bool_), np.empty((0, 2))),
    ]

    def find(
        _image: Image, _keys: Sequence[Point], _origin: Point, **_kwargs: object
    ) -> list[MarkerSeries]:
        warnings.warn("two keys look alike", stacklevel=2)
        return found

    monkeypatch.setattr(cli, "find_marker_series", find)
    assert main([*MARKERS, *MARKER_KEYS[:6]]) == 0

    assert capsys.readouterr().err.startswith(
        "plottracer: two keys look alike\n"
        "plottracer: markers found in each series: 2, 2, 0\n"
        "plottracer: series 3: found no markers like the key at 377,18\n"
        "plottracer: series 1: 2 found, 1 of them ambiguous near x = 1\n"
        "plottracer: series 2: 2 found, 1 of them ambiguous near x = 2.5, and 2 "
        "probably hidden near x = 1.5, 2\n"
    )


def test_tells_curves_of_one_colour_apart_by_style(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*STYLES, "-n", "3"]) == 0

    captured = capsys.readouterr()
    options = "-c 000000:solid -c 000000:dashed -c 000000:dotted"  # most pixels first
    assert captured.err == (
        "plottracer: the image is greyscale, so curves of different colours may "
        f"share a grey\nplottracer: colours: {options}\n"
    )
    assert_recovers_styles(captured.out, [decay, growth, wave])


def test_traces_the_given_styles(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*STYLES, "-c", "000000:dotted", "-c", "000000:solid"]) == 0

    assert_recovers_styles(capsys.readouterr().out, [wave, decay])


def assert_recovers_styles(text: str, curves: Sequence[Curve]) -> None:
    """Check that each series in CSV output follows its true curve, in order.

    A dashed or dotted curve may start and end a gap in from the axes.
    """
    data = read_csv(text, header="series,x,y")
    assert set(data[:, 0]) == set(range(1, len(curves) + 1))
    for series, curve in enumerate(curves, start=1):
        x, y = data[data[:, 0] == series, 1:].T
        assert x.min() < 0.15
        assert x.max() > 9.85
        np.testing.assert_allclose(y, curve(x), atol=0.04)


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
    assert captured.err == f"plottracer: locations: {' '.join(LOCATIONS)}\n"
    assert_recovers(captured.out, spectrum)


def test_reads_the_axes_without_reference_points(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preview = tmp_path / "preview.png"

    assert main([LABELLED, "--preview", str(preview)]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        'plottracer: x-axis "Time (s)" with ticks at 0, 2, 4, 6, 8, 10\n'
        'plottracer: y-axis "Voltage (mV)" with ticks at '
        "-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1\n"
    )
    assert_recovers(captured.out, ringing, header="Time (s),Voltage (mV)")
    image = cv2.imread(str(preview))
    assert image is not None
    assert np.any(np.all(image == (255, 0, 0), axis=-1))  # the ticks, in blue


def test_reads_logarithmic_axes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([LOGARITHMIC]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        'plottracer: logarithmic x-axis "Frequency (Hz)" with ticks at '
        "1, 10, 100, 1000, 10000\n"
        'plottracer: logarithmic y-axis "Power (mW)" with ticks at '
        "0.1, 1, 10, 100, 1000\n"
    )
    assert_follows_lowpass(read_csv(captured.out, "Frequency (Hz),Power (mW)"))


def test_reads_powers_of_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([BINARY]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        'plottracer: logarithmic x-axis "Size (KiB)" with ticks at 1, 2, 4, 8, 16\n'
        'plottracer: logarithmic y-axis "Time (ms)" with ticks at '
        "0.25, 0.5, 1, 2, 4, 8, 16, 32, 64\n"
    )
    data = read_csv(captured.out, "Size (KiB),Time (ms)")
    assert data[:, 0].min() < 1.1
    assert data[:, 0].max() > 15
    # Within 0.03 doublings, or 2 percent, of the true values.
    np.testing.assert_allclose(
        np.log2(data[:, 1]), np.log2(quadratic(data[:, 0])), atol=0.03
    )


def test_calibrates_logarithmic_axes_from_reference_points(
    capsys: pytest.CaptureFixture[str],
) -> None:
    points = ["-p", "1,0.1", "-p", "1e4,0.1", "-p", "1,1e3"]
    # The pixel positions of the points that examples/make_examples.py prints.
    locations = ["-l", "93,66", "-l", "610,66", "-l", "93,380"]

    assert main([LOGARITHMIC, *points, *locations, "--log", "xy"]) == 0
    assert_follows_lowpass(read_csv(capsys.readouterr().out))


def spectrum_panel(
    label: str | None, titles: tuple[str | None, str | None] = ("E", "I")
) -> Panel:
    """Return the spectrum example as a panel, with its axes as though read."""
    calibration = Calibration.from_points(
        [(0, 0), (10, 0), (0, 1)], [(80, 44), (576, 44), (80, 147)]
    )
    axes = Axes(calibration, ((80, 0), (576, 10)), ((44, 0), (147, 1)), *titles)
    return Panel(label, load_image(EXAMPLES / "spectrum.png"), (0, 0), axes)


def test_quotes_titles_in_the_csv_header(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    panel = spectrum_panel(None, ("E, eV", None))
    monkeypatch.setattr(cli, "read_panels", lambda _image: [panel])

    assert main([str(EXAMPLES / "spectrum.png")]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        'plottracer: x-axis "E, eV" with ticks at 0, 10\n'
        "plottracer: y-axis with ticks at 0, 1\n"
    )
    assert_recovers(captured.out, spectrum, header='"E, eV",y')


def test_reads_each_panel_of_a_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output, preview = tmp_path / "panels.csv", tmp_path / "preview.png"

    assert main([PANELS, "-o", str(output), "--preview", str(preview)]) == 0

    paths = [tmp_path / f"panels_{letter}.csv" for letter in "abcd"]
    assert capsys.readouterr().err == (
        'plottracer: panel a: x-axis "Energy (eV)" with ticks at 0, 2, 4, 6, 8, 10\n'
        'plottracer: panel a: y-axis "Intensity" with ticks at '
        "0, 0.5, 1, 1.5, 2, 2.5, 3\n"
        'plottracer: panel b: x-axis "Time (s)" with ticks at 0, 2, 4, 6, 8, 10\n'
        'plottracer: panel b: y-axis "Voltage (mV)" with ticks at '
        "-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1\n"
        'plottracer: panel c: x-axis "Depth (m)" with ticks at 0, 2, 4, 6, 8, 10\n'
        'plottracer: panel c: y-axis "Temperature (K)" with ticks at '
        "0, 0.5, 1, 1.5, 2, 2.5, 3\n"
        'plottracer: panel d: logarithmic x-axis "Frequency (Hz)" with ticks at '
        "1, 10, 100, 1000, 10000\n"
        'plottracer: panel d: logarithmic y-axis "Power (mW)" with ticks at '
        "0.1, 1, 10, 100, 1000\n"
        f"plottracer: wrote {', '.join(map(str, paths))}\n"
    )
    assert not output.exists()
    for path, curve, header in zip(
        paths[:3],
        [spectrum, ringing, growth],
        ["Energy (eV),Intensity", "Time (s),Voltage (mV)", "Depth (m),Temperature (K)"],
        strict=True,
    ):
        data = read_csv(path.read_text(), header)
        # The panels are narrower than the other examples, so the curves
        # start and end further from the axes, three pixels in.
        assert data[:, 0].min() < 0.15
        assert data[:, 0].max() > 9.85
        np.testing.assert_allclose(data[:, 1], curve(data[:, 0]), atol=0.04)
    assert_follows_lowpass(read_csv(paths[3].read_text(), "Frequency (Hz),Power (mW)"))
    image = cv2.imread(str(preview))
    assert image is not None
    assert image.shape == (600, 800, 3)


def read_panel_csv(text: str, header: str) -> dict[str, npt.NDArray[np.float64]]:
    """Parse CSV output with a panel column, checking its header."""
    first, *rows = text.splitlines()
    assert first == header
    panels = dict.fromkeys(row.partition(",")[0] for row in rows)
    return {
        panel: np.loadtxt(
            [row.partition(",")[2] for row in rows if row.startswith(f"{panel},")],
            delimiter=",",
            ndmin=2,
        )
        for panel in panels
    }


@pytest.mark.parametrize(
    ("titles", "header"), [(("E", "I"), "panel,E,I"), (("t", None), "panel,x,y")]
)
def test_writes_panels_to_standard_output_with_a_panel_column(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    titles: tuple[str | None, str | None],
    header: str,
) -> None:
    # The second panel has no label, so takes its number.
    found = [spectrum_panel("a"), spectrum_panel(None, titles)]
    monkeypatch.setattr(cli, "read_panels", lambda _image: found)

    assert main([str(EXAMPLES / "spectrum.png"), "-n", "1"]) == 0

    captured = capsys.readouterr()
    y_title = f' "{titles[1]}"' if titles[1] else ""
    assert captured.err.splitlines() == [
        'plottracer: panel a: x-axis "E" with ticks at 0, 10',
        'plottracer: panel a: y-axis "I" with ticks at 0, 1',
        f'plottracer: panel 2: x-axis "{titles[0]}" with ticks at 0, 10',
        f"plottracer: panel 2: y-axis{y_title} with ticks at 0, 1",
        "plottracer: panel a: colours: -c 000000",
        "plottracer: panel 2: colours: -c 000000",
    ]
    data = read_panel_csv(captured.out, header)
    assert list(data) == ["a", "2"]
    for panel in data.values():
        assert_follows(panel, spectrum)


def blank(panel: Panel) -> Panel:
    """Return a panel whose image is blank, so it has no curve to trace."""
    return dataclasses.replace(panel, image=np.full_like(panel.image, 255))


def test_reports_panels_left_out(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def read_panels(_image: Image) -> list[Panel]:
        warnings.warn("left out the panel at 1,2, as ...", stacklevel=1)
        return [spectrum_panel("a"), blank(spectrum_panel("b"))]

    monkeypatch.setattr(cli, "read_panels", read_panels)

    assert main([str(EXAMPLES / "spectrum.png")]) == 0

    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert lines[0] == "plottracer: left out the panel at 1,2, as ..."
    assert lines[-1].startswith(
        "plottracer: left out panel b, as its curves cannot be traced: "
    )
    data = read_panel_csv(captured.out, "panel,E,I")
    assert list(data) == ["a"]


def test_needs_a_panel_whose_curves_can_be_traced(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = [blank(spectrum_panel("a")), blank(spectrum_panel("b"))]
    monkeypatch.setattr(cli, "read_panels", lambda _image: found)

    assert main([str(EXAMPLES / "spectrum.png")]) == 1
    assert capsys.readouterr().err.endswith(
        "plottracer: error: cannot trace the curves of any of the 2 panels\n"
    )


def test_reports_an_invalid_marker_key_across_panels(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = [spectrum_panel("a"), spectrum_panel("b")]
    monkeypatch.setattr(cli, "read_panels", lambda _image: found)

    assert main([str(EXAMPLES / "spectrum.png"), "-m", "148,18"]) == 1
    assert capsys.readouterr().err.endswith(
        "plottracer: error: cannot trace the curves of any of the 2 panels\n"
    )


def test_reads_an_inset_as_a_plot_of_its_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "decay.csv"

    assert main([INSET, "-o", str(output)]) == 0

    paths = [output, tmp_path / "decay_inset.csv"]
    assert capsys.readouterr().err == (
        'plottracer: x-axis "Time (s)" with ticks at 0, 2, 4, 6, 8, 10\n'
        'plottracer: y-axis "Voltage (mV)" with ticks at 0, 0.5, 1, 1.5, 2, 2.5, 3\n'
        "plottracer: panel inset: x-axis with ticks at 6, 7, 8, 9, 10\n"
        "plottracer: panel inset: y-axis with ticks at 0.1, 0.2, 0.3\n"
        f"plottracer: wrote {', '.join(map(str, paths))}\n"
    )
    # The plot's curve is traced without the inset's, and the inset's in
    # its own units, over x from 6 to 10.
    assert_recovers(output.read_text(), decay, header="Time (s),Voltage (mV)")
    data = read_csv(paths[1].read_text())
    assert data[:, 0].min() < 6.1
    assert data[:, 0].max() > 9.9
    np.testing.assert_allclose(data[:, 1], decay(data[:, 0]), atol=0.005)


def inset_of(panel: Panel) -> Panel:
    """Return the spectrum example as an inset of a panel."""
    return dataclasses.replace(spectrum_panel(None), inset_of=panel)


def test_writes_insets_to_files_of_their_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plot = spectrum_panel(None)
    monkeypatch.setattr(cli, "read_panels", lambda _image: [plot, inset_of(plot)])
    output = tmp_path / "out.csv"

    assert main([str(EXAMPLES / "spectrum.png"), "-o", str(output)]) == 0

    # The plot holding the inset has no name, so is written to out.csv.
    paths = [output, tmp_path / "out_inset.csv"]
    assert capsys.readouterr().err.splitlines() == [
        'plottracer: x-axis "E" with ticks at 0, 10',
        'plottracer: y-axis "I" with ticks at 0, 1',
        'plottracer: panel inset: x-axis "E" with ticks at 0, 10',
        'plottracer: panel inset: y-axis "I" with ticks at 0, 1',
        f"plottracer: wrote {', '.join(map(str, paths))}",
    ]
    for path in paths:
        assert_recovers(path.read_text(), spectrum, header="E,I")


def test_writes_insets_to_standard_output_with_a_panel_column(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plot = spectrum_panel(None)
    monkeypatch.setattr(cli, "read_panels", lambda _image: [plot, inset_of(plot)])

    assert main([str(EXAMPLES / "spectrum.png")]) == 0

    # The plot holding the inset has no name, so its panel is empty.
    data = read_panel_csv(capsys.readouterr().out, "panel,E,I")
    assert list(data) == ["", "inset"]
    for panel in data.values():
        assert_follows(panel, spectrum)


def test_reports_a_plot_with_insets_left_out(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plot = blank(spectrum_panel(None))
    monkeypatch.setattr(cli, "read_panels", lambda _image: [plot, inset_of(plot)])

    assert main([str(EXAMPLES / "spectrum.png")]) == 0

    captured = capsys.readouterr()
    assert captured.err.splitlines()[-1].startswith(
        "plottracer: left out the plot, as its curves cannot be traced: "
    )
    assert list(read_panel_csv(captured.out, "panel,E,I")) == ["inset"]


def test_names_insets_after_their_panels() -> None:
    a, b = spectrum_panel("a"), spectrum_panel(None)
    plot = spectrum_panel(None)

    assert _names([a, inset_of(a), b, inset_of(b), inset_of(b)]) == [
        "a",
        "a_inset",
        "2",
        "2_inset1",
        "2_inset2",
    ]
    assert _names([plot, inset_of(plot), inset_of(plot)]) == [None, "inset1", "inset2"]


def test_finds_markers_in_a_plot_with_insets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calibration = Calibration.from_points(
        [(0, 0), (10, 0), (0, 1)], [(55, 83), (627, 83), (55, 185)]
    )
    axes = Axes(calibration, ((55, 0), (627, 10)), ((83, 0), (185, 1)), None, None)
    plot = Panel(None, load_image(EXAMPLES / "markers.png"), (0, 0), axes)
    inset = dataclasses.replace(plot, inset_of=plot)
    monkeypatch.setattr(cli, "read_panels", lambda _image: [plot, inset])

    assert main([str(EXAMPLES / "markers.png"), *MARKER_KEYS]) == 0

    # The inset is left out, as the markers are those of one plot.
    captured = capsys.readouterr()
    assert "inset" not in captured.err
    data = marker_csv(captured.out, ["Decay", "Growth", "Wave", "Trend"])
    assert set(data[:, 0]) == {1, 2, 3, 4}


@pytest.mark.usefixtures("without_ocr")
def test_reports_a_missing_ocr_extra(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([LABELLED]) == 1
    assert "needs the ocr extra" in capsys.readouterr().err


def test_reads_pdf_files_exactly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preview = tmp_path / "preview.png"

    assert main([LABELLED_PDF, "--preview", str(preview)]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        'plottracer: x-axis "Time (s)" with ticks at 0, 2, 4, 6, 8, 10\n'
        'plottracer: y-axis "Voltage (mV)" with ticks at '
        "-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1\n"
    )
    data = read_csv(captured.out, header="Time (s),Voltage (mV)")
    # As exact as the 6 significant figures of the CSV file.
    np.testing.assert_allclose(data[:, 1], ringing(data[:, 0]), atol=1e-5)
    image = cv2.imread(str(preview))
    assert image is not None
    assert image.shape[0] == 2 * 288  # the page, 2 pixels to the point
    assert np.any(np.all(image == (255, 0, 0), axis=-1))  # the ticks, in blue


def test_traces_the_curves_of_pdf_files_by_colour(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*SERIES_PDF, "-n", "4"]) == 0

    captured = capsys.readouterr()
    assert (
        captured.err == "plottracer: colours: -c 2ca02c -c ff7f0e -c 1f77b4 -c 000000\n"
    )
    data = read_csv(captured.out, header="series,x,y")
    for series, curve in enumerate([wave, growth, decay, trend], start=1):
        x, y = data[data[:, 0] == series, 1:].T
        np.testing.assert_allclose(y, curve(x), atol=1e-5)


def test_traces_the_given_colours_in_a_box_of_a_page_of_a_pdf_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    box = ["--page", "1", "--box", "0,0,460.8,288"]

    assert main([*SERIES_PDF, *box, "-c", "1f77b4"]) == 0

    data = read_csv(capsys.readouterr().out)
    np.testing.assert_allclose(data[:, 1], decay(data[:, 0]), atol=1e-5)


def test_asks_for_missing_locations_in_pdf_files_and_prints_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shown: list[Image] = []

    def pick_points(image: Image, _labels: Sequence[str]) -> list[Point]:
        shown.append(image)
        # The page is shown 2 pixels to the point.
        return [(115.2, 63.36), (829.44, 63.36), (115.2, 211.2)]

    monkeypatch.setattr(interactive, "pick_points", pick_points)

    assert main([str(EXAMPLES / "series.pdf"), *POINTS, "-c", "1f77b4"]) == 0
    assert shown[0].shape == (576, 922, 3)
    captured = capsys.readouterr()
    assert captured.err == f"plottracer: locations: {' '.join(PDF_LOCATIONS)}\n"
    data = read_csv(captured.out)
    np.testing.assert_allclose(data[:, 1], decay(data[:, 0]), atol=1e-5)


def test_traces_curves_drawn_as_images_in_pdf_files(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rasterised = str(EXAMPLES / "rasterised.pdf")

    assert main([rasterised, "-n", "2", "--remove-grid"]) == 0

    captured = capsys.readouterr()
    assert captured.err.endswith("plottracer: colours: -c ff7f0e -c 1f77b4\n")
    data = read_csv(captured.out, header="series,Time (s),Voltage (mV)")
    for series, curve in enumerate([growth, decay], start=1):
        assert_follows(data[data[:, 0] == series, 1:], curve)


def test_finds_series_by_their_markers_in_pdf_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preview = tmp_path / "preview.png"

    assert main([MARKERS_PDF, *MARKER_PDF_KEYS, "--preview", str(preview)]) == 0

    captured = capsys.readouterr()
    assert "plottracer: markers found in each series: 19, 19, 19, 19\n" in captured.err
    data = marker_csv(captured.out, ["Decay", "Growth", "Wave", "Trend"])
    for series, curve in enumerate([decay, growth, wave, trend], start=1):
        x, y = data[data[:, 0] == series, 1:].T
        # Every marker, every half unit from 0.5 to 9.5, even where the
        # series cross.
        np.testing.assert_allclose(x, np.arange(0.5, 10, 0.5), atol=1e-5)
        np.testing.assert_allclose(y, curve(x), atol=1e-5)
    image = cv2.imread(str(preview))
    assert image is not None
    for red, green, blue in cli.SERIES_COLOURS[:4]:  # each series' colour
        assert np.any(np.all(image == (blue, green, red), axis=-1))


def test_traces_series_of_markers_in_pdf_files_by_colour(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([str(EXAMPLES / "scatter.pdf"), "-n", "2"]) == 0

    captured = capsys.readouterr()
    assert "plottracer: colours: -c " in captured.err
    data = marker_csv(
        captured.out, ["Growth", "Decay"], header="series,Time (s),Voltage (mV)"
    )
    for series, curve in enumerate([growth, decay], start=1):
        x, y = data[data[:, 0] == series, 1:].T
        np.testing.assert_allclose(x, np.arange(0.5, 10, 0.5), atol=1e-5)
        np.testing.assert_allclose(y, curve(x), atol=1e-5)


def test_tells_curves_of_pdf_files_apart_by_style(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([STYLES_PDF, "-n", "3"]) == 0

    captured = capsys.readouterr()
    options = "-c 000000:dotted -c 000000:dashed -c 000000:solid"  # longest first
    assert captured.err.endswith(f"plottracer: colours: {options}\n")
    data = read_csv(captured.out, header="series,x,y")
    for series, curve in enumerate([wave, growth, decay], start=1):
        x, y = data[data[:, 0] == series, 1:].T
        np.testing.assert_allclose(y, curve(x), atol=1e-5)


def test_picks_curves_of_pdf_files_by_style(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([STYLES_PDF, "-c", "000000:Solid", "-c", "#000000"]) == 0

    data = read_csv(capsys.readouterr().out, header="series,x,y")
    x, y = data[data[:, 0] == 1, 1:].T
    np.testing.assert_allclose(y, decay(x), atol=1e-5)
    # Without a style, every curve of the colour, as one.
    x, y = data[data[:, 0] == 2, 1:].T
    near = [np.isclose(y, curve(x), atol=1e-5) for curve in (decay, growth, wave)]
    assert np.all(np.any(near, axis=0))
    assert np.all(np.any(near, axis=1))


def test_writes_the_options_that_pick_each_series() -> None:
    def series(colour: tuple[int, int, int], style: str) -> Series:
        return Series(colour, style, 1, np.zeros((1, 2)))  # type: ignore[arg-type]

    every = [
        series((0, 0, 0), "solid"),
        series((255, 0, 0), "solid"),
        series((0, 0, 0), "dashed"),
        series((0, 0, 0), "solid"),
    ]

    assert _options(every, every) == [
        "000000:solid",
        "ff0000",
        "000000:dashed",
        "000000:solid2",
    ]


def test_reads_error_bars_with_their_markers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preview = tmp_path / "preview.png"

    arguments = [*ERRORS, *POINTS, *ERROR_LOCATIONS, "--preview", str(preview)]
    assert main(arguments) == 0

    data = read_bounds(capsys.readouterr().out)
    # A unit of x is 49.6 pixels, and of y 77, so 2 pixels are 0.04 and 0.026.
    for series in range(1, 5):
        rows = data[data[:, 0] == series, 1:]
        marked = rows[~np.isnan(rows[:, 2])]
        assert len(marked) == 10  # a row for each marker, with its bounds
        assert len(np.unique(rows[:, 0])) == len(rows)  # and no other there
        x, *bounds = error_bounds(series)
        np.testing.assert_allclose(marked[:, 0], x, atol=0.04)
        # The triangles' middles lie up to 2.4 pixels below their points.
        steps = [0.033, 0.026, 0.026, 0.04, 0.04]
        for column, (truth, step) in enumerate(zip(bounds, steps, strict=True)):
            np.testing.assert_allclose(marked[:, column + 1], truth, atol=step)
        assert np.isnan(rows[np.isnan(rows[:, 2]), 2:]).all()  # the line's own
    image = cv2.imread(str(preview))
    assert image is not None
    # The cap of the first blue marker's bar, 330.5 pixels up, 104.8 across.
    assert np.any(np.all(image[68:72, 101:109] == (180, 119, 31), axis=-1))


def test_reads_error_bars_in_pdf_files_exactly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preview = tmp_path / "preview.png"

    assert main([*ERRORS_PDF, "--preview", str(preview)]) == 0

    captured = capsys.readouterr()
    header = "series,Time (s),Voltage (mV),y_low,y_high,x_low,x_high"
    data = read_bounds(captured.out, header)
    for series in range(1, 5):
        rows = data[data[:, 0] == series, 1:]
        np.testing.assert_allclose(
            rows, np.column_stack(error_bounds(series)), atol=1e-5
        )
    image = cv2.imread(str(preview))
    assert image is not None
    assert np.any(np.all(image == (40, 39, 214), axis=-1))  # the red bars


def test_reads_error_bars_in_order_on_reversed_axes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Both axes reversed: x from 10 to 0, and y from 4 down to 3 a unit up.
    points = ["-p", "10,4", "-p", "0,4", "-p", "10,3"]
    image = str(EXAMPLES / "errorbars.png")
    assert main([image, "-c", "2ca02c", *points, *ERROR_LOCATIONS, "--error-bars"]) == 0

    # The green markers, with bars across and up and down, from the right.
    data = read_bounds(capsys.readouterr().out, "x,y,y_low,y_high,x_low,x_high")
    marked = data[~np.isnan(data[:, 2])][::-1]
    x, _, below, above, left, right = error_bounds(3)
    np.testing.assert_allclose(marked[:, 0], 10 - x, atol=0.04)
    for column, truth in enumerate([4 - above, 4 - below, 10 - right, 10 - left]):
        np.testing.assert_allclose(marked[:, column + 2], truth, atol=0.04)


def test_reads_no_error_bars_where_there_are_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*MARKERS, *MARKER_KEYS, "--error-bars"]) == 0

    # The lines through the markers are not bars, and none run across.
    data = read_bounds(capsys.readouterr().out, header="series,x,y,y_low,y_high")
    assert len(data) >= 4 * 18
    assert np.isnan(data[:, 3:]).all()


@pytest.mark.usefixtures("without_pdf")
def test_reports_a_missing_pdf_extra(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([LABELLED_PDF]) == 1
    assert "needs the pdf extra" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["labelled.svg", "labelled-text.svg"])
def test_reads_svg_files_exactly(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    preview = tmp_path / "preview.png"

    assert main([str(EXAMPLES / name), "--preview", str(preview)]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        'plottracer: x-axis "Time (s)" with ticks at 0, 2, 4, 6, 8, 10\n'
        'plottracer: y-axis "Voltage (mV)" with ticks at '
        "-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1\n"
    )
    data = read_csv(captured.out, header="Time (s),Voltage (mV)")
    # As exact as the 6 significant figures of the CSV file.
    np.testing.assert_allclose(data[:, 1], ringing(data[:, 0]), atol=1e-5)
    image = cv2.imread(str(preview))
    assert image is not None
    assert image.shape[0] == 2 * 288  # the drawing, 2 pixels to the point
    assert np.any(np.all(image == (255, 0, 0), axis=-1))  # the ticks, in blue


def test_reads_svg_files_by_their_start_and_in_a_box(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    figure = tmp_path / "figure"
    figure.write_bytes((EXAMPLES / "series-text.svg").read_bytes())
    compressed = tmp_path / "figure.svg.gz"
    compressed.write_bytes(gzip.compress(figure.read_bytes()))
    box = ["--box", "0,0,460.8,288"]

    assert main([str(figure), *POINTS, *PDF_LOCATIONS, *box, "-c", "1f77b4"]) == 0
    data = read_csv(capsys.readouterr().out)
    np.testing.assert_allclose(data[:, 1], decay(data[:, 0]), atol=1e-5)
    assert main([str(compressed), "-n", "4"]) == 0
    assert capsys.readouterr().err.endswith(
        "plottracer: colours: -c 2ca02c -c 1f77b4 -c ff7f0e -c 000000\n"
    )


@pytest.mark.usefixtures("without_pdf")
def test_reads_svg_files_without_the_pdf_extra(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([LABELLED_SVG, "--preview", str(tmp_path / "preview.png")]) == 0

    data = read_csv(capsys.readouterr().out, header="Time (s),Voltage (mV)")
    np.testing.assert_allclose(data[:, 1], ringing(data[:, 0]), atol=1e-5)


def test_asks_for_missing_locations_in_svg_files_and_prints_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shown: list[Image] = []

    def pick_points(image: Image, _labels: Sequence[str]) -> list[Point]:
        shown.append(image)
        # The drawing is shown 2 pixels to the point.
        return [(115.2, 63.36), (829.44, 63.36), (115.2, 211.2)]

    monkeypatch.setattr(interactive, "pick_points", pick_points)

    assert main([str(EXAMPLES / "series.svg"), *POINTS, "-c", "1f77b4"]) == 0
    assert shown[0].shape == (576, 922, 3)
    captured = capsys.readouterr()
    assert captured.err == f"plottracer: locations: {' '.join(PDF_LOCATIONS)}\n"
    data = read_csv(captured.out)
    np.testing.assert_allclose(data[:, 1], decay(data[:, 0]), atol=1e-5)


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
        (
            [*SERIES, "-n", "5"],
            "found only 4 of 5 curves, -c 2ca02c -c 1f77b4 -c ff7f0e -c 000000",
        ),
        ([*SERIES, "-c", "00ffff"], "no curve of colour #00ffff found"),
        (
            [*STYLES_AT_POINTS, "-c", "000000", "-c", "0" * 6],
            "-c 000000 is given twice",
        ),
        ([*SERIES, "-c", "000000:dashed"], "no dashed series of colour #000000"),
        (
            [*STYLES_AT_POINTS, "-c", "000000:solid2", "-c", "000000:solid2"],
            "-c 000000:solid2 is given twice",
        ),
        (
            [*STYLES_AT_POINTS, "-n", "4"],
            (
                "found only 3 of 4 curves, -c 000000:dotted -c 000000:dashed "
                "-c 000000:solid"
            ),
        ),
        (
            [*STYLES_AT_POINTS, "-c", "000000:dashed2"],
            "found only 1 dashed series of colour #000000",
        ),
        (
            [*STYLES_AT_POINTS, "-c", "000000:markers"],
            "no markers series of colour #000000 found",
        ),
        (
            [*STYLES_AT_POINTS, "-c", "ff0000:solid"],
            "no curve of colour #ff0000 found",
        ),
        (
            [*SPECTRUM, "-l", "80,44", "-l", "5760,44", "-l", "80,147"],
            "the -l location 5760,44 lies outside the 640x400 image",
        ),
        (
            [*MARKERS, "-m", "148,18", "-m", "148,-1"],
            "the -m position 148,-1 lies outside the 640x400 image",
        ),
        ([*MARKERS, "-m", "148,18", "-m", "600,18"], "found no marker near 600,18"),
        ([LABELLED_PDF, "--page", "2"], "'" + LABELLED_PDF + "' has no page 2"),
        (["missing.svg"], "cannot read SVG 'missing.svg'"),
        (
            [str(EXAMPLES / "series.pdf"), *POINTS, "-l", "0,0", "-l", "999,0"],
            "the -l location 999,0 lies outside the page read, 0,0,460.8,288",
        ),
        (
            [MARKERS_PDF, "-m", "999,0"],
            "the -m position 999,0 lies outside the page read, 0,0,460.8,288",
        ),
        (
            [MARKERS_PDF, *POINTS, *MARKER_PDF_LOCATIONS, "-m", "5,280"],
            "found no marker near 5,280",
        ),
        (
            [
                *(MARKERS_PDF, *POINTS, *MARKER_PDF_LOCATIONS),
                *("-m", "108.45,12.91", "-m", "110,13"),
            ],
            "the markers near 108.45,12.91 and 110,13 belong to the same series",
        ),
    ],
)
def test_reports_errors(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 1
    assert capsys.readouterr().err.startswith(f"plottracer: error: {message}")


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
        (["plot.png", "-p", "nan,0"], "expected finite numbers in the point 'nan,0'"),
        (["plot.png", "-p", "0,-inf"], "expected finite numbers in the point"),
        (["plot.png", "-o", "plot.png"], "-o would overwrite the image"),
        (["plot.png", "--preview", "./plot.png"], "--preview would overwrite the"),
        (["plot.png", "-o", "a.png", "--preview", "./a.png"], "name the same file"),
        ([*SERIES, "-n", "0"], "expected a positive whole number, got '0'"),
        ([*SERIES, "-n", "two"], "expected a positive whole number"),
        ([*SERIES, "-c", "red"], "expected a colour written as RRGGBB, got 'red'"),
        ([*SERIES, "-c", "#12345"], "expected a colour written as RRGGBB"),
        ([*SERIES, "-c", "000000:solid0"], "expected a colour written as RRGGBB"),
        (
            [*SERIES, "-c", "000000:wavy"],
            "expected a style of solid, dashed, dotted, dashdot, markers, got",
        ),
        ([*SERIES, "-n", "2", "-c", "000000"], "not allowed with argument"),
        ([*SERIES, "-c", "000000", "-m", "1,1"], "not allowed with argument"),
        (["plot.png", "-l", "80,44"], "give the reference points of the -l"),
        (["plot.png", "--log", "x"], "--log needs -p"),
        ([*SPECTRUM, "--log", "z"], "invalid choice: 'z'"),
        (["plot.png", "--page", "2"], "--page and --box choose what to read in a PDF"),
        (["plot.png", "--box", "0,0,1,1"], "--page and --box choose what to read"),
        ([LABELLED_PDF, "--page", "0"], "expected a positive whole number, got '0'"),
        ([LABELLED_SVG, "--page", "1"], "--page chooses a page of a PDF file; an SVG"),
        ([LABELLED_PDF, "--box", "0,0,1"], "expected a box written as x0,y0,x1,y1"),
        ([LABELLED_PDF, "--box", "0,0,inf,1"], "expected finite numbers in the box"),
        ([LABELLED_PDF, "--box", "1,0,0,1"], "whose x1 and y1 exceed its x0 and y0"),
        (["plot.png", "--axes-from", "a.json", *POINTS], "choose --axes-from or -p"),
        (
            ["plot.png", "--axes-from", "a.json", "-o", "a.json"],
            "overwrite the calibration",
        ),
        (["plot.png", "-o", "a.json", "--metadata", "a.json"], "name the same file"),
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
    monkeypatch.setattr(sys, "argv", ["plottracer", "--version"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("plottracer", run_name="__main__")

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"{__version__}\n"


def test_batch_reuses_clicked_points_and_first_colour_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Thicken the orange stroke to make the second image rank orange first.
    second = tmp_path / "series.png"
    bgr = cv2.imread(SERIES[0])
    assert bgr is not None
    orange = (14, 127, 255)
    stroke = np.all(bgr == orange, axis=2).astype(np.uint8)
    bgr[cv2.dilate(stroke, np.ones((3, 3))) > 0] = orange
    cv2.imwrite(str(second), bgr)
    assert main([str(second), *POINTS, *LOCATIONS, "-n", "4"]) == 0
    assert "colours: -c ff7f0e" in capsys.readouterr().err

    clicks = []

    def pick(_image: Image, labels: Sequence[str]) -> list[Point]:
        clicks.append(labels)
        return [(80, 44), (576, 44), (80, 147)]

    monkeypatch.setattr(interactive, "pick_points", pick)
    preview = tmp_path / "previews"
    assert (
        main([SERIES[0], *POINTS, str(second), "-n", "4", "--preview", str(preview)])
        == 0
    )
    captured = capsys.readouterr()
    assert (
        len(clicks)
        == captured.err.count("locations:")
        == captured.err.count("colours:")
        == 1
    )
    rows = list(csv.DictReader(captured.out.splitlines()))
    assert list(rows[0]) == ["source", "series", "x", "y"]
    assert list(dict.fromkeys(row["source"] for row in rows)) == [
        SERIES[0],
        str(second),
    ]
    for source in [SERIES[0], str(second)]:
        for number, function in enumerate([wave, decay, growth, trend], 1):
            values = np.array(
                [
                    (float(row["x"]), float(row["y"]))
                    for row in rows
                    if row["source"] == source and int(row["series"]) == number
                ]
            )
            assert_follows(values, function)
    assert {path.name for path in preview.iterdir()} == {
        "series.preview.png",
        "series.2.preview.png",
    }


def test_batch_continues_after_failed_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["missing.png", *SERIES, LABELLED_PDF, "--no-level", "-n", "1"]) == 1
    captured = capsys.readouterr()
    assert "missing.png: error: cannot read image" in captured.err
    assert f"{LABELLED_PDF}: error: --level" in captured.err
    rows = list(csv.DictReader(captured.out.splitlines()))
    assert {row["source"] for row in rows} == {SERIES[0]}
    assert_follows(np.array([(float(row["x"]), float(row["y"])) for row in rows]), wave)


def test_batch_keeps_sources_in_panel_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    panels = [spectrum_panel("a"), spectrum_panel("b")]
    monkeypatch.setattr(cli, "read_panels", lambda _image: panels)
    inputs = [SPECTRUM[0], LABELLED]
    output = tmp_path / "out.csv"
    assert main([*inputs, "-n", "1", "-o", str(output)]) == 0
    assert capsys.readouterr().err.count("colours:") == 2
    for name in "ab":
        rows = list(
            csv.DictReader(output.with_stem(f"out_{name}").read_text().splitlines())
        )
        assert list(dict.fromkeys(row["source"] for row in rows)) == inputs


def test_batch_protects_inputs_from_generated_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    protected = tmp_path / "spectrum.preview.png"
    protected.write_bytes(Path(SPECTRUM[0]).read_bytes())
    before = protected.read_bytes()
    assert (
        main([*SPECTRUM, str(protected), *LOCATIONS, "--preview", str(tmp_path)]) == 1
    )
    assert "conflicting preview path" in capsys.readouterr().err
    assert protected.read_bytes() == before

    monkeypatch.setattr(
        cli, "read_panels", lambda _image: [spectrum_panel("a"), spectrum_panel("b")]
    )
    protected = tmp_path / "out_a.png"
    protected.write_bytes(before)
    assert main([SPECTRUM[0], str(protected), "-o", str(tmp_path / "out.png")]) == 1
    assert "panel output would overwrite" in capsys.readouterr().err
    assert protected.read_bytes() == before


def test_batch_with_no_readable_inputs_leaves_output_alone(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    output = tmp_path / "existing.csv"
    output.write_text("previous result\n")
    assert main(["missing1.png", "missing2.png", "-o", str(output)]) == 1
    captured = capsys.readouterr()
    assert "missing1.png: error:" in captured.err
    assert "missing2.png: error:" in captured.err
    assert output.read_text() == "previous result\n"


@pytest.mark.parametrize(
    ("arguments", "kind"),
    [
        ([LABELLED, "-n", "1"], "image"),
        ([LABELLED_PDF], "pdf"),
        ([str(EXAMPLES / "labelled-text.svg")], "svg"),
        ([str(EXAMPLES / "skewed.png")], "image"),
        ([*MARKERS, *MARKER_KEYS], "image"),
        ([*ERRORS, *POINTS, *ERROR_LOCATIONS], "image"),
        ([*GRID, *GRID_LOCATIONS, "--remove-grid"], "image"),
        (
            [
                LOGARITHMIC,
                "-p",
                "1,1",
                "-p",
                "10,1",
                "-p",
                "1,10",
                *LOCATIONS,
                "--log",
                "xy",
            ],
            "image",
        ),
    ],
)
def test_metadata_reproduces_single_plot(
    arguments: list[str],
    kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "points" if "-p" in arguments else "ticks"
    sidecar = tmp_path / "run with spaces.json"
    argv = [*arguments, "--metadata", str(sidecar)]
    assert main(argv) == 0
    first = capsys.readouterr()
    document = json.loads(sidecar.read_text())
    (entry,) = document["inputs"]
    assert document["plottracer"] == __version__
    assert document["argv"] == argv
    assert "+00:00" in document["created_utc"]
    assert all(document["versions"].values())
    assert entry["kind"] == kind
    assert (
        entry["sha256"] == hashlib.sha256(Path(arguments[0]).read_bytes()).hexdigest()
    )
    assert entry["error"] is None
    assert entry["page"] == (1 if kind == "pdf" else None)
    assert (entry["box"] is None) == (kind == "image")
    (plot,) = entry["plots"]
    fit = plot["calibration"]
    assert fit["source"] == source
    for axis in ("x", "y"):
        assert len(fit[axis]["ticks"]) >= 2
        assert fit[axis]["residual_px"] >= 0
    assert sum(s["points"] for s in plot["series"]) == len(first.out.splitlines()) - 1
    if arguments[0] == LABELLED:
        (series,) = plot["series"]
        assert fit["x"]["title"] == "Time (s)"
        assert series["style"] == "solid"
        assert series["width"] > 0
        assert f"colours: -c {series['selector']}" in first.err
    if source == "points":
        assert entry["points"]
        assert entry["locations"]
    if "-m" in arguments:
        assert len(entry["markers"]) == len(plot["series"]) == 4
    if arguments[0].endswith("skewed.png"):
        assert abs(entry["level_degrees"]) > 0.1

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("replay attempted to discover axes, click points or rank colours")

    for name in ("read_panels", "read_pdf_axes", "find_series", "find_pdf_series"):
        monkeypatch.setattr(cli, name, unexpected)
    monkeypatch.setattr(interactive, "pick_points", unexpected)
    # Recording a replay must also retain the calibration for a later replay.
    command = shlex.split(document["reproduce"])
    assert command[0] == "plottracer"
    assert main(command[1:]) == 0
    assert capsys.readouterr().out == first.out
    assert main([*command[1:], "--metadata", str(tmp_path / "replayed.json")]) == 0
    assert capsys.readouterr().out == first.out


@pytest.mark.parametrize("image", [PANELS, INSET])
def test_metadata_records_each_panel(image: str, tmp_path: Path) -> None:
    sidecar = tmp_path / "panels.json"
    assert main([image, "--metadata", str(sidecar)]) == 0
    document = json.loads(sidecar.read_text())
    (entry,) = document["inputs"]
    assert len(entry["plots"]) >= 2
    assert len({p["panel"] for p in entry["plots"]}) == len(entry["plots"])
    assert "reproduce" not in entry
    assert "reproduce" not in document


def test_metadata_records_clicks_warnings_and_failed_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "batch.json"
    picks = [(80, 44), (576, 44), (80, 147)]
    monkeypatch.setattr(interactive, "pick_points", lambda *_args: picks)
    argv = [SERIES[0], str(tmp_path / "missing.png"), LABELLED, *POINTS, "-n", "2"]
    assert main([*argv, "--metadata", str(sidecar)]) == 1
    captured = capsys.readouterr()
    good, missing, grey = json.loads(sidecar.read_text())["inputs"]
    assert good["plots"][0]["calibration"]["source"] == "clicked"
    assert good["locations"] == list(map(list, picks))
    assert f"locations: {' '.join(LOCATIONS)}" in captured.err
    assert missing["sha256"] is None
    assert missing["error"]
    assert grey["sha256"]
    assert grey["error"]
    assert grey["warnings"]
    assert grey["warnings"][0] in captured.err
    assert main(argv) == 1
    assert capsys.readouterr().out == captured.out


def test_metadata_rejects_missing_calibration_and_records_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sidecar = tmp_path / "record.json"
    assert main([LABELLED, "-n", "9", "--metadata", str(sidecar)]) == 1
    document = json.loads(sidecar.read_text())
    assert "found only" in document["inputs"][0]["error"]
    assert main([SERIES[0], "--axes-from", str(sidecar)]) == 1
    assert "no valid single-plot calibration" in capsys.readouterr().err


def test_metadata_reapplies_crop_and_painted_label() -> None:
    image = np.ones((12, 10, 3), dtype=np.uint8)
    part = image[3:10, 1:8].copy()
    part[1:3, 2:6] = [254, 255, 253]
    region = metadata.Region.from_image(image, part, (1, 2))
    assert region.crop == (1, 3, 8, 10)
    assert region.label == (2, 1, 6, 3)
    np.testing.assert_array_equal(region.apply(image), part)
    assert np.all(image == 1)


def test_metadata_records_missing_optional_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def version(name: str) -> str:
        if name == "rapidocr":
            raise packages.PackageNotFoundError(name)
        return "1"

    monkeypatch.setattr(packages, "version", version)
    sidecar = tmp_path / "manual.json"
    assert main([*SERIES, "--metadata", str(sidecar)]) == 0
    assert json.loads(sidecar.read_text())["versions"]["rapidocr"] is None


@pytest.mark.parametrize(
    "image", [PANELS, INSET, LOGARITHMIC, str(EXAMPLES / "skewed.png")]
)
def test_trace_ranges_use_each_plots_calibration(
    image: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([image, "--trace-range"]) == 0
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert rows[0][-2:] == ["trace_low", "trace_high"]
    for row in rows[1:]:
        y, lo, hi = map(float, row[-3:])
        assert lo <= y <= hi


@pytest.mark.parametrize(
    "args", [[LABELLED_PDF], [LABELLED_SVG], [*MARKERS, *MARKER_KEYS]]
)
def test_trace_ranges_leave_vector_and_marker_rows_empty(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*args, "--trace-range"]) == 0
    output = capsys.readouterr().out.splitlines()
    assert output[0].endswith("trace_low,trace_high")
    assert len(output) > 1
    assert all(row.endswith(",,") for row in output[1:])


def test_trace_ranges_and_error_bars_reverse_both_bounds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        *ERRORS,
        "-p",
        "0,0",
        "-p",
        "10,0",
        "-p",
        "0,-1",
        *ERROR_LOCATIONS,
        "--trace-range",
    ]
    sidecar = tmp_path / "err.json"
    assert main([*args, "--metadata", str(sidecar)]) == 0
    first = capsys.readouterr().out
    rows = list(csv.DictReader(first.splitlines()))
    assert list(rows[0])[-6:] == [
        "y_low",
        "y_high",
        "x_low",
        "x_high",
        "trace_low",
        "trace_high",
    ]
    marked = [r for r in rows if r["y_low"]]
    assert marked
    assert any(r["trace_low"] == "" for r in marked)
    traced = [r for r in rows if r["trace_low"]]
    assert traced
    assert all(
        float(r["trace_low"]) <= float(r["y"]) <= float(r["trace_high"]) for r in traced
    )
    assert main(shlex.split(json.loads(sidecar.read_text())["reproduce"])[1:]) == 0
    assert capsys.readouterr().out == first
