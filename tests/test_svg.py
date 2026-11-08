"""Tests for reading plots from SVG files, from their vector drawing."""

import base64
import gzip
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from figrecover import (
    Drawing,
    find_pdf_colours,
    find_pdf_markers,
    load_svg,
    read_pdf_axes,
    remove_pdf_grid,
    render_svg,
    trace_pdf_curves,
)
from figrecover import svg as svg_module
from figrecover.svg import _plain

type Curve = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

EXAMPLES = Path(__file__).parents[1] / "examples"
# Both ways matplotlib writes text: as glyph outlines, and as text.
FONTTYPES = ["", "-text"]
SIZE = 'width="400pt" height="300pt" viewBox="0 0 400 300"'  # 400 by 300 points
X_TICKS = {60: "0", 160: "1", 260: "2", 360: "3"}  # x in points, and label
Y_TICKS = {60: "0", 160: "10", 260: "20"}  # y in points from the bottom, and label
# The curve of a plot whose axes are the ticks above: data (0, 0), (1, 5),
# (2, 15) and (3, 20), with y down from the top of the drawing.
CURVE = '<path d="M 60 240 L 160 190 L 260 90 L 360 40" fill="none" stroke="red"/>'
# Where examples/make_examples.py draws markers, in markers.svg.
MARKED = np.arange(0.5, 10, 0.5)


def ringing(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in labelled.svg."""
    return np.exp(-x / 4) * np.cos(2 * x)


def lowpass(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in logarithmic.svg."""
    return 1e3 / (1 + (x / 100) ** 2)


def decay(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the blue curve that examples/make_examples.py draws in series.svg."""
    return 2.5 * np.exp(-x / 3)


def growth(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the orange curve that examples/make_examples.py draws in series.svg."""
    return 2.5 * (1 - np.exp(-x / 3))


def wave(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the green curve that examples/make_examples.py draws in series.svg."""
    return 1.25 + np.sin(x) / 2


def trend(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the black line that examples/make_examples.py draws in series.svg."""
    return 0.4 + 0.2 * x


# The curves of series.svg and rasterised.svg, by colour.
BY_COLOUR = {
    (31, 119, 180): decay,
    (255, 127, 14): growth,
    (44, 160, 44): wave,
    (0, 0, 0): trend,
}


def write_svg(path: Path, body: str, size: str = SIZE) -> Path:
    """Write an SVG file that draws ``body``, and return its path.

    The drawing is 400 by 300 points, in user units of points, unless
    ``size`` gives the root's width, height and viewBox otherwise.
    """
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" {size}>{body}</svg>',
        encoding="utf-8",
    )
    return path


def drawing(tmp_path: Path, body: str, size: str = SIZE) -> Drawing:
    """Read an SVG file that draws ``body``."""
    return load_svg(write_svg(tmp_path / "plot.svg", body, size))


def label(text: str, x: float, y: float, anchor: str = "middle") -> str:
    """Return a text of 10 points, anchored at x and y, y down from the top."""
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="10">{text}</text>'


def axes() -> str:
    """Return a drawing of axes with the ticks of X_TICKS and Y_TICKS, and titles.

    The axis lines cross at 60, 60 from the bottom-left corner, with tick
    marks outside them and labels centred on the marks, the x-axis's below
    and the y-axis's left of them. The y-axis title is written upwards.
    """
    parts = ['<path d="M 60 240 H 360 M 60 240 V 40" stroke="black" fill="none"/>']
    for x, text in X_TICKS.items():
        parts.append(f'<line x1="{x}" y1="240" x2="{x}" y2="244" stroke="black"/>')
        parts.append(label(text, x, 255))
    for y, text in Y_TICKS.items():
        parts.append(
            f'<line x1="56" y1="{300 - y}" x2="60" y2="{300 - y}" stroke="black"/>'
        )
        parts.append(label(text, 53, 303.6 - y, "end"))
    parts.append(label("Time (s)", 210, 275))
    parts.append(
        '<text transform="rotate(-90 25 150)" x="25" y="150" text-anchor="middle" '
        'font-size="10">Signal (V)</text>'
    )
    return "".join(parts)


def curves(found: Drawing, count: int = 1) -> list[npt.NDArray[np.float64]]:
    """Return the data of the longest curves in a plot whose axes are read."""
    calibration = read_pdf_axes(found).calibration
    colours = find_pdf_colours(found, count, calibration.origin)
    traced = trace_pdf_curves(found, colours, calibration.origin)
    return [calibration.to_data(curve) for curve in traced]


def png(pixels: npt.NDArray[np.uint8]) -> str:
    """Return an image, in OpenCV's order of channels, as a data URI of a PNG."""
    encoded = base64.b64encode(cv2.imencode(".png", pixels)[1].tobytes()).decode()
    return f"data:image/png;base64,{encoded}"


def boxes(found: Drawing) -> dict[str, list[float]]:
    """Return the texts read, each with its box and slant, rounded to 0.01 points."""
    return {
        text.text: [
            round(value, 2)
            for value in (text.left, text.bottom, text.right, text.top, text.slant)
        ]
        for text in found.texts
    }


def test_reads_axes_and_curves_drawn_as_text_and_paths(tmp_path: Path) -> None:
    found = drawing(tmp_path, axes() + CURVE)

    read = read_pdf_axes(found)

    assert found.box == (0.0, 0.0, 400.0, 300.0)
    assert read.x_ticks == ((60.0, 0.0), (160.0, 1.0), (260.0, 2.0), (360.0, 3.0))
    assert read.y_ticks == ((60.0, 0.0), (160.0, 10.0), (260.0, 20.0))
    assert (read.x_title, read.y_title) == ("Time (s)", "Signal (V)")
    np.testing.assert_allclose(
        curves(found)[0], [(0, 0), (1, 5), (2, 15), (3, 20)], atol=1e-9
    )
    assert find_pdf_colours(found) == [(255, 0, 0)]


@pytest.mark.parametrize("fonttype", FONTTYPES)
@pytest.mark.parametrize(
    ("name", "expected", "log"),
    [
        ("labelled", [ringing], False),
        ("logarithmic", [lowpass], True),
        ("series", [wave, growth, decay, trend], False),
    ],
)
def test_reads_the_example_svgs_exactly(
    name: str, expected: list[Curve], log: bool, fonttype: str
) -> None:
    found = load_svg(EXAMPLES / f"{name}{fonttype}.svg")
    calibration = read_pdf_axes(found).calibration
    colours = find_pdf_colours(found, len(expected), calibration.origin)

    traced = trace_pdf_curves(found, colours, calibration.origin)

    # matplotlib writes positions to six decimal places of a point.
    functions = (
        [BY_COLOUR[colour] for colour in colours] if name == "series" else expected
    )
    assert set(functions) == set(expected)
    for curve, function in zip(traced, functions, strict=True):
        x, y = calibration.to_data(curve).T
        if log:
            np.testing.assert_allclose(np.log10(y), np.log10(function(x)), atol=1e-6)
        else:
            np.testing.assert_allclose(y, function(x), atol=1e-6)
        assert x.min() == pytest.approx(1 if log else 0, abs=1e-6)
        assert x.max() == pytest.approx(1e4 if log else 10, rel=1e-6)


@pytest.mark.parametrize("fonttype", FONTTYPES)
def test_reads_the_axes_of_the_example_svgs(fonttype: str) -> None:
    labelled = read_pdf_axes(load_svg(EXAMPLES / f"labelled{fonttype}.svg"))
    logarithmic = read_pdf_axes(load_svg(EXAMPLES / f"logarithmic{fonttype}.svg"))

    assert [value for _, value in labelled.x_ticks] == [0, 2, 4, 6, 8, 10]
    assert [value for _, value in labelled.y_ticks] == [
        -0.75,
        -0.5,
        -0.25,
        0,
        0.25,
        0.5,
        0.75,
        1,
    ]
    assert (labelled.x_title, labelled.y_title) == ("Time (s)", "Voltage (mV)")
    # The positions of (0, 0), (10, 0) and (0, 1) that
    # examples/make_examples.py prints for labelled.pdf, in points.
    np.testing.assert_allclose(
        labelled.calibration.to_data(
            [(73.83, 123.40), (398.49, 123.40), (73.83, 243.36)]
        ),
        [(0, 0), (10, 0), (0, 1)],
        atol=1e-3,
    )
    # matplotlib writes the ticks of logarithmic axes as powers of ten, in
    # mathtext.
    assert [value for _, value in logarithmic.x_ticks] == [1, 10, 100, 1000, 10000]
    assert [value for _, value in logarithmic.y_ticks] == [0.1, 1, 10, 100, 1000]
    assert logarithmic.calibration.x_log
    assert logarithmic.calibration.y_log
    assert (logarithmic.x_title, logarithmic.y_title) == (
        "Frequency (Hz)",
        "Power (mW)",
    )


@pytest.mark.parametrize("fonttype", FONTTYPES)
def test_finds_series_by_their_markers_in_the_example_svgs(fonttype: str) -> None:
    found = load_svg(EXAMPLES / f"markers{fonttype}.svg")
    calibration = read_pdf_axes(found).calibration
    origin = calibration.origin
    # The positions of the legend's markers that examples/make_examples.py
    # prints.
    keys = [(108.47, 12.90), (187.86, 12.90), (272.06, 12.90), (347.51, 12.90)]

    series = find_pdf_markers(found, keys, origin)

    for markers, function in zip(series, [decay, growth, wave, trend], strict=True):
        x, y = calibration.to_data(markers).T
        np.testing.assert_allclose(x, MARKED, atol=1e-6)
        np.testing.assert_allclose(y, function(x), atol=1e-6)
    # By colour, the series are the black markers of all four, and the lines
    # through them are left out.
    assert find_pdf_colours(found, 1, origin) == [(0, 0, 0)]
    assert len(trace_pdf_curves(found, [(0, 0, 0)], origin)[0]) == 4 * len(MARKED)


@pytest.mark.parametrize("fonttype", FONTTYPES)
def test_traces_curves_drawn_as_images_in_the_example_svgs(fonttype: str) -> None:
    found = load_svg(EXAMPLES / f"rasterised{fonttype}.svg")

    data = curves(found, 2)

    assert len(found.pictures) == 1
    # matplotlib draws the image upside down, and turns it over.
    colours = find_pdf_colours(found, 2, read_pdf_axes(found).calibration.origin)
    for curve, colour in zip(data, colours, strict=True):
        x, y = curve.T
        np.testing.assert_allclose(y, BY_COLOUR[colour](x), atol=0.01)
        assert x.min() < 0.02
        assert x.max() > 9.98
    assert len(remove_pdf_grid(found).pictures) == 1


def plotly(tmp_path: Path) -> Drawing:
    """Read a figure drawn as Plotly draws one, in pixels, with a log y-axis.

    The plot area runs from 80 to 620 pixels across and from 60 to 330 down,
    with x from 0 to 4 and y from 10 to the 0 to 10 to the 3. Its blue curve
    runs through x of 0 to 3 by halves, at 10 to the x squared over 3, and
    orange circles mark 10 to the x at x of 0.5, 1.5 and 2.5. There are no
    axis lines or tick marks, only grid lines. The y-axis labels are powers
    of ten, the first two raised by baseline-shift and the others as Plotly
    raises them, by dy, and the y-axis title is turned to run upwards.
    """
    font = "font-family: 'Open Sans', verdana, arial, sans-serif; fill: rgb(68, 68, 68)"
    grid = "stroke: rgb(238, 238, 238); stroke-opacity: 1; stroke-width: 1px"
    parts = [
        '<rect x="0" y="0" width="700" height="450" style="fill: rgb(255, 255, 255)"/>',
        (
            '<defs id="defs-1"><g class="clips"><clipPath id="clip1xyplot">'
            '<rect width="540" height="270"/></clipPath></g></defs>'
        ),
    ]
    parts += [
        f'<path class="xgrid crisp" transform="translate({x},0)" d="M0,60v270" '
        f'style="{grid}"/>'
        for x in (80, 215, 350, 485, 620)
    ]
    parts += [
        f'<path class="ygrid crisp" transform="translate(0,{y})" d="M80,0h540" '
        f'style="{grid}"/>'
        for y in (330, 240, 150, 60)
    ]
    line = "M0,270L67.5,262.5L135,240L202.5,202.5L270,150L337.5,82.5L405,0"
    circle = "M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z"
    points = "".join(
        f'<path class="point" transform="translate({x},{y})" d="{circle}" '
        'style="opacity: 1; stroke-width: 0px; fill: rgb(255, 127, 14)"/>'
        for x, y in ((67.5, 225), (202.5, 135), (337.5, 45))
    )
    parts.append(
        '<g class="plot" transform="translate(80,60)" clip-path="url(#clip1xyplot)">'
        f'<path class="js-line" d="{line}" style="vector-effect: non-scaling-stroke; '
        'fill: none; stroke: rgb(31, 119, 180); stroke-opacity: 1; stroke-width: 2px"/>'
        f'<g class="points">{points}</g></g>'
    )
    for number, x in enumerate((80, 215, 350, 485, 620)):
        parts.append(
            f'<g class="xtick"><text text-anchor="middle" x="0" y="347" '
            f'transform="translate({x},0)" style="{font}; font-size: 12px; '
            f'fill-opacity: 1; white-space: pre">{number}</text></g>'
        )
    for power, y in enumerate((330, 240, 150, 60)):
        raised = (
            f'<tspan baseline-shift="super" font-size="70%">{power}</tspan>'
            if power < 2
            else f'<tspan style="font-size:70%" dy="-0.6em">{power}</tspan>'
            '<tspan dy="0.42em">&#8203;</tspan>'
        )
        parts.append(
            f'<g class="ytick"><text text-anchor="end" x="77" y="4.2" '
            f'transform="translate(0,{y})" style="{font}; font-size: 12px">'
            f"10{raised}</text></g>"
        )
    parts.append(
        f'<text class="xtitle" x="350" y="385" text-anchor="middle" '
        f'style="{font}; font-size: 14px">Time (s)</text>'
    )
    parts.append(
        '<text class="ytitle" transform="rotate(-90,31,195)" x="31" y="195" '
        f'text-anchor="middle" style="{font}; font-size: 14px">Signal (V)</text>'
    )
    size = 'class="main-svg" width="700" height="450" style="" viewBox="0 0 700 450"'
    return drawing(tmp_path, "".join(parts), size)


def test_reads_a_figure_drawn_as_plotly_draws_one(tmp_path: Path) -> None:
    found = plotly(tmp_path)
    read = read_pdf_axes(found)
    origin = read.calibration.origin

    colours = find_pdf_colours(found, 2, origin)
    line, circles = (
        read.calibration.to_data(curve)
        for curve in trace_pdf_curves(found, colours, origin)
    )

    # Pixels are 0.75 points.
    assert found.box == (0.0, 0.0, 525.0, 337.5)
    assert read.x_ticks == tuple(
        (x * 0.75, value) for value, x in enumerate((80, 215, 350, 485, 620))
    )
    assert [value for _, value in read.y_ticks] == [1, 10, 100, 1000]
    assert read.calibration.y_log
    assert (read.x_title, read.y_title) == ("Time (s)", "Signal (V)")
    texts = {text.text for text in found.texts}
    assert {"10\N{SUPERSCRIPT ONE}", "10\N{SUPERSCRIPT THREE}"} <= texts
    assert colours == [(31, 119, 180), (255, 127, 14)]
    x = np.arange(0, 3.5, 0.5)
    np.testing.assert_allclose(
        line, np.column_stack([x, 10 ** (x**2 / 3)]), rtol=1e-9, atol=1e-9
    )
    x = np.array([0.5, 1.5, 2.5])
    np.testing.assert_allclose(circles, np.column_stack([x, 10**x]), rtol=1e-9)


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        ("", [(0, 0), (10, 10)]),
        ("translate(10)", [(10, 0), (20, 10)]),
        ("translate(10, 20)", [(10, 20), (20, 30)]),
        ("scale(2)", [(0, 0), (20, 20)]),
        ("scale(2,3)", [(0, 0), (20, 30)]),
        ("rotate(90)", [(0, 0), (-10, 10)]),
        ("rotate(90 10 10)", [(20, 0), (10, 10)]),
        ("skewX(45)", [(0, 0), (20, 10)]),
        ("skewY(45)", [(0, 0), (10, 20)]),
        ("matrix(1 2 3 4 5 6)", [(5, 6), (45, 66)]),
        ("translate(10) scale(2)", [(10, 0), (30, 20)]),
        ("translate(10)\n,scale(2)", [(10, 0), (30, 20)]),
        ("rotate(1 2)", [(0, 0), (10, 10)]),
        ("scale(1 2 3) unknown(4)", [(0, 0), (10, 10)]),
    ],
)
def test_reads_paths_through_their_transforms(
    tmp_path: Path, transform: str, expected: list[tuple[float, float]]
) -> None:
    body = (
        '<g transform="translate(100 100)">'
        f'<path d="M 0 0 L 10 10" transform="{transform}" stroke="black" fill="none"/>'
        "</g>"
    )

    path = drawing(tmp_path, body).paths[0]

    # y runs down in the file, and up from the bottom of the drawing.
    np.testing.assert_allclose(
        path.subpaths[0], [(100 + x, 200 - y) for x, y in expected], atol=1e-9
    )


@pytest.mark.parametrize(
    ("size", "box", "expected"),
    [
        (SIZE, (0, 0, 400, 300), [(0, 300), (100, 200)]),
        ('width="4in" height="3in" viewBox="0 0 400 300"', (0, 0, 288, 216), None),
        ('width="400" height="300"', (0, 0, 300, 225), [(0, 225), (75, 150)]),
        ('height="225pt" viewBox="0 0 400 300"', (0, 0, 300, 225), None),
        ('width="300pt" viewBox="0 0 400 300"', (0, 0, 300, 225), None),
        ('viewBox="0 0 400 300"', (0, 0, 300, 225), None),
        ('width="100%" viewBox="0 0 400 300"', (0, 0, 300, 225), None),
        ("", (0, 0, 225, 112.5), None),
        (
            'width="200pt" height="100pt" viewBox="0 0 100 100"',
            None,
            [(50, 100), (150, 0)],
        ),
        (
            (
                'width="200pt" height="100pt" viewBox="0 0 100 100" '
                'preserveAspectRatio="xMinYMin slice"'
            ),
            None,
            [(0, 100), (200, -100)],
        ),
        (
            (
                'width="200pt" height="100pt" viewBox="0 0 100 100" '
                'preserveAspectRatio="none"'
            ),
            None,
            [(0, 100), (200, 0)],
        ),
        (
            'width="200pt" height="100pt" viewBox="0 0 -1 100"',
            None,
            [(0, 100), (75, 25)],
        ),
    ],
)
def test_reads_the_size_and_units_of_a_drawing(
    tmp_path: Path,
    size: str,
    box: tuple[float, ...] | None,
    expected: list[tuple[float, float]] | None,
) -> None:
    body = '<path d="M 0 0 L 100 100" stroke="black" fill="none"/>'

    found = drawing(tmp_path, body, size)

    if box is not None:
        assert found.box == pytest.approx(box)
    if expected is not None:
        np.testing.assert_allclose(found.paths[0].subpaths[0], expected)


def test_reads_svgs_nested_in_others(tmp_path: Path) -> None:
    body = (
        '<svg x="100" y="50" width="100" height="100" viewBox="0 0 10 10">'
        '<path d="M 0 0 L 10 10" stroke="black" fill="none"/>'
        '<svg width="50%" height="50%"><circle cx="5" cy="5" r="5" fill="red"/></svg>'
        "</svg>"
        '<svg x="500"><path d="M 0 0 L 10 10" stroke="black" fill="none"/></svg>'
    )

    first, second = drawing(tmp_path, body).paths

    np.testing.assert_allclose(first.subpaths[0], [(100, 250), (200, 150)])
    assert first.clip == (100.0, 150.0, 200.0, 250.0)
    # The inner viewport is half the outer one, 5 user units, or 50 points.
    np.testing.assert_allclose(second.subpaths[0][0], (200, 200))
    assert second.clip == (100.0, 200.0, 150.0, 250.0)


@pytest.mark.parametrize(
    ("data", "subpaths", "closed", "segments"),
    [
        (
            "M 10 10 L 20 10 20 20 Z",
            [[(10, 10), (20, 10), (20, 20)]],
            [True],
            [(10, 10, 20, 10), (20, 10, 20, 20), (20, 20, 10, 10)],
        ),
        (
            "m 10 10 l 10 0 h 5 v 5 H 10 V 10 z l 5 5 m 5 5 l 1 1",
            [
                [(10, 10), (20, 10), (25, 10), (25, 15), (10, 15), (10, 10)],
                [(10, 10), (15, 15)],
                [(20, 20), (21, 21)],
            ],
            [True, False, False],
            None,
        ),
        (
            "M 0 0 C 1 1 2 2 3 3 S 5 5 6 6 Q 7 7 8 8 T 9 9 c 1 1 2 2 3 3",
            [[(0, 0), (3, 3), (6, 6), (8, 8), (9, 9), (12, 12)]],
            [False],
            [],
        ),
        ("M 0 0 A 10 10 0 0 1 20 0", [[(0, 0), (10, -10), (20, 0)]], [False], []),
        ("M 0 0 A 10 10 0 0 0 20 0", [[(0, 0), (10, 10), (20, 0)]], [False], []),
        ("M 0 0 A 1 1 0 0 1 20 0", [[(0, 0), (10, -10), (20, 0)]], [False], []),
        (
            "M 0 0 A 10 10 0 1 1 10 10",
            [[(0, 0), (10, -10), (20, 0), (10, 10)]],
            [False],
            [],
        ),
        (
            "M 0 0 A 20 10 90 0 1 0 40",
            [[(0, 0), (10, 20), (0, 40)]],
            [False],
            [],
        ),
        ("M0 0a10 10 0 0020 0", [[(0, 0), (10, 10), (20, 0)]], [False], []),
        ("M 0 0 A 0 5 0 0 1 20 0", [[(0, 0), (20, 0)]], [False], [(0, 0, 20, 0)]),
        ("M 0 0 A 5 5 0 0 1 0 0", [[(0, 0)]], [False], []),
        ("M0,0L10-10.5.5.5", [[(0, 0), (10, -10.5), (0.5, 0.5)]], [False], None),
        ("M 0 0 L 10", [[(0, 0)]], [False], []),
        ("M 0 0 L 10 0 Z 5 5", [[(0, 0), (10, 0)]], [True], None),
        ("M 1e1 0 L 2E1 0 junk", [[(10, 0), (20, 0)]], [False], None),
    ],
)
def test_reads_path_data(
    tmp_path: Path,
    data: str,
    subpaths: list[list[tuple[float, float]]],
    closed: list[bool],
    segments: list[tuple[float, float, float, float]] | None,
) -> None:
    body = (
        f'<path d="{data}" stroke="black" fill="none" '
        'transform="translate(100 100) scale(1 -1)"/>'
    )

    path = drawing(tmp_path, body).paths[0]

    # Flipped back to y up, and moved by 100, 200.
    assert len(path.subpaths) == len(subpaths)
    for found, expected in zip(path.subpaths, subpaths, strict=True):
        np.testing.assert_allclose(found - (100, 200), expected, atol=1e-9)
    assert list(path.closed) == closed
    if segments is not None:
        np.testing.assert_allclose(
            path.segments - (100, 200, 100, 200),
            np.reshape(segments, (-1, 4)),
            atol=1e-9,
        )


def test_reads_shapes(tmp_path: Path) -> None:
    body = "".join(
        f'<{shape} stroke="black" fill="none" {attributes}/>'
        for shape, attributes in [
            ("rect", 'x="10" y="20" width="30" height="40" rx="5"'),
            ("rect", 'width="50%" height="10"'),
            ("rect", 'x="10" y="20" height="40"'),  # no width: not drawn
            ("circle", 'cx="100" cy="100" r="10"'),
            ("ellipse", 'cx="100" cy="100" rx="20" ry="10"'),
            ("circle", 'cx="100" cy="100"'),  # no radius
            ("line", 'x1="1" y1="2" x2="3" y2="4" style="fill: red"'),
            ("polyline", 'points="0,0 10,0 10,10"'),
            ("polygon", 'points="0 0 10 0 10 10 5"'),
            ("polygon", 'points=""'),  # no points
            ("path", 'd="L 10 10 M 0 0 L 9 9"'),  # not starting with a move
        ]
    )

    found = drawing(tmp_path, body).paths

    flipped = [path.subpaths[0] * (1, -1) + (0, 300) for path in found]
    rect, half, circle, ellipse, line, polyline, polygon = flipped
    np.testing.assert_allclose(rect, [(10, 20), (40, 20), (40, 60), (10, 60)])
    assert found[0].closed == (True,)
    assert len(found[0].segments) == 4
    np.testing.assert_allclose(half, [(0, 0), (200, 0), (200, 10), (0, 10)])
    np.testing.assert_allclose(circle, [(110, 100), (100, 110), (90, 100), (100, 90)])
    np.testing.assert_allclose(ellipse, [(120, 100), (100, 110), (80, 100), (100, 90)])
    assert not len(found[2].segments)
    np.testing.assert_allclose(line, [(1, 2), (3, 4)])
    assert found[4].fill is None
    assert found[4].closed == (False,)
    np.testing.assert_allclose(polyline, [(0, 0), (10, 0), (10, 10)])
    assert found[5].closed == (False,)
    np.testing.assert_allclose(polygon, [(0, 0), (10, 0), (10, 10)])
    assert found[6].closed == (True,)
    assert len(found[6].segments) == 3


def test_reads_colours_and_how_elements_inherit_them(tmp_path: Path) -> None:
    square = 'd="M 0 0 h 10 v 10 h -10 z"'
    body = "".join(
        [
            f'<g fill="none" stroke="blue"><path {square}/></g>',
            f'<path {square} fill="red" style="fill: green; stroke : ; "/>',
            (
                f'<g color="#123456"><path {square} stroke="currentColor" '
                'fill="none"/></g>'
            ),
            f'<g stroke="blue"><path {square} stroke="inherit" fill="none"/></g>',
            f"<path {square}/>",
            f'<path {square} fill="#abc" stroke="#aabbcc80"/>',
            f'<path {square} fill="#0000" stroke="rgb(10%, 20%, 30%)"/>',
            f'<path {square} fill="rgba(1, 2, 3, 0)" stroke="rgb(300 2 3 / 50%)"/>',
            f'<path {square} fill="rgb(1, 2)" stroke="steelblue"/>',
            f'<path {square} fill="transparent" stroke="nonsense"/>',
            f'<path {square} fill="black" fill-opacity="0" stroke="red"/>',
            f'<g opacity="0"><path {square}/></g>',
            f'<g opacity="50%"><path {square} fill="red" style="opacity: 0.5"/></g>',
            f'<path {square} style="display: none"/>',
            (
                f'<g visibility="hidden"><path {square}/>'
                f'<path {square} visibility="visible" fill="lime"/></g>'
            ),
            (
                '<linearGradient id="g"><stop offset="0" stop-color="#ff0000"/>'
                "</linearGradient>"
                '<radialGradient id="h"><stop style="stop-color: blue"/>'
                "</radialGradient>"
                '<pattern id="p"/>'
            ),
            f'<path {square} fill="url(#g)" stroke="url(#h)"/>',
            f'<path {square} fill="url(#missing) green" stroke="url(#p)"/>',
            f'<path {square} fill="url(&apos;#g&apos;)" stroke="url(#missing)"/>',
        ]
    )

    found = drawing(tmp_path, body).paths

    painted = [(path.stroke, path.fill) for path in found]
    assert painted == [
        ((0, 0, 255), None),
        (None, (0, 128, 0)),
        ((18, 52, 86), None),
        ((0, 0, 255), None),
        (None, (0, 0, 0)),
        ((170, 187, 204), (170, 187, 204)),
        ((26, 51, 76), None),
        ((255, 2, 3), None),
        ((70, 130, 180), None),
        ((255, 0, 0), None),
        (None, (255, 0, 0)),
        (None, (0, 255, 0)),
        ((0, 0, 255), (255, 0, 0)),
        ((0, 0, 0), (0, 128, 0)),
        ((0, 0, 0), (255, 0, 0)),
    ]


def test_reads_clip_paths_as_boxes(tmp_path: Path) -> None:
    curve = 'd="M 0 0 L 400 300" stroke="black" fill="none"'
    body = "".join(
        [
            '<clipPath id="c"><rect width="100" height="100"/></clipPath>',
            (
                '<clipPath id="moved" transform="translate(100 0)">'
                '<use href="#square"/><circle cx="50" cy="150" r="10"/></clipPath>'
            ),
            '<defs><rect id="square" width="100" height="100"/></defs>',
            (
                '<clipPath id="empty"/><clipPath id="away">'
                '<rect x="900" width="9" height="9"/>'
                "</clipPath>"
            ),
            f'<path {curve} clip-path="url(#c)"/>',
            f'<g clip-path="url(#c)" transform="translate(10 10)"><path {curve}/></g>',
            f'<path {curve} clip-path="url(#moved)"/>',
            f'<path {curve} clip-path="url(#missing)"/>',
            f'<path {curve} clip-path="url(#square)"/>',
            f'<path {curve} clip-path="none"/>',
            f'<path {curve} clip-path="url(#empty)"/>',  # hidden
            f'<path {curve} clip-path="url(#away)"/>',  # hidden off the drawing
            '<path d="M 0 0 L 9 9" stroke="black" clip-path="url(#away)"/>',
            '<g clip-path="url(#c)"><path d="M 300 0 L 310 10" stroke="black"/></g>',
        ]
    )

    found = drawing(tmp_path, body).paths

    assert [path.clip for path in found] == [
        (0, 200, 100, 300),
        (10, 190, 110, 290),
        (100, 140, 200, 300),
        (0, 0, 400, 300),
        (0, 0, 400, 300),
        (0, 0, 400, 300),
    ]


def test_reads_markers_drawn_with_use(tmp_path: Path) -> None:
    body = "".join(
        [
            (
                '<defs><path id="m" d="M -2 -2 L 2 -2 L 2 2 L -2 2 Z"/>'
                '<path id="glyph0-1" d="M 0 0 L 5 0 L 5 5 Z"/></defs>'
            ),
            '<symbol id="s"><circle r="2" fill="green"/></symbol>',
            '<g fill="blue">',
            *(
                f'<use xlink:href="#m" x="{x}" y="{300 - y}"/>'
                for x, y in [(100, 100), (200, 150), (300, 200)]
            ),
            "</g>",
            *(
                f'<use href="#s" x="{x}" y="{300 - y}"/>'
                for x, y in [(100, 50), (200, 60)]
            ),
            '<g id="loop"><use href="#loop"/><path d="M 0 0 L 9 9" stroke="red"/></g>',
            '<use href="#glyph0-1"/><use href="#nothing"/><use href="other.svg#m"/>',
            label("Blue", 340, 100, "start"),
        ]
    )
    found = drawing(tmp_path, body)

    blue, green = find_pdf_markers(found, [(100, 100), (200, 60)], (0, 0))

    np.testing.assert_allclose(blue, [(100, 100), (200, 150), (300, 200)])
    np.testing.assert_allclose(green, [(100, 50), (200, 60)])
    assert [path.anchor for path in found.paths[:2]] == [(100, 100), (200, 150)]
    assert found.paths[0].fill == (0, 0, 255)
    assert found.paths[-1].anchor is None
    assert len(found.paths) == 6  # the loop is drawn once


def test_reads_the_widths_and_dashes_of_lines(tmp_path: Path) -> None:
    line = 'd="M 0 0 L 10 10" stroke="black" fill="none"'
    body = "".join(
        [
            (
                f'<path {line} stroke-width="2" stroke-dasharray="5, 3" '
                'transform="scale(2)"/>'
            ),
            f'<path {line} style="stroke-width: 0.5em; stroke-dasharray: 1 2 3"/>',
            f'<path {line} stroke-dasharray="none"/>',
            f'<path {line} stroke-dasharray="-1 2"/>',
            f'<path {line} stroke-dasharray="0 0"/>',
            f'<path {line} stroke-dasharray="1 x"/>',
        ]
    )

    found = drawing(tmp_path, body).paths

    assert [(path.width, path.dashes) for path in found] == [
        (4, (10, 6)),
        (8, (1, 2, 3, 1, 2, 3)),
        (1, ()),
        (1, ()),
        (1, ()),
        (1, ()),
    ]


def test_reads_text_where_it_is_placed(tmp_path: Path) -> None:
    body = (
        '<g font-size="10">'
        # Characters 10 points apart, though 6 wide, with spaces between.
        '<text x="10 20 30" y="50">abcd</text>'
        '<text x="10" y="80" dx="5">  spaced \n\t out  </text>'
        '<text x="200" y="80" text-anchor="middle">mm</text>'
        '<text x="300" y="80" style="text-anchor: end">Ij</text>'
        '<text x="10" y="110" textLength="100" font-size="20">four</text>'
        '<text x="200" y="110" font-size="12pt">x'
        '<tspan font-size="50%">y</tspan><tspan font-size="2em">z</tspan></text>'
        '<text transform="rotate(-90 100 200)" x="100" y="200">up</text>'
        '<text transform="rotate(90 150 200)" x="150" y="200">down</text>'
        '<text x="300" y="200"><tspan x="300" y="200">e</tspan>'
        '<tspan x="310" y="200">f</tspan><!-- a comment --><title>g</title>'
        "</text>"
        '<text x="10" y="150">CO<tspan baseline-shift="sub" font-size="70%">2'
        '</tspan> and m<tspan baseline-shift="super" font-size="7">-3</tspan>'
        "</text>"
        '<text x="10" y="180">10<tspan dy="-4" font-size="7">4</tspan>'
        '<tspan dy="4">&#8203;</tspan></text>'
        '<text x="200" y="150">h<tspan display="none">i</tspan>'
        '<tspan visibility="hidden">j</tspan>k<tspan baseline-shift="-30%">l'
        '</tspan><tspan baseline-shift="baseline">m</tspan></text>'
        '<text x="200" y="250" display="none">hidden</text>'
        "</g>"
    )

    found = boxes(drawing(tmp_path, body))

    # Characters are 0.6 font sizes wide, or 0.3 if narrow and 0.9 if wide,
    # and reach 0.73 font sizes above their baseline, and those with
    # descenders 0.21 below it.
    assert found == {
        "a b cd": [10, 250, 42, 257.3, 0],
        "spaced out": [15, 217.9, 69, 227.3, 0],
        "mm": [191, 220, 209, 227.3, 0],
        "Ij": [294, 217.9, 300, 227.3, 0],
        "four": [10, 190, 110, 204.6, 0],
        # 12pt is 16 pixels, the user units of SVG's text.
        "xyz": [200, 188.32, 233.6, 213.36, 0],
        "up": [92.7, 100, 102.1, 112, 90],
        "down": [150, 73, 157.3, 100, -90],
        "e f": [300, 100, 313, 107.3, 0],
        "CO\N{SUBSCRIPT TWO} and m\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT THREE}": [
            10,
            148,
            67.5,
            159.11,
            0,
        ],
        "10\N{SUPERSCRIPT FOUR}": [10, 120, 26.2, 129.11, 0],
        # A hidden character leaves its space.
        "h klm": [200, 147, 227, 157.3, 0],
    }


def test_reads_text_that_matplotlib_draws_as_glyph_outlines(tmp_path: Path) -> None:
    glyph = '<path id="g1" d="M 0 0 L 600 0 L 600 700 L 0 700 Z"/>'
    body = "".join(
        [
            f"<defs>{glyph}</defs>",
            (
                '<g id="text_1"><!-- $\\mathdefault{10^{-1}}$ -->'
                '<g transform="translate(100 200) scale(0.01 -0.01)">'
                '<use xlink:href="#g1"/><use xlink:href="#g1" x="700"/></g></g>'
            ),
            (
                '<g id="text_2"><!-- Price &amp; -&#8209;$ -->'
                '<g transform="translate(100 150) rotate(-90) scale(0.01 -0.01)">'
                '<defs><path id="g2" d="M 0 0 L 300 0 L 300 300 Z"/></defs>'
                '<use xlink:href="#g2"/><use xlink:href="#missing"/></g></g>'
            ),
            '<g id="text_3"><!-- empty --><g/><path d="M 0 0 L 9 9" stroke="red"/></g>',
            '<g id="patch_1"><!-- not text --><path d="M 0 0 L 9 0" stroke="red"/></g>',
            # With svg.fonttype none, matplotlib writes mathtext a character at
            # a time, grouped by font: here the small circle first.
            (
                '<g id="text_4"><!-- T ($^\\circ$C) -->'
                '<g transform="translate(200 250)">'
                '<text><tspan x="10" y="-4" style="font-size: 7px">&#8728;</tspan>'
                '<tspan x="0" y="0" font-size="10">T</tspan>'
                '<tspan x="6" y="0" font-size="10">(</tspan>'
                '<tspan x="14" y="0" font-size="10">C</tspan>'
                '<tspan x="21" y="0" font-size="10">)</tspan></text></g></g>'
            ),
            '<g id="text_5"><!-- blank --><g><text> </text></g></g>',
        ]
    )

    found = drawing(tmp_path, body)

    assert boxes(found) == {
        "10\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}": [100, 100, 113, 107, 0],
        "Price & -\N{NON-BREAKING HYPHEN}$": [97, 150, 100, 153, 90],
        # Brackets reach 0.13 font sizes below their line and 0.76 above.
        "T (\N{DEGREE SIGN}C)": [200, 48.7, 224, 59.11, 0],
    }
    assert len(found.paths) == 2


@pytest.mark.parametrize(
    ("written", "plain"),
    [
        ("plain text", "plain text"),
        ("Price $5", "Price $5"),
        (r"$\mathdefault{10^{-1}}$", "10\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}"),
        (
            r"$\times\mathdefault{10^{3}}\mathdefault{}$",
            "\N{MULTIPLICATION SIGN}10\N{SUPERSCRIPT THREE}",
        ),
        (r"mg L$^{-1}$", "mg L\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}"),
        (r"CO$_2$", "CO\N{SUBSCRIPT TWO}"),
        (r"T ($^\circ$C)", "T (\N{DEGREE SIGN}C)"),
        (
            r"$\mu$m, $\lambda$, $\Delta T$",
            (
                "\N{GREEK SMALL LETTER MU}m, \N{GREEK SMALL LETTER LAMDA}, "
                "\N{GREEK CAPITAL LETTER DELTA}T"
            ),
        ),
        (r"$\frac{a}{b}$ $\sqrt{x}$", "a/b \N{SQUARE ROOT}x"),
        (r"$\log_{10} x \, \% \!$", "log\N{SUBSCRIPT ONE}\N{SUBSCRIPT ZERO}x %"),
        (r"$x$ costs \$", "x costs $"),
        (r"$x^$", "x"),
    ],
)
def test_writes_mathtext_as_plain_text(written: str, plain: str) -> None:
    assert _plain(written) == plain


def test_reads_images_as_they_are_shown(tmp_path: Path) -> None:
    pixels = np.full((3, 4, 3), 255, dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)  # blue, top left, in OpenCV's order
    pixels[-1, -1] = (0, 0, 255)  # red, bottom right
    png_bytes = cv2.imencode(".png", pixels)[1]
    see_through = np.zeros((2, 2, 4), dtype=np.uint8)  # black, half transparent
    see_through[..., 3] = 128
    cv2.imwrite(str(tmp_path / "pixels.png"), pixels)
    body = "".join(
        [
            f'<image x="100" y="50" width="40" height="30" href="{png(pixels)}"/>',
            (
                '<image transform="scale(1 -1)" x="200" y="-80" width="40" '
                f'height="30" '
                f'xlink:href="{png(pixels[::-1])}"/>'
            ),
            (
                f'<image transform="rotate(90 300 50)" x="300" y="50" width="40" '
                f'height="30" href="{png(pixels)}"/>'
            ),
            (
                '<image transform="scale(-1 1)" x="-340" y="150" width="40" '
                f'height="30" '
                f'href="{png(pixels[:, ::-1])}"/>'
            ),
            '<image x="100" y="150" href="pixels.png"/>',
            (
                '<image x="200" y="150" width="40" height="40" '
                f'href="{png(pixels[..., 0])}"/>'
            ),
            (
                f'<image x="10" y="10" width="20" height="20" opacity="0.5" '
                f'href="{png(see_through)}"/>'
            ),
            (
                '<image x="10" y="250" width="4" height="3" href="data:image/png,'
                f'{"".join(f"%{byte:02x}" for byte in png_bytes)}"/>'
            ),
            '<image href="missing.png"/><image href="https://example.com/a.png"/>',
            '<image href="data:image/png;base64,abc"/><image href="data:,"/>',
            '<image href="data:text/plain,hello"/><image href=""/>',
            f'<image visibility="hidden" href="{png(pixels)}"/>',
            f'<image x="500" href="{png(pixels)}"/>',
            '<clipPath id="c"><rect width="10" height="10"/></clipPath>',
            f'<image x="50" y="50" clip-path="url(#c)" href="{png(pixels)}"/>',
            '<clipPath id="edge"><rect width="400" height="3"/></clipPath>',
            (
                '<image width="40" height="40" clip-path="url(#edge)" '
                f'href="{png(pixels)}"/>'
            ),
        ]
    )

    found = drawing(tmp_path, body).pictures

    assert [picture.box for picture in found] == [
        (100, 220, 140, 250),
        (200, 220, 240, 250),
        (270, 210, 300, 250),
        (300, 120, 340, 150),
        (100, 147, 104, 150),
        (200, 115, 240, 145),  # fitted to its width, in the middle
        (10, 270, 30, 290),
        (10, 47, 14, 50),
    ]
    blue, red = [0, 0, 255], [255, 0, 0]
    corners = [
        (picture.image[0, 0].tolist(), picture.image[-1, -1].tolist())
        for picture in found[:5]
    ]
    # matplotlib's way: the image is turned over and drawn upside down.
    assert corners == [
        (blue, red),
        (blue, red),
        ([255] * 3, [255] * 3),
        (blue, red),
        (blue, red),
    ]
    # Turned a quarter turn clockwise, the top-left pixel is at the top right.
    assert found[2].image.shape == (4, 3, 3)
    assert found[2].image[0, -1].tolist() == blue
    assert found[5].image[0, 0].tolist() == [255] * 3
    assert found[6].image[0, 0].tolist() == [191] * 3  # a quarter as dark
    assert found[7].image[0, 0].tolist() == blue


def test_reads_files_compressed_with_gzip(tmp_path: Path) -> None:
    write_svg(tmp_path / "plot.svg", axes() + CURVE)
    data = gzip.compress((tmp_path / "plot.svg").read_bytes())
    (tmp_path / "plot.svgz").write_bytes(data)
    (tmp_path / "broken.svgz").write_bytes(data[:-20])

    found = load_svg(tmp_path / "plot.svgz")

    np.testing.assert_allclose(
        curves(found)[0], [(0, 0), (1, 5), (2, 15), (3, 20)], atol=1e-9
    )
    with pytest.raises(ValueError, match=r"cannot read SVG .*broken\.svgz"):
        load_svg(tmp_path / "broken.svgz")


def test_refuses_files_that_declare_entities_at_once(tmp_path: Path) -> None:
    # The billion laughs: ten levels of ten references, a billion lols.
    entities = '<!ENTITY lol0 "lol">' + "".join(
        f'<!ENTITY lol{level} "{f"&lol{level - 1};" * 10}">' for level in range(1, 10)
    )
    path = tmp_path / "laughs.svg"
    path.write_text(
        f'<?xml version="1.0"?><!DOCTYPE svg [{entities}]>'
        '<svg xmlns="http://www.w3.org/2000/svg"><text>&lol9;</text></svg>'
    )
    start = time.perf_counter()

    with pytest.raises(ValueError, match="declares XML entities"):
        load_svg(path)

    assert time.perf_counter() - start < 1


def test_reads_what_matplotlib_writes_around_its_drawing(tmp_path: Path) -> None:
    path = tmp_path / "plot.svg"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
        '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        "<!-- Created by hand -->\n"
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
        f"{SIZE}>"
        '<metadata><title>Not drawn</title></metadata><sodipodi:namedview id="n"/>'
        '<style type="text/css">*{stroke-linejoin: round}</style>'
        f"{CURVE}</svg>"
    )

    found = load_svg(path)

    assert len(found.paths) == 1


def test_reports_files_that_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "text.svg").write_text("not XML")
    (tmp_path / "page.svg").write_text("<html><body/></html>")
    write_svg(tmp_path / "plot.svg", "<g><g><g><path d='M 0 0 L 9 9'/></g></g></g>")
    nested = '<use href="#a"/>'  # drawing a, which draws b twice, and so on
    for name in "abc":
        nested += f'<g id="{name}"><use href="#{chr(ord(name) + 1)}"/>'
        nested += f'<use href="#{chr(ord(name) + 1)}"/></g>'
    write_svg(tmp_path / "uses.svg", f'{nested}<path id="d" d="M 0 0 L 9 9"/>')

    with pytest.raises(ValueError, match=r"cannot read SVG .*No such file"):
        load_svg(tmp_path / "missing.svg")
    with pytest.raises(ValueError, match=r"cannot read SVG .*syntax error: line 1"):
        load_svg(tmp_path / "text.svg")
    with pytest.raises(ValueError, match="is not an SVG file: its root is <html>"):
        load_svg(tmp_path / "page.svg")
    with pytest.raises(ValueError, match="the box 500,0,600,10 lies off the drawing"):
        load_svg(tmp_path / "plot.svg", box=(500, 0, 600, 10))
    monkeypatch.setattr(svg_module, "_MAX_DRAWN", 10)
    with pytest.raises(ValueError, match="draws more than 10 elements"):
        load_svg(tmp_path / "uses.svg")
    monkeypatch.setattr(svg_module, "_MAX_DEPTH", 3)
    with pytest.raises(ValueError, match="nests elements more than 3 deep"):
        load_svg(tmp_path / "plot.svg")
    monkeypatch.setattr(svg_module, "_MAX_SIZE", 10)
    with pytest.raises(ValueError, match="is larger than 0 MiB"):
        load_svg(tmp_path / "plot.svg")


def test_reads_a_box_of_a_drawing(tmp_path: Path) -> None:
    content = axes() + CURVE + '<path d="M 380 290 L 390 280" stroke="blue"/>'
    path = write_svg(tmp_path / "plot.svg", content)

    found = load_svg(path, box=(0, 0, 370, 300))

    assert found.box == (0.0, 0.0, 370.0, 300.0)
    assert find_pdf_colours(found, 1, (60, 60)) == [(255, 0, 0)]
    with pytest.raises(ValueError, match="found only 1 of 2"):
        find_pdf_colours(found, 2, (60, 60))
    # The y-axis title lies left of 30 points.
    assert len(load_svg(path, box=(30, 0, 370, 300)).texts) == len(found.texts) - 1


def test_draws_what_was_read(tmp_path: Path) -> None:
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)  # black
    body = "".join(
        [
            '<rect width="400" height="300" fill="white"/>',
            (
                '<rect x="10" y="10" width="20" height="20" fill="lime" stroke="blue" '
                'stroke-width="2"/>'
            ),
            '<path d="M 100 100 L 200 100 L 200 200" stroke="red" fill="none"/>',
            f'<image x="300" y="200" width="20" height="20" href="{png(pixels)}"/>',
            label("Across", 100, 250),
            (
                '<text transform="rotate(-90 40 200)" x="40" y="200" font-size="10">'
                "Up</text>"
            ),
            (
                '<text transform="rotate(90 60 200)" x="60" y="200" font-size="10">Down'
                "</text>"
            ),
            label("10<tspan font-size='5' dy='-4'>-3</tspan> \N{MICRO SIGN}", 200, 280),
            (
                '<text transform="rotate(-45 300 100)" x="300" y="100" font-size="10">'
                "Slanted</text>"
            ),
            '<text x="350" y="20" font-size="0.1">tiny</text>',
        ]
    )
    found = drawing(tmp_path, body)

    image = render_svg(found)
    half = render_svg(load_svg(tmp_path / "plot.svg", box=(0, 0, 200, 150)), scale=1)

    assert image.shape == (600, 800, 3)
    assert half.shape == (150, 200, 3)
    assert image.dtype == np.uint8
    assert image[40, 40].tolist() == [0, 255, 0]  # the filled square
    assert image[20, 40].tolist() == [0, 0, 255]  # its outline
    assert image[200, 300].tolist() == [255, 0, 0]  # the line
    assert image[410, 610].tolist() == [0, 0, 0]  # the image
    for text in found.texts[:-2]:  # each text is written in black in its box
        left, top = round(text.left * 2), round((300 - text.top) * 2)
        right, bottom = round(text.right * 2), round((300 - text.bottom) * 2)
        assert image[top:bottom, left:right].min() < 64
    slanted = found.texts[-2]  # drawn as the outline of its box
    assert (
        image[round((300 - slanted.top) * 2), round(slanted.x * 2)].tolist()
        == [128] * 3
    )


@pytest.mark.usefixtures("without_pdf")
def test_reads_svg_files_without_the_pdf_extra(tmp_path: Path) -> None:
    found = drawing(tmp_path, axes() + CURVE)

    assert len(curves(found)[0]) == 4
    assert render_svg(found).shape == (600, 800, 3)
