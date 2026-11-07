"""Tests for reading plots from PDF files, from their vector drawing."""

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pypdfium2
import pytest

from figrecover import (
    Drawing,
    find_pdf_colours,
    find_pdf_markers,
    load_pdf,
    read_pdf_axes,
    remove_pdf_grid,
    render_pdf,
    trace_pdf_curves,
)

type Curve = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

EXAMPLES = Path(__file__).parents[1] / "examples"
X_TICKS = {60: "0", 160: "1", 260: "2", 360: "3"}  # x in points, and label
Y_TICKS = {60: "0", 160: "10", 260: "20"}  # y in points, and label
# The curve of a plot whose axes are the ticks above: data (0, 0), (1, 5),
# (2, 15) and (3, 20).
CURVE = "1 0 0 RG 60 60 m 160 110 l 260 210 l 360 260 l S"
DIGIT = 5.56  # the width of a digit of Helvetica at 10 points
# Where examples/make_examples.py draws markers, in markers.pdf and scatter.pdf.
MARKED = np.arange(0.5, 10, 0.5)


def ringing(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in labelled.pdf."""
    return np.exp(-x / 4) * np.cos(2 * x)


def lowpass(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in logarithmic.pdf."""
    return 1e3 / (1 + (x / 100) ** 2)


def decay(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the blue curve that examples/make_examples.py draws in series.pdf."""
    return 2.5 * np.exp(-x / 3)


def growth(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the orange curve that examples/make_examples.py draws in series.pdf."""
    return 2.5 * (1 - np.exp(-x / 3))


def wave(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the green curve that examples/make_examples.py draws in series.pdf."""
    return 1.25 + np.sin(x) / 2


def trend(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the black line that examples/make_examples.py draws in series.pdf."""
    return 0.4 + 0.2 * x


def write_pdf(
    path: Path,
    content: str,
    forms: Sequence[str] = (),
    size: tuple[float, float] = (400, 300),
) -> Path:
    """Write a PDF of one page that draws ``content``, and return its path.

    The page has Helvetica as font /F1, a graphics state that draws nothing
    as /GS1, and each of the forms, which draw their content in their own
    space, as /X1, /X2 and so on.
    """
    names = " ".join(f"/X{index} {index + 7} 0 R" for index in range(1, len(forms) + 1))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %g %g] /Resources 5 0 R "
        b"/Contents 4 0 R >>" % size,
        _stream(b"", content),
        b"<< /Font << /F1 6 0 R >> /ExtGState << /GS1 7 0 R >> /XObject << %s >> >>"
        % names.encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /ExtGState /CA 0 /ca 0 >>",
        *(
            _stream(
                b"/Type /XObject /Subtype /Form /BBox [-9999 -9999 9999 9999] "
                b"/Resources 5 0 R ",
                form,
            )
            for form in forms
        ),
    ]
    data = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(data))
        data += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    table = len(data)
    data += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    data += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    data += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    data += b"startxref\n%d\n%%%%EOF\n" % table
    path.write_bytes(data)
    return path


def _stream(entries: bytes, content: str) -> bytes:
    """Return a stream object with the given dictionary entries and content."""
    body = content.encode("latin-1")
    return b"<< %s/Length %d >>\nstream\n%s\nendstream" % (entries, len(body), body)


def text(words: str, x: float, y: float, size: float = 10, angle: float = 0) -> str:
    """Return content that writes words in Helvetica, from x and y, at an angle."""
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    words = words.replace("(", "\\(").replace(")", "\\)")
    return (
        f"BT /F1 {size:g} Tf {cos:.6f} {sin:.6f} {-sin:.6f} {cos:.6f} {x:g} {y:g} Tm "
        f"({words}) Tj ET"
    )


def axes(*, lines: bool = True, marks: bool = True, x_labels: bool = True) -> str:
    """Return content that draws axes with the ticks of X_TICKS and Y_TICKS.

    The axis lines cross at 60, 60, with tick marks outside them and labels
    centred on the marks, the x-axis's below and the y-axis's left of them.
    """
    parts = ["0 0 0 RG 1 w"]
    if lines:
        parts += ["60 60 m 360 60 l S", "60 60 m 60 260 l S"]
    for x, label in X_TICKS.items():
        if marks:
            parts.append(f"{x} 60 m {x} 56 l S")
        if x_labels:
            parts.append(text(label, x - DIGIT * len(label) / 2, 45))
    for y, label in Y_TICKS.items():
        if marks:
            parts.append(f"56 {y} m 60 {y} l S")
        parts.append(text(label, 53 - DIGIT * len(label), y - 3.6))
    return "\n".join(parts)


def plot(*parts: str) -> str:
    """Return content that draws axes, titles and the given parts, such as CURVE."""
    titles = [text("Time (s)", 192, 25), text("Signal (V)", 25, 138, angle=90)]
    return "\n".join([axes(), *titles, *parts])


def image(width: int, height: int, rows: Sequence[int]) -> str:
    """Return content that draws an image with a red line through it, in a unit square.

    The line runs through the given row of each column, from the top, with
    the row below it, on white.
    """
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    for column, row in enumerate(rows):
        pixels[row : row + 2, column] = (255, 0, 0)
    return (
        f"BI /W {width} /H {height} /CS /RGB /BPC 8 /F /AHx ID "
        f"{pixels.tobytes().hex()}> EI"
    )


def squares(y: float, paint: str) -> str:
    """Return content that draws squares of side 4 around x of 100, 200 and 300 at y.

    Each is painted by the given operator, such as f to fill it.
    """
    return " ".join(f"{x - 2} {y - 2} 4 4 re {paint}" for x in (100, 200, 300))


def drawing(tmp_path: Path, content: str, forms: Sequence[str] = ()) -> Drawing:
    """Read a page drawn with the given content and forms."""
    return load_pdf(write_pdf(tmp_path / "plot.pdf", content, forms))


def curves(found: Drawing, count: int = 1) -> list[npt.NDArray[np.float64]]:
    """Return the data of the longest curves in a plot whose axes are read."""
    calibration = read_pdf_axes(found).calibration
    colours = find_pdf_colours(found, count, calibration.origin)
    traced = trace_pdf_curves(found, colours, calibration.origin)
    return [calibration.to_data(curve) for curve in traced]


def test_reads_axes_and_curves_drawn_as_paths(tmp_path: Path) -> None:
    found = drawing(tmp_path, plot(CURVE))

    read = read_pdf_axes(found)

    assert read.x_ticks == ((60.0, 0.0), (160.0, 1.0), (260.0, 2.0), (360.0, 3.0))
    assert read.y_ticks == ((60.0, 0.0), (160.0, 10.0), (260.0, 20.0))
    assert read.x_title == "Time (s)"
    assert read.y_title == "Signal (V)"
    assert read.calibration.origin == (60.0, 60.0)
    np.testing.assert_allclose(
        curves(found)[0], [(0, 0), (1, 5), (2, 15), (3, 20)], atol=1e-6
    )
    assert find_pdf_colours(found) == [(255, 0, 0)]


@pytest.mark.parametrize(
    ("name", "expected", "log"),
    [
        ("labelled", [ringing], False),
        ("logarithmic", [lowpass], True),
        ("series", [wave, growth, decay, trend], False),
    ],
)
def test_reads_the_example_pdfs_exactly(
    name: str, expected: list[Curve], log: bool
) -> None:
    found = load_pdf(EXAMPLES / f"{name}.pdf")

    data = curves(found, len(expected))

    # The curves are drawn through points that lie on them, to within the
    # precision with which PDFium reads them.
    for curve, function in zip(data, expected, strict=True):
        x, y = curve.T
        if log:
            np.testing.assert_allclose(np.log10(y), np.log10(function(x)), atol=1e-6)
        else:
            np.testing.assert_allclose(y, function(x), atol=1e-6)
        assert x.min() == pytest.approx(1 if log else 0, abs=1e-6)
        assert x.max() == pytest.approx(1e4 if log else 10, rel=1e-6)


def test_reads_the_axes_of_the_example_pdfs() -> None:
    labelled = read_pdf_axes(load_pdf(EXAMPLES / "labelled.pdf"))
    logarithmic = read_pdf_axes(load_pdf(EXAMPLES / "logarithmic.pdf"))

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
    # examples/make_examples.py prints.
    np.testing.assert_allclose(
        labelled.calibration.to_data(
            [(73.83, 123.40), (398.49, 123.40), (73.83, 243.36)]
        ),
        [(0, 0), (10, 0), (0, 1)],
        atol=1e-3,
    )
    # matplotlib writes the ticks of logarithmic axes as powers of ten, with
    # superscripts.
    assert [value for _, value in logarithmic.x_ticks] == [1, 10, 100, 1000, 10000]
    assert [value for _, value in logarithmic.y_ticks] == [0.1, 1, 10, 100, 1000]
    assert logarithmic.calibration.x_log
    assert logarithmic.calibration.y_log
    assert (logarithmic.x_title, logarithmic.y_title) == (
        "Frequency (Hz)",
        "Power (mW)",
    )


def test_reads_the_colours_of_the_example_pdfs_exactly() -> None:
    found = load_pdf(EXAMPLES / "series.pdf")

    colours = find_pdf_colours(found, 4, read_pdf_axes(found).calibration.origin)

    # matplotlib's colours, longest curve first.
    assert colours == [(44, 160, 44), (255, 127, 14), (31, 119, 180), (0, 0, 0)]


def test_reads_the_markers_of_the_example_pdfs_exactly() -> None:
    found = load_pdf(EXAMPLES / "scatter.pdf")

    data = curves(found, 2)

    # The caps of the error bars, and the markers in the legend, which lies
    # inside the axes, are left out.
    with pytest.raises(ValueError, match="found only 2 of 3 curves"):
        find_pdf_colours(found, 3, read_pdf_axes(found).calibration.origin)
    # The markers mark the points they are drawn at, to within the precision
    # with which PDFium reads where they are placed.
    for curve, function in zip(data, [growth, decay], strict=True):
        x, y = curve.T
        np.testing.assert_allclose(x, MARKED, atol=1e-5)
        np.testing.assert_allclose(y, function(x), atol=1e-5)


def test_finds_series_by_their_markers_in_the_example_pdf() -> None:
    found = load_pdf(EXAMPLES / "markers.pdf")
    calibration = read_pdf_axes(found).calibration
    origin = calibration.origin
    # The positions of the legend's markers that examples/make_examples.py
    # prints.
    keys = [(108.45, 12.91), (187.85, 12.91), (272.06, 12.91), (347.51, 12.91)]

    series = find_pdf_markers(found, keys, origin)

    for markers, function in zip(series, [decay, growth, wave, trend], strict=True):
        x, y = calibration.to_data(markers).T
        np.testing.assert_allclose(x, MARKED, atol=1e-5)
        np.testing.assert_allclose(y, function(x), atol=1e-5)
    # A marker in the plot picks its series as its legend's marker does.
    picked = find_pdf_markers(found, [(series[2][5, 0], series[2][5, 1])], origin)
    np.testing.assert_array_equal(picked[0], series[2])
    # By colour, the series are the black markers of all four, and the lines
    # through them are left out.
    assert find_pdf_colours(found, 1, origin) == [(0, 0, 0)]
    assert len(trace_pdf_curves(found, [(0, 0, 0)], origin)[0]) == 4 * len(MARKED)


def test_reads_markers_drawn_one_by_one_at_their_middles(tmp_path: Path) -> None:
    content = plot(
        "0 0 1 rg",
        *(f"{x - 2} {y - 2} 4 4 re f" for x, y in [(60, 60), (160, 110), (260, 210)]),
        "q 1 0 0 1 360 260 cm 0 -3 m 3 0 l 0 3 l -3 0 l h f Q",  # a diamond
        "q 1 0 0 1 60 160 cm -3 -3 6 6 re f Q",  # beside a tick label
        "0 0 1 RG 60 60 m 360 260 l S",  # a line through them
        "q 1 0 0 1 200 282 cm -3 -3 6 6 re f Q",  # a legend's marker
        text("Squares", 210, 279),  # and its label
        text("Diamonds", 300, 279),  # a label before its marker
        "q 1 0 0 1 353 282 cm 0 -3 m 3 0 l 0 3 l -3 0 l h f Q",
        "q 1 0 0 1 300 100 cm 0 0 m 6 0 l 3 6 l h f Q",  # a shape drawn once
    )
    found = drawing(tmp_path, content)

    data = curves(found)[0]

    np.testing.assert_allclose(
        data, [(0, 0), (1, 5), (2, 15), (3, 20), (0, 10)], atol=1e-6
    )
    with pytest.raises(ValueError, match="found only 1 of 2"):
        find_pdf_colours(found, 2, (60, 60))


def test_reads_markers_drawn_as_forms_at_their_origins(tmp_path: Path) -> None:
    forms = [
        "0 0 1 rg 0 0 m 6 0 l 0 6 l h f",  # a triangle, its corner at the origin
        "1 0 0 rg 100 100 4 4 re f",  # a square far from the origin
    ]
    content = plot(
        *(f"q 1 0 0 1 {x} {y} cm /X1 Do Q" for x, y in [(60, 60), (160, 110)]),
        "q /X2 Do Q q 1 0 0 1 100 50 cm /X2 Do Q",
    )
    found = drawing(tmp_path, content, forms)

    blue, red = trace_pdf_curves(found, [(0, 0, 255), (255, 0, 0)], (60, 60))

    np.testing.assert_allclose(blue, [(60, 60), (160, 110)], atol=1e-6)
    np.testing.assert_allclose(red, [(102, 102), (202, 152)], atol=1e-6)


def test_tells_series_of_markers_apart_by_colour(tmp_path: Path) -> None:
    content = plot(
        "1 1 1 rg 0 0 0 RG " + squares(80, "B"),  # open
        "0 1 0 rg 0 0 0 RG " + squares(120, "B"),
        "1 1 1 rg " + squares(160, "f"),  # white
        # Filled, then outlined apart.
        "0 0 1 rg 1 0 0 RG "
        + " ".join(f"{x - 2} 198 4 4 re f {x - 2} 198 4 4 re S" for x in (100, 200)),
        *(f"q 1 0 0 1 {x} 240 cm /X1 Do Q" for x in (100, 200, 300)),
    )
    forms = ["1 0 1 RG -3 -3 m 3 3 l -3 3 m 3 -3 l S"]  # crosses
    found = drawing(tmp_path, content, forms)

    colours = find_pdf_colours(found, 5, (60, 60))

    assert sorted(colours) == [
        (0, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 255),
        (255, 255, 255),
    ]
    # The squares filled in blue and outlined in red apart are one series.
    with pytest.raises(ValueError, match="found only 5 of 6"):
        find_pdf_colours(found, 6, (60, 60))


def test_leaves_out_shapes_that_are_not_markers(tmp_path: Path) -> None:
    content = plot(
        "1 0 0 rg 100 100 40 40 re f 200 100 40 40 re f",  # too large
        *(f"q 1 0 0 1 {x} 200 cm /X1 Do Q" for x in (100, 200, 300)),
        "0 0 1 RG 100 100 m 104 104 l S 120 120 m 124 124 l S",  # dashes
    )
    forms = ["0 1 0 RG -3 0 m 3 0 l S"]  # the caps of error bars
    found = drawing(tmp_path, content, forms)

    colours = find_pdf_colours(found, 1, (60, 60))

    # The dashes, drawn one by one, are pieces of a curve.
    assert colours == [(0, 0, 255)]
    np.testing.assert_allclose(
        trace_pdf_curves(found, colours, (60, 60))[0],
        [(100, 100), (104, 104), (120, 120), (124, 124)],
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="found only 1 of 2"):
        find_pdf_colours(found, 2, (60, 60))


def test_finds_series_by_the_shapes_and_colours_of_markers(tmp_path: Path) -> None:
    diamond = "0 -3 m 3 0 l 0 3 l -3 0 l h f"
    content = plot(
        "0 0 1 rg " + squares(100, "f"),
        "1 0 0 rg " + squares(150, "f"),
        "0 0 1 rg",
        *(f"q 1 0 0 1 {x} 200 cm {diamond} Q" for x in (100, 200, 300)),
        "30 30 4 4 re f",  # left of and below the axes
        "228 278 4 4 re f",  # a legend's marker
        text("Squares", 240, 277),
    )
    found = drawing(tmp_path, content)

    blue, diamonds = find_pdf_markers(found, [(230, 280), (200, 201)], (60, 60))

    np.testing.assert_allclose(blue, [(100, 100), (200, 100), (300, 100)])
    np.testing.assert_allclose(diamonds, [(100, 200), (200, 200), (300, 200)])
    with pytest.raises(ValueError, match="found no marker near 10,290"):
        find_pdf_markers(found, [(10, 290)], (60, 60))
    with pytest.raises(
        ValueError,
        match="the markers near 230,280 and 100,100 belong to the same series",
    ):
        find_pdf_markers(found, [(230, 280), (100, 100)], (60, 60))

    found = load_pdf(EXAMPLES / "rasterised.pdf")
    read = read_pdf_axes(found)

    data = curves(found, 2)

    assert [value for _, value in read.y_ticks] == [0, 0.5, 1, 1.5, 2, 2.5, 3]
    assert (read.x_title, read.y_title) == ("Time (s)", "Voltage (mV)")
    # The curves are drawn at 100 dots per inch, and traced to within half a
    # pixel, from the first column of pixels to the last.
    for curve, function in zip(data, [growth, decay], strict=True):
        x, y = curve.T
        np.testing.assert_allclose(y, function(x), atol=0.01)
        assert x.min() < 0.02
        assert x.max() > 9.98


def test_reads_a_figure_in_a_box_on_a_page(tmp_path: Path) -> None:
    # A page of a paper holds its figures as forms, scaled to fit.
    paper = pypdfium2.PdfDocument.new()
    page = paper.new_page(595, 842)
    for name, bottom in (("labelled", 450), ("series", 150)):
        figure = pypdfium2.PdfDocument(EXAMPLES / f"{name}.pdf")
        placed = figure.page_as_xobject(0, paper).as_pageobject()
        placed.transform(pypdfium2.PdfMatrix().scale(0.75, 0.75).translate(80, bottom))
        page.insert_obj(placed)
    page.gen_content()
    paper.save(tmp_path / "paper.pdf")

    upper = load_pdf(tmp_path / "paper.pdf", box=(80, 450, 426, 666))
    lower = load_pdf(tmp_path / "paper.pdf", 1, (80, 150, 426, 366))

    assert upper.box == (80.0, 450.0, 426.0, 666.0)
    x, y = curves(upper)[0].T
    np.testing.assert_allclose(y, ringing(x), atol=1e-6)
    assert read_pdf_axes(upper).x_title == "Time (s)"
    data = curves(lower, 4)
    for curve, function in zip(data, [wave, growth, decay, trend], strict=True):
        x, y = curve.T
        np.testing.assert_allclose(y, function(x), atol=1e-6)


def test_reads_superscripts_subscripts_and_spaces(tmp_path: Path) -> None:
    title = [
        text("CO", 150, 25),
        text("2", 164.5, 22, size=7),  # lowered and small: a subscript
        text("(mg", 172, 25),  # 3.6 points on: a space
        text("L", 192, 25),
        text("-1", 199, 29, size=7),  # raised and small: a superscript
        text(")", 207, 25),
        text(" ", 300, 25),  # a space alone is no text
    ]

    read = read_pdf_axes(drawing(tmp_path, "\n".join([axes(), *title])))

    assert read.x_title == "CO₂ (mg L⁻¹)"
    assert read.y_title is None


def test_reads_slanted_labels_by_their_ends(tmp_path: Path) -> None:
    # Labels of 13.9 points written at 45 degrees, ending just left of their
    # ticks, as plotting libraries rotate them.
    labels = [
        text(f"{label}.0", x - 10.8, 40, angle=45) for x, label in X_TICKS.items()
    ]
    content = "\n".join([axes(x_labels=False), *labels])

    read = read_pdf_axes(drawing(tmp_path, content))

    assert read.x_ticks == ((60.0, 0.0), (160.0, 1.0), (260.0, 2.0), (360.0, 3.0))


def test_reads_titles_written_upwards_in_pieces(tmp_path: Path) -> None:
    content = "\n".join(
        [
            axes(),
            text("Signal", 30, 120, angle=90),
            text("(V)", 30, 156, angle=90),  # too far on to join Signal
            text("Time", 180, 20, angle=-90),  # downwards
        ]
    )

    read = read_pdf_axes(drawing(tmp_path, content))

    assert read.y_title == "Signal (V)"


def test_reads_decimal_commas(tmp_path: Path) -> None:
    labels = {60: "0,0", 160: "0,5", 260: "1,0", 360: "1,5"}
    content = "\n".join(
        [
            axes(x_labels=False),
            *(text(label, x - 6.95, 45) for x, label in labels.items()),
            text("Time, in s", 180, 25),
        ]
    )

    read = read_pdf_axes(drawing(tmp_path, content))

    assert [value for _, value in read.x_ticks] == [0, 0.5, 1, 1.5]
    assert read.x_title == "Time, in s"


def test_reads_titles_written_across_and_over_lines(tmp_path: Path) -> None:
    content = "\n".join(
        [
            axes(),
            text("Photon energy", 180, 28),
            text("(eV)", 195, 18),
            text("Absorbance", 2, 160, size=7),
        ]
    )

    read = read_pdf_axes(drawing(tmp_path, content))

    assert read.x_title == "Photon energy (eV)"
    assert read.y_title == "Absorbance"


def test_reads_axes_without_lines_or_tick_marks(tmp_path: Path) -> None:
    content = "\n".join([axes(lines=False, marks=False), CURVE])

    read = read_pdf_axes(drawing(tmp_path, content))

    # Each tick lies at the middle of its label, and the axes cross just
    # beyond the labels.
    np.testing.assert_allclose(
        [pixel for pixel, _ in read.x_ticks], list(X_TICKS), atol=0.5
    )
    assert 53 < read.calibration.origin[0] < 60
    assert 52 < read.calibration.origin[1] < 60


def test_keeps_labels_far_from_tick_marks_at_their_middles(tmp_path: Path) -> None:
    # Marks a third of the tick spacing beside the ticks.
    content = (
        axes(marks=False)
        + "\n"
        + "\n".join(f"{x + 33} 60 m {x + 33} 56 l S" for x in X_TICKS)
    )

    read = read_pdf_axes(drawing(tmp_path, content))

    np.testing.assert_allclose(
        [pixel for pixel, _ in read.x_ticks], list(X_TICKS), atol=0.5
    )


def test_leaves_out_lines_that_are_not_curves(tmp_path: Path) -> None:
    content = plot(
        "0 0 1 RG 60 60 300 200 re S",  # a frame
        "0 0 1 rg 100 100 m 200 150 l 300 100 l f",  # a filled area
        "0 0 1 RG 0 0 1 rg 100 200 m 200 250 l 300 200 l B",  # outlined too
        "q /GS1 gs 0 0 1 RG 100 100 m 200 250 l S Q",  # transparent
        "0 0 1 RG 60 210 m 360 210 l S 210 60 m 210 260 l S",  # grid lines
        "0 1 0 RG 0 0 m 50 40 l S",  # left of and below the axes
        CURVE,
    )

    with pytest.raises(ValueError, match="found only 1 of 2 curves"):
        find_pdf_colours(drawing(tmp_path, content), 2, (60, 60))


def test_reads_curved_paths_through_their_ends(tmp_path: Path) -> None:
    content = plot("0 0 1 RG 60 60 m 100 200 200 200 260 160 c 300 60 l S")

    data = curves(drawing(tmp_path, content))[0]

    np.testing.assert_allclose(data, [(0, 0), (2, 10), (2.4, 0)], atol=1e-6)


def test_leaves_out_points_hidden_by_clips(tmp_path: Path) -> None:
    clipped = "q 60 60 300 150 re W n {} Q"
    form = "0 0 1 RG 0 0 m 100 50 l 200 150 l 300 200 l S"
    content = plot(
        clipped.format(CURVE),
        clipped.format("q 1 0 0 1 60 60 cm /X1 Do Q"),
        "q 0 0 10 10 re W n 0 1 0 RG 60 60 m 100 200 l S Q",  # hidden
        "q 500 500 10 10 re W n 0 1 0 RG 60 60 m 100 200 l S Q",  # off the page
        "q 0 0 10 10 re W n /X1 Do Q",  # in a hidden form
    )

    found = drawing(tmp_path, content, [form])
    red, blue = curves(found, 2)

    # The point at y = 20 is hidden, above the clip at y = 15.
    np.testing.assert_allclose(red, [(0, 0), (1, 5), (2, 15)], atol=1e-6)
    np.testing.assert_allclose(blue, [(0, 0), (1, 5), (2, 15)], atol=1e-6)


def test_reads_a_box_of_a_page(tmp_path: Path) -> None:
    content = plot(CURVE, "0 0 1 RG 380 10 m 390 20 l S")  # outside the box
    path = write_pdf(tmp_path / "plot.pdf", content)

    found = load_pdf(path, box=(0, 0, 370, 300))

    assert found.box == (0.0, 0.0, 370.0, 300.0)
    assert find_pdf_colours(found, 1, (60, 60)) == [(255, 0, 0)]
    with pytest.raises(ValueError, match="found only 1 of 2"):
        find_pdf_colours(found, 2, (60, 60))
    assert len(load_pdf(path, box=(-50, -50, 500, 500)).paths) == len(
        load_pdf(path).paths
    )


# A line rising from the bottom-left corner of an image of 60 by 40 pixels,
# to the top-right.
DIAGONAL = image(60, 40, [round(38 - column * 38 / 59) for column in range(60)])


def test_traces_images_within_the_axes_from_their_edges(tmp_path: Path) -> None:
    content = "\n".join(
        [
            f"q 300 0 0 200 60 60 cm {DIAGONAL} Q",
            f"q 40 0 0 40 0 200 cm {DIAGONAL} Q",  # left of the axes
            f"q 40 0 0 40 200 0 cm {DIAGONAL} Q",  # below the axes
        ]
    )
    found = drawing(tmp_path, content)

    colours = find_pdf_colours(found, 1, (60, 60))
    traced = trace_pdf_curves(found, colours, (60, 60))[0]

    assert colours == [(255, 0, 0)]
    with pytest.raises(ValueError, match="found only 1 of 2"):
        find_pdf_colours(found, 2, (60, 60))
    # The centres of the first and last columns of pixels, 5 points wide.
    assert traced[:, 0].min() == pytest.approx(62.5)
    assert traced[:, 0].max() == pytest.approx(357.5)


def test_traces_images_around_the_axes_beyond_them(tmp_path: Path) -> None:
    found = drawing(tmp_path, f"q 300 0 0 200 0 0 cm {DIAGONAL} Q")

    traced = trace_pdf_curves(found, [(255, 0, 0)], (60, 60))[0]

    # The axis lines in the image, and a margin of 3 pixels beyond them, are
    # left out.
    assert traced.min(axis=0).tolist() > [75, 75]
    assert traced[:, 0].max() == pytest.approx(297.5)


def test_leaves_out_images_outside_the_box(tmp_path: Path) -> None:
    content = f"q 40 0 0 40 300 250 cm {image(4, 4, [0, 1, 2, 2])} Q"
    path = write_pdf(tmp_path / "plot.pdf", content)

    assert len(load_pdf(path).pictures) == 1
    assert not load_pdf(path, box=(0, 0, 200, 200)).pictures


def test_removes_grid_lines_from_images(tmp_path: Path) -> None:
    pixels = ["ff" * 3 * 200] * 199 + ["00" * 3 * 200]  # a line along the bottom
    content = (
        "q 200 0 0 200 0 0 cm BI /W 200 /H 200 /CS /RGB /BPC 8 /F /AHx ID "
        + "".join(pixels)
        + "> EI Q"
    )
    found = drawing(tmp_path, content)

    cleaned = remove_pdf_grid(found)

    assert found.pictures[0].image.min() == 0
    assert cleaned.pictures[0].image.min() > 0
    assert cleaned.texts == found.texts


def test_draws_the_part_of_the_page_read(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "plot.pdf", plot(CURVE))

    whole = render_pdf(load_pdf(path))
    part = render_pdf(load_pdf(path, box=(60, 60, 360, 260)), scale=1)

    assert whole.shape == (600, 800, 3)
    assert part.shape == (200, 300, 3)
    assert whole.dtype == np.uint8
    assert whole[0, 0].tolist() == [255, 255, 255]
    assert ((part[..., 0] > 200) & (part[..., 1] < 100)).any()  # the red curve


def test_reports_colours_matching_no_curve_or_the_same_one(tmp_path: Path) -> None:
    found = drawing(tmp_path, plot(CURVE))

    with pytest.raises(ValueError, match="no curve of colour #00ff00"):
        trace_pdf_curves(found, [(0, 255, 0)], (60, 60))
    with pytest.raises(ValueError, match="#ff0000 and #ee0000 both match"):
        trace_pdf_curves(found, [(255, 0, 0), (238, 0, 0)], (60, 60))
    with pytest.raises(ValueError, match="no curve found"):
        find_pdf_colours(drawing(tmp_path, axes()), 1, (60, 60))


def test_reports_a_page_without_text(tmp_path: Path) -> None:
    found = drawing(tmp_path, CURVE)

    assert found.texts == ()
    with pytest.raises(ValueError, match="found no row of 3 or more numbers"):
        read_pdf_axes(found)


def test_reports_unreadable_files_and_missing_pages(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "plot.pdf", plot(CURVE))
    (tmp_path / "text.pdf").write_text("not a PDF")

    with pytest.raises(ValueError, match="cannot read PDF"):
        load_pdf(tmp_path / "text.pdf")
    with pytest.raises(ValueError, match="has no page 2, only 1"):
        load_pdf(path, 2)
    with pytest.raises(ValueError, match="the box 500,0,600,10 lies off the page"):
        load_pdf(path, box=(500, 0, 600, 10))


@pytest.mark.usefixtures("without_pdf")
def test_reports_a_missing_pdf_extra(tmp_path: Path) -> None:
    with pytest.raises(ModuleNotFoundError, match="pdf extra"):
        load_pdf(tmp_path / "plot.pdf")
