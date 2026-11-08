"""Tests for reading the axes of a plot from the text in its image."""

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from figrecover import Axes, Image, axes, load_image, read_axes
from figrecover.axes import (
    _PLAIN,
    _factor,
    _in_line_along,
    _merged,
    _read_sideways,
    _read_text,
    _read_words,
    _split,
    _Text,
)

EXAMPLES = Path(__file__).parents[1] / "examples"
HEIGHT, WIDTH = 300, 400
AXIS_LINES = (60, 60)  # pixel position where the axis lines cross


@pytest.fixture(scope="module")
def labelled() -> Axes:
    """Read the axes of the example plot with axis titles, once."""
    return read_axes(load_image(EXAMPLES / "labelled.png"))


def test_reads_tick_values_and_titles(labelled: Axes) -> None:
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
    assert labelled.x_title == "Time (s)"
    assert labelled.y_title == "Voltage (mV)"


def test_calibrates_from_the_ticks(labelled: Axes) -> None:
    # The pixel positions of (0, 0), (10, 0) and (0, 1) that
    # examples/make_examples.py prints.
    pixels = [(103, 171), (553, 171), (103, 338)]

    data = labelled.calibration.to_data(pixels)

    np.testing.assert_allclose(data, [(0, 0), (10, 0), (0, 1)], atol=0.03)
    assert not labelled.calibration.x_log
    assert not labelled.calibration.y_log


def test_reads_logarithmic_axes() -> None:
    found = read_axes(load_image(EXAMPLES / "logarithmic.png"))
    # The pixel positions of (1, 0.1), (10^4, 0.1) and (1, 10^3) that
    # examples/make_examples.py prints.
    pixels = [(93, 66), (610, 66), (93, 380)]

    data = found.calibration.to_data(pixels)

    assert [value for _, value in found.x_ticks] == [1, 10, 100, 1000, 10000]
    assert [value for _, value in found.y_ticks] == [0.1, 1, 10, 100, 1000]
    assert found.calibration.x_log
    assert found.calibration.y_log
    expected = [(1, 0.1), (1e4, 0.1), (1, 1e3)]
    np.testing.assert_allclose(np.log10(data), np.log10(expected), atol=0.01)
    assert found.x_title == "Frequency (Hz)"
    assert found.y_title == "Power (mW)"


def test_reads_powers_of_ten_whose_superscripts_ocr_missed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def read_text(image: Image) -> list[_Text]:
        """Read the text as OCR does, but with every superscript written plain."""
        return [
            dataclasses.replace(line, text=line.text.translate(_PLAIN))
            for line in _read_text(image)
        ]

    monkeypatch.setattr(axes, "_read_text", read_text)

    found = read_axes(load_image(EXAMPLES / "logarithmic.png"))

    # The x-axis labels read as 100 to 104, so only their raised last digits
    # show that they are powers of ten.
    assert [value for _, value in found.x_ticks] == [1, 10, 100, 1000, 10000]
    assert found.calibration.x_log


def test_reads_negative_exponents_whose_minus_ocr_missed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def read_text(image: Image) -> list[_Text]:
        """Read the text as OCR does, but with superscripts plain and no minus."""
        return [
            dataclasses.replace(line, text=line.text.translate(_PLAIN).replace("-", ""))
            for line in _read_text(image)
        ]

    monkeypatch.setattr(axes, "_read_text", read_text)

    found = read_axes(load_image(EXAMPLES / "logarithmic.png"))

    # 10 to the -1 reads as 101, and only its raised minus sign shows it.
    assert [value for _, value in found.y_ticks] == [0.1, 1, 10, 100, 1000]


def test_reads_powers_of_two() -> None:
    found = read_axes(load_image(EXAMPLES / "binary.png"))
    # The pixel positions of (1, 1/4), (16, 1/4) and (1, 64) that
    # examples/make_examples.py prints.
    pixels = [(84, 66), (610, 66), (84, 380)]

    data = found.calibration.to_data(pixels)

    # OCR misses the superscripts of the x-axis labels, reading 2 to the 0
    # to 4 as 20 to 24, and of some of the y-axis labels.
    assert [value for _, value in found.x_ticks] == [1, 2, 4, 8, 16]
    assert [value for _, value in found.y_ticks] == [2**n for n in range(-2, 7)]
    assert found.calibration.x_log
    assert found.calibration.y_log
    expected = [(1, 0.25), (16, 0.25), (1, 64)]
    np.testing.assert_allclose(np.log2(data), np.log2(expected), atol=0.02)
    assert found.x_title == "Size (KiB)"
    assert found.y_title == "Time (ms)"


def test_reads_titles_of_two_lines() -> None:
    found = read_axes(load_image(EXAMPLES / "titled.png"))
    # The pixel positions of (0, 0), (10, 0) and (0, 1) that
    # examples/make_examples.py prints.
    pixels = [(76, 68), (627, 68), (76, 175)]

    data = found.calibration.to_data(pixels)

    np.testing.assert_allclose(data, [(0, 0), (10, 0), (0, 1)], atol=0.03)
    assert found.x_title == "Photon energy (eV)"
    assert found.y_title == "Absorbance (arb. units)"


def test_reads_slanted_tick_labels() -> None:
    found = read_axes(load_image(EXAMPLES / "slanted.png"))
    # The pixel positions of (0, 0), (10, 0) and (0, 1) that
    # examples/make_examples.py prints.
    pixels = [(56, 56), (625, 56), (56, 190)]

    data = found.calibration.to_data(pixels)

    # As drawn, OCR reads the x-axis labels 2 as a Chinese character and 6 as
    # 9, and the 0 of 10 lies higher than the 1, like an exponent.
    assert [value for _, value in found.x_ticks] == [0, 2, 4, 6, 8, 10]
    np.testing.assert_allclose(data, [(0, 0), (10, 0), (0, 1)], atol=0.03)


@pytest.mark.parametrize(
    ("name", "pixels", "x_marks", "y_marks"),
    [
        # x labels at 30 degrees, y labels at 45 degrees.
        (
            "tilted",
            [(59, 55), (625, 55), (59, 189)],
            [59.5, 172.5, 285.5, 398.5, 511.5, 625.5],
            [54.5, 121.5, 188.5, 255.5, 322.5, 390.5],
        ),
        # x labels at 60 degrees, y labels written upwards.
        (
            "steep",
            [(48, 56), (625, 56), (48, 190)],
            [48.5, 163.5, 279.5, 394.5, 510.5, 625.5],
            [55.5, 122.5, 189.5, 256.5, 323.5, 390.5],
        ),
        # x and y labels written downwards, and the y title too.
        (
            "downwards",
            [(48, 52), (629, 52), (48, 187)],
            [48.5, 164.5, 280.5, 396.5, 513.5, 629.5],
            [51.5, 119.5, 186.5, 254.5, 322.5, 390.5],
        ),
    ],
)
def test_reads_rotated_tick_labels(
    name: str,
    pixels: list[tuple[int, int]],
    x_marks: list[float],
    y_marks: list[float],
) -> None:
    # The pixels are the positions of (0, 0), (10, 0) and (0, 1) that
    # examples/make_examples.py prints, and the marks the centres of the
    # pixels of the tick marks drawn.
    found = read_axes(load_image(EXAMPLES / f"{name}.png"))

    data = found.calibration.to_data(pixels)

    assert [value for _, value in found.x_ticks] == [0, 2, 4, 6, 8, 10]
    assert [value for _, value in found.y_ticks] == [0, 0.5, 1, 1.5, 2, 2.5]
    np.testing.assert_allclose([pixel for pixel, _ in found.x_ticks], x_marks, atol=1)
    np.testing.assert_allclose([pixel for pixel, _ in found.y_ticks], y_marks, atol=1)
    np.testing.assert_allclose(data, [(0, 0), (10, 0), (0, 1)], atol=0.03)
    assert (found.x_title, found.y_title) == ("Time (s)", "Voltage (mV)")


def test_refuses_y_labels_written_upwards_read_without_their_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Read as they lie, without turning them, OCR reads the y-axis labels of
    # steep.png as 0.0, 05, 1.0, 15 and 25, losing the points of three.
    monkeypatch.setattr(axes, "_read_sideways", lambda _image, _right, **_options: [])

    with pytest.raises(ValueError, match="OCR lost characters"):
        read_axes(load_image(EXAMPLES / "steep.png"))


@pytest.mark.parametrize("name", ["labelled", "spectrum"])
def test_finds_where_the_axis_lines_cross(name: str) -> None:
    # The axes of labelled.png start before the first ticks, and those of
    # spectrum.png at them.
    origin = read_axes(load_image(EXAMPLES / f"{name}.png")).calibration.origin

    assert origin == pytest.approx((80, 44), abs=1)


def test_needs_tick_labels() -> None:
    blank = np.full((100, 200, 3), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="found no row of 3 or more numbers"):
        read_axes(blank)


@pytest.mark.usefixtures("without_ocr")
def test_needs_the_ocr_extra() -> None:
    with pytest.raises(ModuleNotFoundError, match="needs the ocr extra"):
        read_axes(load_image(EXAMPLES / "labelled.png"))


def text(content: str, x: float, y: float, width: float = 20) -> _Text:
    """Return a line of text 14 pixels high, centred on a pixel position."""
    return _Text(content, x - width / 2, y - 7, x + width / 2, y + 7)


def x_labels(values: Sequence[str]) -> list[_Text]:
    """Return x-axis tick labels, 80 pixels apart from x = 80."""
    return [text(value, 80 + 80 * i, 45) for i, value in enumerate(values)]


def y_labels(values: Sequence[str]) -> list[_Text]:
    """Return y-axis tick labels, 60 pixels apart from y = 80, aligned right."""
    return [text(value, 40, 80 + 60 * i) for i, value in enumerate(values)]


def tall(content: str, x: float, y: float, height: float) -> _Text:
    """Return a line of text written upwards, 14 pixels wide, centred on a pixel."""
    return _Text(content, x - 7, y - height / 2, x + 7, y + height / 2)


def plot(*, lines: bool = True) -> Image:
    """Draw a blank plot, with or without axis lines through ``AXIS_LINES``."""
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    if lines:
        corner = (AXIS_LINES[0], HEIGHT - AXIS_LINES[1])
        cv2.line(image, (AXIS_LINES[0], 0), corner, (0, 0, 0), 1)
        cv2.line(image, corner, (WIDTH - 1, corner[1]), (0, 0, 0), 1)
    return image


def read(
    monkeypatch: pytest.MonkeyPatch, texts: Sequence[_Text], *, lines: bool = True
) -> Axes:
    """Read the axes of a blank plot as though OCR had found the texts."""
    monkeypatch.setattr(axes, "_read_text", lambda _image: list(texts))
    return read_axes(plot(lines=lines))


def ocr(monkeypatch: pytest.MonkeyPatch, texts: Sequence[_Text]) -> None:
    """Make OCR find the texts in a plot, as RapidOCR gives them.

    RapidOCR gives the corners of each text's box clockwise from the top left,
    in pixels from the top-left corner of the image.
    """
    boxes = [
        [
            (t.left, HEIGHT - t.top),
            (t.right, HEIGHT - t.top),
            (t.right, HEIGHT - t.bottom),
            (t.left, HEIGHT - t.bottom),
        ]
        for t in texts
    ]
    found = SimpleNamespace(
        boxes=np.array(boxes, dtype=np.float32),
        txts=[t.text for t in texts],
        scores=[t.score for t in texts],
    )
    monkeypatch.setattr(axes, "_engine", lambda: lambda _image, **_options: found)


@pytest.mark.parametrize(
    ("misread", "written"),
    [
        # Symbols that OCR reads as either of two that look alike.
        ("\N{INCREMENT}E", "\N{GREEK CAPITAL LETTER DELTA}E"),
        ("\N{MICRO SIGN}m", "\N{GREEK SMALL LETTER MU}m"),
        ("\N{OHM SIGN}", "\N{GREEK CAPITAL LETTER OMEGA}"),
        ("\N{ANGSTROM SIGN}", "\N{LATIN CAPITAL LETTER A WITH RING ABOVE}"),
        ("\N{FULLWIDTH LEFT PARENTHESIS}5\N{FULLWIDTH PERCENT SIGN} ", "(5%"),
        # The raised signs of superscripts, read full size.
        (
            "10\N{MINUS SIGN}\N{SUPERSCRIPT THREE}",
            "10\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT THREE}",
        ),
        ("m-\N{SUPERSCRIPT THREE}", "m\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT THREE}"),
        ("10+\N{SUPERSCRIPT TWO}", "10\N{SUPERSCRIPT PLUS SIGN}\N{SUPERSCRIPT TWO}"),
        # Subscripts, and signs before full-size digits, stay as they are.
        ("CO\N{SUBSCRIPT TWO} -2", "CO\N{SUBSCRIPT TWO} -2"),
    ],
)
def test_writes_symbols_read_in_several_ways_one_way(
    monkeypatch: pytest.MonkeyPatch, misread: str, written: str
) -> None:
    ocr(monkeypatch, [_Text(misread, 20, 8, 380, 22, 0.9)])

    assert _read_text(plot()) == [_Text(written, 20, 8, 380, 22, 0.9)]


def test_reads_powers_whose_minus_ocr_read_full_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minus = "\N{MINUS SIGN}"
    powers = ["\N{SUPERSCRIPT THREE}", "\N{SUPERSCRIPT TWO}", "\N{SUPERSCRIPT ONE}"]
    labels = [
        *x_labels([f"10{minus}{power}" for power in powers]),
        *y_labels(["0", "1", "2"]),
    ]
    ocr(monkeypatch, labels)

    found = read_axes(plot())

    assert [value for _, value in found.x_ticks] == [0.001, 0.01, 0.1]
    assert found.calibration.x_log


def test_fits_the_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2", "3"]), *y_labels(["0", "0.5", "1", "1.5"])]

    found = read(monkeypatch, labels)

    assert list(found.x_ticks) == [(80, 0), (160, 1), (240, 2), (320, 3)]
    assert list(found.y_ticks) == [(80, 0), (140, 0.5), (200, 1), (260, 1.5)]
    # The centres of the pixels of the lines through AXIS_LINES.
    assert found.calibration.origin == (60.5, 59.5)
    np.testing.assert_allclose(
        found.calibration.to_data([(80, 80), (320, 260)]),
        [(0, 0), (3, 1.5)],
        atol=1e-12,
    )
    assert found.x_title is None
    assert found.y_title is None


def test_reads_reversed_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Values fall from left to right, and from bottom to top, as on a depth axis.
    labels = [*x_labels(["3", "2", "1", "0"]), *y_labels(["100", "10", "1"])]

    found = read(monkeypatch, labels)

    assert list(found.x_ticks) == [(80, 3), (160, 2), (240, 1), (320, 0)]
    assert list(found.y_ticks) == [(80, 100), (140, 10), (200, 1)]
    assert found.calibration.y_log
    assert found.calibration.origin == (60.5, 59.5)
    np.testing.assert_allclose(
        found.calibration.to_data([(80, 80), (320, 200)]),
        [(3, 100), (0, 1)],
        atol=1e-12,
    )


def test_finds_axis_lines_past_stretched_label_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OCR stretched the boxes of the labels at the corner over the axis lines,
    # but not so far that they no longer line up with the others.
    near = [text(value, 44, 80 + 60 * i) for i, value in enumerate("012")]
    labels = [*x_labels(["0", "1", "2"]), *near]
    labels[0] = dataclasses.replace(labels[0], top=64)
    labels[3] = dataclasses.replace(labels[3], right=61)

    found = read(monkeypatch, labels)

    assert found.calibration.origin == (60.5, 59.5)


@pytest.mark.parametrize(
    "minus",
    [
        "\N{MINUS SIGN}",
        # The dashes that OCR can read for a minus sign.
        "\N{HYPHEN}",
        "\N{NON-BREAKING HYPHEN}",
        "\N{FIGURE DASH}",
        "\N{EN DASH}",
        "\N{EM DASH}",
        "\N{HORIZONTAL BAR}",
        "\N{SMALL HYPHEN-MINUS}",
        "\N{FULLWIDTH HYPHEN-MINUS}",
    ],
)
def test_reads_minus_signs(monkeypatch: pytest.MonkeyPatch, minus: str) -> None:
    labels = [*x_labels(["-1", "0", "1"]), *y_labels([f"{minus}1", "0", "1"])]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [-1, 0, 1]
    assert [value for _, value in found.y_ticks] == [-1, 0, 1]


def test_reads_letters_misread_for_digits(monkeypatch: pytest.MonkeyPatch) -> None:
    # OCR read 0 as O or o, and 1 as l, I or |. A lone O has no digit to show
    # that it is one, and l after a digit may be a unit, so both are dropped.
    labels = [
        *x_labels(["O", "l0", "2O", "3o", "4l"]),
        *y_labels(["0", "0.5", "I.0", "|.5"]),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [10, 20, 30]
    assert [value for _, value in found.y_ticks] == [0, 0.5, 1, 1.5]


def test_reads_separators_percentages_and_exponents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [
        *x_labels(["0", "1,000", "2,000", "3,000"]),
        *y_labels(["0%", "50%", "100%"]),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [0, 1000, 2000, 3000]
    assert [value for _, value in found.y_ticks] == [0, 50, 100]


@pytest.mark.parametrize(
    "space",
    [" ", "\N{NO-BREAK SPACE}", "\N{THIN SPACE}", "\N{NARROW NO-BREAK SPACE}"],
)
def test_reads_thousands_separators_of_spaces(
    monkeypatch: pytest.MonkeyPatch, space: str
) -> None:
    labels = [
        *x_labels(["0", f"5{space}000", f"10{space}000"]),
        *y_labels(["0", "1", "2"]),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [0, 5000, 10000]


def test_reads_decimal_commas(monkeypatch: pytest.MonkeyPatch) -> None:
    # As German plots write them: 2.000 is two thousand on the x-axis, as the
    # decimal commas on the y-axis show.
    labels = [
        *x_labels(["0", "2.000", "4.000", "6.000"]),
        *y_labels(["-0,5", "0,0", "0,5", "1,0"]),
        text("Länge, gesamt (m)", 200, 20, width=120),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [0, 2000, 4000, 6000]
    assert [value for _, value in found.y_ticks] == [-0.5, 0, 0.5, 1]
    assert found.x_title == "Länge, gesamt (m)"


def test_one_misread_comma_does_not_make_decimal_commas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The second y label, 0.5, was misread as 0,5.
    labels = [
        *x_labels(["0", "1,000", "2,000", "3,000"]),
        *y_labels(["0.0", "0,5", "1.0", "1.5"]),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [0, 1000, 2000, 3000]
    assert [value for _, value in found.y_ticks] == [0, 1, 1.5]


def test_reads_scientific_notation(monkeypatch: pytest.MonkeyPatch) -> None:
    # R labels a logarithmic axis 1e+01, 1e+02, ...
    labels = [
        *x_labels(["1e+01", "1e+02", "1e+03"]),
        *y_labels(["0.0E0", "2.5E-3", "5.0e-3"]),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [10, 100, 1000]
    assert found.calibration.x_log
    assert [value for _, value in found.y_ticks] == [0, 0.0025, 0.005]


def test_ignores_tick_marks_read_as_dashes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without axis lines, OCR can read the tick above an x-axis label as a
    # minus sign, making the label's box taller, and the tick after a y-axis
    # label as a dash.
    ticked = [_Text(f"-{i}", 70 + 80 * i, 38, 90 + 80 * i, 62) for i in range(3)]
    labels = [*ticked, *y_labels(["0 -", "1 -", "2 -"])]

    found = read(monkeypatch, labels, lines=False)

    assert [value for _, value in found.x_ticks] == [0, 1, 2]
    assert [value for _, value in found.y_ticks] == [0, 1, 2]


def test_keeps_minus_signs_of_labels_of_the_usual_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*x_labels(["-2", "-1", "0"]), *y_labels(["0", "1", "2"])]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [-2, -1, 0]


def test_drops_misread_tick_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    # The third x label, 2, was misread as 1; the third y label, 1, as 10.
    labels = [
        *x_labels(["0", "1", "1", "3", "4"]),
        *y_labels(["0", "0.5", "10", "1.5"]),
    ]

    found = read(monkeypatch, labels)

    assert list(found.x_ticks) == [(80, 0), (160, 1), (320, 3), (400, 4)]
    assert list(found.y_ticks) == [(80, 0), (140, 0.5), (260, 1.5)]


def test_drops_tick_values_off_the_even_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 750 misread as 758, close enough to its tick to be in line.
    labels = [
        *x_labels(["0", "250", "500", "758", "1000"]),
        *y_labels(["0", "1", "2"]),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [0, 250, 500, 1000]


def write_across(image: Image, x: float, y: float, content: str) -> _Text:
    """Write text across, centred on a pixel position, in a box 3 pixels around it.

    Returns:
        The text as OCR reads it, and its box.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), _ = cv2.getTextSize(content, font, 0.5, 1)
    left, baseline = round(x - width / 2), HEIGHT - round(y - height / 2)
    cv2.putText(image, content, (left, baseline), font, 0.5, (0, 0, 0), 1)
    return _Text(
        content, left - 3, y - height / 2 - 4, left + width + 3, y + height / 2 + 4
    )


@pytest.mark.parametrize(
    ("read_as", "values"),
    [
        (["0.5", "10", "1.5", "2.0"], [0.5, 1.5, 2.0]),  # 1.0 lost its point
        (["5", "1.0", "1.5", "2.0"], [1.0, 1.5, 2.0]),  # 0.5 lost two characters
    ],
)
def test_drops_tick_labels_that_lost_characters(
    monkeypatch: pytest.MonkeyPatch, read_as: list[str], values: list[float]
) -> None:
    # OCR read a label as fewer characters than are drawn in its box.
    image = plot()
    drawn = [
        write_across(image, 30, 80 + 40 * i, value)
        for i, value in enumerate(["0.5", "1.0", "1.5", "2.0"])
    ]
    read_labels = [
        dataclasses.replace(label, text=text)
        for label, text in zip(drawn, read_as, strict=True)
    ]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*x_labels(["0", "1", "2"]), *read_labels]
    )

    found = read_axes(image)

    assert [value for _, value in found.y_ticks] == values


def test_refuses_tick_labels_that_all_lost_their_decimal_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Read as 5, 10 and 15, the labels are in line, at even steps, but each
    # value is ten times too large.
    image = plot()
    drawn = [
        write_across(image, 30, 80 + 60 * i, v)
        for i, v in enumerate(["0.5", "1.0", "1.5"])
    ]
    read_as = [
        dataclasses.replace(label, text=label.text.replace(".", "").lstrip("0"))
        for label in drawn
    ]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*x_labels(["0", "1", "2"]), *read_as]
    )

    with pytest.raises(
        ValueError, match="OCR lost characters, such as decimal points, of 3 of the 3 y"
    ):
        read_axes(image)


def test_needs_tick_values_at_even_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    # 250 and 750 misread as 258 and 758: no three labels take even steps.
    labels = [*x_labels(["0", "258", "500", "758"]), *y_labels(["0", "1", "2"])]

    with pytest.raises(
        ValueError, match="only 2 of the 4 x-axis tick labels are in line, on a"
    ):
        read(monkeypatch, labels)


def test_needs_tick_labels_in_line(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["1", "3", "2", "4"]), *y_labels(["0", "1", "2"])]

    with pytest.raises(
        ValueError, match="only 2 of the 4 x-axis tick labels are in line, on a"
    ):
        read(monkeypatch, labels)


def test_reads_powers_of_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    # As OCR reads 10 to the -1, 0, 1 and 2, missing some superscripts.
    minus_one = "10\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}"
    labels = [
        *x_labels([minus_one, "100", "101", "10^{2"]),
        *y_labels(["10-1", "10\N{DEGREE SIGN}", "10^\N{SUPERSCRIPT ONE}"]),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [0.1, 1, 10, 100]
    assert [value for _, value in found.y_ticks] == [0.1, 1, 10]
    assert found.calibration.x_log
    assert found.calibration.y_log
    np.testing.assert_allclose(
        found.calibration.to_data([(120, 110)]), [(10**-0.5, 10**-0.5)]
    )


def test_reads_powers_of_other_bases(monkeypatch: pytest.MonkeyPatch) -> None:
    # As OCR reads 2 to the 0 to 3, and e to the -1, 0 and 1 as matplotlib
    # writes them.
    e = "2.718281828459045"
    labels = [
        *x_labels(["2\N{SUPERSCRIPT ZERO}", "2^1", "2^{2}", "2\N{SUPERSCRIPT THREE}"]),
        *y_labels(
            [
                f"{e}\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}",
                f"{e}\N{DEGREE SIGN}",
                f"{e}^1",
            ]
        ),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [1, 2, 4, 8]
    y_values = [value for _, value in found.y_ticks]
    np.testing.assert_allclose(y_values, [1 / math.e, 1, math.e])
    assert found.calibration.x_log
    assert found.calibration.y_log


def test_one_misread_label_does_not_make_powers_of_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 101 misread with a degree sign for a superscript 0.
    labels = [
        *x_labels(["100", "10\N{DEGREE SIGN}", "102", "103"]),
        *y_labels(["0", "1", "2"]),
    ]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [100, 102, 103]
    assert not found.calibration.x_log


def test_reads_multiples_of_powers_of_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    # matplotlib labels minor ticks too on an axis spanning about a decade.
    one = "10\N{DEGREE SIGN}"
    values = {one: 1, f"2 \N{MULTIPLICATION SIGN} {one}": 2, "3x10^0": 3}
    minor_labels = [
        text(label, 40, 80 + 200 * math.log10(value)) for label, value in values.items()
    ]

    found = read(monkeypatch, [*x_labels(["0", "1", "10", "100"]), *minor_labels])

    # 0 is not in line on the logarithmic x-axis.
    assert [value for _, value in found.x_ticks] == [1, 10, 100]
    assert [value for _, value in found.y_ticks] == [1, 2, 3]
    assert found.calibration.x_log
    assert found.calibration.y_log


def test_reads_numbers_like_102_without_powers_of_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*x_labels(["100", "101", "102"]), *y_labels(["1", "10", "100"])]

    found = read(monkeypatch, labels)

    # 100, 101 and 102 fall in line on both scales, and the linear one wins.
    assert [value for _, value in found.x_ticks] == [100, 101, 102]
    assert [value for _, value in found.y_ticks] == [1, 10, 100]
    assert not found.calibration.x_log
    assert found.calibration.y_log


def write(image: Image, x: float, y: float, content: str, superscript: str) -> _Text:
    """Write text centred on a pixel position, then a smaller, raised superscript.

    Returns:
        The text as OCR reads it if it misses the superscript, and its box.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), _ = cv2.getTextSize(content, font, 0.5, 1)
    left, baseline = round(x - width / 2), HEIGHT - round(y - height / 2)
    cv2.putText(image, content, (left, baseline), font, 0.5, (0, 0, 0), 1)
    above = (left + width, baseline - height // 2)
    cv2.putText(image, superscript, above, font, 0.35, (0, 0, 0), 1)
    (raised, _), _ = cv2.getTextSize(superscript, font, 0.35, 1)
    return _Text(
        content + superscript,
        x - width / 2 - 2,
        y - 10,
        x + width / 2 + raised + 3,
        y + 10,
    )


@pytest.mark.parametrize(
    ("raised", "values", "log"),
    [(True, [1, 10, 100], True), (False, [100, 101, 102], False)],
)
def test_reads_raised_last_digits_as_exponents(
    monkeypatch: pytest.MonkeyPatch, raised: bool, values: list[float], log: bool
) -> None:
    # OCR reads 10 to the 0, 1 and 2 as 100, 101 and 102, like the numbers.
    image = plot()
    labels = []
    for i in range(3):
        content, superscript = ("10", f"{i}") if raised else (f"10{i}", "")
        labels.append(write(image, 80 + 80 * i, 45, content, superscript))
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*labels, *y_labels(["0", "1", "2"])]
    )

    found = read_axes(image)

    assert [label.text for label in labels] == ["100", "101", "102"]
    assert [value for _, value in found.x_ticks] == values
    assert found.calibration.x_log is log


@pytest.mark.parametrize(
    ("exponents", "values"),
    [(["0", "1", "2"], [1, 2, 4]), (["10", "11", "12"], [1024, 2048, 4096])],
)
def test_reads_raised_digits_as_exponents_of_other_bases(
    monkeypatch: pytest.MonkeyPatch, exponents: list[str], values: list[float]
) -> None:
    # OCR reads 2 to the 0, 1 and 2 as 20, 21 and 22, and 2 to the 10 as 210.
    image = plot()
    labels = [write(image, 80 + 80 * i, 45, "2", e) for i, e in enumerate(exponents)]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*labels, *y_labels(["0", "1", "2"])]
    )

    found = read_axes(image)

    assert [value for _, value in found.x_ticks] == values
    assert found.calibration.x_log


def test_reads_powers_of_the_base_most_labels_show(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 2 to the 1 is drawn without a raised digit, and 2 to the 3 is misread
    # as 9 to the 3.
    image = plot()
    drawn = [("2", "0"), ("21", ""), ("2", "2"), ("9", "3")]
    labels = [write(image, 80 + 80 * i, 45, *parts) for i, parts in enumerate(drawn)]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*labels, *y_labels(["0", "1", "2"])]
    )

    found = read_axes(image)

    # Most labels are powers of 2, so 21 is one too, but 93 is out of line.
    assert [label.text for label in labels] == ["20", "21", "22", "93"]
    assert [value for _, value in found.x_ticks] == [1, 2, 4]
    assert found.calibration.x_log


def write_upwards(
    image: Image, x: float, y: float, content: str, *, upwards: bool = True
) -> _Text:
    """Write text upwards, or downwards, centred on a pixel, as rotated labels are.

    Returns:
        The text as OCR reads it, and its box.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(content, font, 0.5, 1)
    across = np.full((height + baseline + 4, width + 4, 3), 255, dtype=np.uint8)
    cv2.putText(across, content, (2, height + 2), font, 0.5, (0, 0, 0), 1)
    turn = cv2.ROTATE_90_COUNTERCLOCKWISE if upwards else cv2.ROTATE_90_CLOCKWISE
    turned = cv2.rotate(across, turn)
    tall, wide = turned.shape[:2]
    top, left = HEIGHT - round(y + tall / 2), round(x - wide / 2)
    image[top : top + tall, left : left + wide] = turned
    return _Text(content, left, HEIGHT - top - tall, left + wide, HEIGHT - top)


def test_reads_labels_written_upwards(monkeypatch: pytest.MonkeyPatch) -> None:
    # Written upwards, each label's last digit stands above the others.
    image = plot()
    labels = [write_upwards(image, 80 + 80 * i, 40, f"{2 * i + 2}0") for i in range(3)]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*labels, *y_labels(["0", "1", "2"])]
    )

    found = read_axes(image)

    assert [value for _, value in found.x_ticks] == [20, 40, 60]
    assert not found.calibration.x_log


def write_slanted(
    image: Image, x: float, y: float, content: str, angle: float = 45
) -> _Text:
    """Write text rising at an angle, in degrees, centred on a pixel position.

    Returns:
        The text as OCR reads it, and the box around it, tilted as the text is.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), _ = cv2.getTextSize(content, font, 0.5, 1)
    side = width + 8
    across = np.full((side, side, 3), 255, dtype=np.uint8)
    cv2.putText(across, content, (4, (side + height) // 2), font, 0.5, (0, 0, 0), 1)
    turn = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1)
    slanted = cv2.warpAffine(across, turn, (side, side), borderValue=(255, 255, 255))
    top, left = HEIGHT - round(y + side / 2), round(x - side / 2)
    image[top : top + side, left : left + side] = slanted
    return _Text(
        content, left, HEIGHT - top - side, left + side, HEIGHT - top, 1, angle
    )


def test_reads_slanted_labels_without_exponents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rising to the right, each label's last digit lies above the others.
    image = plot()
    labels = [write_slanted(image, 80 + 80 * i, 40, f"{2 * i + 2}0") for i in range(3)]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*labels, *y_labels(["0", "1", "2"])]
    )

    found = read_axes(image)

    assert [value for _, value in found.x_ticks] == [20, 40, 60]


def test_reads_slightly_slanted_labels_in_line_as_they_are(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rising at 25 degrees, the 0 of each label lies higher than the digit
    # before it, but is as tall, so it is no exponent. OCR read the labels
    # well, in boxes not tilted, so they are not read again turned.
    image = plot()
    labels = [
        dataclasses.replace(
            write_slanted(image, 80 + 80 * i, 40, f"{i + 1}0", 25), slant=0
        )
        for i in range(3)
    ]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*labels, *y_labels(["0", "1", "2"])]
    )

    def read_turned(_image: Image, _top: float, angle: float) -> list[_Text]:
        raise AssertionError(angle)

    monkeypatch.setattr(axes, "_read_turned", read_turned)

    found = read_axes(image)

    assert [value for _, value in found.x_ticks] == [10, 20, 30]
    assert not found.calibration.x_log


@pytest.mark.parametrize("missed", [1, 2])
def test_reads_slanted_labels_again_turned_level(
    monkeypatch: pytest.MonkeyPatch, missed: int
) -> None:
    # Written at 60 degrees, OCR reads 2 as a Chinese character, or misses
    # 4, leaving a gap. Turned level, by minus 60 degrees, it reads them all,
    # and the title below them.
    image = plot()
    drawn = [write_slanted(image, 80 + 80 * i, 40, v, 60) for i, v in enumerate("0246")]
    if missed == 1:
        drawn[1] = dataclasses.replace(drawn[1], text="\N{CJK UNIFIED IDEOGRAPH-4E59}")
    else:
        del drawn[2]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*drawn, *y_labels(["0", "1", "2"])]
    )
    angles: list[float] = []

    def read_turned(_image: Image, _top: float, angle: float) -> list[_Text]:
        angles.append(angle)
        numbers = [text(v, 80 + 80 * i, 41, 12) for i, v in enumerate("0246")]
        return [*numbers, text("Time", 200, 10, 40)]

    monkeypatch.setattr(axes, "_read_turned", read_turned)

    found = read_axes(image)

    assert [value for _, value in found.x_ticks] == [0, 2, 4, 6]
    assert [pixel for pixel, _ in found.x_ticks] == pytest.approx(
        [80, 160, 240, 320], abs=1
    )
    assert angles == [pytest.approx(-60, abs=10)]


def test_keeps_slanted_labels_read_no_better_turned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = plot()
    drawn = [write_slanted(image, 80 + 80 * i, 40, v) for i, v in enumerate("0246")]
    drawn[1] = dataclasses.replace(drawn[1], text="\N{CJK UNIFIED IDEOGRAPH-4E59}")
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*drawn, *y_labels(["0", "1", "2"])]
    )
    monkeypatch.setattr(axes, "_read_turned", lambda _image, _top, _angle: [])

    found = read_axes(image)

    assert [value for _, value in found.x_ticks] == [0, 4, 6]


@pytest.mark.parametrize("sure", [-90, 90])
def test_reads_x_labels_written_up_or_down_again_turned_the_surer_way(
    monkeypatch: pytest.MonkeyPatch, sure: int
) -> None:
    # OCR reads the tick mark above 1.0, written upwards, as a minus sign.
    # Turned a quarter turn each way, it reads the labels in line, but less
    # surely upside down, misread as other numbers.
    image = plot()
    drawn = [
        write_upwards(image, 80 + 80 * i, 40, value)
        for i, value in enumerate(["0.5", "1.0", "1.5"])
    ]
    drawn[1] = dataclasses.replace(drawn[1], text="-1.0")
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*drawn, *y_labels(["0", "1", "2"])]
    )

    def read_turned(_image: Image, _top: float, angle: float) -> list[_Text]:
        values, score = (["0.5", "1.0", "1.5"], 1) if angle == sure else ("123", 0.7)
        return [
            dataclasses.replace(text(value, 80 + 80 * i, 40), score=score, slant=-angle)
            for i, value in enumerate(values)
        ]

    monkeypatch.setattr(axes, "_read_turned", read_turned)

    found = read_axes(image)

    assert [value for _, value in found.x_ticks] == [0.5, 1.0, 1.5]


def test_merges_labels_read_turned_into_those_first_read() -> None:
    labels = [
        text("1", 50, 40),  # holds the middle of -5, as does the next
        text("!", 60, 40),
        text("8", 100, 40),  # holds the middles of 3 and 4
        text("9", 160, 40),  # holds the middle of 6 alone
        text("x", 220, 40),  # holds none
    ]
    turned = [
        text("-5", 55, 40, 10),
        text("3", 96, 40, 4),
        text("4", 104, 40, 4),
        dataclasses.replace(text("6", 161, 41, 10), score=0.9, slant=60),
        text("7", 300, 40),  # in no label's box
    ]

    merged = _merged(labels, turned)

    assert merged == [
        labels[4],
        *turned[:3],
        dataclasses.replace(labels[3], text="6", score=0.9, slant=60),
        turned[4],
    ]


def sideways(
    values: Sequence[str], y: float, *, upwards: bool, score: float = 1
) -> list[_Text]:
    """Return y-axis tick labels written up or down, as read turned to run across.

    The labels are 14 pixels wide and 24 tall, 60 pixels apart from y up.
    """
    slant = 90 if upwards else -90
    return [
        dataclasses.replace(tall(value, 40, y + 60 * i, 24), score=score, slant=slant)
        for i, value in enumerate(values)
    ]


@pytest.mark.parametrize("upwards", [True, False])
def test_reads_labels_written_up_or_down_again_turned(
    monkeypatch: pytest.MonkeyPatch, upwards: bool
) -> None:
    # OCR runs the labels above 0.5 together, reading them from the bottom
    # if written upwards and from the top if downwards. Turned the other way,
    # it reads the labels less surely, misread without their points. Turned
    # either way, it reads the number in the title as a word of its own.
    merged = "1.0 1.5 2.0" if upwards else "2.0 1.5 1.0"
    labels = [tall("0.5", 40, 80, 30), tall(merged, 40, 200, 150)]
    title = tall("Run 2 (V)", 10, 170, 90)
    turned = {
        upwards: sideways(["0.5", "1.0", "1.5", "2.0"], 80, upwards=upwards),
        not upwards: sideways(
            ["05", "10", "15", "20"], 80, upwards=not upwards, score=0.7
        ),
    }
    for way, words in turned.items():
        words.append(
            dataclasses.replace(tall("2", 10, 170, 10), slant=90 if way else -90)
        )
    monkeypatch.setattr(
        axes, "_read_sideways", lambda _image, _right, *, upwards: turned[upwards]
    )

    found = read(monkeypatch, [*x_labels(["0", "1", "2"]), *labels, title])

    assert found.y_ticks == ((80.0, 0.5), (140.0, 1.0), (200.0, 1.5), (260.0, 2.0))
    assert found.y_title == "Run 2 (V)"


def test_keeps_labels_read_as_no_numbers_turned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [tall(value, 40, 80 + 60 * i, 30) for i, value in enumerate("012")]
    labels = [dataclasses.replace(label, text=f"{label.text}.0") for label in labels]
    turned = {True: [], False: [tall("ll", 40, 80, 30)]}
    monkeypatch.setattr(
        axes, "_read_sideways", lambda _image, _right, *, upwards: turned[upwards]
    )

    found = read(monkeypatch, [*x_labels(["0", "1", "2"]), *labels])

    assert found.y_ticks == ((80.0, 0.0), (140.0, 1.0), (200.0, 2.0))


@pytest.mark.parametrize(
    "y_axis",
    [
        # Digits written across are taller than wide, as labels written up
        # or down are.
        [
            _Text(value, 35, 73 + 60 * i, 45, 87 + 60 * i)
            for i, value in enumerate("246")
        ],
        # A number misread from text written across can be tall too, but
        # most numbers above the x-axis tick labels are not.
        [*y_labels(["0.5", "1.0", "1.5"]), tall("08", 200, 150, 30)],
    ],
)
def test_reads_labels_written_across_as_they_are(
    monkeypatch: pytest.MonkeyPatch, y_axis: list[_Text]
) -> None:
    def read_sideways(_image: Image, _right: int, *, upwards: bool) -> list[_Text]:
        raise AssertionError(upwards)

    monkeypatch.setattr(axes, "_read_sideways", read_sideways)

    found = read(monkeypatch, [*x_labels(["0", "1", "2"]), *y_axis])

    assert [pixel for pixel, _ in found.y_ticks] == [80, 140, 200]


def test_reads_the_lowest_label_written_up_or_down_level_with_the_x_axis_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The lowest y-axis label lies level with the x-axis labels, beside its
    # tick at the corner, where its value, 0, is in line with theirs.
    corner = text("0", 60, 45)
    labels = [corner, text("1", 220, 45), text("2", 380, 45)]
    labels += [tall("0.0", 40, 45, 24), tall("0.5 1.0 1.5", 40, 165, 144)]
    turned = {
        True: sideways(["0.0", "0.5", "1.0", "1.5"], 45, upwards=True),
        False: [],
    }
    monkeypatch.setattr(
        axes, "_read_sideways", lambda _image, _right, *, upwards: turned[upwards]
    )

    found = read(monkeypatch, labels)

    assert found.x_ticks == ((60.0, 0.0), (220.0, 1.0), (380.0, 2.0))
    assert found.y_ticks == ((45.0, 0.0), (105.0, 0.5), (165.0, 1.0), (225.0, 1.5))


def test_reads_the_lowest_label_reaching_down_beside_the_x_axis_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Written at a slant, the y-axis labels' boxes are 30 pixels square, and
    # the lowest's middle lies below the tops of the x-axis labels.
    slanted = [
        _Text(value, 25, 33 + 60 * i, 55, 63 + 60 * i, slant=45)
        for i, value in enumerate(["0.0", "0.5", "1.0"])
    ]

    found = read(monkeypatch, [*x_labels(["0", "1", "2"]), *slanted])

    assert found.y_ticks == ((48.0, 0.0), (108.0, 0.5), (168.0, 1.0))


def single_digits_written_downwards(image: Image) -> list[_Text]:
    """Write y-axis tick labels 0, 2, 4 and 6 downwards, 60 pixels apart from y = 80."""
    return [
        write_upwards(image, 40, 80 + 60 * i, digit, upwards=False)
        for i, digit in enumerate("0246")
    ]


def test_reads_single_digits_written_downwards_again_turned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OCR reads single digits written downwards in boxes wider than tall,
    # and 6 as 9, out of line. Their characters run sideways, so they are
    # read again turned each way, less surely upside down.
    image = plot()
    drawn = single_digits_written_downwards(image)
    first = [*drawn[:3], dataclasses.replace(drawn[3], text="9")]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*x_labels(["0", "1", "2"]), *first]
    )
    turned = {
        False: [dataclasses.replace(label, slant=-90) for label in drawn],
        True: [
            dataclasses.replace(label, text=upside_down, score=0.7, slant=90)
            for label, upside_down in zip(drawn, "0hp9", strict=True)
        ],
    }
    monkeypatch.setattr(
        axes, "_read_sideways", lambda _image, _right, *, upwards: turned[upwards]
    )

    found = read_axes(image)

    assert [value for _, value in found.y_ticks] == [0, 2, 4, 6]


def test_reads_single_digits_written_downwards_as_they_are_if_read_well(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = plot()
    drawn = single_digits_written_downwards(image)
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*x_labels(["0", "1", "2"]), *drawn]
    )

    def read_sideways(_image: Image, _right: int, *, upwards: bool) -> list[_Text]:
        raise AssertionError(upwards)

    monkeypatch.setattr(axes, "_read_sideways", read_sideways)

    found = read_axes(image)

    assert [value for _, value in found.y_ticks] == [0, 2, 4, 6]


@pytest.mark.parametrize("upwards", [True, False])
def test_parts_numbers_run_together_by_their_characters(upwards: bool) -> None:
    # Written up or down, -1 and -2 run together as -1-2: four characters,
    # each a run of six rows of ink, from y = 10 up, four rows apart.
    ink = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for bottom in (10, 20, 30, 40):
        ink[HEIGHT - bottom - 6 : HEIGHT - bottom, 20:30] = 255
    slant = 90 if upwards else -90
    word = _Text("-1-2", 15, 8, 35, 48, 0.9, slant)

    parted = _split(word, ink)

    # Written upwards, the first number is the lowest; downwards, the highest.
    low, high = (10, 26), (30, 46)
    first, second = (low, high) if upwards else (high, low)
    assert parted == [
        _Text("-1", 15, first[0], 35, first[1], 0.9, slant),
        _Text("-2", 15, second[0], 35, second[1], 0.9, slant),
    ]
    # A word of one number stays as it is, and one whose characters cannot
    # all be told apart is left out.
    assert _split(dataclasses.replace(word, text="-12"), ink) == [
        dataclasses.replace(word, text="-12")
    ]
    assert _split(dataclasses.replace(word, text="-1-25"), ink) == []


def test_reads_no_labels_in_line_of_fewer_than_three() -> None:
    labels = x_labels(["0", "1"])
    ink = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    assert _in_line_along(labels, lambda t: t.x, ink, 14) == []


@pytest.mark.parametrize("upwards", [True, False])
def test_reads_words_sideways_where_they_lie(
    monkeypatch: pytest.MonkeyPatch, upwards: bool
) -> None:
    def read_words(image: Image) -> list[_Text]:
        """Read each piece of ink in an image as a word in its box, as OCR would."""
        ink = (image.min(axis=2) < 128).astype(np.uint8)
        stats = cv2.connectedComponentsWithStats(ink)[2][1:]
        height = image.shape[0]
        return [
            _Text("1", left, height - top - tall, left + wide, height - top, 0.9)
            for left, top, wide, tall, _ in stats.tolist()
        ]

    image = plot(lines=False)
    image[HEIGHT - 120 : HEIGHT - 90, 25:35] = 0  # x from 25 to 35, y from 90 to 120
    monkeypatch.setattr(axes, "_read_words", read_words)

    words = _read_sideways(image, 50, upwards=upwards)

    assert words == [_Text("1", 25, 90, 35, 120, 0.9, 90 if upwards else -90)]


def test_reads_words_as_lines_lie_and_lines_turned_upright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RapidOCR keeps the options of its last call.
    found = SimpleNamespace(
        boxes=np.zeros((1, 4, 2), dtype=np.float32),
        txts=["1.25 1.50"],
        scores=[0.9],
        word_results=[
            [
                ("1.25", 1.0, [[10, 5], [40, 5], [40, 20], [10, 20]]),
                ("1.50", 0.8, [[50, 5], [80, 5], [80, 20], [50, 20]]),
            ]
        ],
    )
    options: list[dict[str, bool]] = []

    def engine(_image: Image, **given: bool) -> SimpleNamespace:
        options.append(given)
        return found

    monkeypatch.setattr(axes, "_engine", lambda: engine)
    image = np.full((30, 100, 3), 255, dtype=np.uint8)

    words = _read_words(image)
    _read_text(image)

    assert words == [
        _Text("1.25", 10, 10, 40, 25, 0.9),
        _Text("1.50", 50, 10, 80, 25, 0.9),
    ]
    assert options == [
        {"use_cls": False, "return_word_box": True},
        {"use_cls": True, "return_word_box": False},
    ]


def test_reads_no_words_where_ocr_finds_none(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr(monkeypatch, [])
    monkeypatch.setattr(
        axes, "_engine", lambda: lambda _image, **_options: SimpleNamespace(boxes=None)
    )

    assert _read_words(plot()) == []


def right_labels(values: Sequence[str]) -> list[_Text]:
    """Return labels of a second y-axis, beyond the x-axis line, aligned left."""
    return [
        _Text(value, 360, 73 + 60 * i, 380, 87 + 60 * i)
        for i, value in enumerate(values)
    ]


def short_plot() -> Image:
    """Draw a blank plot whose x-axis line ends at x = 340."""
    image = plot()
    image[HEIGHT - AXIS_LINES[1], 341:] = 255
    return image


def test_refuses_a_second_y_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [
        *x_labels(["0", "1", "2"]),
        *y_labels(["0", "1", "2"]),
        *right_labels(["0", "50", "100"]),
    ]
    monkeypatch.setattr(axes, "_read_text", lambda _image: labels)

    with pytest.raises(ValueError, match="second y-axis, on the right"):
        read_axes(short_plot())


@pytest.mark.parametrize(
    "right",
    [
        ["0", "1", "2"],  # the left axis's values again
        ["0", "5", "1"],  # not in line, so not an axis
        ["0", "1"],  # too few labels
    ],
)
def test_accepts_other_labels_on_the_right(
    monkeypatch: pytest.MonkeyPatch, right: list[str]
) -> None:
    labels = [
        *x_labels(["0", "1", "2"]),
        *y_labels(["0", "1", "2"]),
        *right_labels(right),
    ]
    monkeypatch.setattr(axes, "_read_text", lambda _image: labels)

    read_axes(short_plot())


def test_finds_a_second_y_axis_without_axis_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [
        *x_labels(["0", "1", "2"]),
        *y_labels(["0", "1", "2"]),
        *right_labels(["5", "6", "7"]),
    ]

    with pytest.raises(ValueError, match="second y-axis"):
        read(monkeypatch, labels, lines=False)


def test_needs_y_axis_tick_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="no column of 3 or more numbers"):
        read(monkeypatch, [*x_labels(["0", "1", "2"]), *y_labels(["0", "1"])])


def test_starts_beyond_the_labels_without_axis_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]

    found = read(monkeypatch, labels, lines=False)

    # Beyond the right of the y labels and the top of the x labels by 0.6 of
    # a label's height, for their tick marks.
    assert found.calibration.origin == pytest.approx((50 + 8.4, 52 + 8.4))


def test_keeps_the_first_tick_in_without_axis_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The first y tick lies 14 pixels above the top of the x labels.
    labels = [
        *x_labels(["0", "1", "2"]),
        *(text(v, 40, 66 + 60 * i) for i, v in enumerate("012")),
    ]

    found = read(monkeypatch, labels, lines=False)

    assert found.calibration.origin[1] == pytest.approx((52 + 66) / 2)


def marked_plot(x_marks: Sequence[int], y_marks: Sequence[int], line: int = 1) -> Image:
    """Draw a blank plot with axis lines and tick marks outside them."""
    image = plot()
    corner = (AXIS_LINES[0], HEIGHT - AXIS_LINES[1])
    cv2.line(image, (corner[0], 0), corner, (0, 0, 0), line)
    cv2.line(image, corner, (WIDTH - 1, corner[1]), (0, 0, 0), line)
    reach = line // 2 + 5
    for x in x_marks:
        cv2.line(image, (x, corner[1]), (x, corner[1] + reach), (0, 0, 0), 1)
    for y in y_marks:
        cv2.line(
            image,
            (corner[0] - reach, HEIGHT - y),
            (corner[0], HEIGHT - y),
            (0, 0, 0),
            1,
        )
    return image


@pytest.mark.parametrize("line", [1, 3])
def test_moves_ticks_to_the_marks_drawn(
    monkeypatch: pytest.MonkeyPatch, line: int
) -> None:
    # Rotated x labels end at their ticks, rather than centring on them, and
    # are as long as their texts, so their middles lie unlike beside them.
    rotated = [
        text(value, 80 + 80 * i - width / 2, 45, width=width)
        for i, (value, width) in enumerate(zip("012", (20, 30, 40), strict=True))
    ]
    labels = [*rotated, *y_labels(["0", "1", "2"])]
    monkeypatch.setattr(axes, "_read_text", lambda _image: labels)

    found = read_axes(marked_plot([80, 160, 240], [81, 141, 201], line))

    # The centres of the marks' pixels; the y labels are centred on theirs.
    assert [pixel for pixel, _ in found.x_ticks] == [80.5, 160.5, 240.5]
    assert [pixel for pixel, _ in found.y_ticks] == [80, 140, 200]


def test_moves_ticks_of_rotated_labels_centred_on_the_marks_to_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The x labels, written at 45 degrees, lie centred on their marks, but
    # their middles 1.5 pixels off; those of the y labels, written across,
    # half a pixel off, and they stay.
    image = marked_plot([80, 160, 240], [81, 141, 201])
    slanted = [write_slanted(image, 79 + 80 * i, 40, v) for i, v in enumerate("012")]
    labels = [*slanted, *y_labels(["0", "1", "2"])]
    monkeypatch.setattr(axes, "_read_text", lambda _image: labels)

    found = read_axes(image)

    assert [label.x for label in slanted] == [79, 159, 239]
    assert [pixel for pixel, _ in found.x_ticks] == [80.5, 160.5, 240.5]
    assert [pixel for pixel, _ in found.y_ticks] == [80, 140, 200]


def test_moves_ticks_to_marks_beside_the_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rotated y labels can lie alike to one side of their ticks, here 9.5
    # pixels below them, and as tall as wide, so their ends lie further off.
    rotated = [
        _Text(value, 10, 60 + 60 * i, 50, 100 + 60 * i) for i, value in enumerate("012")
    ]
    labels = [*x_labels(["0", "1", "2"]), *rotated]
    monkeypatch.setattr(axes, "_read_text", lambda _image: labels)

    found = read_axes(marked_plot([80, 160, 240], [90, 150, 210]))

    assert [pixel for pixel, _ in found.y_ticks] == [89.5, 149.5, 209.5]


def test_keeps_ticks_at_labels_beside_minor_marks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Minor tick marks lie alike beside the labels too, nearer than the
    # major ones, so the labels stay rather than take the minor marks.
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    monkeypatch.setattr(axes, "_read_text", lambda _image: labels)

    found = read_axes(marked_plot([80, 160, 240], [84, 90, 144, 150, 204, 210]))

    assert [pixel for pixel, _ in found.y_ticks] == [80, 140, 200]


def test_keeps_ticks_at_labels_far_from_the_marks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    monkeypatch.setattr(axes, "_read_text", lambda _image: labels)

    found = read_axes(marked_plot([120, 200, 280], [110, 170, 230]))  # between

    assert [pixel for pixel, _ in found.x_ticks] == [80, 160, 240]
    assert [pixel for pixel, _ in found.y_ticks] == [80, 140, 200]


def test_moves_ticks_to_marks_beside_the_labels_past_ink_at_the_corner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The labels of each axis lie alike beside their marks, 9.5 pixels below
    # or right of them. Beyond each axis line, the ink of the other axis's
    # first label, as of a 0, runs over the columns or rows of the marks.
    labels = [
        *(text(value, 90 + 80 * i, 45, width=40) for i, value in enumerate("012")),
        *(
            _Text(value, 10, 60 + 60 * i, 50, 100 + 60 * i)
            for i, value in enumerate("012")
        ),
    ]
    monkeypatch.setattr(axes, "_read_text", lambda _image: labels)
    image = marked_plot([80, 160, 240], [90, 150, 210])
    for y in (35, 45):  # rows of a 0 below the x-axis line
        image[HEIGHT - y, 54:60] = 0
    for x in (40, 48):  # columns of a 0 left of the y-axis line
        image[HEIGHT - 59 : HEIGHT - 50, x] = 0

    found = read_axes(image)

    assert [pixel for pixel, _ in found.x_ticks] == [80.5, 160.5, 240.5]
    assert [pixel for pixel, _ in found.y_ticks] == [89.5, 149.5, 209.5]


def test_finds_the_x_axis_line_at_a_tick_beside_its_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The y labels lie 9.5 pixels below their ticks, the first of which is
    # at the corner, as rotated labels do; so the x-axis line lies further
    # than usual past the first label.
    rotated = [
        _Text(value, 10, 30 + 60 * i, 50, 70 + 60 * i) for i, value in enumerate("012")
    ]
    labels = [*x_labels(["0", "1", "2"]), *rotated]
    monkeypatch.setattr(axes, "_read_text", lambda _image: labels)

    found = read_axes(marked_plot([80, 160, 240], [60, 120, 180]))

    assert found.calibration.origin == (60.5, 59.5)
    assert [pixel for pixel, _ in found.y_ticks] == [59.5, 119.5, 179.5]


def test_applies_scale_factors(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    above_y_axis = text("1e-5", 60, 290)
    beside_x_axis = text("\N{MULTIPLICATION SIGN}10\N{SUPERSCRIPT THREE}", 360, 30)

    found = read(monkeypatch, [*labels, above_y_axis, beside_x_axis])

    assert [value for _, value in found.x_ticks] == [0, 1000, 2000]
    assert [value for _, value in found.y_ticks] == pytest.approx([0, 1e-5, 2e-5])
    assert found.x_title is None


def test_applies_scale_factors_beside_the_top_tick_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The topmost tick is at the top of the axis, and its label reaches
    # above it, beside the scale factor.
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2", "3"])]
    above_y_axis = text("1e6", 75, 270)

    found = read(monkeypatch, [*labels, above_y_axis])

    assert [value for _, value in found.y_ticks] == [0, 1e6, 2e6, 3e6]


def test_ignores_scale_factors_below_the_start_of_the_x_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # As the y-axis's scale factor of a plot below would lie.
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    below_x_axis = text("1e5", 75, 10)

    found = read(monkeypatch, [*labels, below_x_axis])

    assert [value for _, value in found.x_ticks] == [0, 1, 2]


def test_ignores_text_like_scale_factors_in_the_plot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    legend = [text("1e-3", 300, 250), text("1e-2", 300, 230)]
    annotation = text("\N{MINUS SIGN}2 eV", 200, 150, width=40)
    above_the_plot = text("1e-5", 300, 290)  # above the y labels, but not the axis

    found = read(monkeypatch, [*labels, *legend, annotation, above_the_plot])

    assert [value for _, value in found.y_ticks] == [0, 1, 2]


def test_refuses_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]

    with pytest.raises(ValueError, match=r"ticks are offset by \+1e3"):
        read(monkeypatch, [*labels, text("+1e3", 60, 290)])


def test_reads_the_nearest_titles(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    x_title = [text("(s)", 175, 25), text("Time", 140, 25, width=40)]
    y_title = [tall("(mV)", 10, 175, 30), tall("Voltage", 10, 130, 50)]
    others = [
        text("Figure 1", 200, 8, width=60),  # a caption below the x title
        text("(a)", 10, 290),  # a panel label above the y-axis
        text("Legend", 300, 250),  # in the plot
    ]
    monkeypatch.setattr(
        axes, "_read_text", lambda _image: [*labels, *x_title, *y_title, *others]
    )
    image = plot()
    image[:30, AXIS_LINES[0]] = 255  # the y-axis line ends at y = 270

    found = read_axes(image)

    assert found.x_title == "Time (s)"
    assert found.y_title == "Voltage (mV)"


def upright(content: str, x: float, y: float, height: float, score: float = 1) -> _Text:
    """Return a line of text written upwards as OCR reads it turned upright.

    The text is 14 pixels wide and centred on a pixel position of the plot.
    Turned a quarter turn clockwise, the plot's left side runs along the top
    of a square as tall as the plot, which OCR reads.
    """
    return _Text(
        content, y - height / 2, HEIGHT - x - 7, y + height / 2, HEIGHT - x + 7, score
    )


def read_turned(
    monkeypatch: pytest.MonkeyPatch, texts: Sequence[_Text], turned: Sequence[_Text]
) -> Axes:
    """Read the axes of a plot with ink left of the y labels, as OCR read the texts.

    Turned upright, OCR reads the turned texts left of the y-axis tick labels.
    """
    image = plot()
    image[100:200, :20] = 0

    def read_text(image: Image) -> list[_Text]:
        return list(turned if image.shape[0] == image.shape[1] else texts)

    monkeypatch.setattr(axes, "_read_text", read_text)
    return read_axes(image)


def test_reads_titles_over_several_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    # Each line lies within a line's height of the one before, aligned with
    # it; the first line of a title written upwards is the leftmost.
    x_title = [text("Temperature", 160, 28, width=80), text("(K)", 160, 15)]
    y_title = [tall("Resistance", 8, 140, 80), tall("(Ohm)", 22, 140, 40)]

    found = read(monkeypatch, [*labels, *x_title, *y_title])

    assert found.x_title == "Temperature (K)"
    assert found.y_title == "Resistance (Ohm)"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # A short first line, of letters without descenders.
        (_Text("T", 155, 26, 165, 36), text("(degrees Celsius)", 160, 18, width=110)),
        # Lines spaced widely, at 1.8 times the font size.
        (text("Photon energy", 160, 30, width=90), text("(eV)", 160, 13, width=30)),
    ],
)
def test_reads_titles_over_lines_apart_by_more_than_a_short_line(
    monkeypatch: pytest.MonkeyPatch, first: _Text, second: _Text
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]

    found = read(monkeypatch, [*labels, first, second])

    assert found.x_title == f"{first.text} {second.text}"


def test_leaves_out_second_lines_of_tick_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each label has its unit below it, and the scale factor lies beyond the
    # last label. The title lies under the middle label, but alone.
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    units = [text("s", 80 + 80 * i, 28) for i in range(3)]
    scale = text("\N{MULTIPLICATION SIGN}10\N{SUPERSCRIPT THREE}", 360, 28)
    title = text("Time", 160, 10, width=40)

    found = read(monkeypatch, [*labels, *units, scale, title])

    assert found.x_title == "Time"
    assert [value for _, value in found.x_ticks] == [0, 1000, 2000]


def test_leaves_out_lines_not_aligned_with_a_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    x_title = text("Time (s)", 140, 28, width=60)
    note = text("Source: lab", 320, 15, width=60)  # under the plot's right end

    found = read(monkeypatch, [*labels, x_title, note])

    assert found.x_title == "Time (s)"


def test_leaves_out_text_level_with_a_line_but_far_along(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The labels of two plots below a wide one, level with each other, would
    # make a line across the title's middle.
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    x_title = text("Time (s)", 200, 26, width=60)
    others = [text("(b)", 70, 10), text("(c)", 330, 10)]

    found = read(monkeypatch, [*labels, x_title, *others])

    assert found.x_title == "Time (s)"


def test_reads_titles_aligned_with_an_end_of_their_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    # Beyond the last tick labels, but alongside the axis lines.
    x_title = text("Distance (km)", 340, 25, width=100)
    y_title = tall("Height (m)", 10, 250, 70)

    found = read(monkeypatch, [*labels, x_title, y_title])

    assert found.x_title == "Distance (km)"
    assert found.y_title == "Height (m)"


def test_reads_titles_written_across(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    y_title = [text("Voltage", 14, 150, width=24), text("(mV)", 14, 137, width=24)]

    found = read(monkeypatch, [*labels, *y_title])

    assert found.y_title == "Voltage (mV)"


def test_applies_scale_factors_in_titles(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    times, minus = "\N{MULTIPLICATION SIGN}", "\N{SUPERSCRIPT MINUS}"
    x_title = text(f"Current ({times}10{minus}\N{SUPERSCRIPT THREE} A)", 160, 25, 120)
    y_title = text("Energy / 10\N{SUPERSCRIPT THREE} J", 14, 140, width=28)

    found = read(monkeypatch, [*labels, x_title, y_title])

    assert found.x_title == "Current (A)"
    assert [value for _, value in found.x_ticks] == [0, 0.001, 0.002]
    assert found.y_title == "Energy / J"
    assert [value for _, value in found.y_ticks] == [0, 1000, 2000]
    np.testing.assert_allclose(
        found.calibration.to_data([(80, 80), (240, 200)]),
        [(0, 0), (0.002, 2000)],
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("title", "rest", "factor"),
    [
        ("Count (\N{MULTIPLICATION SIGN}10\N{SUPERSCRIPT THREE})", "Count", 1e3),
        ("Pressure (10^3 Pa)", "Pressure (Pa)", 1e3),
        ("Signal [x10^{-2}]", "Signal", 1e-2),
        ("Rate (10\N{SUPERSCRIPT ONE}\N{SUPERSCRIPT ZERO} Hz)", "Rate (Hz)", 1e10),
        ("(\N{MULTIPLICATION SIGN}10\N{SUPERSCRIPT THREE})", None, 1e3),
        # Prefixes of units, and tens that are not powers, stay.
        ("Voltage (mV)", "Voltage (mV)", 1),
        ("Year (10 BC)", "Year (10 BC)", 1),
        (None, None, 1),
    ],
)
def test_finds_scale_factors_in_titles(
    title: str | None, rest: str | None, factor: float
) -> None:
    assert _factor(title) == (rest, factor)


def test_reads_titles_written_upwards_turned_upright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    # OCR split the title into pieces, and missed its first line, until the
    # text was turned upright.
    pieces = [
        tall("Volt", 22, 115, 30),
        tall("e", 22, 137, 10),
        tall("age (mV)", 22, 170, 50),
    ]
    turned = [upright("Measured", 8, 150, 60), upright("Voltage (mV)", 22, 150, 100)]

    found = read_turned(monkeypatch, [*labels, *pieces], turned)

    assert found.y_title == "Measured Voltage (mV)"


@pytest.mark.parametrize(
    ("first", "turned", "title"),
    [
        # Turned, OCR read the delta twice, less surely than as first read.
        (
            "\N{GREEK CAPITAL LETTER DELTA}E (%)",
            0.85,
            "\N{GREEK CAPITAL LETTER DELTA}E (%)",
        ),
        # Turned, OCR read surely what is written twice, as in delta delta G.
        (
            "\N{GREEK CAPITAL LETTER DELTA}G",
            0.95,
            "\N{GREEK CAPITAL LETTER DELTA}" * 2 + "G",
        ),
    ],
)
def test_reads_titles_with_a_character_doubled_turned_the_surer_way(
    monkeypatch: pytest.MonkeyPatch, first: str, turned: float, title: str
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    upwards = dataclasses.replace(tall(first, 8, 150, 60), score=0.9)
    doubled = first[0] + first

    found = read_turned(
        monkeypatch, [*labels, upwards], [upright(doubled, 8, 150, 70, turned)]
    )

    assert found.y_title == title


@pytest.mark.parametrize(
    ("across", "turned", "title"),
    [
        # A V written upwards, read as > until turned upright.
        (_Text(">", 1, 133, 15, 147, 0.9), upright("V", 8, 140, 14, 0.98), "V"),
        # A y written across, misread when turned.
        (_Text("y", 1, 133, 15, 147), upright("A", 8, 140, 14, 0.6), "y"),
    ],
)
def test_reads_single_letters_whichever_way_ocr_is_surer_of(
    monkeypatch: pytest.MonkeyPatch, across: _Text, turned: _Text, title: str
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]

    found = read_turned(monkeypatch, [*labels, across], [turned])

    assert found.y_title == title
