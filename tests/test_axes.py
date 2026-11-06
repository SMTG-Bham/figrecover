"""Tests for reading the axes of a plot from the text in its image."""

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from figrecover import Axes, Image, axes, load_image, read_axes
from figrecover.axes import _PLAIN, _read_text, _Text

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


def test_reads_minus_signs(monkeypatch: pytest.MonkeyPatch) -> None:
    minus = "\N{MINUS SIGN}"
    labels = [*x_labels(["-1", "0", "1"]), *y_labels([f"{minus}1", "0", "1"])]

    found = read(monkeypatch, labels)

    assert [value for _, value in found.x_ticks] == [-1, 0, 1]
    assert [value for _, value in found.y_ticks] == [-1, 0, 1]


def test_drops_misread_tick_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    # The third x label, 2, was misread as 1; the third y label, 1, as 10.
    labels = [
        *x_labels(["0", "1", "1", "3", "4"]),
        *y_labels(["0", "0.5", "10", "1.5"]),
    ]

    found = read(monkeypatch, labels)

    assert list(found.x_ticks) == [(80, 0), (160, 1), (320, 3), (400, 4)]
    assert list(found.y_ticks) == [(80, 0), (140, 0.5), (260, 1.5)]


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
    return _Text(
        content + superscript, x - width / 2 - 2, y - 10, x + width / 2 + 10, y + 10
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


def test_needs_y_axis_tick_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="no column of 3 or more numbers"):
        read(monkeypatch, [*x_labels(["0", "1", "2"]), *y_labels(["0", "1"])])


def test_uses_the_first_ticks_without_axis_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]

    found = read(monkeypatch, labels, lines=False)

    assert found.calibration.origin == (80.0, 80.0)


def test_applies_scale_factors(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    above_y_axis = text("1e-5", 60, 290)
    beside_x_axis = text("\N{MULTIPLICATION SIGN}10\N{SUPERSCRIPT THREE}", 360, 30)

    found = read(monkeypatch, [*labels, above_y_axis, beside_x_axis])

    assert [value for _, value in found.x_ticks] == [0, 1000, 2000]
    assert [value for _, value in found.y_ticks] == pytest.approx([0, 1e-5, 2e-5])
    assert found.x_title is None


def test_refuses_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]

    with pytest.raises(ValueError, match=r"ticks are offset by \+1e3"):
        read(monkeypatch, [*labels, text("+1e3", 60, 290)])


def test_reads_the_nearest_titles(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [*x_labels(["0", "1", "2"]), *y_labels(["0", "1", "2"])]
    x_title = [text("(s)", 175, 25), text("Time", 140, 25, width=40)]
    y_title = [text("(mV)", 10, 170, width=14), text("Voltage", 10, 110, width=14)]
    others = [
        text("Figure 1", 200, 8, width=60),  # a caption below the x title
        text("(a)", 10, 290),  # a panel label above the y-axis
        text("Legend", 300, 250),  # in the plot
    ]

    found = read(monkeypatch, [*labels, *x_title, *y_title, *others])

    assert found.x_title == "Time (s)"
    assert found.y_title == "Voltage (mV)"
