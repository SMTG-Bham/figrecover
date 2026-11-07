"""Tests for finding the plots in an image of several, and reading each."""

import dataclasses
import string
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from figrecover import Image, Panel, load_image, panels, read_panels
from figrecover.axes import _Text
from figrecover.digitize import _Box
from figrecover.panels import (
    _beside,
    _boxes,
    _in_order,
    _in_sequence,
    _written,
)

EXAMPLES = Path(__file__).parents[1] / "examples"
HEIGHT, WIDTH = 400, 600
# The frames of the axes of four plots, two above two.
GRID = [
    _Box(20, 170, 60, 280),
    _Box(20, 170, 360, 580),
    _Box(220, 370, 60, 280),
    _Box(220, 370, 360, 580),
]


def text(content: str, x: float, y: float, width: float = 20) -> _Text:
    """Return a line of text 14 pixels high, centred on a pixel position."""
    return _Text(content, x - width / 2, y - 7, x + width / 2, y + 7)


def ticks(box: _Box, x_values: Sequence[str], y_values: Sequence[str]) -> list[_Text]:
    """Return the tick labels of a plot whose axes are framed by a box.

    The x-axis labels lie below the frame, 60 pixels apart from 20 pixels
    right of its corner, and the y-axis labels left of it, 40 pixels apart
    from 20 pixels above its corner.
    """
    left, bottom = box.left, HEIGHT - box.bottom
    return [
        *(
            text(value, left + 20 + 60 * i, bottom - 15)
            for i, value in enumerate(x_values)
        ),
        *(
            text(value, left - 20, bottom + 20 + 40 * i)
            for i, value in enumerate(y_values)
        ),
    ]


def label(content: str, box: _Box) -> _Text:
    """Return a plot's label, inside the top-left corner of its frame."""
    return text(content, box.left + 20, HEIGHT - box.top - 12, width=24)


def figure(texts: Sequence[_Text], boxes: Sequence[_Box] = GRID) -> Image:
    """Draw the frames of plots, and a block of ink in the middle of each text."""
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    for box in boxes:
        corners = (box.left, box.top), (box.right - 1, box.bottom - 1)
        cv2.rectangle(image, *corners, (0, 0, 0), 1)
    for line in texts:
        rows = slice(HEIGHT - round(line.top) + 4, HEIGHT - round(line.bottom) - 4)
        image[rows, round(line.left) + 4 : round(line.right) - 4] = 0
    return image


def read_parts(
    monkeypatch: pytest.MonkeyPatch, image: Image, texts: Sequence[_Text]
) -> None:
    """Make OCR find the texts, placed in the whole image, in each part read.

    Each part read is a view of the image, so its place in the image's memory
    gives its place in the image. A text is in a part if its middle is.
    """

    def read_text(part: Image) -> list[_Text]:
        start = part.__array_interface__["data"][0] - image.ctypes.data
        top, left = divmod(start // image.strides[1], WIDTH)
        bottom = HEIGHT - top - part.shape[0]
        return [
            dataclasses.replace(
                line,
                left=line.left - left,
                bottom=line.bottom - bottom,
                right=line.right - left,
                top=line.top - bottom,
            )
            for line in texts
            if left <= line.x < left + part.shape[1]
            and bottom <= line.y < bottom + part.shape[0]
        ]

    monkeypatch.setattr(panels, "_read_text", read_text)


def four_plots() -> list[_Text]:
    """Return the tick labels of the four plots of GRID, each with its own."""
    return [
        *ticks(GRID[0], ["0", "1", "2"], ["0", "1", "2"]),
        *ticks(GRID[1], ["0", "10", "20"], ["5", "6", "7"]),
        *ticks(GRID[2], ["0", "0.5", "1"], ["0", "2", "4"]),
        *ticks(GRID[3], ["5", "6", "7"], ["-1", "0", "1"]),
    ]


def test_reads_each_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [
        label(f"({letter})", box) for letter, box in zip("abcd", GRID, strict=True)
    ]
    texts = [*four_plots(), *labels]
    image = figure(texts)
    read_parts(monkeypatch, image, texts)

    found = read_panels(image)

    assert [panel.label for panel in found] == ["a", "b", "c", "d"]
    assert [[value for _, value in panel.axes.x_ticks] for panel in found] == [
        [0, 1, 2],
        [0, 10, 20],
        [0, 0.5, 1],
        [5, 6, 7],
    ]
    assert [[value for _, value in panel.axes.y_ticks] for panel in found] == [
        [0, 1, 2],
        [5, 6, 7],
        [0, 2, 4],
        [-1, 0, 1],
    ]
    # Each panel's part runs down to just above the axes below it, and right
    # to the tick labels of those beside it, but its image only up to the top
    # and right of its own axes.
    assert [panel.offset for panel in found] == [(0, 183), (283, 183), (0, 0), (283, 0)]
    assert [panel.image.shape[:2] for panel in found] == [
        (197, 280),
        (197, 297),
        (180, 280),
        (180, 297),
    ]
    # The pixel centres of the frames' bottom-left corners, in the image.
    corners = [
        tuple(np.add(panel.axes.calibration.origin, panel.offset)) for panel in found
    ]
    assert corners == [(60.5, 230.5), (360.5, 230.5), (60.5, 30.5), (360.5, 30.5)]


def test_paints_over_the_labels_inside_the_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OCR stretches the box of (a) over the top of the frame; (c) lies
    # outside its axes, above and left of them.
    labels = [
        label("(a)", GRID[0]),
        label("(b)", GRID[1]),
        text("(c)", GRID[2].left - 25, HEIGHT - GRID[2].top + 10),
        label("(d)", GRID[3]),
    ]
    texts = [*four_plots(), *labels]
    image = figure(texts)
    stretched = dataclasses.replace(labels[0], top=HEIGHT - GRID[0].top + 3)
    read_parts(monkeypatch, image, [*four_plots(), stretched, *labels[1:]])

    found = read_panels(image)

    # The top row of (a)'s image is the top of its frame, which is kept, but
    # the ink of its label, 9 to 15 pixels below, is painted white.
    assert (found[0].image[0, 60:280] == 0).all()
    assert found[0].image[3:20, 63:277].min() == 255
    assert found[2].label == "c"


def test_reads_misread_labels_in_reading_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OCR reads (b) as (p), (c) as (), and misses (d).
    labels = [
        label(content, box)
        for content, box in zip(["(a)", "(p)", "()"], GRID, strict=False)
    ]
    texts = [*four_plots(), *labels]
    image = figure(texts)
    read_parts(monkeypatch, image, texts)

    found = read_panels(image)

    assert [panel.label for panel in found] == ["a", "b", "c", "d"]
    # The label read as () is painted over too: its ink lies 9 to 15 pixels
    # below the top of (c)'s frame.
    assert found[2].image[9:16, 72:88].min() == 255


def test_leaves_out_panels_whose_axes_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [
        label(f"({letter})", box) for letter, box in zip("abcd", GRID, strict=True)
    ]
    plots = four_plots()
    misread = ticks(GRID[1], ["0", "10", "20"], ["5", "8", "6"])  # out of line
    texts = [*plots[:6], *misread, *plots[12:], *labels]
    image = figure(texts)
    read_parts(monkeypatch, image, texts)

    with pytest.warns(
        UserWarning,
        match=r"left out the panel \(b\) at 360,230, as its axes cannot be read: "
        "only 2 of the 3 y-axis tick labels are in line",
    ):
        found = read_panels(image)

    # Most of the labels read are no longer in order, so are kept as read.
    assert [panel.label for panel in found] == ["a", "c", "d"]


def test_reads_axes_shared_with_panels_beside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The plots share both axes, labelled only on the outer panels: (a) has
    # y-axis labels, (d) x-axis labels, (c) both and (b) none. (b) shares its
    # x-axis with (d), and its y-axis with (a), each read only after it.
    labels = [
        label(f"({letter})", box) for letter, box in zip("abcd", GRID, strict=True)
    ]
    plots = ticks(GRID[2], ["0", "0.5", "1"], ["0", "2", "4"])
    shared = ticks(GRID[3], ["0", "0.5", "1"], [])
    left = ticks(GRID[0], [], ["0", "2", "4"])
    titles = [
        *(text("Time (s)", box.left + 110, 3, width=60) for box in GRID[2:]),
        *(text("Voltage", 15, HEIGHT - box.bottom + 75, width=30) for box in GRID[::2]),
    ]
    texts = [*left, *plots, *shared, *titles, *labels]
    image = figure(texts)
    read_parts(monkeypatch, image, texts)

    found = read_panels(image)

    assert [panel.label for panel in found] == ["a", "b", "c", "d"]
    for panel in found:
        assert [value for _, value in panel.axes.x_ticks] == [0, 0.5, 1]
        assert [value for _, value in panel.axes.y_ticks] == [0, 2, 4]
        assert panel.axes.x_title == "Time (s)"
        assert panel.axes.y_title == "Voltage"
    # Each panel's ticks lie where those of the panel it shares them with
    # do, in the whole image.
    x_ticks, y_ticks = [], []
    for panel in found:
        x, y = panel.offset
        x_ticks.append([pixel + x for pixel, _ in panel.axes.x_ticks])
        y_ticks.append([pixel + y for pixel, _ in panel.axes.y_ticks])
    assert x_ticks[0] == x_ticks[2] == [80, 140, 200]
    assert x_ticks[1] == x_ticks[3] == [380, 440, 500]
    assert y_ticks[0] == y_ticks[1] == [250, 290, 330]
    assert y_ticks[2] == y_ticks[3] == [50, 90, 130]


def test_leaves_out_panels_sharing_an_axis_with_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # (b) has no x-axis labels, and no panel below it to share them.
    texts = [
        *ticks(GRID[0], ["0", "1", "2"], ["0", "1", "2"]),
        *ticks(GRID[1], [], ["5", "6", "7"]),
        *ticks(GRID[2], ["0", "1", "2"], ["0", "1", "2"]),
    ]
    image = figure(texts, boxes=GRID[:3])
    read_parts(monkeypatch, image, texts)

    with pytest.warns(
        UserWarning,
        match=r"left out the panel at 360,230, as its axes cannot be read: "
        "found no row of 3 or more numbers to read as x-axis ticks",
    ):
        found = read_panels(image)

    assert len(found) == 2


def test_reads_no_label_of_another_panel_as_a_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # (c)'s label lies above its axes, as matplotlib draws a title, below the
    # x-axis tick labels of (a), which has no x-axis title of its own.
    labels = [
        label("(a)", GRID[0]),
        label("(b)", GRID[1]),
        text("(c)", GRID[2].left + 15, HEIGHT - GRID[2].top + 10),
        label("(d)", GRID[3]),
    ]
    texts = [*four_plots(), *labels]
    image = figure(texts)
    read_parts(monkeypatch, image, texts)

    found = read_panels(image)

    assert found[0].axes.x_title is None
    assert found[2].label == "c"


def test_needs_a_panel_whose_axes_can_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = figure([])
    read_parts(monkeypatch, image, [])

    with (
        pytest.warns(UserWarning, match=r"left out the panel at \d+,\d+, as its axes"),
        pytest.raises(ValueError, match="cannot read the axes of any of the 4 panels"),
    ):
        read_panels(image)


def test_reads_a_plot_without_axis_lines_from_the_whole_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = [*ticks(GRID[2], ["0", "1", "2"], ["0", "1", "2"]), label("(a)", GRID[2])]
    image = figure(texts, boxes=[])
    read_parts(monkeypatch, image, texts)

    (found,) = read_panels(image)

    assert found.label is None
    assert found.image is image
    assert found.offset == (0.0, 0.0)
    assert [value for _, value in found.axes.x_ticks] == [0, 1, 2]


@pytest.mark.parametrize(("content", "letter"), [("A", "A"), ("Figure", None)])
def test_reads_a_plot_from_the_whole_image(
    monkeypatch: pytest.MonkeyPatch, content: str, letter: str | None
) -> None:
    # An inset, inside the axes, is not a plot of its own.
    box, inset = GRID[2], _Box(260, 330, 180, 270)
    texts = [*ticks(box, ["0", "1", "2"], ["0", "1", "2"]), label(content, box)]
    image = figure(texts, boxes=[box, inset])
    read_parts(monkeypatch, image, texts)

    (found,) = read_panels(image)

    assert found.label == letter
    assert found.offset == (0.0, 0.0)
    assert found.image.shape == image.shape
    # The label is painted over only if it is one.
    assert (found.image[HEIGHT - 168, 80] == 255).all() == (letter is not None)
    assert found.axes.calibration.origin == (60.5, 30.5)


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (0.25, "25e-2"),
        (-0.75, "-75e-2"),
        (0.0, "0e-1"),
        (1e-5, "1e-5"),
        (1e3, "10000e-1"),
    ],
)
def test_writes_shared_ticks_without_points_or_commas(
    value: float, written: str
) -> None:
    assert _written(value) == written
    assert float(written) == value


def test_finds_plots_by_axis_lines_that_meet() -> None:
    # A line across the image, as a flat stretch of a curve may be, frames
    # nothing.
    image = figure([], boxes=GRID[:2])
    image[300, 50:550] = 0

    assert _boxes(255 - image[..., 0]) == GRID[:2]


def test_orders_panels_in_rows() -> None:
    # A tall plot beside two stacked ones comes first.
    tall, top, bottom = _Box(20, 370, 60, 280), GRID[1], GRID[3]

    assert _in_order([bottom, top, tall]) == [tall, top, bottom]
    assert _in_order(GRID[::-1]) == GRID


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ("##..###...", 104),  # tick marks of the plot on the left, then text
        ("..........", 110),  # no text: up to the plot on the right
        ("##########", 110),
        ("", 100),
    ],
)
def test_finds_the_text_of_the_plot_on_the_right(columns: str, expected: int) -> None:
    ink = np.zeros((50, 120), dtype=np.bool_)
    ink[20:30, 100 : 100 + len(columns)] = [char == "#" for char in columns]
    box, other = _Box(10, 40, 50, 100), _Box(5, 45, 100 + len(columns), 120)

    assert _beside(ink, box, other) == expected


@pytest.mark.parametrize(
    ("letters", "expected"),
    [
        (["a", "p", "c", None], ["a", "b", "c", "d"]),
        (["A", "B", None], ["A", "B", "C"]),
        (["a", "B", "c", "p"], ["a", "b", "c", "d"]),
        # Labelled down the columns: half the letters are in reading order.
        (["a", "c", "b", "d"], ["a", "c", "b", "d"]),
        (["x", "x", "y"], [None, None, "y"]),
        ([None, None], [None, None]),
        ([None] * 26 + ["a"], [None] * 26 + ["a"]),
    ],
)
def test_takes_letters_in_reading_order(
    letters: list[str | None], expected: list[str | None]
) -> None:
    assert _in_sequence(letters) == expected


def test_labels_more_panels_than_letters() -> None:
    letters = list(string.ascii_lowercase)

    assert _in_sequence(letters) == letters
    assert _in_sequence([*letters, "a"]) == [None, *letters[1:], None]


@pytest.fixture(scope="module")
def example() -> list[Panel]:
    """Read the panels of the example figure, once."""
    return read_panels(load_image(EXAMPLES / "panels.png"))


def test_reads_the_panels_of_a_figure(example: list[Panel]) -> None:
    assert [panel.label for panel in example] == ["a", "b", "c", "d"]
    assert [(panel.axes.x_title, panel.axes.y_title) for panel in example] == [
        ("Energy (eV)", "Intensity"),
        ("Time (s)", "Voltage (mV)"),
        ("Depth (m)", "Temperature (K)"),
        ("Frequency (Hz)", "Power (mW)"),
    ]
    assert [value for _, value in example[1].axes.y_ticks] == [
        -1 + 0.25 * step for step in range(9)
    ]
    assert [value for _, value in example[3].axes.x_ticks] == [1, 10, 100, 1e3, 1e4]
    assert example[3].axes.calibration.x_log
    assert example[3].axes.calibration.y_log


@pytest.mark.parametrize(
    ("index", "pixels", "points"),
    [
        # The pixel positions of the points that examples/make_examples.py
        # prints for each panel.
        (0, [(56, 349), (376, 349), (56, 424)], [(0, 0), (10, 0), (0, 1)]),
        (1, [(466, 462), (787, 462), (466, 574)], [(0, 0), (10, 0), (0, 1)]),
        (2, [(56, 50), (376, 50), (56, 125)], [(0, 0), (10, 0), (0, 1)]),
        (3, [(481, 60), (772, 60), (481, 265)], [(1, 0.1), (1e4, 0.1), (1, 1e3)]),
    ],
)
def test_calibrates_each_panel_of_a_figure(
    example: list[Panel],
    index: int,
    pixels: list[tuple[int, int]],
    points: list[tuple[float, float]],
) -> None:
    panel = example[index]

    data = panel.axes.calibration.to_data(np.subtract(pixels, panel.offset))

    if index == 3:  # within 0.02 decades, or a pixel and a half
        np.testing.assert_allclose(np.log10(data), np.log10(points), atol=0.02)
    else:
        np.testing.assert_allclose(data, points, atol=0.03)


@pytest.fixture(scope="module")
def shared() -> list[Panel]:
    """Read the panels of the example figure whose plots share their axes, once."""
    return read_panels(load_image(EXAMPLES / "shared.png"))


def test_reads_the_panels_of_a_figure_sharing_axes(shared: list[Panel]) -> None:
    assert [panel.label for panel in shared] == ["a", "b", "c", "d"]
    for panel in shared:
        assert [value for _, value in panel.axes.x_ticks] == [0, 2, 4, 6, 8, 10]
        assert [value for _, value in panel.axes.y_ticks] == [
            0.5 * step for step in range(7)
        ]
        assert panel.axes.x_title == "Time (s)"
        assert panel.axes.y_title == "Voltage (mV)"


@pytest.mark.parametrize(
    ("index", "pixels"),
    [
        # The pixel positions of (0, 0), (10, 0) and (0, 1) that
        # examples/make_examples.py prints for each panel.
        (0, [(56, 330), (410, 330), (56, 412)]),
        (1, [(432, 330), (787, 330), (432, 412)]),
        (2, [(56, 48), (410, 48), (56, 129)]),
        (3, [(432, 48), (787, 48), (432, 129)]),
    ],
)
def test_calibrates_each_panel_of_a_figure_sharing_axes(
    shared: list[Panel], index: int, pixels: list[tuple[int, int]]
) -> None:
    panel = shared[index]

    data = panel.axes.calibration.to_data(np.subtract(pixels, panel.offset))

    np.testing.assert_allclose(data, [(0, 0), (10, 0), (0, 1)], atol=0.03)
