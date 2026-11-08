"""Tests for reading images and tracing curves."""

import contextlib
import itertools
import re
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from figrecover import (
    Calibration,
    Colour,
    Image,
    Point,
    find_colours,
    find_marker_series,
    find_markers,
    load_image,
    remove_grid,
    trace_curve,
    trace_curves,
)

type Function = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

EXAMPLES = Path(__file__).parents[1] / "examples"

HEIGHT, WIDTH = 200, 400
ORIGIN = (40, 30)  # pixel position of the axes, from the bottom-left corner
PIXELS_PER_UNIT = (35, 40)
BLACK, BLUE, ORANGE = (0, 0, 0), (31, 119, 180), (255, 127, 14)
GREEN, LIGHT_BLUE = (44, 160, 44), (143, 187, 217)  # the second half-way to white
PALE_BLUE = (200, 220, 240)  # 55 levels from white
PINK, GREY = (218, 139, 195), (234, 234, 242)  # from seaborn's palette and style


def curve(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the function drawn in the synthetic plot."""
    return 1.5 + np.sin(x)


def rising(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a line that crosses ``curve`` and ``falling``."""
    return 0.5 + 0.2 * x


def falling(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a line that crosses ``curve`` and ``rising``."""
    return 3 - 0.25 * x


def to_row(y_pixel: float) -> int:
    """Convert a height above the bottom edge to an image row."""
    return round(HEIGHT - y_pixel)


def synthetic_plot(
    grid: int | None = None,
    curves: Sequence[tuple[Colour, Function]] = ((BLACK, curve),),
    x_range: tuple[float, float] = (0.5, 10),
) -> tuple[Image, Calibration]:
    """Draw axes and known curves, and return the image and its calibration.

    Args:
        grid: The grey level of grid lines to draw, if any.
        curves: The colour and function of each curve, drawn in order.
        x_range: The data range to draw the curves over.
    """
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    if grid is not None:
        for column in range(60, WIDTH, 50):
            cv2.line(image, (column, 0), (column, HEIGHT), (grid,) * 3, 1)
        for row in range(10, HEIGHT, 50):
            cv2.line(image, (0, row), (WIDTH, row), (grid,) * 3, 1)
    cv2.line(image, (ORIGIN[0], 0), (ORIGIN[0], to_row(ORIGIN[1])), BLACK, 2)
    cv2.line(
        image, (ORIGIN[0], to_row(ORIGIN[1])), (WIDTH, to_row(ORIGIN[1])), BLACK, 2
    )

    x = np.linspace(*x_range, 500)
    columns = ORIGIN[0] + x * PIXELS_PER_UNIT[0]
    for colour, function in curves:
        heights = ORIGIN[1] + function(x) * PIXELS_PER_UNIT[1]
        # OpenCV draws point (c, r) at the pixel centred on (c + 0.5, HEIGHT - r - 0.5),
        # here to a sixteenth of a pixel.
        points = np.column_stack((columns - 0.5, HEIGHT - heights - 0.5))
        cv2.polylines(
            image,
            [(points * 16).round().astype(np.int32)],
            isClosed=False,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_AA,
            shift=4,
        )

    data = [(0.0, 0.0), (10.0, 0.0), (0.0, 1.0)]
    pixels = [
        ORIGIN,
        (ORIGIN[0] + 10 * PIXELS_PER_UNIT[0], ORIGIN[1]),
        (ORIGIN[0], ORIGIN[1] + PIXELS_PER_UNIT[1]),
    ]
    return image, Calibration.from_points(data, pixels)


def assert_follows(
    pixels: npt.NDArray[np.float64],
    calibration: Calibration,
    function: Function,
    atol: float = 0.05,
) -> None:
    """Check that traced pixels follow a function across most of the plot."""
    data = calibration.to_data(pixels)

    assert len(data) > 300
    assert np.all(np.diff(data[:, 0]) > 0)
    np.testing.assert_allclose(data[:, 1], function(data[:, 0]), atol=atol)


def assert_traces_curve(image: Image, calibration: Calibration) -> None:
    """Check that the curve traced in a synthetic plot follows ``curve``."""
    assert_follows(trace_curve(image, calibration.origin), calibration, curve)


def test_trace_curve_recovers_a_known_curve() -> None:
    assert_traces_curve(*synthetic_plot())


def test_trace_curve_ignores_the_axes() -> None:
    image, calibration = synthetic_plot()
    pixels = trace_curve(image, calibration.origin)

    assert pixels[:, 0].min() > ORIGIN[0] + 2
    assert pixels[:, 1].min() > ORIGIN[1] + 2


def frame(image: Image, colour: Colour = BLACK) -> None:
    """Close the axes of a synthetic plot with a frame, as matplotlib does."""
    top, right = 10, WIDTH - 20
    cv2.line(image, (ORIGIN[0], top), (right, top), colour, 2)
    cv2.line(image, (right, top), (right, to_row(ORIGIN[1])), colour, 2)


def test_trace_curve_ignores_a_frame() -> None:
    image, calibration = synthetic_plot()
    frame(image)

    pixels = trace_curve(image, calibration.origin)

    assert pixels[:, 0].max() < WIDTH - 20
    assert_follows(pixels, calibration, curve)


def test_find_colours_ignores_a_frame() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, curve)], x_range=(0.5, 3))
    frame(image)

    assert find_colours(image, 1, calibration.origin) == [BLUE]


def test_a_band_around_the_plot_is_not_a_frame() -> None:
    image, calibration = synthetic_plot()
    cv2.rectangle(image, (ORIGIN[0], 0), (WIDTH, 25), (110, 110, 110), -1)
    cv2.line(image, (WIDTH - 5, 0), (WIDTH - 5, HEIGHT), (110, 110, 110), 2)
    cv2.line(image, (ORIGIN[0], 12), (WIDTH - 10, 12), BLUE, 2)  # on the band

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    assert np.all(pixels[:, 1] > HEIGHT - 15)


def test_a_line_along_the_top_is_not_a_frame() -> None:
    image, calibration = synthetic_plot()
    cv2.line(image, (ORIGIN[0], 10), (WIDTH - 20, 10), BLUE, 2)  # no right side

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    assert np.all(pixels[:, 1] > HEIGHT - 12)


def boxed(image: Image, rows: slice, columns: slice, colour: Colour = BLACK) -> None:
    """Paint part of a synthetic plot white, and draw a line around it."""
    image[rows, columns] = 255
    corners = (columns.start, rows.start), (columns.stop - 1, rows.stop - 1)
    cv2.rectangle(image, *corners, colour, 1)


def test_trace_curves_leaves_out_an_inset() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, curve)])
    boxed(image, slice(10, 61), slice(230, 371))  # above the curve
    for row in (20, 35, 50):  # tick marks, left of the inset
        cv2.line(image, (225, row), (229, row), BLACK, 1)
    # The inset's curve, in the same colour, meets the frame at a tick mark.
    columns = np.arange(231, 370)
    rows = 35 - 20 * np.sin((columns - 231) / 8)
    points = np.column_stack((columns, rows)).round().astype(np.int32)
    cv2.polylines(image, [points], isClosed=False, color=BLUE, thickness=2)

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    assert_follows(pixels, calibration, curve)


@pytest.mark.parametrize(
    ("top", "left", "right"),
    [(15, 280, 380), (0, 300, WIDTH)],  # inside the plot, and at its corner
)
def test_trace_curves_leaves_out_a_legend(top: int, left: int, right: int) -> None:
    # A light grey frame, as matplotlib draws, around a key and its text.
    image, calibration = synthetic_plot(curves=[(BLUE, curve)])
    boxed(image, slice(top, top + 31), slice(left, right), (214,) * 3)
    cv2.line(image, (left + 10, top + 15), (left + 35, top + 15), BLUE, 2)
    image[top + 11 : top + 20, left + 42 : left + 80] = 0

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    assert_follows(pixels, calibration, curve)


def test_trace_curves_keeps_a_curve_in_a_box_drawn_around_it() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, curve)])
    cv2.rectangle(image, (150, 60), (250, 140), BLACK, 1)

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    assert_follows(pixels, calibration, curve)
    assert np.count_nonzero((pixels[:, 0] > 150) & (pixels[:, 0] < 250)) > 90


def test_trace_curves_keeps_a_curve_in_a_shaded_box() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, curve)], x_range=(2.5, 7.5))
    shaded = image[60:161, 110:321]
    shaded[np.all(shaded == 255, axis=-1)] = 220
    cv2.rectangle(image, (110, 60), (320, 160), BLACK, 1)

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    data = calibration.to_data(pixels)
    np.testing.assert_allclose(data[:, 1], curve(data[:, 0]), atol=0.05)
    assert data[:, 0].min() < 2.6
    assert data[:, 0].max() > 7.4


def pulses(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a step curve that rises from 1 to 2.5 over x from 3 to 5, and 7 to 9."""
    return np.where(((x >= 3) & (x < 5)) | ((x >= 7) & (x < 9)), 2.5, 1.0)


def test_trace_curve_keeps_the_stretches_of_a_step_curve() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, pulses)])

    data = calibration.to_data(trace_curve(image, calibration.origin))

    flat = np.all(np.abs(data[:, :1] - [3, 5, 7, 9]) > 0.1, axis=1)
    assert np.count_nonzero(flat) > 300
    np.testing.assert_allclose(data[flat, 1], pulses(data[flat, 0]), atol=0.05)


def test_trace_curves_keeps_a_curve_between_grid_lines() -> None:
    # The grid lines around the curve close a box, but run on past it.
    image, calibration = synthetic_plot(
        grid=200, curves=[(BLUE, lambda x: 2.1 + 0.3 * np.sin(x))], x_range=(1, 8.5)
    )

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    assert len(pixels) > 250


def test_trace_curves_keeps_a_box_closed_by_two_curves() -> None:
    # The blue curve's pulse stands on a stretch of the orange one.
    def under(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where((x >= 3) & (x < 6), 1.8, 0.8)

    def over(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where((x >= 3) & (x < 6), 2.5, 1.0)

    image, calibration = synthetic_plot(curves=[(ORANGE, under), (BLUE, over)])

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    data = calibration.to_data(pixels)
    pulse = data[(data[:, 0] > 3.2) & (data[:, 0] < 5.8)]
    assert len(pulse) > 80
    np.testing.assert_allclose(pulse[:, 1], 2.5, atol=0.05)


def test_trace_curve_ignores_a_light_grid() -> None:
    assert_traces_curve(*synthetic_plot(grid=200))


def test_trace_curve_handles_dark_backgrounds() -> None:
    image, calibration = synthetic_plot()
    assert_traces_curve(255 - image, calibration)


def test_remove_grid_lets_the_curve_be_traced() -> None:
    image, calibration = synthetic_plot(grid=0)
    original = image.copy()

    cleaned = remove_grid(image)

    assert np.array_equal(image, original)  # not modified
    assert np.all(cleaned[10, :] == 255)  # a grid line is gone
    assert_traces_curve(cleaned, calibration)


def test_remove_grid_handles_dark_backgrounds() -> None:
    image, calibration = synthetic_plot(grid=0)

    cleaned = remove_grid(255 - image)

    assert np.all(cleaned[10, :] == 0)  # a grid line is gone
    assert_traces_curve(cleaned, calibration)


def test_find_colours_puts_the_longest_curve_first() -> None:
    image, calibration = synthetic_plot(
        curves=[(ORANGE, rising), (BLUE, curve)], x_range=(0.5, 6)
    )
    cv2.line(image, (300, 50), (WIDTH, 50), BLUE, 2)  # makes the blue curve longer

    colours = find_colours(image, 2, calibration.origin)

    np.testing.assert_allclose(colours, [BLUE, ORANGE], atol=8)


def test_trace_curves_separates_crossing_curves() -> None:
    curves = [(BLUE, curve), (ORANGE, rising), (BLACK, falling)]
    image, calibration = synthetic_plot(curves=curves)

    colours = find_colours(image, 3, calibration.origin)
    traced = trace_curves(image, [BLUE, ORANGE, BLACK], calibration.origin)

    assert len(colours) == 3
    for pixels, (_, function) in zip(traced, curves, strict=True):
        # A curve loses some pixels where a later one is drawn over it.
        assert_follows(pixels, calibration, function, atol=0.075)


def test_trace_curves_matches_similar_colours() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, curve), (ORANGE, rising)])

    (pixels,) = trace_curves(image, [(60, 130, 160)], calibration.origin)

    # The orange line is drawn over the curve where they cross.
    assert_follows(pixels, calibration, curve, atol=0.075)


def test_find_colours_tells_close_colours_apart() -> None:
    # Neighbouring colours of eight from viridis, 36 levels apart.
    teal, green = (39, 127, 142), (31, 161, 135)
    image, calibration = synthetic_plot(curves=[(teal, rising), (green, falling)])

    colours = find_colours(image, 2, calibration.origin)
    traced = trace_curves(image, [teal, green], calibration.origin)

    np.testing.assert_allclose(sorted(colours), sorted([teal, green]), atol=8)
    for pixels, function in zip(traced, [rising, falling], strict=True):
        assert_follows(pixels, calibration, function, atol=0.075)


def test_trace_curves_needs_a_curve_for_each_colour_given() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, curve), (ORANGE, rising)])

    with pytest.raises(ValueError, match="colours #1f77b4 and #3c82a0 both match"):
        trace_curves(image, [BLUE, (60, 130, 160)], calibration.origin)


def test_trace_curves_needs_a_curve_of_each_colour() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, curve)])
    with pytest.raises(ValueError, match="no curve of colour #00ff00 found"):
        trace_curves(image, [BLUE, (0, 255, 0)], calibration.origin)


def test_pale_curves_can_be_traced_but_are_not_found() -> None:
    image, calibration = synthetic_plot(curves=[(PALE_BLUE, curve), (BLACK, rising)])

    with pytest.raises(ValueError, match="found only 1 of 2 curves"):
        find_colours(image, 2, calibration.origin)
    (pixels,) = trace_curves(image, [PALE_BLUE], calibration.origin)
    # The black line is drawn over the curve where they cross.
    assert_follows(pixels, calibration, curve, atol=0.075)


def thin_plot(
    curves: Sequence[tuple[Colour, Function]], thickness: float
) -> tuple[Image, Calibration]:
    """Draw curves as antialiased lines a fraction of a pixel wide.

    The lines are drawn four times as large, ``thickness`` pixels wide at
    the plot's size, which is then shrunk to that size, averaging the pixels
    each covers, as a plotting library's antialiasing does.
    """
    image, calibration = synthetic_plot(curves=[])
    large = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    x = np.linspace(0.5, 10, 500)
    for colour, function in curves:
        points = np.column_stack(
            (
                ORIGIN[0] + x * PIXELS_PER_UNIT[0],
                HEIGHT - ORIGIN[1] - function(x) * PIXELS_PER_UNIT[1],
            )
        )
        cv2.polylines(
            large,
            [(points * 4 * 16).round().astype(np.int32)],
            isClosed=False,
            color=colour,
            thickness=round(4 * thickness),
            lineType=cv2.LINE_AA,
            shift=4,
        )
    shrunk = cv2.resize(large, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return shrunk.astype(np.uint8), calibration


RED, GREEN = (214, 39, 40), (44, 160, 44)


def ray_angle(colour: Colour, other: Colour) -> float:
    """Return the angle in degrees between two colours' directions from white."""
    first, second = (np.subtract(c, 255) for c in (colour, other))
    cosine = first @ second / np.linalg.norm(first) / np.linalg.norm(second)
    return float(np.degrees(np.arccos(min(cosine, 1))))


def test_find_colours_finds_lines_thinner_than_a_pixel() -> None:
    # Their pixels spread over many pale shades, none common on its own,
    # but all on the ray from the background towards each line's ink.
    inks, functions = [RED, BLUE, GREEN], [curve, rising, falling]
    image, calibration = thin_plot(list(zip(inks, functions, strict=True)), 0.5)

    found = find_colours(image, 3, calibration.origin)

    nearest = [min(inks, key=lambda ink: ray_angle(colour, ink)) for colour in found]
    assert sorted(nearest) == sorted(inks)
    assert max(map(ray_angle, found, nearest, strict=True)) < 3
    traced = trace_curves(image, inks, calibration.origin)
    for pixels, function in zip(traced, functions, strict=True):
        assert_follows(pixels, calibration, function, atol=0.075)


def test_trace_curves_matches_a_translucent_line_by_its_colour() -> None:
    # Drawn at half opacity, the line shows a pale blue far from BLUE, but
    # on its ray from the background.
    image, calibration = synthetic_plot(curves=[(BLUE, curve)])
    faded = ((image.astype(np.float64) + 255) / 2).astype(np.uint8)

    (pixels,) = trace_curves(faded, [BLUE], calibration.origin)

    assert_follows(pixels, calibration, curve)


def banded(image: Image, colour: Colour, function: Function, opacity: float) -> Image:
    """Shade a band 0.75 either side of a function over a plot, translucently."""
    shaded = image.astype(np.float64)
    x = np.linspace(0.5, 10, 500)
    columns = ORIGIN[0] + x * PIXELS_PER_UNIT[0]
    edges = [
        HEIGHT - ORIGIN[1] - (function(x) + side) * PIXELS_PER_UNIT[1]
        for side in (0.75, -0.75)
    ]
    outline = np.concatenate(
        [
            np.column_stack((columns, edges[0])),
            np.column_stack((columns, edges[1]))[::-1],
        ]
    )
    band = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(band, [outline.round().astype(np.int32)], 1)
    inside = band > 0
    shaded[inside] = (1 - opacity) * shaded[inside] + opacity * np.array(colour)
    return shaded.round().astype(np.uint8)


def test_find_colours_skips_where_translucent_bands_overlap() -> None:
    # Each curve has a band; where the bands overlap, they show a colour of
    # their own, which is no curve's.
    image, calibration = synthetic_plot(curves=[(BLUE, curve), (ORANGE, rising)])
    for colour, function in ((BLUE, curve), (ORANGE, rising)):
        image = banded(image, colour, function, 0.3)

    found = find_colours(image, 2, calibration.origin)

    assert sorted(found) == sorted([BLUE, ORANGE])


def test_find_colours_finds_a_curve_inside_its_own_band() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, curve)])
    image = banded(image, BLUE, curve, 0.2)

    assert find_colours(image, 1, calibration.origin) == [BLUE]


def test_find_colours_leaves_out_a_thin_lines_fringe() -> None:
    # A fainter ink alongside a thin line, as compression gives one, is the
    # line's fringe, not a curve of its own.
    image, calibration = thin_plot(
        [(RED, curve), (PINK, lambda x: curve(x) + 1.5 / PIXELS_PER_UNIT[1])], 0.5
    )

    (found,) = find_colours(image, 1, calibration.origin)

    assert ray_angle(found, RED) < 3
    with pytest.raises(ValueError, match="found only 1 of 2 curves"):
        find_colours(image, 2, calibration.origin)


def test_find_colours_finds_a_thin_line_beside_a_paler_one() -> None:
    # The red line runs 2 pixels from the pink one, but is stronger, so it
    # is no fringe of it.
    pink = (218, 139, 195)
    image, calibration = thin_plot(
        [(RED, curve), (pink, lambda x: curve(x) + 2 / PIXELS_PER_UNIT[1])], 0.5
    )

    found = find_colours(image, 2, calibration.origin)

    assert sorted(min(ray_angle(c, ink) for c in found) for ink in (RED, pink)) == [
        pytest.approx(0, abs=3),
        pytest.approx(0, abs=3),
    ]


def test_find_colours_skips_dashed_lines_across_the_plot() -> None:
    # A dashed horizontal line, as a threshold is drawn, is grid-like.
    image, calibration = synthetic_plot(curves=[(BLUE, curve)])
    row = to_row(ORIGIN[1] + 2.5 * PIXELS_PER_UNIT[1])
    for column in range(ORIGIN[0], WIDTH, 13):
        cv2.line(image, (column, row), (column + 7, row), ORANGE, 2)

    with pytest.raises(ValueError, match=r"#ff7f0e .* as it is drawn in lines across"):
        find_colours(image, 2, calibration.origin)


def test_find_colours_says_what_it_found_and_what_it_left_out() -> None:
    image, calibration = synthetic_plot(
        grid=100, curves=[(PALE_BLUE, curve), (BLACK, rising)]
    )

    with pytest.raises(ValueError, match="found only") as raised:
        find_colours(image, 3, calibration.origin)

    found, _, left_out = str(raised.value).partition("; left out ")
    assert re.fullmatch(r"found only 1 of 3 curves, #000000 \(\d+ pixels\)", found)
    assert re.fullmatch(
        r"#646464 \(\d+ pixels\), as it is drawn in lines across the plot, as grid "
        r"lines are; #c8dcf0 \(\d+ pixels\), as it is too pale, 54 levels from the "
        r"background",
        left_out,
    )


def test_trace_curves_leaves_a_darker_curves_edges_out_of_a_lighter_one() -> None:
    # The edges of the black line, a third covered, show the light grey of
    # the other curve.
    light_grey = (179, 179, 179)
    image, calibration = synthetic_plot(curves=[(light_grey, curve), (BLACK, rising)])

    pixels, _ = trace_curves(image, [light_grey, BLACK], calibration.origin)

    assert_follows(pixels, calibration, curve, atol=0.075)


def test_find_colours_finds_mid_tones_on_a_grey_background() -> None:
    image, calibration = synthetic_plot(grid=254, curves=[(PINK, curve)])
    image[np.all(image == 255, axis=-1)] = GREY
    image[np.all(image == 254, axis=-1)] = 255  # a white grid, as seaborn draws

    assert find_colours(image, 1, calibration.origin) == [PINK]


def test_find_colours_skips_grid_lines_of_any_shade() -> None:
    image, calibration = synthetic_plot(grid=100, curves=[(BLUE, curve)])

    assert find_colours(image, 1, calibration.origin) == [BLUE]


def filled(image: Image) -> None:
    """Fill a synthetic plot blue below a line high in the plot, over most of it."""
    corners = [
        (ORIGIN[0], 40),
        (WIDTH, 60),
        (WIDTH, to_row(ORIGIN[1])),
        (ORIGIN[0], 199),
    ]
    cv2.fillPoly(image, [np.array(corners, dtype=np.int32)], BLUE)


def test_find_colours_skips_filled_areas() -> None:
    image, calibration = synthetic_plot(curves=[])
    filled(image)
    curves = synthetic_plot(curves=[(BLACK, curve)])[0]
    image[np.all(curves == 0, axis=-1)] = BLACK

    assert find_colours(image, 1, calibration.origin) == [BLACK]
    assert_traces_curve(image, calibration)


def test_trace_curves_refuses_filled_areas() -> None:
    image, calibration = synthetic_plot()
    filled(image)

    with pytest.raises(ValueError, match="colour #1f77b4 fills areas"):
        trace_curves(image, [BLUE], calibration.origin)


def test_trace_curve_follows_both_sides_of_a_loop() -> None:
    image, calibration = synthetic_plot(curves=[])
    # An ellipse centred on (5, 2), 3 wide and 1 tall in data units.
    centre = (
        ORIGIN[0] + 5 * PIXELS_PER_UNIT[0],
        to_row(ORIGIN[1] + 2 * PIXELS_PER_UNIT[1]),
    )
    axes = (3 * PIXELS_PER_UNIT[0], 1 * PIXELS_PER_UNIT[1])
    cv2.ellipse(image, centre, axes, 0, 0, 360, BLACK, 2, cv2.LINE_AA)

    with pytest.warns(UserWarning, match="crosses most columns 2 times: it may be 2"):
        data = calibration.to_data(trace_curve(image, calibration.origin))

    middle = data[np.abs(data[:, 0] - 5) < 2]
    np.testing.assert_allclose(
        np.abs(middle[:, 1] - 2), np.sqrt(1 - ((middle[:, 0] - 5) / 3) ** 2), atol=0.05
    )
    assert np.count_nonzero(middle[:, 1] > 2) == np.count_nonzero(middle[:, 1] < 2)


def test_trace_curve_ignores_short_marks_beside_a_curve() -> None:
    image, calibration = synthetic_plot()
    xs = np.arange(1.0, 10.0)
    heights = ORIGIN[1] + (curve(xs) + 0.6) * PIXELS_PER_UNIT[1]
    for x, height in zip(xs, heights, strict=True):  # like caps of error bars
        column = round(ORIGIN[0] + x * PIXELS_PER_UNIT[0])
        row = to_row(float(height))
        cv2.line(image, (column - 4, row), (column + 4, row), BLACK, 2)

    assert_traces_curve(image, calibration)


def test_trace_curves_ignores_ink_of_other_colours() -> None:
    # An annotation's arrow, in black, too short to be a curve of its own,
    # but as long as a branch of the blue curve.
    image, calibration = synthetic_plot(curves=[(BLUE, curve)])
    cv2.arrowedLine(image, (200, 20), (260, 50), BLACK, 2, cv2.LINE_AA)

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    assert_follows(pixels, calibration, curve)


def test_trace_curves_ignores_specks() -> None:
    image, calibration = synthetic_plot(curves=[(BLUE, curve)], x_range=(0.5, 6))
    for column in range(300, WIDTH, 10):  # beyond the curve
        image[column % 37 + 20, column : column + 3] = BLUE

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    assert calibration.to_data(pixels)[:, 0].max() < 6.1


def test_trace_curve_needs_a_curve() -> None:
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    image[50:60, 50:60] = 0  # too small to be a curve
    with pytest.raises(ValueError, match="no curve found"):
        trace_curve(image)


def test_trace_curve_needs_a_plot_area() -> None:
    image, _ = synthetic_plot()
    with pytest.raises(ValueError, match="no plot area"):
        trace_curve(image, (WIDTH, 0))


def test_trace_curve_needs_a_plot_area_inside_a_frame() -> None:
    # The axes leave a few rows of ink, taken for a frame's top, with nothing
    # below it.
    image = np.full((20, 30, 3), 255, dtype=np.uint8)
    image[:4, 15:] = 0
    with pytest.raises(ValueError, match="no plot area"):
        trace_curve(image, (12, 13))


# The middle of the pixel where the axis lines of each example cross, as a
# click on it gives, and the positions of the legend keys of markers.png.
CORNERS = {"spectrum": (80.5, 43.5), "grid": (80.5, 43.5), "markers": (55.5, 82.5)}
MARKER_KEYS = [(148, 18), (260, 18), (377, 18), (483, 18)]


def traced(name: str, origin: Point) -> list[npt.NDArray[np.float64]]:
    """Trace an example's curve, or find its markers, from an origin."""
    image = load_image(EXAMPLES / f"{name}.png")
    if name == "markers":
        return find_markers(image, MARKER_KEYS, origin)
    return [trace_curve(image, origin)]


@pytest.mark.parametrize("name", CORNERS)
@pytest.mark.parametrize("offset", [(-10, 0), (0, -10), (-5, -5), (-4, -6), (-1, -10)])
def test_axes_given_below_or_left_of_their_lines_are_snapped_to_them(
    name: str, offset: tuple[int, int]
) -> None:
    corner = CORNERS[name]
    origin = (corner[0] + offset[0], corner[1] + offset[1])

    for found, expected in zip(traced(name, origin), traced(name, corner), strict=True):
        np.testing.assert_array_equal(found, expected)


@pytest.mark.parametrize("name", CORNERS)
def test_axes_given_anywhere_near_their_lines_give_a_trace_or_an_error(
    name: str,
) -> None:
    corner = CORNERS[name]
    for dx, dy in itertools.product(range(-15, 16, 5), repeat=2):
        with contextlib.suppress(ValueError), warnings.catch_warnings():
            # Any warning or ValueError, such as for an axis line traced
            # with a curve, but no other error.
            warnings.simplefilter("ignore")
            traced(name, (corner[0] + dx, corner[1] + dy))


def test_an_open_plots_x_axis_line_is_not_a_frame() -> None:
    # The axes are given too far below the x-axis line to be snapped to it,
    # and the line, across the plot from the y-axis, with tick marks below
    # it, is not taken for the top of a frame.
    image, calibration = synthetic_plot()
    for column in range(ORIGIN[0], WIDTH, 35):
        cv2.line(image, (column, to_row(ORIGIN[1])), (column, to_row(5)), BLACK, 2)
    x, y = calibration.origin

    # The x-axis line is traced as a second curve of the same colour.
    with pytest.warns(UserWarning, match="it may be 2 curves of that colour"):
        pixels = trace_curve(image, (x, y - 20))

    assert_follows(pixels[pixels[:, 1] > ORIGIN[1] + 2], calibration, curve)


MARKER_XS = np.arange(0.5, 10, 0.5)  # where the synthetic markers lie
LEGEND_ROW = 185  # below the x-axis


def wavy(level: int) -> Function:
    """Return a gentle wave that keeps clear of those of the other levels."""
    return lambda x: 0.5 + 0.9 * level + 0.2 * np.sin(x)


def draw_marker(
    image: Image, shape: str, column: int, row: int, colour: Colour = BLACK
) -> None:
    """Draw a marker 9 pixels across, centred on a pixel, over what is there.

    A disc and an ellipse are filled, and an ellipse is 17 pixels wide.
    """
    match shape:
        case "square":
            cv2.rectangle(
                image, (column - 4, row - 4), (column + 4, row + 4), colour, -1
            )
        case "diamond":
            corners = [(column, row - 5), (column + 5, row), (column, row + 5)]
            corners.append((column - 5, row))
            cv2.fillPoly(image, [np.array(corners)], colour, cv2.LINE_AA)
        case "circle":
            cv2.circle(image, (column, row), 4, (255, 255, 255), -1)
            cv2.circle(image, (column, row), 4, colour, 1, cv2.LINE_AA)
        case "disc":
            cv2.circle(image, (column, row), 4, colour, -1, cv2.LINE_AA)
        case "ellipse":
            cv2.ellipse(image, (column, row), (8, 3), 0, 0, 360, colour, -1)
        case _:  # a cross
            for end in (-3, 3):
                start, stop = (column - 3, row - end), (column + 3, row + end)
                cv2.line(image, start, stop, colour, 1, cv2.LINE_AA)


def marker_plot(
    shapes: Sequence[str],
    legend_row: int = LEGEND_ROW,
    *,
    key_lines: bool = True,
    colours: Sequence[Colour] = (),
) -> tuple[Image, Calibration, list[Point], list[npt.NDArray[np.float64]]]:
    """Draw axes and series of markers joined by lines, with a legend.

    Series ``k`` follows ``wavy(k)``, with markers at ``MARKER_XS``, in
    ``colours[k]``, or black. The legend's keys lie in a row, each with a
    short line through its marker unless ``key_lines`` is false.

    Returns:
        The image, its calibration, the position of each key's marker, and
        the true pixel position of each series' markers.
    """
    image, calibration = synthetic_plot(curves=[])
    keys: list[Point] = []
    markers = []
    for level, shape in enumerate(shapes):
        colour = colours[level] if colours else BLACK
        columns = np.round(ORIGIN[0] + MARKER_XS * PIXELS_PER_UNIT[0]).astype(int)
        heights = ORIGIN[1] + wavy(level)(MARKER_XS) * PIXELS_PER_UNIT[1]
        rows = np.round(HEIGHT - heights).astype(int)
        line = np.column_stack((columns, rows)).astype(np.int32)
        cv2.polylines(image, [line], isClosed=False, color=colour, lineType=cv2.LINE_AA)
        for column, row in zip(columns, rows, strict=True):
            draw_marker(image, shape, int(column), int(row), colour)
        markers.append(np.column_stack((columns + 0.5, HEIGHT - rows - 0.5)))

        column = 70 + 80 * level
        if key_lines:
            start, stop = (column - 12, legend_row), (column + 12, legend_row)
            cv2.line(image, start, stop, colour, 1, cv2.LINE_AA)
        draw_marker(image, shape, column, legend_row, colour)
        keys.append((column + 0.5, HEIGHT - legend_row - 0.5))
    return image, calibration, keys, markers


def assert_finds_markers(
    found: Sequence[npt.NDArray[np.float64]],
    markers: Sequence[npt.NDArray[np.float64]],
) -> None:
    """Check that each series' markers are found, each to half a pixel."""
    assert len(found) == len(markers)
    for pixels, true in zip(found, markers, strict=True):
        assert pixels.shape == true.shape
        np.testing.assert_allclose(pixels, true, atol=0.5)


SHAPES = ["square", "diamond", "circle", "cross"]


def test_find_markers_tells_series_apart_by_their_shapes() -> None:
    image, calibration, keys, markers = marker_plot(SHAPES)

    found = find_markers(image, keys, calibration.origin)

    assert_finds_markers(found, markers)


def test_find_markers_reads_keys_without_lines() -> None:
    image, calibration, keys, markers = marker_plot(SHAPES, key_lines=False)
    assert_finds_markers(find_markers(image, keys, calibration.origin), markers)


def test_find_markers_skips_a_legend_in_the_plot_area() -> None:
    image, calibration, keys, markers = marker_plot(SHAPES[:3], legend_row=12)
    assert_finds_markers(find_markers(image, keys, calibration.origin), markers)


def test_find_markers_skips_markers_in_an_inset() -> None:
    image, calibration, keys, markers = marker_plot(SHAPES[:2])
    boxed(image, slice(10, 81), slice(230, 371))  # above the series
    for column in range(250, 360, 25):
        draw_marker(image, SHAPES[0], column, 45)

    assert_finds_markers(find_markers(image, keys, calibration.origin), markers)


def test_find_markers_handles_dark_backgrounds() -> None:
    image, calibration, keys, markers = marker_plot(SHAPES)
    assert_finds_markers(find_markers(255 - image, keys, calibration.origin), markers)


def test_find_markers_gives_no_markers_for_a_series_not_drawn() -> None:
    image, _, keys, markers = marker_plot(SHAPES[:2])
    draw_marker(image, "circle", 300, LEGEND_ROW)  # a key, but no series

    found = find_markers(image, [*keys, (300.5, HEIGHT - LEGEND_ROW - 0.5)])

    assert_finds_markers(found, [*markers, np.empty((0, 2))])


def test_find_markers_needs_a_plot_area_bigger_than_a_marker() -> None:
    image, _, keys, _ = marker_plot(SHAPES[:2])

    found = find_markers(image, keys, (WIDTH - 6, HEIGHT - 6))

    assert_finds_markers(found, [np.empty((0, 2))] * 2)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ((WIDTH + 1, 50), f"the marker position {WIDTH + 1},50 lies outside"),
        ((-1, 50), "the marker position -1,50 lies outside"),
        ((380.5, 190.5), "found no marker near 380.5,190.5"),
        ((350.5, 9.5), "found only a line, and no marker, at 350.5,9.5"),
    ],
)
def test_find_markers_needs_a_marker_at_each_key(key: Point, message: str) -> None:
    image, calibration, keys, _ = marker_plot(SHAPES[:1])
    cv2.line(image, (330, 190), (370, 190), BLACK, 1)  # a key without a marker

    with pytest.raises(ValueError, match=message):
        find_markers(image, [*keys, key], calibration.origin)


def test_find_markers_tells_series_of_one_shape_apart_by_colour() -> None:
    colours = [BLUE, ORANGE, GREEN, BLACK]
    image, calibration, keys, markers = marker_plot(["disc"] * 4, colours=colours)
    assert_finds_markers(find_markers(image, keys, calibration.origin), markers)


def test_find_markers_tells_series_apart_by_shape_and_colour() -> None:
    # Two shapes in each of two colours, one drawn lighter in one series.
    shapes = ["disc", "disc", "square", "square"]
    colours = [LIGHT_BLUE, ORANGE, ORANGE, BLUE]
    image, calibration, keys, markers = marker_plot(shapes, colours=colours)
    assert_finds_markers(find_markers(image, keys, calibration.origin), markers)


def test_find_markers_reads_wide_markers_without_lines() -> None:
    image, calibration, keys, markers = marker_plot(
        ["ellipse", "square"], key_lines=False
    )
    assert_finds_markers(find_markers(image, keys, calibration.origin), markers)


def test_find_marker_series_flags_markers_of_keys_alike() -> None:
    image, calibration, keys, markers = marker_plot(["square", "square"])

    with pytest.warns(UserWarning, match="look alike at this size") as caught:
        found = find_marker_series(image, keys, calibration.origin)

    assert len(caught) == 1
    points, true = (
        np.array(sorted(np.concatenate(each).round(1).tolist()))
        for each in ([series.points for series in found], markers)
    )
    assert_finds_markers([points], [true])
    assert all(np.all(series.ambiguous) for series in found)


def test_find_marker_series_finds_markers_partly_hidden_under_others() -> None:
    image, calibration = synthetic_plot(curves=[])
    columns = np.round(ORIGIN[0] + MARKER_XS * PIXELS_PER_UNIT[0]).astype(int)
    heights = ORIGIN[1] + wavy(1)(MARKER_XS) * PIXELS_PER_UNIT[1]
    rows = np.round(HEIGHT - heights).astype(int)
    hidden = 9  # where a diamond hides a square altogether
    for number, (column, row) in enumerate(zip(columns, rows, strict=True)):
        if number != hidden:
            draw_marker(image, "square", int(column), int(row))
    lifts = np.where(np.arange(len(rows)) == hidden, 0, 5)  # diamonds 5 pixels above
    for column, row in zip(columns, rows - lifts, strict=True):
        draw_marker(image, "diamond", int(column), int(row))
    keys: list[Point] = []
    for column, shape in [(70, "square"), (150, "diamond")]:
        cv2.line(image, (column - 12, LEGEND_ROW), (column + 12, LEGEND_ROW), BLACK, 1)
        draw_marker(image, shape, column, LEGEND_ROW)
        keys.append((column + 0.5, HEIGHT - LEGEND_ROW - 0.5))

    squares, diamonds = find_marker_series(image, keys, calibration.origin)

    centres = np.column_stack((columns + 0.5, HEIGHT - rows - 0.5))
    assert_finds_markers(
        [squares.points, diamonds.points],
        [
            np.delete(centres, hidden, axis=0),
            centres + np.column_stack((0 * lifts, lifts)),
        ],
    )
    np.testing.assert_allclose(squares.hidden, centres[[hidden]], atol=1)
    assert len(diamonds.hidden) == 0
    assert not np.any(squares.ambiguous)
    assert not np.any(diamonds.ambiguous)


def test_load_image_reads_colour_stretched_to_the_full_range(tmp_path: Path) -> None:
    path = tmp_path / "colour.png"
    image = np.full((10, 20, 3), (200, 180, 160), dtype=np.uint8)  # BGR
    image[5, :] = (40, 50, 60)
    cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (10, 20, 3)
    assert loaded.dtype == np.uint8
    assert np.all(loaded[0, :] == (191, 223, 255))  # RGB
    assert np.all(loaded[5, :] == (32, 16, 0))


def test_load_image_reads_greyscale_as_colour(tmp_path: Path) -> None:
    path = tmp_path / "grey.png"
    image = np.full((10, 20), 200, dtype=np.uint8)
    image[5, :] = 100
    cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (10, 20, 3)
    assert np.all(loaded[5, :] == 0)
    assert np.all(np.delete(loaded, 5, axis=0) == 255)


def test_load_image_puts_transparent_pixels_on_white(tmp_path: Path) -> None:
    path = tmp_path / "transparent.png"
    image = np.zeros((10, 20, 4), dtype=np.uint8)  # transparent black
    image[5, :, 3] = 255  # an opaque black line
    cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert np.all(loaded[5, :] == 0)
    assert np.all(np.delete(loaded, 5, axis=0) == 255)


def test_load_image_scales_16_bit_images(tmp_path: Path) -> None:
    path = tmp_path / "deep.png"
    image = np.zeros((10, 20), dtype=np.uint16)
    image[:, 5:] = 60000
    cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert np.all(loaded[:, :5] == 0)
    assert np.all(loaded[:, 5:] == 255)


def test_load_image_reports_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read image"):
        load_image(tmp_path / "missing.png")


@pytest.mark.parametrize("colour", [128, (200, 180, 160)])
def test_load_image_rejects_blank_images(
    tmp_path: Path, colour: int | tuple[int, int, int]
) -> None:
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.full((10, 20, 3), colour, dtype=np.uint8))
    with pytest.raises(ValueError, match="single colour"):
        load_image(path)
