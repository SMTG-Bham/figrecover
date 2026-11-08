"""Tests for reading images and tracing curves."""

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
    find_error_bars,
    find_marker_error_bars,
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


MARKER_XS = np.arange(0.5, 10, 0.5)  # where the synthetic markers lie
LEGEND_ROW = 185  # below the x-axis


def wavy(level: int) -> Function:
    """Return a gentle wave that keeps clear of those of the other levels."""
    return lambda x: 0.5 + 0.9 * level + 0.2 * np.sin(x)


def draw_marker(image: Image, shape: str, column: int, row: int) -> None:
    """Draw a marker 9 pixels across, centred on a pixel, over what is there."""
    match shape:
        case "square":
            cv2.rectangle(
                image, (column - 4, row - 4), (column + 4, row + 4), BLACK, -1
            )
        case "diamond":
            corners = [(column, row - 5), (column + 5, row), (column, row + 5)]
            corners.append((column - 5, row))
            cv2.fillPoly(image, [np.array(corners)], BLACK, cv2.LINE_AA)
        case "circle":
            cv2.circle(image, (column, row), 4, (255, 255, 255), -1)
            cv2.circle(image, (column, row), 4, BLACK, 1, cv2.LINE_AA)
        case _:  # a cross
            for end in (-3, 3):
                start, stop = (column - 3, row - end), (column + 3, row + end)
                cv2.line(image, start, stop, BLACK, 1, cv2.LINE_AA)


def marker_plot(
    shapes: Sequence[str], legend_row: int = LEGEND_ROW, *, key_lines: bool = True
) -> tuple[Image, Calibration, list[Point], list[npt.NDArray[np.float64]]]:
    """Draw axes and series of markers joined by lines, with a legend.

    Series ``k`` follows ``wavy(k)``, with markers at ``MARKER_XS``. The
    legend's keys lie in a row, each with a short line through its marker
    unless ``key_lines`` is false.

    Returns:
        The image, its calibration, the position of each key's marker, and
        the true pixel position of each series' markers.
    """
    image, calibration = synthetic_plot(curves=[])
    keys: list[Point] = []
    markers = []
    for level, shape in enumerate(shapes):
        columns = np.round(ORIGIN[0] + MARKER_XS * PIXELS_PER_UNIT[0]).astype(int)
        heights = ORIGIN[1] + wavy(level)(MARKER_XS) * PIXELS_PER_UNIT[1]
        rows = np.round(HEIGHT - heights).astype(int)
        line = np.column_stack((columns, rows)).astype(np.int32)
        cv2.polylines(image, [line], isClosed=False, color=BLACK, lineType=cv2.LINE_AA)
        for column, row in zip(columns, rows, strict=True):
            draw_marker(image, shape, int(column), int(row))
        markers.append(np.column_stack((columns + 0.5, HEIGHT - rows - 0.5)))

        column = 70 + 80 * level
        if key_lines:
            start, stop = (column - 12, legend_row), (column + 12, legend_row)
            cv2.line(image, start, stop, BLACK, 1, cv2.LINE_AA)
        draw_marker(image, shape, column, legend_row)
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


def at(x: float, y: float) -> tuple[int, int]:
    """Return the column and row of the pixel at a point of a synthetic plot."""
    return (
        round(ORIGIN[0] + x * PIXELS_PER_UNIT[0] - 0.5),
        round(HEIGHT - ORIGIN[1] - y * PIXELS_PER_UNIT[1] - 0.5),
    )


def bar(
    image: Image,
    ends: tuple[tuple[float, float], tuple[float, float]],
    colour: Colour = BLUE,
    *,
    caps: bool,
) -> None:
    """Draw an error bar between two points of a synthetic plot, with or without caps.

    The bar is a pixel wide, and its caps, across its ends, 9 pixels long.
    """
    cv2.line(image, at(*ends[0]), at(*ends[1]), colour, 1)
    for end in ends if caps else ():
        column, row = at(*end)
        if ends[0][0] == ends[1][0]:
            cv2.line(image, (column - 4, row), (column + 4, row), colour, 1)
        else:
            cv2.line(image, (column, row - 4), (column, row + 4), colour, 1)


def test_find_error_bars_follows_bars_to_their_ends() -> None:
    image, calibration = synthetic_plot(curves=[(ORANGE, lambda x: 3.6 + 0 * x)])
    points = [(1, 1.5), (3, 2.0), (5, 2.5), (7, 2.0), (9, 1.0), (9.5, 3.0)]
    line = np.array([at(*point) for point in points[:5]])
    cv2.polylines(image, [line], isClosed=False, color=BLUE, lineType=cv2.LINE_AA)
    bar(image, ((1, 1.2), (1, 2.2)), caps=True)  # longer above than below
    bar(image, ((3, 1.7), (3, 2.4)), caps=False)  # crossed by an orange mark
    cv2.line(image, at(2.8, 2.25), at(3.2, 2.25), ORANGE, 2)
    bar(image, ((5, 2.2), (5, 3.1)), caps=True)  # its upper end under one
    cv2.line(image, at(4.8, 3.1), at(5.2, 3.1), ORANGE, 3)
    bar(image, ((6.6, 2.0), (7.4, 2.0)), caps=True)  # across
    bar(image, ((9, -0.5), (9, 1.4)), caps=False)  # down past the x-axis
    for point in points:  # the last without bars or a line
        cv2.circle(image, at(*point), 4, BLUE, -1, cv2.LINE_AA)

    bars, orange = find_error_bars(image, [BLUE, ORANGE], calibration.origin)

    # The ends of bars hidden by another colour, or beyond the plot, are not
    # known. The bars end within a pixel, about 0.03 in x and y.
    nan = np.nan
    np.testing.assert_allclose(calibration.to_data(bars.points), points, atol=0.03)
    low = [(nan, 1.2), (nan, 1.7), (nan, 2.2), (6.6, nan), (nan, nan), (nan, nan)]
    high = [(nan, 2.2), (nan, 2.4), (nan, nan), (7.4, nan), (nan, 1.4), (nan, nan)]
    np.testing.assert_allclose(calibration.to_data(bars.low), low, atol=0.03)
    np.testing.assert_allclose(calibration.to_data(bars.high), high, atol=0.03)
    assert len(orange.points) == 0
    # The bars are left out of the line, which strays 0.27 with them.
    (traced,) = trace_curves(image, [BLUE], calibration.origin)
    data = calibration.to_data(traced)
    lined = data[data[:, 0] < 9.1]
    expected = np.interp(lined[:, 0], *np.array(points[:5]).T)
    assert np.abs(lined[:, 1] - expected).max() < 0.1
    assert data[:, 1].min() > 0.9


# The four series that examples/make_examples.py draws in errorbars.png,
# from the top: each colour and curve, its errors below and above, and to
# each side, if it has bars across.
ERROR_MARKED = np.arange(0.5, 10, 1.0)
ERROR_SERIES: list[tuple[Colour, Function, tuple[float, float], float | None]] = [
    ((31, 119, 180), lambda x: 3.4 + np.sin(x) / 4, (0.2, 0.2), None),
    ((255, 127, 14), lambda x: 2.4 + np.cos(x) / 4, (0.1, 0.3), None),
    ((44, 160, 44), lambda x: 1.5 + np.sin(x / 2) / 5, (0.15, 0.15), 0.3),
    ((214, 39, 40), lambda x: 0.6 + 0.15 * np.cos(x / 2), (0.2, 0.2), None),
]
# Its calibration, from where matplotlib draws (0, 0), (10, 0) and (0, 1).
ERROR_CALIBRATION = Calibration.from_points(
    [(0, 0), (10, 0), (0, 1)], [(80, 44), (576, 44), (80, 121)]
)


def test_find_error_bars_reads_the_example_plot() -> None:
    image = load_image(EXAMPLES / "errorbars.png")
    origin = ERROR_CALIBRATION.origin

    found = find_error_bars(image, [colour for colour, *_ in ERROR_SERIES], origin)

    # matplotlib draws 49.6 pixels to a unit of x and 77 to one of y. The
    # middles of the triangles, which are wider below, lie lower than their
    # points, by up to 2.5 pixels; the rest within a pixel. Every bar ends
    # within 2 pixels of its bound.
    scale = np.array([ERROR_CALIBRATION.x_scale, ERROR_CALIBRATION.y_scale])
    x = ERROR_MARKED
    for bars, (colour, function, (below, above), across) in zip(
        found, ERROR_SERIES, strict=True
    ):
        sideways = np.full_like(x, np.nan if across is None else across)
        points = np.column_stack([x, function(x)])
        low = np.column_stack([x - sideways, function(x) - below])
        high = np.column_stack([x + sideways, function(x) + above])
        off = 2.5 if colour == (214, 39, 40) else 1
        np.testing.assert_allclose(
            ERROR_CALIBRATION.to_data(bars.points), points, atol=off * scale.max()
        )
        for ends, truth in ((bars.low, low), (bars.high, high)):
            error = np.abs(ERROR_CALIBRATION.to_data(ends) - truth) / scale
            np.testing.assert_array_equal(np.isnan(error), np.isnan(truth))
            assert np.nanmax(error) <= 2


def test_trace_curves_leaves_out_error_bars() -> None:
    image = load_image(EXAMPLES / "errorbars.png")
    colours = [colour for colour, *_ in ERROR_SERIES]

    traced = trace_curves(image, colours, ERROR_CALIBRATION.origin)

    # The line through the orange markers, whose bars are longer above,
    # strays no more than 0.01 of the y range (4) from the line through
    # their points, and the green markers, with bars across, give points
    # only where they lie, about 6 pixels each side.
    _, orange, green, _ = (ERROR_CALIBRATION.to_data(curve) for curve in traced)
    x = ERROR_MARKED
    expected = np.interp(orange[:, 0], x, ERROR_SERIES[1][1](x))
    assert np.abs(orange[:, 1] - expected).max() <= 0.01 * 4
    assert np.abs(green[:, 0, np.newaxis] - x).min(axis=1).max() < 7 / 49.6


def test_find_marker_error_bars_follows_bars_in_any_colour() -> None:
    image, calibration, keys, markers = marker_plot(SHAPES[:2])
    for x, y in markers[0][::6]:  # every sixth square, 12 pixels each way
        column, row = round(x - 0.5), round(HEIGHT - y - 0.5)
        cv2.line(image, (column, row - 12), (column, row + 12), BLACK, 1)
        for end in (row - 12, row + 12):
            cv2.line(image, (column - 4, end), (column + 4, end), BLACK, 1)

    found = find_marker_error_bars(image, keys, calibration.origin)

    markers = find_markers(image, keys, calibration.origin)
    for bars, points in zip(found, markers, strict=True):
        np.testing.assert_array_equal(bars.points, points)
    # The caps lie 12 pixels from the middles of the squares, and the
    # diamonds have no bars.
    barred = np.zeros(len(markers[0]), dtype=np.bool_)
    barred[::6] = True
    lows, highs = found[0].low[:, 1], found[0].high[:, 1]
    np.testing.assert_allclose(lows[barred], markers[0][barred, 1] - 12, atol=1)
    np.testing.assert_allclose(highs[barred], markers[0][barred, 1] + 12, atol=1)
    assert np.isnan(lows[~barred]).all()
    assert np.isnan(highs[~barred]).all()
    assert np.isnan(found[0].low[:, 0]).all()
    assert np.isnan(found[1].low).all()
    assert np.isnan(found[1].high).all()


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
