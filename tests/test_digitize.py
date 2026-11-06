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
    find_colours,
    load_image,
    remove_grid,
    trace_curve,
    trace_curves,
)

type Function = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

HEIGHT, WIDTH = 200, 400
ORIGIN = (40, 30)  # pixel position of the axes, from the bottom-left corner
PIXELS_PER_UNIT = (35, 40)
BLACK, BLUE, ORANGE = (0, 0, 0), (31, 119, 180), (255, 127, 14)
LIGHT_BLUE = (166, 206, 227)


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


def test_a_line_along_the_top_is_not_a_frame() -> None:
    image, calibration = synthetic_plot()
    cv2.line(image, (ORIGIN[0], 10), (WIDTH - 20, 10), BLUE, 2)  # no right side

    (pixels,) = trace_curves(image, [BLUE], calibration.origin)

    assert np.all(pixels[:, 1] > HEIGHT - 12)


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


def test_light_curves_can_be_traced_but_are_not_found() -> None:
    image, calibration = synthetic_plot(curves=[(LIGHT_BLUE, curve), (BLACK, rising)])

    with pytest.raises(ValueError, match="found only 1 of 2 curves"):
        find_colours(image, 2, calibration.origin)
    (pixels,) = trace_curves(image, [LIGHT_BLUE], calibration.origin)
    # The black line is drawn over the curve where they cross.
    assert_follows(pixels, calibration, curve, atol=0.075)


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
