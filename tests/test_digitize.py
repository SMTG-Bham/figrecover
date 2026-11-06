"""Tests for reading images and tracing curves."""

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from figrecover import Calibration, Image, load_image, remove_grid, trace_curve

HEIGHT, WIDTH = 200, 400
ORIGIN = (40, 30)  # pixel position of the axes, from the bottom-left corner
PIXELS_PER_UNIT = (35, 40)


def curve(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the function drawn in the synthetic plot."""
    return 1.5 + np.sin(x)


def to_row(y_pixel: float) -> int:
    """Convert a height above the bottom edge to an image row."""
    return round(HEIGHT - y_pixel)


def synthetic_plot(grid: int | None = None) -> tuple[Image, Calibration]:
    """Draw axes and a known curve, and return the image and its calibration.

    Args:
        grid: The grey level of grid lines to draw, if any.
    """
    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    if grid is not None:
        for column in range(60, WIDTH, 50):
            cv2.line(image, (column, 0), (column, HEIGHT), grid, 1)
        for row in range(10, HEIGHT, 50):
            cv2.line(image, (0, row), (WIDTH, row), grid, 1)
    cv2.line(image, (ORIGIN[0], 0), (ORIGIN[0], to_row(ORIGIN[1])), 0, 2)
    cv2.line(image, (ORIGIN[0], to_row(ORIGIN[1])), (WIDTH, to_row(ORIGIN[1])), 0, 2)

    x = np.linspace(0.5, 10, 500)
    columns = ORIGIN[0] + x * PIXELS_PER_UNIT[0]
    heights = ORIGIN[1] + curve(x) * PIXELS_PER_UNIT[1]
    points = np.column_stack((columns, HEIGHT - heights)).round().astype(np.int32)
    cv2.polylines(image, [points], isClosed=False, color=0, thickness=2)

    data = [(0.0, 0.0), (10.0, 0.0), (0.0, 1.0)]
    pixels = [
        ORIGIN,
        (ORIGIN[0] + 10 * PIXELS_PER_UNIT[0], ORIGIN[1]),
        (ORIGIN[0], ORIGIN[1] + PIXELS_PER_UNIT[1]),
    ]
    return image, Calibration.from_points(data, pixels)


def assert_traces_curve(image: Image, calibration: Calibration) -> None:
    """Check that the curve traced in a synthetic plot follows ``curve``."""
    data = calibration.to_data(trace_curve(image, calibration.origin))

    assert len(data) > 300
    assert np.all(np.diff(data[:, 0]) > 0)
    np.testing.assert_allclose(data[:, 1], curve(data[:, 0]), atol=0.05)


def test_trace_curve_recovers_a_known_curve() -> None:
    assert_traces_curve(*synthetic_plot())


def test_trace_curve_ignores_the_axes() -> None:
    image, calibration = synthetic_plot()
    pixels = trace_curve(image, calibration.origin)

    assert pixels[:, 0].min() > ORIGIN[0] + 2
    assert pixels[:, 1].min() > ORIGIN[1] + 2


def test_trace_curve_ignores_a_light_grid() -> None:
    assert_traces_curve(*synthetic_plot(grid=200))


def test_remove_grid_lets_the_curve_be_traced() -> None:
    image, calibration = synthetic_plot(grid=0)
    original = image.copy()

    cleaned = remove_grid(image)

    assert np.array_equal(image, original)  # not modified
    assert np.all(cleaned[10, :] == 255)  # a grid line is gone
    assert_traces_curve(cleaned, calibration)


def test_trace_curve_needs_a_curve() -> None:
    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    image[50:60, 50:60] = 0  # too small to be a curve
    with pytest.raises(ValueError, match="no curve found"):
        trace_curve(image)


def test_trace_curve_needs_a_plot_area() -> None:
    image, _ = synthetic_plot()
    with pytest.raises(ValueError, match="no plot area"):
        trace_curve(image, (WIDTH, 0))


def test_load_image_converts_colour_to_full_range_greyscale(tmp_path: Path) -> None:
    path = tmp_path / "colour.png"
    image = np.full((10, 20, 3), (200, 180, 160), dtype=np.uint8)
    image[5, :] = (40, 50, 60)
    cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (10, 20)
    assert loaded.dtype == np.uint8
    assert loaded.min() == 0
    assert loaded.max() == 255
    assert np.all(loaded[5, :] == 0)


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


def test_load_image_inverts_dark_images(tmp_path: Path) -> None:
    path = tmp_path / "dark.png"
    image, calibration = synthetic_plot()
    cv2.imwrite(str(path), 255 - image)

    loaded = load_image(path)

    assert np.array_equal(loaded, image)
    assert_traces_curve(loaded, calibration)


def test_load_image_reports_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read image"):
        load_image(tmp_path / "missing.png")


def test_load_image_rejects_blank_images(tmp_path: Path) -> None:
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.full((10, 20), 128, dtype=np.uint8))
    with pytest.raises(ValueError, match="single colour"):
        load_image(path)
