"""Tests for mapping pixels to data coordinates."""

import numpy as np
import pytest

from figrecover import Calibration, Point, Tick


def test_calibration_maps_reference_points_exactly() -> None:
    data = [(0.0, 0.0), (10.0, 0.0), (0.0, 1.0)]
    pixels = [(80.0, 44.0), (576.0, 44.0), (80.0, 147.0)]
    calibration = Calibration.from_points(data, pixels)

    np.testing.assert_allclose(calibration.to_data(pixels), data, atol=1e-12)
    np.testing.assert_allclose(calibration.to_data((328, 95.5)), [[5.0, 0.5]])
    assert calibration.origin == (80.0, 44.0)


def test_calibration_fits_inexact_points_by_least_squares() -> None:
    data = [(0.0, 0.0), (10.0, 0.0), (0.0, 1.0)]
    pixels = [(80.0, 44.0), (576.0, 45.0), (81.0, 147.0)]
    calibration = Calibration.from_points(data, pixels)

    assert calibration.x_scale == pytest.approx(14865 / 736563)
    assert calibration.y_scale == pytest.approx(205 / 21014)
    assert calibration.origin == (80.0, 44.0)


def test_calibration_handles_reversed_axes() -> None:
    calibration = Calibration.from_points(
        [(0, 0), (10, 0), (0, 5)], [(100, 0), (0, 0), (100, 50)]
    )
    np.testing.assert_allclose(calibration.to_data([(50, 25)]), [[5.0, 2.5]])


def test_calibration_needs_matching_point_counts() -> None:
    with pytest.raises(ValueError, match="3 data points but 2 pixel locations"):
        Calibration.from_points([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0)])


def test_calibration_needs_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        Calibration.from_points([(0, 0), (1, 1)], [(0, 0), (1, 1)])


@pytest.mark.parametrize(
    ("data", "pixels", "axis"),
    [
        ([(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 1), (2, 2)], "x"),
        ([(0, 0), (1, 1), (2, 2)], [(0, 5), (1, 5), (2, 5)], "y"),
    ],
)
def test_calibration_needs_points_spanning_each_axis(
    data: list[Point], pixels: list[Point], axis: str
) -> None:
    with pytest.raises(ValueError, match=f"span the {axis}-axis"):
        Calibration.from_points(data, pixels)


def test_calibration_fits_ticks_on_each_axis() -> None:
    x_ticks = [(80.0, 0.0), (179.0, 2.0), (278.0, 4.0)]
    y_ticks = [(44.0, 0.0), (147.0, 1.0)]
    calibration = Calibration.from_ticks(x_ticks, y_ticks, origin=(70, 40))

    np.testing.assert_allclose(
        calibration.to_data([(80, 44), (278, 147)]), [(0, 0), (4, 1)], atol=1e-12
    )
    assert calibration.origin == (70, 40)


@pytest.mark.parametrize(
    ("x_ticks", "y_ticks", "axis"),
    [
        ([], [(0, 0), (1, 1)], "x"),
        ([(0, 0), (1, 1)], [(5, 0), (5, 1)], "y"),
    ],
)
def test_calibration_needs_ticks_spanning_each_axis(
    x_ticks: list[Tick], y_ticks: list[Tick], axis: str
) -> None:
    with pytest.raises(ValueError, match=f"the ticks must span the {axis}-axis"):
        Calibration.from_ticks(x_ticks, y_ticks, origin=(0, 0))
