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


def test_calibration_maps_logarithmic_axes() -> None:
    data = [(1.0, 0.01), (1e4, 0.01), (1.0, 100.0)]
    pixels = [(80.0, 44.0), (576.0, 44.0), (80.0, 147.0)]
    calibration = Calibration.from_points(data, pixels, x_log=True, y_log=True)

    np.testing.assert_allclose(calibration.to_data(pixels), data)
    np.testing.assert_allclose(calibration.to_data((328, 95.5)), [[100.0, 1.0]])
    assert calibration.x_scale == pytest.approx(4 / 496)  # decades per pixel
    assert calibration.origin == (80.0, 44.0)


def test_calibration_maps_one_logarithmic_axis() -> None:
    x_ticks = [(80.0, 0.0), (576.0, 10.0)]
    y_ticks = [(44.0, 1.0), (95.5, 10.0), (147.0, 100.0)]
    calibration = Calibration.from_ticks(x_ticks, y_ticks, (80, 44), y_log=True)

    np.testing.assert_allclose(calibration.to_data([(328, 121.25)]), [[5.0, 10**1.5]])
    assert not calibration.x_log
    assert calibration.y_log


@pytest.mark.parametrize("axis", ["x", "y"])
def test_calibration_needs_positive_points_on_logarithmic_axes(axis: str) -> None:
    data = [(0, 0), (10, 0), (0, 1)]
    pixels = [(80, 44), (576, 44), (80, 147)]

    with pytest.raises(ValueError, match=f"on the logarithmic {axis}-axis must be"):
        Calibration.from_points(data, pixels, x_log=axis == "x", y_log=axis == "y")


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
