"""Tests for following error bars in images, case by case."""

import numpy as np

from figrecover.errorbars import _capped, _filled_markers, _line_depth, _run


def test_a_colour_without_pixels_has_lines_of_no_depth() -> None:
    assert _line_depth(np.zeros((5, 5), dtype=np.bool_)) == 0


def test_runs_reach_the_edges() -> None:
    pixels = np.array([True, True, False, True, True])

    assert _run(pixels, slice(0, 1)) == (0, 1)
    assert _run(pixels, slice(4, 5)) == (3, 4)


def test_a_cap_is_no_thicker_than_its_bar_is_wide() -> None:
    # Three wide runs, centred on column 5, end a bar a pixel wide; four do not.
    runs = [(5, 5), (5, 5), (1, 9), (1, 9), (1, 9)]

    assert _capped(runs, 5, 1) == 3
    assert _capped(runs[2:], 5, 1) == 3
    assert _capped([*runs, (1, 9)], 5, 1) == 0


def test_a_marker_barely_thicker_than_its_lines_keeps_its_box() -> None:
    # A line 2 pixels wide, through a marker 5 pixels across, rows 2 to 6 and
    # columns 8 to 12, whose middle is a row of 3 pixels: too thin to narrow
    # by the width of a bar. Widened by its depth, its box is a pixel wider
    # each side than the marker.
    in_colour = np.zeros((9, 20), dtype=np.bool_)
    in_colour[4:6, :] = True
    in_colour[2:7, 8:13] = True

    np.testing.assert_array_equal(_filled_markers(in_colour), [(2, 7, 7, 14)])
