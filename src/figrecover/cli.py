"""Command-line interface."""

import argparse
import csv
import io
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import numpy.typing as npt

from . import __version__, interactive
from .axes import read_axes
from .calibration import MIN_POINTS, Calibration, Point
from .digitize import (
    Colour,
    Image,
    find_colours,
    load_image,
    remove_grid,
    trace_curves,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: The arguments, excluding the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.locations and not arguments.points:
        parser.error("give the reference points of the -l locations with -p")
    if 0 < len(arguments.points) < MIN_POINTS:
        parser.error(f"give at least {MIN_POINTS} reference points with -p")

    try:
        _digitize(arguments)
    except (ImportError, OSError, ValueError, cv2.error) as error:
        print(f"figrecover: error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="figrecover",
        description="Recover the curves in an image of a plot as CSV, with "
        "columns for x and y, and a series column if there are several curves.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("image", type=Path, help="image of the plot")
    parser.add_argument(
        "-p",
        "--point",
        dest="points",
        action="append",
        default=[],
        type=_point,
        metavar="X,Y",
        help="data coordinates of a reference point, such as the origin or a "
        f"labelled tick; repeat for at least {MIN_POINTS} points (default: read "
        "the tick labels, and name the columns after the axis titles; this needs "
        "the ocr extra)",
    )
    parser.add_argument(
        "-l",
        "--location",
        dest="locations",
        action="append",
        default=[],
        type=_point,
        metavar="X,Y",
        help="pixel position of each reference point, in the same order, measured "
        "from the bottom-left corner (default: click on them, then reuse the "
        "printed -l options)",
    )
    curves = parser.add_mutually_exclusive_group()
    curves.add_argument(
        "-n",
        "--series",
        type=_count,
        metavar="N",
        help="trace the N most common curve colours, then reuse the printed -c "
        "options (default: 1)",
    )
    curves.add_argument(
        "-c",
        "--colour",
        dest="colours",
        action="append",
        default=[],
        type=_colour,
        metavar="RRGGBB",
        help="hexadecimal colour of a curve to trace, such as 1f77b4; repeat for "
        "several curves",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="CSV file to write (default: standard output)",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        metavar="IMAGE",
        help="also save a faded copy of the image with each recovered curve "
        "drawn in its colour",
    )
    parser.add_argument(
        "--remove-grid", action="store_true", help="paint over grid lines first"
    )
    return parser


def _digitize(arguments: argparse.Namespace) -> None:
    """Write the curves in the image as CSV."""
    image = load_image(arguments.image)
    if arguments.points:
        pixels = arguments.locations or _click_locations(image, arguments.points)
        calibration = Calibration.from_points(arguments.points, pixels)
        names = ("x", "y")
    else:
        calibration, pixels, names = _read_axes(image)
    traced = remove_grid(image) if arguments.remove_grid else image
    colours = arguments.colours or _find_colours(
        traced, arguments.series, calibration.origin
    )
    curves = trace_curves(traced, colours, calibration.origin)

    if arguments.preview:
        _save_preview(arguments.preview, image, curves, colours, pixels)
    data = [calibration.to_data(curve) for curve in curves]
    _write_csv(arguments.output or sys.stdout, data, names)


def _click_locations(image: Image, points: Sequence[Point]) -> list[Point]:
    """Ask the user to click on each point, and print the -l options to reuse."""
    pixels = interactive.pick_points(image, [f"{x:g},{y:g}" for x, y in points])
    options = " ".join(f"-l {x:g},{y:g}" for x, y in pixels)
    print(f"figrecover: locations: {options}", file=sys.stderr)
    return pixels


def _read_axes(image: Image) -> tuple[Calibration, list[Point], tuple[str, str]]:
    """Read the axes, and print what was read so it can be checked.

    Returns:
        The calibration, the pixel position of each tick on its axis line, and
        the names of the x and y columns: the axis titles, or x and y.
    """
    axes = read_axes(image)
    x_origin, y_origin = axes.calibration.origin
    for name, title, ticks in (
        ("x", axes.x_title, axes.x_ticks),
        ("y", axes.y_title, axes.y_ticks),
    ):
        values = ", ".join(f"{value:g}" for _, value in ticks)
        titled = f' "{title}"' if title else ""
        message = f"figrecover: {name}-axis{titled} with ticks at {values}"
        print(message, file=sys.stderr)
    pixels = [(x, y_origin) for x, _ in axes.x_ticks]
    pixels += [(x_origin, y) for y, _ in axes.y_ticks]
    return axes.calibration, pixels, (axes.x_title or "x", axes.y_title or "y")


def _find_colours(image: Image, count: int | None, origin: Point) -> list[Colour]:
    """Find the curve colours, and print the -c options to reuse if -n was given."""
    colours = find_colours(image, count or 1, origin)
    if count is not None:
        options = " ".join(f"-c {bytes(colour).hex()}" for colour in colours)
        print(f"figrecover: colours: {options}", file=sys.stderr)
    return colours


def _write_csv(
    file: Path | TextIO,
    curves: Sequence[npt.NDArray[np.float64]],
    names: tuple[str, str],
) -> None:
    """Write the curves as CSV, with a series column if there are several."""
    if len(curves) == 1:
        table, columns, fmt = curves[0], [*names], ["%.6g"] * 2
    else:
        table = np.vstack(
            [
                np.column_stack((np.full(len(curve), series), curve))
                for series, curve in enumerate(curves, start=1)
            ]
        )
        columns, fmt = ["series", *names], ["%d", "%.6g", "%.6g"]
    header = io.StringIO()
    csv.writer(header, lineterminator="").writerow(columns)
    np.savetxt(
        file, table, fmt=fmt, delimiter=",", header=header.getvalue(), comments=""
    )


def _save_preview(
    path: Path,
    image: Image,
    curves: Sequence[npt.NDArray[np.float64]],
    colours: Sequence[Colour],
    pixels: Sequence[Point],
) -> None:
    """Save a faded copy of the image with the curves and reference points on it.

    Each curve is drawn in its own colour, and the reference points in blue.
    """
    height = image.shape[0]
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    preview = cv2.cvtColor(255 - (255 - grey) // 3, cv2.COLOR_GRAY2BGR)
    for curve, (red, green, blue) in zip(curves, colours, strict=True):
        for x, y in curve:
            centre = (round(x), round(height - y))
            cv2.circle(preview, centre, 1, (blue, green, red), -1)
    for x, y in pixels:
        cv2.circle(preview, (round(x), round(height - y)), 6, (255, 0, 0), 2)
    if not cv2.imwrite(os.fspath(path), preview, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        msg = f"cannot write image {os.fspath(path)!r}"
        raise OSError(msg)


def _point(text: str) -> Point:
    """Parse a point written as ``x,y``."""
    try:
        x, y = (float(value) for value in text.split(","))
    except ValueError:
        msg = f"expected a point written as x,y, got {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    return x, y


def _count(text: str) -> int:
    """Parse a positive whole number."""
    try:
        count = int(text)
    except ValueError:
        count = 0
    if count < 1:
        msg = f"expected a positive whole number, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return count


def _colour(text: str) -> Colour:
    """Parse a colour written in hexadecimal as ``RRGGBB``, with or without a ``#``."""
    match = re.fullmatch(r"#?([0-9a-f]{6})", text.strip(), re.IGNORECASE)
    if match is None:
        msg = f"expected a colour written as RRGGBB, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    red, green, blue = bytes.fromhex(match[1])
    return red, green, blue
