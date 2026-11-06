"""Command-line interface."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from . import __version__, interactive
from .calibration import MIN_POINTS, Calibration, Point
from .digitize import Image, load_image, remove_grid, trace_curve


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
    if len(arguments.points) < MIN_POINTS:
        parser.error(f"give at least {MIN_POINTS} reference points with -p")

    try:
        _digitize(arguments)
    except (OSError, ValueError, cv2.error) as error:
        print(f"figrecover: error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="figrecover",
        description="Recover the curve in an image of a plot as CSV, with "
        "columns x and y.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("image", type=Path, help="image of the plot")
    parser.add_argument(
        "-p",
        "--point",
        dest="points",
        action="append",
        required=True,
        type=_point,
        metavar="X,Y",
        help="data coordinates of a reference point, such as the origin or a "
        f"labelled tick; repeat for at least {MIN_POINTS} points",
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
        help="also save the image with the recovered curve drawn in red",
    )
    parser.add_argument(
        "--remove-grid", action="store_true", help="paint over grid lines first"
    )
    return parser


def _digitize(arguments: argparse.Namespace) -> None:
    """Write the curve in the image as CSV."""
    image = load_image(arguments.image)
    pixels = arguments.locations or _click_locations(image, arguments.points)
    calibration = Calibration.from_points(arguments.points, pixels)
    traced = remove_grid(image) if arguments.remove_grid else image
    curve = trace_curve(traced, calibration.origin)

    if arguments.preview:
        _save_preview(arguments.preview, image, curve, pixels)
    np.savetxt(
        arguments.output or sys.stdout,
        calibration.to_data(curve),
        fmt="%.6g",
        delimiter=",",
        header="x,y",
        comments="",
    )


def _click_locations(image: Image, points: Sequence[Point]) -> list[Point]:
    """Ask the user to click on each point, and print the -l options to reuse."""
    pixels = interactive.pick_points(image, [f"{x:g},{y:g}" for x, y in points])
    options = " ".join(f"-l {x:g},{y:g}" for x, y in pixels)
    print(f"figrecover: locations: {options}", file=sys.stderr)
    return pixels


def _save_preview(
    path: Path,
    image: Image,
    curve: npt.NDArray[np.float64],
    pixels: Sequence[Point],
) -> None:
    """Save the image with the curve in red and the reference points in blue."""
    height = image.shape[0]
    preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in curve:
        cv2.circle(preview, (round(x), round(height - y)), 1, (0, 0, 255), -1)
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
