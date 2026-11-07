"""Command-line interface."""

import argparse
import csv
import io
import math
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import numpy.typing as npt

from . import __version__, interactive
from .axes import Axes, read_axes
from .calibration import MIN_POINTS, Calibration, Point
from .digitize import (
    Colour,
    Image,
    find_colours,
    find_markers,
    load_image,
    remove_grid,
    trace_curves,
)
from .pdf import (
    PREVIEW_SCALE,
    Box,
    Drawing,
    find_pdf_colours,
    load_pdf,
    read_pdf_axes,
    remove_pdf_grid,
    render_pdf,
    trace_pdf_curves,
)

SERIES_COLOURS: list[Colour] = [  # matplotlib's, to draw series of markers in
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


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
    if arguments.log and not arguments.points:
        parser.error(
            "--log needs -p; figrecover finds logarithmic axes itself when reading them"
        )
    if _is_pdf(arguments.image) and arguments.markers:
        parser.error("-m finds markers in images, not in PDF files")
    if not _is_pdf(arguments.image) and (arguments.page or arguments.box):
        parser.error("--page and --box choose what to read in a PDF file")
    outputs = {"-o": arguments.output, "--preview": arguments.preview}
    for option, path in outputs.items():
        if path is not None and path.resolve() == arguments.image.resolve():
            parser.error(f"{option} would overwrite the image")
    if (
        None not in outputs.values()
        and len({p.resolve() for p in outputs.values()}) == 1
    ):
        parser.error("-o and --preview name the same file")

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
        description="Recover the curves in an image of a plot, or a PDF file, as "
        "CSV, with columns for x and y, and a series column if there are several "
        "curves.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "image",
        type=Path,
        help="image of the plot, or a PDF file, whose curves are read exactly "
        "from its drawing (this needs the pdf extra); positions in a PDF are in "
        "points from the bottom-left corner of the page",
    )
    parser.add_argument(
        "--page",
        type=_count,
        metavar="N",
        help="the page of the PDF file to read, from 1 (default: 1)",
    )
    parser.add_argument(
        "--box",
        type=_box,
        metavar="X0,Y0,X1,Y1",
        help="the left, bottom, right and top edges of the part of the PDF's page "
        "to read, such as one figure, in points (default: the whole page)",
    )
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
    parser.add_argument(
        "--log",
        choices=["x", "y", "xy"],
        default="",
        help="the axes of the -p reference points that are logarithmic (default: "
        "neither)",
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
    curves.add_argument(
        "-m",
        "--marker",
        dest="markers",
        action="append",
        default=[],
        type=_point,
        metavar="X,Y",
        help="pixel position of the marker in a series' legend key, measured from "
        "the bottom-left corner; repeat for each series, to find the series by "
        "the shapes of their markers rather than by colour",
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
        "drawn in its colour, or each series of markers in a colour of its own",
    )
    parser.add_argument(
        "--remove-grid", action="store_true", help="paint over grid lines first"
    )
    return parser


def _digitize(arguments: argparse.Namespace) -> None:
    """Write the curves in the image, or in the PDF file, as CSV."""
    if _is_pdf(arguments.image):
        _digitize_pdf(arguments)
        return
    image = load_image(arguments.image)
    _check_locations(arguments.markers, image, "-m position")
    if arguments.points:
        _check_locations(arguments.locations, image, "-l location")
        pixels = arguments.locations or _click_locations(image, arguments.points)
        calibration = Calibration.from_points(
            arguments.points,
            pixels,
            x_log="x" in arguments.log,
            y_log="y" in arguments.log,
        )
        names = ("x", "y")
    else:
        calibration, pixels, names = _read_axes(read_axes(image))
    traced = remove_grid(image) if arguments.remove_grid else image
    rings: list[tuple[Point, int, Colour]] = [
        (pixel, 6, (0, 0, 255)) for pixel in pixels
    ]
    if arguments.markers:
        curves = _find_markers(traced, arguments.markers, calibration.origin)
        colours = [
            SERIES_COLOURS[series % len(SERIES_COLOURS)]
            for series in range(len(curves))
        ]
        for key, curve, colour in zip(arguments.markers, curves, colours, strict=True):
            rings += [(key, 6, colour), *((marker, 4, colour) for marker in curve)]
    else:
        colours = arguments.colours or _find_colours(
            lambda count: find_colours(traced, count, calibration.origin),
            arguments.series,
        )
        curves = trace_curves(traced, colours, calibration.origin)

    if arguments.preview:
        _save_preview(arguments.preview, image, curves, colours, rings)
    data = [calibration.to_data(curve) for curve in curves]
    _write_csv(arguments.output or sys.stdout, data, names)


def _digitize_pdf(arguments: argparse.Namespace) -> None:
    """Write the curves in a plot on a page of the PDF file as CSV.

    Positions on the page are in points, and the preview shows the part of
    the page read, ``PREVIEW_SCALE`` pixels to the point.
    """
    drawing = load_pdf(arguments.image, arguments.page or 1, arguments.box)
    if arguments.remove_grid:
        drawing = remove_pdf_grid(drawing)
    if arguments.points:
        _check_positions(arguments.locations, drawing.box)
        positions = arguments.locations or _click_positions(drawing, arguments.points)
        calibration = Calibration.from_points(
            arguments.points,
            positions,
            x_log="x" in arguments.log,
            y_log="y" in arguments.log,
        )
        names = ("x", "y")
    else:
        calibration, positions, names = _read_axes(read_pdf_axes(drawing))
    colours = arguments.colours or _find_colours(
        lambda count: find_pdf_colours(drawing, count, calibration.origin),
        arguments.series,
    )
    curves = trace_pdf_curves(drawing, colours, calibration.origin)

    if arguments.preview:
        corner = np.array(drawing.box[:2])
        pixels = [(curve - corner) * PREVIEW_SCALE for curve in curves]
        rings: list[tuple[Point, int, Colour]] = [
            ((x, y), 6, (0, 0, 255))
            for x, y in (np.array(positions) - corner) * PREVIEW_SCALE
        ]
        image = render_pdf(drawing, PREVIEW_SCALE)
        _save_preview(arguments.preview, image, pixels, colours, rings)
    data = [calibration.to_data(curve) for curve in curves]
    _write_csv(arguments.output or sys.stdout, data, names)


def _is_pdf(path: Path) -> bool:
    """Return whether a file is a PDF file, by its name."""
    return path.suffix.lower() == ".pdf"


def _check_positions(positions: Sequence[Point], box: Box) -> None:
    """Check that each -l position lies on the part of a PDF's page that is read."""
    left, bottom, right, top = box
    for x, y in positions:
        if not (left <= x <= right and bottom <= y <= top):
            edges = ",".join(f"{edge:g}" for edge in box)
            msg = f"the -l location {x:g},{y:g} lies outside the page read, {edges}"
            raise ValueError(msg)


def _click_positions(drawing: Drawing, points: Sequence[Point]) -> list[Point]:
    """Ask the user to click on each point on a PDF's page, and print the -l options.

    The page is shown ``PREVIEW_SCALE`` pixels to the point.
    """
    pixels = interactive.pick_points(
        render_pdf(drawing, PREVIEW_SCALE), [f"{x:g},{y:g}" for x, y in points]
    )
    left, bottom = drawing.box[:2]
    positions = [
        (left + x / PREVIEW_SCALE, bottom + y / PREVIEW_SCALE) for x, y in pixels
    ]
    options = " ".join(f"-l {x:g},{y:g}" for x, y in positions)
    print(f"figrecover: locations: {options}", file=sys.stderr)
    return positions


def _check_locations(locations: Sequence[Point], image: Image, name: str) -> None:
    """Check that each pixel position, such as an -l location, lies on the image."""
    height, width = image.shape[:2]
    for x, y in locations:
        if not (0 <= x <= width and 0 <= y <= height):
            msg = f"the {name} {x:g},{y:g} lies outside the {width}x{height} image"
            raise ValueError(msg)


def _click_locations(image: Image, points: Sequence[Point]) -> list[Point]:
    """Ask the user to click on each point, and print the -l options to reuse."""
    pixels = interactive.pick_points(image, [f"{x:g},{y:g}" for x, y in points])
    options = " ".join(f"-l {x:g},{y:g}" for x, y in pixels)
    print(f"figrecover: locations: {options}", file=sys.stderr)
    return pixels


def _read_axes(axes: Axes) -> tuple[Calibration, list[Point], tuple[str, str]]:
    """Print the axes that were read, so they can be checked.

    Returns:
        The calibration, the position of each tick on its axis line, and the
        names of the x and y columns: the axis titles, or x and y.
    """
    x_origin, y_origin = axes.calibration.origin
    for name, title, ticks, log in (
        ("x", axes.x_title, axes.x_ticks, axes.calibration.x_log),
        ("y", axes.y_title, axes.y_ticks, axes.calibration.y_log),
    ):
        values = ", ".join(f"{value:g}" for _, value in ticks)
        scale = "logarithmic " if log else ""
        titled = f' "{title}"' if title else ""
        message = f"figrecover: {scale}{name}-axis{titled} with ticks at {values}"
        print(message, file=sys.stderr)
    pixels = [(x, y_origin) for x, _ in axes.x_ticks]
    pixels += [(x_origin, y) for y, _ in axes.y_ticks]
    return axes.calibration, pixels, (axes.x_title or "x", axes.y_title or "y")


def _find_colours(
    find: Callable[[int], list[Colour]], count: int | None
) -> list[Colour]:
    """Find the curve colours, and print the -c options to reuse if -n was given.

    ``find`` finds the colours of a given number of curves.
    """
    colours = find(count or 1)
    if count is not None:
        options = " ".join(f"-c {bytes(colour).hex()}" for colour in colours)
        print(f"figrecover: colours: {options}", file=sys.stderr)
    return colours


def _find_markers(
    image: Image, keys: Sequence[Point], origin: Point
) -> list[npt.NDArray[np.float64]]:
    """Find the markers of each series, and print how many were found."""
    markers = find_markers(image, keys, origin)
    counts = ", ".join(str(len(series)) for series in markers)
    print(f"figrecover: markers found in each series: {counts}", file=sys.stderr)
    return markers


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
    rings: Sequence[tuple[Point, int, Colour]],
) -> None:
    """Save a faded copy of the image with the curves and reference points on it.

    Each curve is drawn in its own colour, and rings of the given radii
    around points, such as the reference points, in theirs.
    """
    height = image.shape[0]
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    preview = cv2.cvtColor(255 - (255 - grey) // 3, cv2.COLOR_GRAY2BGR)
    for curve, (red, green, blue) in zip(curves, colours, strict=True):
        for x, y in curve:
            centre = (round(x), round(height - y))
            cv2.circle(preview, centre, 1, (blue, green, red), -1)
    for (x, y), radius, (red, green, blue) in rings:
        centre = (round(x), round(height - y))
        cv2.circle(preview, centre, radius, (blue, green, red), radius // 3)
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
    if not (math.isfinite(x) and math.isfinite(y)):
        msg = f"expected finite numbers in the point {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return x, y


def _box(text: str) -> Box:
    """Parse a box written as ``x0,y0,x1,y1``, its left, bottom, right and top."""
    try:
        left, bottom, right, top = (float(value) for value in text.split(","))
    except ValueError:
        msg = f"expected a box written as x0,y0,x1,y1, got {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if not all(map(math.isfinite, (left, bottom, right, top))):
        msg = f"expected finite numbers in the box {text!r}"
        raise argparse.ArgumentTypeError(msg)
    if left >= right or bottom >= top:
        msg = f"expected a box whose x1 and y1 exceed its x0 and y0, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return left, bottom, right, top


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
