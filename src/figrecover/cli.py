"""Command-line interface."""

import argparse
import csv
import io
import math
import os
import re
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import numpy.typing as npt

from . import __version__, interactive
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
from .panels import read_panels

type Ring = tuple[Point, int, Colour]
"""A ring to draw in the preview: its centre, its radius and its colour."""

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


@dataclass(frozen=True, slots=True, eq=False)
class _Plot:
    """A plot to trace: the whole image, or one of its panels.

    Attributes:
        name: The name of the panel, the letter of its label or its number,
            or None if the image is one plot.
        image: The plot's image, in which to trace its curves.
        offset: The pixel position, in the whole image, of the bottom-left
            corner of ``image``.
        calibration: The map from pixels of ``image`` to data.
        pixels: The pixel position of each reference point or tick, to ring
            in the preview.
        columns: The names of the x and y columns: the axis titles, or x and
            y.
    """

    name: str | None
    image: Image
    offset: Point
    calibration: Calibration
    pixels: list[Point]
    columns: tuple[str, str]


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
        description="Recover the curves in an image of a plot as CSV, with "
        "columns for x and y, and a series column if there are several curves. "
        "An image of several plots, such as the panels of a figure, gives a CSV "
        "for each when its axes are read.",
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
        help="CSV file to write (default: standard output); for an image of "
        "several panels, one for each, named after the panel, such as out_a.csv "
        "for -o out.csv (default: standard output, with a panel column)",
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
    """Write the curves in the image, or in each of its panels, as CSV.

    A panel whose curves cannot be traced is left out, with a warning.
    """
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
        plots = [_Plot(None, image, (0, 0), calibration, pixels, ("x", "y"))]
    else:
        plots = _read_panels(image)
    if arguments.markers and len(plots) > 1:
        msg = (
            f"-m finds the markers of one plot, but the image has {len(plots)} "
            "panels; crop it to one"
        )
        raise ValueError(msg)

    tables: list[tuple[_Plot, list[npt.NDArray[np.float64]]]] = []
    curves: list[npt.NDArray[np.float64]] = []
    colours: list[Colour] = []
    rings: list[Ring] = []
    for plot in plots:
        try:
            traced, traced_colours, marked = _trace(plot, arguments)
        except ValueError as error:
            if plot.name is None:
                raise
            message = f"left out panel {plot.name}, as its curves cannot be traced"
            print(f"figrecover: {message}: {error}", file=sys.stderr)
            continue
        tables.append((plot, [plot.calibration.to_data(curve) for curve in traced]))
        x, y = plot.offset
        curves += [curve + np.array([x, y]) for curve in traced]
        colours += traced_colours
        rings += [
            ((cx + x, cy + y), radius, colour) for (cx, cy), radius, colour in marked
        ]
    if not tables:
        msg = f"cannot trace the curves of any of the {len(plots)} panels"
        raise ValueError(msg)

    if arguments.preview:
        _save_preview(arguments.preview, image, curves, colours, rings)
    _write(arguments.output, tables)


def _trace(
    plot: _Plot, arguments: argparse.Namespace
) -> tuple[list[npt.NDArray[np.float64]], list[Colour], list[Ring]]:
    """Trace the curves of a plot, by colour or by markers.

    Returns:
        The pixel positions of the points of each curve, its colour in the
        preview, and the rings to draw in the preview around the reference
        points or ticks, and the markers and legend keys.
    """
    traced = remove_grid(plot.image) if arguments.remove_grid else plot.image
    origin = plot.calibration.origin
    rings: list[Ring] = [(pixel, 6, (0, 0, 255)) for pixel in plot.pixels]
    if arguments.markers:
        curves = _find_markers(traced, arguments.markers, origin)
        colours = [
            SERIES_COLOURS[series % len(SERIES_COLOURS)]
            for series in range(len(curves))
        ]
        for key, curve, colour in zip(arguments.markers, curves, colours, strict=True):
            rings += [(key, 6, colour), *((marker, 4, colour) for marker in curve)]
    else:
        colours = arguments.colours or _find_colours(
            traced, arguments.series, origin, _prefix(plot.name)
        )
        curves = trace_curves(traced, colours, origin)
    return curves, colours, rings


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


def _read_panels(image: Image) -> list[_Plot]:
    """Read the axes of the image's panels, and print them so they can be checked.

    Panels left out, as their axes cannot be read, are reported too. An
    image of one plot is one panel, without a name. Each panel's pixel
    positions to ring in the preview are those of its ticks, on their axis
    lines, and its columns are named after its axis titles.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            panels = read_panels(image)
        finally:
            for warning in caught:
                print(f"figrecover: {warning.message}", file=sys.stderr)
    plots = []
    for number, panel in enumerate(panels, start=1):
        name = (panel.label or str(number)) if len(panels) > 1 else None
        axes = panel.axes
        for axis, title, ticks, log in (
            ("x", axes.x_title, axes.x_ticks, axes.calibration.x_log),
            ("y", axes.y_title, axes.y_ticks, axes.calibration.y_log),
        ):
            values = ", ".join(f"{value:g}" for _, value in ticks)
            scale = "logarithmic " if log else ""
            titled = f' "{title}"' if title else ""
            message = (
                f"{_prefix(name)}{scale}{axis}-axis{titled} with ticks at {values}"
            )
            print(f"figrecover: {message}", file=sys.stderr)
        x_origin, y_origin = axes.calibration.origin
        pixels = [(x, y_origin) for x, _ in axes.x_ticks]
        pixels += [(x_origin, y) for y, _ in axes.y_ticks]
        columns = (axes.x_title or "x", axes.y_title or "y")
        plots.append(
            _Plot(name, panel.image, panel.offset, axes.calibration, pixels, columns)
        )
    return plots


def _prefix(name: str | None) -> str:
    """Return the start of a message about a panel, naming it if it has a name."""
    return f"panel {name}: " if name else ""


def _find_colours(
    image: Image, count: int | None, origin: Point, prefix: str
) -> list[Colour]:
    """Find the curve colours, and print the -c options to reuse if -n was given.

    ``prefix`` starts the message, naming the panel.
    """
    colours = find_colours(image, count or 1, origin)
    if count is not None:
        options = " ".join(f"-c {bytes(colour).hex()}" for colour in colours)
        print(f"figrecover: {prefix}colours: {options}", file=sys.stderr)
    return colours


def _find_markers(
    image: Image, keys: Sequence[Point], origin: Point
) -> list[npt.NDArray[np.float64]]:
    """Find the markers of each series, and print how many were found."""
    markers = find_markers(image, keys, origin)
    counts = ", ".join(str(len(series)) for series in markers)
    print(f"figrecover: markers found in each series: {counts}", file=sys.stderr)
    return markers


def _write(
    output: Path | None,
    tables: Sequence[tuple[_Plot, Sequence[npt.NDArray[np.float64]]]],
) -> None:
    """Write the curves of each plot as CSV, to the output file or standard output.

    The curves of an image of several panels go to a file for each panel,
    named after it, such as out_a.csv for out.csv, or to standard output
    with a panel column, whose x and y columns are named after the axis
    titles if every panel's are the same, or x and y.
    """
    if tables[0][0].name is None:
        plot, curves = tables[0]
        _write_csv(output or sys.stdout, [(None, curves)], plot.columns)
    elif output is not None:
        paths = [output.with_stem(f"{output.stem}_{plot.name}") for plot, _ in tables]
        for path, (plot, curves) in zip(paths, tables, strict=True):
            _write_csv(path, [(None, curves)], plot.columns)
        print(f"figrecover: wrote {', '.join(map(str, paths))}", file=sys.stderr)
    else:
        columns = {plot.columns for plot, _ in tables}
        names = columns.pop() if len(columns) == 1 else ("x", "y")
        _write_csv(sys.stdout, [(plot.name, curves) for plot, curves in tables], names)


def _write_csv(
    file: Path | TextIO,
    tables: Sequence[tuple[str | None, Sequence[npt.NDArray[np.float64]]]],
    names: tuple[str, str],
) -> None:
    """Write curves as CSV, from tables of curves named after their panels.

    There is a panel column if the tables are named, rather than one named
    None for an image of one plot, and a series column if a table has
    several curves.
    """
    series = any(len(curves) > 1 for _, curves in tables)
    named = tables[0][0] is not None
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow([*(["panel"] * named), *(["series"] * series), *names])
    for name, curves in tables:
        for number, curve in enumerate(curves, start=1):
            start = [*([name] if name is not None else []), *[str(number)] * series]
            writer.writerows([*start, f"{x:.6g}", f"{y:.6g}"] for x, y in curve)
    if isinstance(file, Path):
        file.write_text(text.getvalue(), encoding="utf-8")
    else:
        file.write(text.getvalue())


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
