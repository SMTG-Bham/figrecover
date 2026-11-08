"""Command-line interface."""

import argparse
import contextlib
import csv
import io
import math
import os
import re
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import numpy.typing as npt

from . import __version__, interactive
from .axes import Axes
from .calibration import MIN_POINTS, Calibration, Point
from .digitize import (
    Colour,
    Image,
    _greyscale,
    find_colours,
    find_markers,
    load_image,
    remove_grid,
    trace_curves,
)
from .level import LEVEL_THRESHOLD, Levelling, level_image
from .panels import Panel, read_panels
from .pdf import (
    PREVIEW_SCALE,
    Box,
    Drawing,
    find_pdf_colours,
    find_pdf_markers,
    load_pdf,
    read_pdf_axes,
    remove_pdf_grid,
    render_pdf,
    trace_pdf_curves,
)
from .svg import load_svg, render_svg

type Ring = tuple[Point, int, Colour]
"""A ring to draw in the preview: its centre, its radius and its colour."""

_SNIFF = 4096  # a file that is not named as one is an SVG file by this many bytes

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
            or None if the image is one plot; an inset's is named after its
            panel, such as a_inset, or inset in an image of one plot.
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
    if _is_svg(arguments.image) and arguments.page:
        parser.error("--page chooses a page of a PDF file; an SVG file has only one")
    if not _is_drawing(arguments.image) and (arguments.page or arguments.box):
        parser.error("--page and --box choose what to read in a PDF or SVG file")
    if _is_drawing(arguments.image) and arguments.level is not None:
        parser.error("--level and --no-level level images, not PDF or SVG files")
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
        description="Recover the curves in an image of a plot, or a PDF or SVG file, "
        "as CSV, with columns for x and y, and a series column if there are several "
        "curves. An image of several plots, such as the panels of a figure, gives "
        "a CSV for each when its axes are read.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "image",
        type=Path,
        help="image of the plot, or a PDF or SVG file, whose curves are read "
        "exactly from its drawing (a PDF file needs the pdf extra); positions in "
        "them are in points from the bottom-left corner of the page",
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
        help="the left, bottom, right and top edges of the part of the PDF's page, "
        "or the SVG's drawing, to read, such as one figure, in points (default: the "
        "whole page)",
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
        help="CSV file to write (default: standard output); for an image of "
        "several panels, or with insets, one for each, named after the panel, "
        "such as out_a.csv or out_inset.csv for -o out.csv (default: standard "
        "output, with a panel column)",
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
    levelling = parser.add_mutually_exclusive_group()
    levelling.add_argument(
        "--level",
        type=_angle,
        metavar="DEGREES",
        help="the angle by which the image's axes are turned anticlockwise, or "
        "clockwise if negative, to turn it back by before reading it; -l and -m "
        "positions are in the image as given (default: find the angle from the "
        "image's long straight lines, and turn the image back if it is turned "
        f"{LEVEL_THRESHOLD:g}° or more)",
    )
    levelling.add_argument(
        "--no-level",
        dest="level",
        action="store_const",
        const=0.0,
        help="read the image as it is, for a plot drawn turned on purpose",
    )
    return parser


def _digitize(arguments: argparse.Namespace) -> None:
    """Write the curves in the image, each of its panels or the PDF file, as CSV.

    A panel whose curves cannot be traced is left out, with a warning. With
    ``-m``, insets are left out, as the markers are those of one plot.

    An image whose axes are turned is levelled before it is read, as
    ``_level`` levels it, and the -l and -m positions, in the image as
    given, are turned with it. The preview is drawn on the image as given,
    with the curves and rings turned back onto it.
    """
    if _is_drawing(arguments.image):
        _digitize_pdf(arguments)
        return
    given = load_image(arguments.image)
    _check_locations(arguments.markers, given, "-m position")
    _check_locations(arguments.locations, given, "-l location")
    if max(arguments.series or 1, len(arguments.colours)) > 1 and _greyscale(given):
        message = "the image is greyscale, so curves are told apart by grey level only"
        print(f"figrecover: {message}", file=sys.stderr)
    locations = arguments.locations
    if arguments.points and not locations:
        locations = _click_locations(given, arguments.points)
    image, levelling = _level(given, arguments.level)
    # From here on, -m positions are those in the levelled image.
    arguments.markers = _as_points(levelling.points(arguments.markers))
    if arguments.points:
        pixels = _as_points(levelling.points(locations))
        calibration = Calibration.from_points(
            arguments.points,
            pixels,
            x_log="x" in arguments.log,
            y_log="y" in arguments.log,
        )
        plots = [_Plot(None, image, (0, 0), calibration, pixels, ("x", "y"))]
    else:
        plots = _read_panels(image, insets=not arguments.markers)
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
            if len(plots) == 1:
                raise
            called = f"panel {plot.name}" if plot.name else "the plot"
            message = f"left out {called}, as its curves cannot be traced"
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
        # Drawn on the image as given, so positions are turned back to it.
        curves = [levelling.back(curve) for curve in curves]
        centres = _as_points(levelling.back([centre for centre, _, _ in rings]))
        rings = [
            (centre, radius, colour)
            for centre, (_, radius, colour) in zip(centres, rings, strict=True)
        ]
        _save_preview(arguments.preview, given, curves, colours, rings)
    _write(arguments.output, tables)


def _level(image: Image, angle: float | None) -> tuple[Image, Levelling]:
    """Level an image whose axes are turned, and say so, as ``level_image`` does.

    ``angle`` is that given with --level, 0 with --no-level, or None to find
    it.
    """
    levelled, levelling = level_image(image, angle)
    if levelling.angle:
        turn = "clockwise" if levelling.angle > 0 else "anticlockwise"
        message = (
            f"turned the image {abs(levelling.angle):.2f}° {turn} to level its axes"
        )
        print(f"figrecover: {message}", file=sys.stderr)
    return levelled, levelling


def _as_points(positions: npt.NDArray[np.float64]) -> list[Point]:
    """Return an ``(n, 2)`` array of positions as a list of points."""
    return [(float(x), float(y)) for x, y in positions]


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
        curves = _counted(find_markers(traced, arguments.markers, origin))
        colours = _series_colours(len(curves))
        rings += _marker_rings(arguments.markers, curves, colours)
    else:
        colours = arguments.colours or _find_colours(
            lambda count: find_colours(traced, count, origin),
            arguments.series,
            _prefix(plot.name),
        )
        with _warnings_printed(_prefix(plot.name)):
            curves = trace_curves(traced, colours, origin)
    return curves, colours, rings


@contextlib.contextmanager
def _warnings_printed(prefix: str) -> Iterator[None]:
    """Print the warnings given in the block, as figrecover's messages.

    ``prefix`` starts each message, naming the panel.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for warning in caught:
                print(f"figrecover: {prefix}{warning.message}", file=sys.stderr)


def _digitize_pdf(arguments: argparse.Namespace) -> None:
    """Write the curves in a plot on a page of the PDF file, or the SVG file, as CSV.

    Positions on the page are in points, and the preview shows the part of
    the page read, ``PREVIEW_SCALE`` pixels to the point.
    """
    drawing = (
        load_svg(arguments.image, arguments.box)
        if _is_svg(arguments.image)
        else load_pdf(arguments.image, arguments.page or 1, arguments.box)
    )
    _check_positions(arguments.markers, drawing.box, "-m position")
    if arguments.remove_grid:
        drawing = remove_pdf_grid(drawing)
    if arguments.points:
        _check_positions(arguments.locations, drawing.box, "-l location")
        positions = arguments.locations or _click_positions(drawing, arguments.points)
        calibration = Calibration.from_points(
            arguments.points,
            positions,
            x_log="x" in arguments.log,
            y_log="y" in arguments.log,
        )
        names = ("x", "y")
    else:
        calibration, positions, names = _read_axes(read_pdf_axes(drawing), None)
    if arguments.markers:
        curves = _counted(
            find_pdf_markers(drawing, arguments.markers, calibration.origin)
        )
        colours = _series_colours(len(curves))
    else:
        colours = arguments.colours or _find_colours(
            lambda count: find_pdf_colours(drawing, count, calibration.origin),
            arguments.series,
            "",
        )
        with _warnings_printed(""):
            curves = trace_pdf_curves(drawing, colours, calibration.origin)

    if arguments.preview:
        corner = np.array(drawing.box[:2])
        pixels = [(curve - corner) * PREVIEW_SCALE for curve in curves]
        rings: list[Ring] = [
            ((x, y), 6, (0, 0, 255))
            for x, y in (np.array(positions) - corner) * PREVIEW_SCALE
        ]
        if arguments.markers:
            keys = (np.array(arguments.markers) - corner) * PREVIEW_SCALE
            rings += _marker_rings([(x, y) for x, y in keys], pixels, colours)
        image = _render(drawing)
        _save_preview(arguments.preview, image, pixels, colours, rings)
    data = [calibration.to_data(curve) for curve in curves]
    _write_csv(arguments.output or sys.stdout, [(None, data)], names)


def _is_pdf(path: Path) -> bool:
    """Return whether a file is a PDF file, by its name."""
    return path.suffix.lower() == ".pdf"


def _is_svg(path: Path) -> bool:
    """Return whether a file is an SVG file, by its name, or else its start.

    A file named neither .pdf nor .svg, .svgz or .svg.gz is an SVG file if it
    starts with a tag and an ``<svg`` tag starts within ``_SNIFF`` bytes.
    """
    suffix = path.suffix.lower()
    if path.name.lower().endswith((".svg", ".svgz", ".svg.gz")):
        return True
    try:
        with path.open("rb") as file:
            start = file.read(_SNIFF)
    except OSError:
        return False
    return suffix != ".pdf" and start.lstrip().startswith(b"<") and b"<svg" in start


def _is_drawing(path: Path) -> bool:
    """Return whether a file is read from its drawing: a PDF or SVG file."""
    return _is_pdf(path) or _is_svg(path)


def _render(drawing: Drawing) -> Image:
    """Draw the part of the page of a PDF or SVG file read, as for a preview.

    It is drawn ``PREVIEW_SCALE`` pixels to the point.
    """
    if _is_svg(Path(drawing.path)):
        return render_svg(drawing, PREVIEW_SCALE)
    return render_pdf(drawing, PREVIEW_SCALE)


def _check_positions(positions: Sequence[Point], box: Box, name: str) -> None:
    """Check that each position, such as an -l location, lies on the page read."""
    left, bottom, right, top = box
    for x, y in positions:
        if not (left <= x <= right and bottom <= y <= top):
            edges = ",".join(f"{edge:g}" for edge in box)
            msg = f"the {name} {x:g},{y:g} lies outside the page read, {edges}"
            raise ValueError(msg)


def _click_positions(drawing: Drawing, points: Sequence[Point]) -> list[Point]:
    """Ask the user to click on each point on a PDF's page, and print the -l options.

    The page is shown ``PREVIEW_SCALE`` pixels to the point.
    """
    pixels = interactive.pick_points(
        _render(drawing), [f"{x:g},{y:g}" for x, y in points]
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


def _read_panels(image: Image, *, insets: bool) -> list[_Plot]:
    """Read the axes of the image's panels, and print them so they can be checked.

    Panels left out, as their axes cannot be read, are reported too. An
    image of one plot is one panel, without a name, as ``_names`` names
    them. Insets are left out unless ``insets`` is true. Each panel's pixel
    positions to ring in the preview are those of its ticks, on their axis
    lines, and its columns are named after its axis titles.
    """
    with _warnings_printed(""):
        panels = read_panels(image)
    if not insets:
        panels = [panel for panel in panels if panel.inset_of is None]
    plots = []
    for panel, name in zip(panels, _names(panels), strict=True):
        calibration, pixels, columns = _read_axes(panel.axes, name)
        plots.append(
            _Plot(name, panel.image, panel.offset, calibration, pixels, columns)
        )
    return plots


def _names(panels: Sequence[Panel]) -> list[str | None]:
    """Return the name of each panel: the letter of its label or its number.

    Panels are numbered in reading order, and the only one has no name. An
    inset is named after its panel, such as a_inset, or inset for the only
    panel's, and its number, such as a_inset2, if the panel has several.
    """
    plots = [panel for panel in panels if panel.inset_of is None]
    names: dict[Panel, str | None] = {
        panel: (panel.label or str(number)) if len(plots) > 1 else None
        for number, panel in enumerate(plots, start=1)
    }
    for panel in panels:
        if panel.inset_of is not None:
            insets = [other for other in panels if other.inset_of is panel.inset_of]
            number = str(insets.index(panel) + 1) if len(insets) > 1 else ""
            plot = names[panel.inset_of]
            names[panel] = f"{plot}_inset{number}" if plot else f"inset{number}"
    return [names[panel] for panel in panels]


def _read_axes(
    axes: Axes, name: str | None
) -> tuple[Calibration, list[Point], tuple[str, str]]:
    """Print the axes that were read, of the panel so named, so they can be checked.

    Returns:
        The calibration, the position of each tick on its axis line, and the
        names of the x and y columns: the axis titles, or x and y.
    """
    for axis, title, ticks, log in (
        ("x", axes.x_title, axes.x_ticks, axes.calibration.x_log),
        ("y", axes.y_title, axes.y_ticks, axes.calibration.y_log),
    ):
        values = ", ".join(f"{value:g}" for _, value in ticks)
        scale = "logarithmic " if log else ""
        titled = f' "{title}"' if title else ""
        message = f"{_prefix(name)}{scale}{axis}-axis{titled} with ticks at {values}"
        print(f"figrecover: {message}", file=sys.stderr)
    x_origin, y_origin = axes.calibration.origin
    positions = [(x, y_origin) for x, _ in axes.x_ticks]
    positions += [(x_origin, y) for y, _ in axes.y_ticks]
    return axes.calibration, positions, (axes.x_title or "x", axes.y_title or "y")


def _prefix(name: str | None) -> str:
    """Return the start of a message about a panel, naming it if it has a name."""
    return f"panel {name}: " if name else ""


def _find_colours(
    find: Callable[[int], list[Colour]], count: int | None, prefix: str
) -> list[Colour]:
    """Find the curve colours, and print the -c options to reuse if -n was given.

    ``find`` finds the colours of a given number of curves, and ``prefix``
    starts the message, naming the panel.
    """
    colours = find(count or 1)
    if count is not None:
        options = " ".join(f"-c {bytes(colour).hex()}" for colour in colours)
        print(f"figrecover: {prefix}colours: {options}", file=sys.stderr)
    return colours


def _counted(markers: list[npt.NDArray[np.float64]]) -> list[npt.NDArray[np.float64]]:
    """Print how many markers were found in each series, and return them."""
    counts = ", ".join(str(len(series)) for series in markers)
    print(f"figrecover: markers found in each series: {counts}", file=sys.stderr)
    return markers


def _series_colours(count: int) -> list[Colour]:
    """Return a colour of its own for each of a number of series of markers."""
    return [SERIES_COLOURS[series % len(SERIES_COLOURS)] for series in range(count)]


def _marker_rings(
    keys: Sequence[Point],
    markers: Sequence[npt.NDArray[np.float64]],
    colours: Sequence[Colour],
) -> list[Ring]:
    """Return rings around each series' key and markers, in its colour, to preview."""
    rings: list[Ring] = []
    for key, series, colour in zip(keys, markers, colours, strict=True):
        rings += [(key, 6, colour), *((marker, 4, colour) for marker in series)]
    return rings


def _write(
    output: Path | None,
    tables: Sequence[tuple[_Plot, Sequence[npt.NDArray[np.float64]]]],
) -> None:
    """Write the curves of each plot as CSV, to the output file or standard output.

    The curves of an image of several panels, or with insets, go to a file
    for each panel, named after it, such as out_a.csv for out.csv, or
    out.csv itself for a panel without a name, as the plot that holds the
    insets in an image of one plot is; or to standard output with a panel
    column, empty for a panel without a name, whose x and y columns are
    named after the axis titles if every panel's are the same, or x and y.
    """
    if len(tables) == 1 and tables[0][0].name is None:
        plot, curves = tables[0]
        _write_csv(output or sys.stdout, [(None, curves)], plot.columns)
    elif output is not None:
        paths = [
            output.with_stem(f"{output.stem}_{plot.name}") if plot.name else output
            for plot, _ in tables
        ]
        for path, (plot, curves) in zip(paths, tables, strict=True):
            _write_csv(path, [(None, curves)], plot.columns)
        print(f"figrecover: wrote {', '.join(map(str, paths))}", file=sys.stderr)
    else:
        columns = {plot.columns for plot, _ in tables}
        names = columns.pop() if len(columns) == 1 else ("x", "y")
        _write_csv(
            sys.stdout, [(plot.name or "", curves) for plot, curves in tables], names
        )


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


def _angle(text: str) -> float:
    """Parse an angle in degrees."""
    try:
        angle = float(text)
    except ValueError:
        angle = math.nan
    if not math.isfinite(angle):
        msg = f"expected an angle in degrees, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return angle


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
