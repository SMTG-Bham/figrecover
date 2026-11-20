"""Command-line interface."""

import argparse
import contextlib
import csv
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import numpy.typing as npt

from . import __version__, annotations, interactive, metadata
from .axes import SMALL_IMAGE, Axes, _Text
from .calibration import MIN_POINTS, Calibration, Point
from .digitize import (
    STYLES,
    Colour,
    Image,
    Series,
    Style,
    _greyscale,
    _left_out,
    _stroke_ranges,
    find_error_bars,
    find_marker_error_bars,
    find_marker_series,
    find_series,
    load_image,
    remove_grid,
    trace_curves,
    trace_series,
)
from .errorbars import ErrorBars
from .level import LEVEL_THRESHOLD, Levelling, level_image
from .masks import painted
from .panels import Panel, read_panels
from .pdf import (
    PREVIEW_SCALE,
    Box,
    Drawing,
    find_pdf_error_bars,
    find_pdf_markers,
    find_pdf_series,
    load_pdf,
    read_pdf_axes,
    remove_pdf_grid,
    render_pdf,
    trace_pdf_curves,
    trace_pdf_series,
)
from .svg import load_svg, render_svg

type TraceRanges = list[dict[Point, Point]]

type Ring = tuple[Point, int, Colour]
"""A ring to draw in the preview: its centre, its radius and its colour."""

PLACE_PIXELS = 5
"""Pixels across which the x positions of markers are rounded, when printed."""

BOUNDS = ("y_low", "y_high", "x_low", "x_high")
"""The columns that --error-bars adds to the CSV, after x and y."""

_SNIFF = 4096  # a file that is not named as one is an SVG file by this many bytes
_CAP = 3  # an error bar's cap runs this many pixels each side of it in the preview

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


@dataclass(frozen=True, slots=True)
class _Selector:
    """The curve that a -c option picks.

    Attributes:
        colour: The colour of the curve.
        style: The style it is drawn in, or None for every curve of the
            colour, as one.
        number: Which of the curves of that colour and style it is, from 1,
            longest first.
    """

    colour: Colour
    style: Style | None = None
    number: int = 1


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
    axes: Axes | None = None
    texts: tuple[_Text, ...] | None = None


@dataclass(frozen=True, slots=True, eq=False)
class _Table:
    """The data rows and labels from one plot, ready to write as CSV."""

    source: str | None
    panel: str | None
    columns: tuple[str, str]
    curves: Sequence[npt.NDArray[np.float64]]
    ranges: TraceRanges | None = None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: The arguments, excluding the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    parser = _build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_intermixed_args(raw)
    arguments.argv = raw
    if arguments.axes_from and arguments.points:
        parser.error("choose --axes-from or -p")
    if arguments.locations and not arguments.points:
        parser.error("give the reference points of the -l locations with -p")
    if 0 < len(arguments.points) < MIN_POINTS:
        parser.error(f"give at least {MIN_POINTS} reference points with -p")
    if arguments.log and not arguments.points:
        parser.error(
            "--log needs -p; figrecover finds logarithmic axes itself when reading them"
        )
    sources = arguments.images
    if len(sources) == 1 and (error := _input_error(Path(sources[0]), arguments)):
        parser.error(error)
    try:
        return _run(arguments, _protected_paths(arguments, parser))
    except (ImportError, OSError, ValueError, cv2.error) as error:
        print(f"figrecover: error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _protected_paths(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> set[Path]:
    """Protect the sources and metadata from all generated filenames."""
    inputs = {Path(source).resolve() for source in arguments.images}
    outputs = {
        "-o": arguments.output,
        "--preview": arguments.preview,
        "--metadata": arguments.metadata,
    }
    if arguments.axes_from:
        inputs.add(arguments.axes_from.resolve())
    destinations: dict[Path, str] = {}
    for option, path in outputs.items():
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in inputs:
            what = (
                "calibration"
                if arguments.axes_from and resolved == arguments.axes_from.resolve()
                else "image"
            )
            parser.error(f"{option} would overwrite the {what}")
        if resolved in destinations:
            parser.error(f"{destinations[resolved]} and {option} name the same file")
        destinations[resolved] = option
    if arguments.metadata:
        inputs.add(arguments.metadata.resolve())
    return inputs


def _input_error(image: Path, arguments: argparse.Namespace) -> str | None:
    """Return the reason a file cannot use the given options, if any."""
    if arguments.figure and not _is_pdf(image):
        return "--figure chooses a captioned figure on a PDF page"
    if arguments.page and _is_svg(image):
        return "--page chooses a page of a PDF file; an SVG file has only one"
    if (arguments.page or arguments.box) and not _is_drawing(image):
        return "--page and --box choose what to read in a PDF or SVG file"
    if arguments.level is not None and _is_drawing(image):
        return "--level and --no-level level images, not PDF or SVG files"
    return None


def _run(options: argparse.Namespace, inputs: set[Path]) -> int:
    """Collect tables from successful inputs, reusing clicks and series choices."""
    batch = len(options.images) > 1
    records: list[metadata.InputRecord] = []
    if batch and options.preview:
        options.preview.mkdir(parents=True, exist_ok=True)
    previews = {options.preview.resolve()} if options.preview and not batch else set()
    selections: dict[str | None, list[_Selector]] = {}
    tables: list[_Table] = []
    status = 0
    for source in options.images:
        arguments = argparse.Namespace(**vars(options))
        arguments.image = Path(source)
        arguments.source = source if batch else None
        arguments.selections = selections.copy()
        arguments.measured = {}
        arguments.saved_annotations = None
        arguments.calibration_source = "points" if options.locations else "clicked"
        arguments.record = (
            _start_record(arguments, source) if options.metadata else None
        )
        if arguments.record:
            records.append(arguments.record)
        try:
            if arguments.record:
                arguments.record.sha256 = metadata.digest(arguments.image)
            if batch:
                _prepare_batch_input(arguments, options, previews, inputs)
            tables.extend(_digitize(arguments))
            selections.update(arguments.selections)
            if arguments.record:
                _finish_record(arguments.record, arguments)
        except (ImportError, OSError, ValueError, cv2.error) as error:
            if arguments.record:
                arguments.record.error = str(error)
                arguments.record.locations = list(arguments.locations)
            if not batch and not options.metadata:
                raise
            prefix = f"{source}: " if batch else ""
            print(f"figrecover: {prefix}error: {error}", file=sys.stderr)
            status = 1
    if tables:
        _write(options.output, tables, inputs | previews)
    if options.metadata:
        metadata.save(options.metadata, options.argv, records)
    return status


def _prepare_batch_input(
    arguments: argparse.Namespace,
    options: argparse.Namespace,
    previews: set[Path],
    inputs: set[Path],
) -> None:
    """Validate input-specific options and choose a distinct preview destination."""
    if error := _input_error(arguments.image, arguments):
        raise ValueError(error)
    if options.preview:
        arguments.preview = _preview_path(options.preview, arguments.image, previews)
        if arguments.preview.resolve() in inputs or (
            options.output and arguments.preview.resolve() == options.output.resolve()
        ):
            msg = f"conflicting preview path: {arguments.preview}"
            raise ValueError(msg)


def _start_record(arguments: argparse.Namespace, source: str) -> metadata.InputRecord:
    """Retain original image coordinates before levelling or collecting clicks."""
    kind = (
        "pdf"
        if _is_pdf(arguments.image)
        else "svg"
        if _is_svg(arguments.image)
        else "image"
    )
    return metadata.InputRecord(
        source,
        kind,
        page=arguments.page,
        box=arguments.box,
        points=arguments.points,
        locations=list(arguments.locations),
        markers=arguments.markers,
    )


def _finish_record(record: metadata.InputRecord, arguments: argparse.Namespace) -> None:
    """Keep the options needed to repeat a single plot."""
    record.locations = list(arguments.locations)
    if (
        record.plot_count == 1
        and len(record.plots) == 1
        and record.plots[0].panel is None
    ):
        extra: list[str] = []
        if arguments.log:
            extra.extend(("--log", arguments.log))
        if arguments.no_annotations:
            extra.append("--no-annotations")
        if arguments.remove_grid:
            extra.append("--remove-grid")
        for mask in arguments.mask:
            extra.extend(("--mask", ",".join(map(str, mask))))
        if arguments.error_bars:
            extra.append("--error-bars")
        if arguments.trace_range:
            extra.append("--trace-range")
        record.reproduce = metadata.reproduce(record, arguments.metadata, extra)


def _preview_path(directory: Path, image: Path, used: set[Path]) -> Path:
    """Choose a distinct preview name when inputs share their stem."""
    path = directory / f"{image.stem}.preview.png"
    number = 1
    while path.resolve() in used:
        number += 1
        path = directory / f"{image.stem}.{number}.preview.png"
    used.add(path.resolve())
    return path


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
        "images",
        nargs="+",
        metavar="image",
        help="one or more images of plots, or a PDF or SVG file, whose curves are read "
        "exactly from its drawing (a PDF file needs the pdf extra); positions in "
        "them are in points from the bottom-left corner of the page",
    )
    parser.add_argument(
        "--page",
        type=_count,
        metavar="N",
        help="the page of the PDF file to read, from 1 (default: 1)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--figure",
        metavar="LABEL",
        help="read the PDF figure with this caption label, such as 2 or S1",
    )
    selection.add_argument(
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
        help="trace N curves, told apart by colour, then by style: in an image "
        "the most common colours first, in a PDF file the longest curves first; "
        "then reuse the printed -c options (default: 1)",
    )
    curves.add_argument(
        "-c",
        "--colour",
        dest="colours",
        action="append",
        default=[],
        type=_selector,
        metavar="RRGGBB[:STYLE]",
        help="hexadecimal colour of a curve to trace, such as 1f77b4, taking "
        "every curve of that colour as one; follow it with a style, one of "
        f"{', '.join(STYLES)}, for the first curve of that colour drawn so, such "
        "as 000000:dashed, or with a number too for another, such as "
        "000000:solid2; repeat for several curves",
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
        "the shapes and colours of their markers, rather than as curves of a colour",
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
        "drawn in its colour; with several inputs give a directory",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="keep plot text, arrows and brackets when tracing",
    )
    parser.add_argument(
        "--mask",
        type=_box,
        action="append",
        default=[],
        metavar="X0,Y0,X1,Y1",
        help="exclude this rectangle, in pixels or page points from the bottom left; "
        "repeat for more regions",
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
    parser.add_argument(
        "--error-bars",
        action="store_true",
        help="also read the error bars of each marker, and add their bounds to "
        "the CSV as y_low and y_high columns, and x_low and x_high if any run "
        "across, in the plot's units",
    )
    parser.add_argument(
        "--trace-range",
        action="store_true",
        help="add the measured lower and upper edges of each raster stroke to the CSV",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        metavar="FILE",
        help="record input hashes, axes, series and a reproduction command as JSON",
    )
    parser.add_argument(
        "--axes-from",
        type=Path,
        metavar="FILE",
        help="reuse the ticks of a matching single-plot input from a metadata file",
    )
    return parser


def _digitize(arguments: argparse.Namespace) -> list[_Table]:
    """Write the curves in the image, each of its panels or the PDF file, as CSV.

    A panel whose curves cannot be traced is left out, with a warning. With
    ``-m``, insets are left out, as the markers are those of one plot.

    An image whose axes are turned is levelled before it is read, as
    ``_level`` levels it, and the -l and -m positions, in the image as
    given, are turned with it. The preview is drawn on the image as given,
    with the curves and rings turned back onto it.
    """
    if _is_drawing(arguments.image):
        return [_digitize_pdf(arguments)]
    given, levelling, plots = _plots(arguments)
    if arguments.markers and len(plots) > 1:
        msg = (
            f"-m finds the markers of one plot, but the image has {len(plots)} "
            "panels; crop it to one"
        )
        raise ValueError(msg)

    tables: list[_Table] = []
    curves: list[npt.NDArray[np.float64]] = []
    colours: list[Colour] = []
    rings: list[Ring] = []
    for plot in plots:
        if max(given.shape[:2]) <= SMALL_IMAGE:
            _print_resolution(plot)
        try:
            traced, traced_colours, marked, bars, ranges = _trace(plot, arguments)
        except ValueError as error:
            if len(plots) == 1:
                raise
            called = f"panel {plot.name}" if plot.name else "the plot"
            message = f"left out {called}, as its curves cannot be traced"
            _notice(f"{message}: {error}", arguments.record)
            continue
        aligned = bool(arguments.markers)  # error bars through the markers traced
        tables.append(
            _Table(
                arguments.source,
                plot.name,
                plot.columns,
                _data(plot.calibration, traced, bars, aligned=aligned),
                ranges,
            )
        )
        if arguments.record:
            _record_plot(
                arguments,
                tables[-1],
                (plot.calibration, plot.pixels, plot.axes),
                plot.offset,
            )
        x, y = plot.offset
        curves += [curve + np.array([x, y]) for curve in traced]
        colours += traced_colours
        if bars is not None:  # drawn over the curves, in their colours
            curves += _bar_dots(bars, (x, y))
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
    return tables


def _print_resolution(plot: _Plot) -> None:
    """Report data units per pixel, or the factor per pixel for a logarithmic axis."""
    fit = plot.calibration
    steps = [
        f"a factor of {10 ** abs(step):.4g}" if log else f"{abs(step):.4g} data units"
        for step, log in ((fit.x_scale, fit.x_log), (fit.y_scale, fit.y_log))
    ]
    print(
        f"figrecover: {_prefix(plot.name)}one pixel spans {steps[0]} on x "
        f"and {steps[1]} on y",
        file=sys.stderr,
    )


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


def _plots(arguments: argparse.Namespace) -> tuple[Image, Levelling, list[_Plot]]:
    """Load the image, level it, and find its plots, from the -p points or its axes.

    The -l and -m positions are checked against the image as given, and
    turned with it as it is levelled; -m positions are changed to those in
    the levelled image. With several curves asked for, a greyscale image is
    reported, as curves of different colours may share a grey in it.

    Returns:
        The image as given, how it was levelled, and its plots, in the
        levelled image.
    """
    given = load_image(arguments.image)
    _check_locations(arguments.markers, given, "-m position")
    _check_locations(arguments.locations, given, "-l location")
    if max(arguments.series or 1, len(arguments.colours)) > 1 and _greyscale(given):
        message = (
            "the image is greyscale, so curves of different colours may share a grey"
        )
        _notice(message, arguments.record)
    locations = arguments.locations
    if arguments.points and not locations:
        locations.extend(_click_locations(given, arguments.points))
    image, levelling = _level(painted(given, arguments.mask), arguments.level)
    if arguments.record:
        arguments.record.level_degrees = levelling.angle
    # From here on, -m positions are those in the levelled image.
    arguments.markers = _as_points(levelling.points(arguments.markers))
    if arguments.axes_from:
        axes, region = metadata.recorded_axes(arguments.axes_from, arguments.image)
        left, _, _, bottom = region.crop or (0, 0, image.shape[1], image.shape[0])
        offset = (left, image.shape[0] - bottom)
        image = region.apply(image)
        arguments.saved_annotations = region.annotations
        if arguments.record:
            arguments.record.region = region
        calibration, pixels, names = _read_axes(axes, None)
        return (
            given,
            levelling,
            [_Plot(None, image, offset, calibration, pixels, names, axes)],
        )
    if not arguments.points:
        plots = _read_panels(image, arguments.record, insets=not arguments.markers)
        if arguments.record and len(plots) == 1:
            only = plots[0]
            arguments.record.region = metadata.Region.from_image(
                image, only.image, only.offset
            )
        return given, levelling, plots
    pixels = _as_points(levelling.points(locations))
    calibration = Calibration.from_points(
        arguments.points,
        pixels,
        x_log="x" in arguments.log,
        y_log="y" in arguments.log,
    )
    return (
        given,
        levelling,
        [_Plot(None, image, (0, 0), calibration, pixels, ("x", "y"))],
    )


def _trace(
    plot: _Plot, arguments: argparse.Namespace
) -> tuple[
    list[npt.NDArray[np.float64]],
    list[Colour],
    list[Ring],
    list[ErrorBars] | None,
    TraceRanges | None,
]:
    """Trace the curves of a plot, by colour or by markers.

    Returns:
        The pixel positions of the points of each curve, its colour in the
        preview, and the rings to draw in the preview around the reference
        points or ticks, and the markers and legend keys; and with
        --error-bars, the error bars of each curve's markers.
    """
    origin = plot.calibration.origin
    traced = plot.image
    if not arguments.no_annotations and not arguments.markers:
        traced, boxes, shafts = annotations.excluded(
            traced, origin, plot.texts, arguments.saved_annotations
        )
        if arguments.record:
            arguments.record.region.annotations = boxes
        if boxes:
            _notice(
                f"{_prefix(plot.name)}left out {len(boxes)} text "
                f"region{'s' if len(boxes) != 1 else ''} and {shafts} "
                f"arrow or bracket object{'s' if shafts != 1 else ''} in the plot area",
                arguments.record,
            )
    if arguments.remove_grid:
        traced = remove_grid(traced)
    rings: list[Ring] = [(pixel, 6, (0, 0, 255)) for pixel in plot.pixels]
    inks: list[Colour] = []  # the colour of each curve, as drawn
    styles: list[Style | None] = []  # and its style, if known
    selectors = arguments.colours or arguments.selections.get(plot.name)
    if arguments.markers:
        curves = _found_markers(
            traced, arguments.markers, plot.calibration, arguments.record
        )
        colours = _series_colours(len(curves))
        rings += _marker_rings(arguments.markers, curves, colours)
    elif selectors:
        with _warnings_printed(_prefix(plot.name), arguments.record):
            curves = _chosen(
                selectors,
                lambda colours: trace_curves(traced, colours, origin),
                lambda colours: trace_series(traced, colours, origin),
            )
        inks = [selector.colour for selector in selectors]
        styles = [selector.style for selector in selectors]
        colours = _distinct(inks)
    else:
        found = _found(
            find_series(traced, origin),
            arguments,
            plot.name,
            lambda: _left_out(traced, origin),
        )
        curves = [series.points for series in found]
        inks = [series.colour for series in found]
        styles = [series.style for series in found]
        colours = _distinct(inks)
    bars = None
    if arguments.error_bars and arguments.markers:
        bars = find_marker_error_bars(traced, arguments.markers, origin)
    elif arguments.error_bars:
        bars = _bars_by_colour(traced, inks, origin)
        # A series of markers gives the middles that their bars run from.
        curves = [
            series.points if style == "markers" else curve
            for curve, series, style in zip(curves, bars, styles, strict=True)
        ]
    ranges = (
        _trace_ranges(plot.calibration, traced, curves, inks, styles)
        if arguments.trace_range
        else None
    )
    return curves, colours, rings, bars, ranges


def _trace_ranges(
    calibration: Calibration,
    image: Image,
    curves: Sequence[npt.NDArray[np.float64]],
    inks: Sequence[Colour],
    styles: Sequence[Style | None],
) -> TraceRanges:
    """Index measured pixel edges by point, leaving inserted error-bar rows empty."""
    result: TraceRanges = [{} for _ in curves]
    lines = [i for i, style in enumerate(styles) if style != "markers"]
    if lines:
        measured = _stroke_ranges(
            image,
            [inks[i] for i in lines],
            [curves[i] for i in lines],
            calibration.origin,
        )
        for i, spans in zip(lines, measured, strict=True):
            lo = calibration.to_data(np.column_stack((curves[i][:, 0], spans[:, 0])))[
                :, 1
            ]
            hi = calibration.to_data(np.column_stack((curves[i][:, 0], spans[:, 1])))[
                :, 1
            ]
            result[i] = {
                (x, y): (min(a, b), max(a, b))
                for (x, y), a, b in zip(
                    calibration.to_data(curves[i]), lo, hi, strict=True
                )
            }
    return result


def _bars_by_colour(
    image: Image, colours: Sequence[Colour], origin: Point
) -> list[ErrorBars]:
    """Return the error bars of each curve's markers, found by the curve's colour.

    Curves that share a colour, as markers and a line through them do, find
    the same markers, so the first curve of a colour, its markers if it has
    any, takes their error bars, and the others none.
    """
    first = list(dict.fromkeys(colours))
    found = dict(zip(first, find_error_bars(image, first, origin), strict=True))
    none = ErrorBars(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)))
    return [found.pop(colour) if colour in found else none for colour in colours]


def _notice(message: str, record: metadata.InputRecord | None) -> None:
    """Report a diagnostic and retain it when recording the run."""
    print(f"figrecover: {message}", file=sys.stderr)
    if record:
        record.warnings.append(message)


@contextlib.contextmanager
def _warnings_printed(
    prefix: str, record: metadata.InputRecord | None
) -> Iterator[None]:
    """Print the warnings given in the block, as figrecover's messages.

    ``prefix`` starts each message, naming the panel.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for warning in caught:
                _notice(f"{prefix}{warning.message}", record)


def _digitize_pdf(arguments: argparse.Namespace) -> _Table:
    """Write the curves in a plot on a page of the PDF file, or the SVG file, as CSV.

    Positions on the page are in points, and the preview shows the part of
    the page read, ``PREVIEW_SCALE`` pixels to the point.
    """
    drawing = (
        load_svg(arguments.image, arguments.box)
        if _is_svg(arguments.image)
        else load_pdf(
            arguments.image, arguments.page or 1, arguments.box, figure=arguments.figure
        )
    )
    drawing = replace(drawing, masks=tuple(arguments.mask))
    if arguments.record:
        arguments.record.box = drawing.box
        arguments.record.page = drawing.page if arguments.record.kind == "pdf" else None
    axes = None
    _check_positions(arguments.markers, drawing.box, "-m position")
    if arguments.remove_grid:
        drawing = remove_pdf_grid(drawing)
    if arguments.points:
        _check_positions(arguments.locations, drawing.box, "-l location")
        positions = arguments.locations
        if not positions:
            positions.extend(_click_positions(drawing, arguments.points))
        calibration = Calibration.from_points(
            arguments.points,
            positions,
            x_log="x" in arguments.log,
            y_log="y" in arguments.log,
        )
        names = ("x", "y")
    else:
        axes = (
            metadata.recorded_axes(arguments.axes_from, arguments.image)[0]
            if arguments.axes_from
            else read_pdf_axes(drawing)
        )
        calibration, positions, names = _read_axes(axes, None)
    selectors = arguments.colours or arguments.selections.get(None)
    if arguments.markers:
        curves = _counted(
            find_pdf_markers(drawing, arguments.markers, calibration.origin),
            arguments.markers,
            arguments.record,
        )
        colours = _series_colours(len(curves))
    elif selectors:
        origin = calibration.origin
        curves = _chosen(
            selectors,
            lambda colours: trace_pdf_curves(drawing, colours, origin),
            lambda colours: trace_pdf_series(drawing, colours, origin),
        )
        colours = _distinct([selector.colour for selector in selectors])
    else:
        found = _found(find_pdf_series(drawing, calibration.origin), arguments, None)
        curves = [series.points for series in found]
        colours = _distinct([series.colour for series in found])
    bars = find_pdf_error_bars(drawing, curves) if arguments.error_bars else None

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
        if bars is not None:  # drawn over the curves, in their colours
            pixels += _bar_dots(bars, -corner * PREVIEW_SCALE, PREVIEW_SCALE)
            colours = [*colours, *colours]
        image = _render(drawing)
        _save_preview(arguments.preview, image, pixels, colours, rings)
    data = _data(calibration, curves, bars, aligned=True)
    ranges: TraceRanges | None = [{} for _ in curves] if arguments.trace_range else None
    table = _Table(arguments.source, None, names, data, ranges)
    if arguments.record:
        _record_plot(arguments, table, (calibration, positions, axes))
    return table


def _record_plot(
    arguments: argparse.Namespace,
    table: _Table,
    measurements: tuple[Calibration, Sequence[Point], Axes | None],
    offset: Point = (0, 0),
) -> None:
    """Record the measurements and CSV row counts for a successful plot."""
    calibration, pixels, axes = measurements
    measured: Sequence[Series]
    selectors: Sequence[_Selector]
    measured, selectors = arguments.measured.get(
        table.panel,
        ([], arguments.colours or arguments.selections.get(table.panel, [])),
    )
    series = []
    for i, curve in enumerate(table.curves):
        selected = selectors[i] if selectors else None
        found = measured[i] if measured else None
        colour = found.colour if found else selected.colour if selected else None
        style = found.style if found else selected.style if selected else "markers"
        series.append(
            metadata.SeriesRecord(
                i + 1,
                bytes(colour).hex() if colour else None,
                style,
                found.width if found else None,
                _written(selected) if selected else None,
                len(curve),
            )
        )
    source = "ticks" if axes else arguments.calibration_source
    fit = metadata.calibration_record(
        calibration, axes, pixels, arguments.points, source
    )
    arguments.record.plots.append(
        metadata.PlotRecord(table.panel, calibration.origin, fit, series, offset)
    )


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


def _read_panels(
    image: Image, record: metadata.InputRecord | None, *, insets: bool
) -> list[_Plot]:
    """Read the axes of the image's panels, and print them so they can be checked.

    Panels left out, as their axes cannot be read, are reported too. An
    image of one plot is one panel, without a name, as ``_names`` names
    them. Insets are left out unless ``insets`` is true. Each panel's pixel
    positions to ring in the preview are those of its ticks, on their axis
    lines, and its columns are named after its axis titles.
    """
    with _warnings_printed("", record):
        panels = read_panels(image)
    if record:
        record.plot_count = len(panels)
    if not insets:
        panels = [panel for panel in panels if panel.inset_of is None]
    plots = []
    for panel, name in zip(panels, _names(panels), strict=True):
        calibration, pixels, columns = _read_axes(panel.axes, name)
        plots.append(
            _Plot(
                name,
                panel.image,
                panel.offset,
                calibration,
                pixels,
                columns,
                panel.axes,
                panel.texts,
            )
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


def _found(
    every: Sequence[Series],
    arguments: argparse.Namespace,
    panel: str | None,
    left_out: Callable[[], str] = str,
) -> list[Series]:
    """Return the first series found, and print the -c options to reuse if -n was given.

    The -n options are kept per panel for subsequent inputs in a batch.
    ``left_out`` says what was left out and why, if anything, as
    ``_left_out`` does, to explain finding too few.

    Raises:
        ValueError: If fewer series were found than were asked for.
    """
    count = arguments.series
    wanted = count or 1
    if len(every) < wanted:
        options = " ".join(f"-c {option}" for option in _options(every, every))
        msg = (
            f"found only {len(every)} of {wanted} curves, {options}"
            if every
            else "no curve found"
        )
        raise ValueError(msg + left_out())
    found = list(every[:wanted])
    if count is not None or arguments.record:
        written = _options(found, every)
        selectors = [_selector(option) for option in written]
        if arguments.record:
            arguments.measured[panel] = found, selectors
        if count is not None:
            arguments.selections[panel] = selectors
            options = " ".join(f"-c {option}" for option in written)
            print(f"figrecover: {_prefix(panel)}colours: {options}", file=sys.stderr)
    return found


def _options(found: Sequence[Series], every: Sequence[Series]) -> list[str]:
    """Return the -c option that picks each series found, among every series.

    That is its colour, if no other series has it, or else its colour and
    style, and its number among those of that colour and style if it is not
    the first, such as 000000:solid2.
    """
    options = []
    for series in found:
        colour = bytes(series.colour).hex()
        alike = [other for other in every if other.colour == series.colour]
        same = [other for other in alike if other.style == series.style]
        number = next(n for n, other in enumerate(same, start=1) if other is series)
        options.append(
            colour
            if len(alike) == 1
            else f"{colour}:{series.style}{number if number > 1 else ''}"
        )
    return options


def _chosen(
    selectors: Sequence[_Selector],
    whole: Callable[[list[Colour]], list[npt.NDArray[np.float64]]],
    styled: Callable[[list[Colour]], list[list[Series]]],
) -> list[npt.NDArray[np.float64]]:
    """Return the points of the curve that each -c option picks.

    ``whole`` traces every curve of each of the colours given as one, and
    ``styled`` finds the curves of each, told apart by style, longest
    first.

    Raises:
        ValueError: If an option is given twice, or picks no curve.
    """
    for first, second in itertools.combinations(selectors, 2):
        if first == second:
            msg = f"-c {_written(first)} is given twice"
            raise ValueError(msg)
    plain = [selector.colour for selector in selectors if selector.style is None]
    named = list(
        dict.fromkeys(selector.colour for selector in selectors if selector.style)
    )
    traced = iter(whole(plain) if plain else [])
    series = dict(zip(named, styled(named) if named else [], strict=True))
    return [
        next(traced)
        if selector.style is None
        else _pick(series[selector.colour], selector)
        for selector in selectors
    ]


def _pick(series: Sequence[Series], selector: _Selector) -> npt.NDArray[np.float64]:
    """Return the points of the series of a colour that a -c option picks by style.

    Raises:
        ValueError: If there are too few series of that style.
    """
    alike = [one for one in series if one.style == selector.style]
    if len(alike) < selector.number:
        colour = bytes(selector.colour).hex()
        msg = (
            f"found only {len(alike)} {selector.style} series of colour #{colour}"
            if alike
            else f"no {selector.style} series of colour #{colour} found"
        )
        raise ValueError(msg)
    return alike[selector.number - 1].points


def _written(selector: _Selector) -> str:
    """Return a -c option as it is written, such as 000000:solid2."""
    colour = bytes(selector.colour).hex()
    if selector.style is None:
        return colour
    number = selector.number if selector.number > 1 else ""
    return f"{colour}:{selector.style}{number}"


def _found_markers(
    image: Image,
    keys: Sequence[Point],
    calibration: Calibration,
    record: metadata.InputRecord | None,
) -> list[npt.NDArray[np.float64]]:
    """Find the series of markers in a plot, and print what was found.

    Warnings, such as of keys whose markers look alike, are printed, and how
    many markers each series has, as ``_counted`` prints them; and for each
    series, how many of its markers are ambiguous and how many are probably
    hidden, and where, in the plot's x units.

    Returns:
        The pixel positions of each series' markers.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = find_marker_series(image, keys, calibration.origin)
    for warning in caught:
        _notice(str(warning.message), record)
    markers = _counted([series.points for series in found], keys, record)
    for number, series in enumerate(found, start=1):
        notes = []
        if np.any(series.ambiguous):
            places = _places(calibration, series.points[series.ambiguous])
            notes.append(
                f"{np.count_nonzero(series.ambiguous)} of them ambiguous near x = "
                f"{places}"
            )
        if len(series.hidden):
            places = _places(calibration, series.hidden)
            notes.append(f"{len(series.hidden)} probably hidden near x = {places}")
        if notes:
            message = f"series {number}: {len(series.points)} found, "
            _notice(message + ", and ".join(notes), record)
    return markers


def _places(calibration: Calibration, pixels: npt.NDArray[np.float64]) -> str:
    """Return the x data coordinates of pixel positions, written out.

    Each is rounded to the power of ten no larger than ``PLACE_PIXELS``
    pixels span there, as markers are several pixels across.
    """
    data = calibration.to_data(pixels)[:, 0]
    beside = calibration.to_data(pixels + np.array([PLACE_PIXELS, 0]))[:, 0]
    digits = -np.floor(np.log10(np.abs(beside - data)))
    return ", ".join(f"{round(x, int(n)):g}" for x, n in zip(data, digits, strict=True))


def _counted(
    markers: list[npt.NDArray[np.float64]],
    keys: Sequence[Point],
    record: metadata.InputRecord | None,
) -> list[npt.NDArray[np.float64]]:
    """Print how many markers were found in each series, and return them.

    A series without markers, whose key is given in ``keys``, gets a warning,
    as its key or the axes may be wrong.
    """
    counts = ", ".join(str(len(series)) for series in markers)
    print(f"figrecover: markers found in each series: {counts}", file=sys.stderr)
    for number, (series, (x, y)) in enumerate(zip(markers, keys, strict=True), 1):
        if not len(series):
            message = f"found no markers like the key at {x:g},{y:g}"
            _notice(f"series {number}: {message}", record)
    return markers


def _distinct(colours: list[Colour]) -> list[Colour]:
    """Return the colours to draw curves in, in the preview: their own, if distinct.

    Curves that share a colour, told apart by style, are each drawn in a
    colour of its own instead, as series of markers are.
    """
    if len(set(colours)) == len(colours):
        return colours
    return _series_colours(len(colours))


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


def _data(
    calibration: Calibration,
    curves: Sequence[npt.NDArray[np.float64]],
    bars: Sequence[ErrorBars] | None,
    *,
    aligned: bool,
) -> list[npt.NDArray[np.float64]]:
    """Return the data of each curve, with the bounds of its error bars, if read.

    Without error bars, a row of a curve's data holds a point's x and y.
    With them, it also holds the bounds that ``BOUNDS`` names, lower first
    even on a reversed axis, NaN where the point has none. If ``aligned``,
    the error bars' points are the curves' own, as markers and the points in
    a PDF are; otherwise each marker adds a row among those of its curve,
    sorted by x, in place of a row of the curve at its point, and the
    curve's other rows have no bounds.
    """
    data = [calibration.to_data(curve) for curve in curves]
    if bars is None:
        return data
    flipped = np.array([calibration.x_scale < 0, calibration.y_scale < 0])
    bounded = []
    for series in bars:
        low, high = calibration.to_data(series.low), calibration.to_data(series.high)
        low, high = np.where(flipped, high, low), np.where(flipped, low, high)
        points = calibration.to_data(series.points)
        bounded.append(
            np.column_stack([points, low[:, 1], high[:, 1], low[:, 0], high[:, 0]])
        )
    if aligned:
        return bounded
    merged = []
    for traced, marked in zip(data, bounded, strict=True):
        middles = {(x, y) for x, y in marked[:, :2]}
        rows = traced[[(x, y) not in middles for x, y in traced]].reshape(-1, 2)
        unbounded = np.column_stack([rows, np.full((len(rows), len(BOUNDS)), np.nan)])
        both = np.vstack([unbounded, marked])
        merged.append(both[np.argsort(both[:, 0], kind="stable")])
    return merged


def _bar_dots(
    bars: Sequence[ErrorBars],
    shift: tuple[float, float] | npt.NDArray[np.float64],
    scale: float = 1,
) -> list[npt.NDArray[np.float64]]:
    """Return each series' error bars as dots a pixel apart, to draw in the preview.

    Each bar runs from its point to one of its ends, with a cap across the
    end, ``_CAP`` pixels each side, and positions are scaled, then shifted,
    into the preview's pixels.
    """
    drawn = []
    for series in bars:
        points = series.points * scale + shift
        dots = [np.empty((0, 2))]
        for ends in (series.low * scale + shift, series.high * scale + shift):
            for axis in (0, 1):
                across = np.zeros(2)
                across[1 - axis] = _CAP
                for point, end in zip(points, ends[:, axis], strict=True):
                    if not math.isnan(end):
                        tip = point.copy()
                        tip[axis] = end
                        dots += [
                            _dotted(point, tip),
                            _dotted(tip - across, tip + across),
                        ]
        drawn.append(np.vstack(dots))
    return drawn


def _dotted(
    start: npt.NDArray[np.float64], end: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return points a pixel apart, or less, along a line from start to end."""
    count = math.ceil(math.dist(start, end)) + 1
    return np.linspace(start, end, count)


def _write(output: Path | None, tables: Sequence[_Table], protected: set[Path]) -> None:
    """Write one combined table to stdout, or one file per named panel."""
    panels = any(table.panel is not None for table in tables)
    if output is None or not panels:
        _write_csv(output or sys.stdout, tables, panels=panels)
        return
    groups: dict[Path, list[_Table]] = {}
    for table in tables:
        path = (
            output.with_stem(f"{output.stem}_{table.panel}") if table.panel else output
        )
        if path.resolve() in protected:
            msg = f"panel output would overwrite an input or preview: {path}"
            raise ValueError(msg)
        groups.setdefault(path, []).append(table)
    for path, group in groups.items():
        _write_csv(path, group, panels=False)
    print(f"figrecover: wrote {', '.join(map(str, groups))}", file=sys.stderr)


def _write_csv(file: Path | TextIO, tables: Sequence[_Table], *, panels: bool) -> None:
    """Write a common header and the rows of each source, panel and series."""
    columns = {table.columns for table in tables}
    names = columns.pop() if len(columns) == 1 else ("x", "y")
    sources = tables[0].source is not None
    series = any(len(table.curves) > 1 for table in tables)
    every = [curve for table in tables for curve in table.curves]
    width = max((curve.shape[1] for curve in every), default=len(names))
    across = len(names) + BOUNDS.index("x_low")  # where bounds across start
    if all(np.isnan(curve[:, across:]).all() for curve in every):
        width = min(width, across)
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(
        [
            *(["source"] * sources),
            *(["panel"] * panels),
            *(["series"] * series),
            *names,
            *BOUNDS[: width - len(names)],
            *(("trace_low", "trace_high") if tables[0].ranges is not None else ()),
        ]
    )
    for table in tables:
        for number, curve in enumerate(table.curves, start=1):
            start = [
                *[table.source] * sources,
                *[table.panel or ""] * panels,
                *[str(number)] * series,
            ]
            padding = [""] * max(0, width - curve.shape[1])
            if table.ranges is None:
                writer.writerows(
                    [*start, *map(_cell, row), *padding] for row in curve[:, :width]
                )
            else:
                extents = table.ranges[number - 1]
                for row in curve[:, :width]:
                    span = extents.get((row[0], row[1]), (math.nan, math.nan))
                    writer.writerow(
                        [*start, *map(_cell, row), *padding, *map(_cell, span)]
                    )
    if isinstance(file, Path):
        file.write_text(text.getvalue(), encoding="utf-8")
    else:
        file.write(text.getvalue())


def _cell(value: float) -> str:
    """Return a value as written in the CSV, or nothing if it is NaN."""
    return "" if math.isnan(value) else f"{value:.6g}"


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


def _selector(text: str) -> _Selector:
    """Parse a -c option: a colour, and perhaps a style and number.

    The colour is written in hexadecimal as ``RRGGBB``, with or without a
    ``#``, and may be followed by a colon, a style and a number from 1, such
    as ``000000:dashed`` or ``000000:solid2``.
    """
    match = re.fullmatch(
        r"#?([0-9a-f]{6})(?::([a-z]+)([1-9][0-9]*)?)?", text.strip(), re.IGNORECASE
    )
    if match is None:
        msg = f"expected a colour written as RRGGBB, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    red, green, blue = bytes.fromhex(match[1])
    if match[2] is None:
        return _Selector((red, green, blue))
    styles = [style for style in STYLES if style == match[2].lower()]
    if not styles:
        msg = f"expected a style of {', '.join(STYLES)}, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return _Selector((red, green, blue), styles[0], int(match[3] or 1))
