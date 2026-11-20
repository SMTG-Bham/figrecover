"""Record the inputs, fitted axes and series used for a digitization."""

import hashlib
import json
import math
import platform
import shlex
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from . import __version__, legends
from .axes import Axes
from .calibration import Calibration, Point, Tick
from .digitize import Image


@dataclass
class AxisRecord:
    """The observed ticks as (value, position) pairs and their largest fit residual."""

    title: str | None
    log: bool
    ticks: list[tuple[float, float]]
    residual_px: float


@dataclass
class SeriesRecord:
    """The chosen series and the number of rows written for it."""

    number: int
    colour: str | None
    style: str | None
    width: float | None
    selector: str | None
    points: int


@dataclass
class Region:
    """Crop and painted panel label in levelled image array coordinates."""

    crop: tuple[int, int, int, int] | None = None
    label: tuple[int, int, int, int] | None = None
    fill: tuple[int, int, int] | None = None
    annotations: list[tuple[float, float, float, float]] = field(default_factory=list)
    legend: list[legends.Entry] = field(default_factory=list)

    @classmethod
    def from_image(cls, image: Image, part: Image, offset: Point) -> Region:
        """Keep the crop and any label rectangle painted before tracing."""
        height, width = part.shape[:2]
        left = int(offset[0])
        top = image.shape[0] - int(offset[1]) - height
        crop = (left, top, left + width, top + height)
        region = cls(crop if image.shape != part.shape else None)
        original = image[top : top + height, left : left + width]
        rows, cols = np.nonzero(np.any(original != part, axis=2))
        if len(rows):
            region.label = (
                int(cols.min()),
                int(rows.min()),
                int(cols.max() + 1),
                int(rows.max() + 1),
            )
            r, g, b = part[rows[0], cols[0]]
            region.fill = int(r), int(g), int(b)
        return region

    def apply(self, image: Image) -> Image:
        """Prepare the same image for tracing without running OCR."""
        if self.crop:
            left, top, right, bottom = self.crop
            image = image[top:bottom, left:right]
        if self.label and self.fill:
            left, top, right, bottom = self.label
            image = image.copy()
            image[top:bottom, left:right] = self.fill
        return image


@dataclass
class PlotRecord:
    """Fitted axes and the series selected for one plot."""

    panel: str | None
    origin: Point
    calibration: dict[str, object]
    series: list[SeriesRecord]
    offset: Point


@dataclass
class InputRecord:
    """Decisions and diagnostics for one input, including unsuccessful inputs."""

    path: str
    kind: str
    page: int | None = None
    box: tuple[float, float, float, float] | None = None
    sha256: str | None = None
    level_degrees: float | None = None
    points: list[Point] = field(default_factory=list)
    locations: list[Point] = field(default_factory=list)
    markers: list[Point] = field(default_factory=list)
    plot_count: int = 1
    region: Region = field(default_factory=Region)
    plots: list[PlotRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    reproduce: str | None = None


def digest(path: Path) -> str:
    """Return the SHA-256 of the original file."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _axis(
    ticks: Sequence[Tick], title: str | None, log: bool, scale: float, offset: float
) -> AxisRecord:
    residual = max(
        abs((math.log10(value) if log else value) - (pixel * scale + offset))
        for pixel, value in ticks
    )
    return AxisRecord(
        title, log, [(value, pixel) for pixel, value in ticks], residual / abs(scale)
    )


def calibration_record(
    calibration: Calibration,
    axes: Axes | None,
    pixels: Sequence[Point],
    points: Sequence[Point],
    source: str,
) -> dict[str, object]:
    """Describe each fitted axis from the measurements used to fit it."""
    if axes:
        xs, ys = list(axes.x_ticks), list(axes.y_ticks)
        titles = axes.x_title, axes.y_title
    else:
        xs = [(p[0], d[0]) for p, d in zip(pixels, points, strict=True)]
        ys = [(p[1], d[1]) for p, d in zip(pixels, points, strict=True)]
        titles = None, None
    return {
        "source": source,
        "x": asdict(
            _axis(
                xs,
                titles[0],
                calibration.x_log,
                calibration.x_scale,
                calibration.x_offset,
            )
        ),
        "y": asdict(
            _axis(
                ys,
                titles[1],
                calibration.y_log,
                calibration.y_scale,
                calibration.y_offset,
            )
        ),
    }


def recorded_axes(path: Path, image: Path) -> tuple[Axes, Region]:
    """Read the measured ticks for the matching single-plot input."""
    document = json.loads(path.read_text())
    fingerprint = digest(image)
    try:
        entry = next(
            entry
            for entry in document["inputs"]
            if entry["sha256"] == fingerprint
            and entry["plot_count"] == 1
            and len(entry["plots"]) == 1
            and entry["plots"][0]["panel"] is None
        )
        plot = entry["plots"][0]
        x, y = (plot["calibration"][axis] for axis in ("x", "y"))
        xs, ys = (
            tuple((float(pixel), float(value)) for value, pixel in axis["ticks"])
            for axis in (x, y)
        )
        ox, oy = plot["origin"]
        fit = Calibration.from_ticks(
            xs, ys, (float(ox), float(oy)), x_log=x["log"], y_log=y["log"]
        )
        region = Region(**entry["region"])
        region.legend = [
            legends.Entry(
                row["name"],
                tuple(row["colour"]),
                tuple(row["key"]),
                tuple(row["box"]),
                row["width"],
                row["height"],
                row.get("broken"),
            )
            for row in entry["region"].get("legend", [])
        ]
        return Axes(fit, xs, ys, x["title"], y["title"]), region
    except (StopIteration, KeyError, TypeError, IndexError, ValueError) as error:
        msg = f"{path} has no valid single-plot calibration for {image}"
        raise ValueError(msg) from error


def reproduce(
    record: InputRecord,
    sidecar: Path,
    extra: Sequence[str],
) -> str:
    """Build a shell-quoted command using the original positions and selected series."""
    args = ["figrecover", record.path]
    if record.points and not (record.region.annotations or record.region.legend):
        for flag, pairs in (("-p", record.points), ("-l", record.locations)):
            for x, y in pairs:
                args.extend((flag, f"{float(x)!r},{float(y)!r}"))
    else:
        args.extend(("--axes-from", str(sidecar)))
    args.extend(extra)
    if record.level_degrees is not None:
        args.extend(("--level", repr(record.level_degrees)))
    if record.page is not None:
        args.extend(("--page", str(record.page)))
    if record.box is not None:
        args.extend(("--box", ",".join(map(str, record.box))))
    for series in record.plots[0].series:
        if series.selector:
            selector = series.selector
            if series.width is not None and ":" not in selector:
                selector = f"{selector}:{series.style}"
            args.extend(("-c", selector))
    for x, y in record.markers:
        args.extend(("-m", f"{float(x)!r},{float(y)!r}"))
    return shlex.join(args)


def save(path: Path, argv: Sequence[str], inputs: Sequence[InputRecord]) -> None:
    """Write a JSON record of the run, including installed optional dependencies."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "opencv-python", "rapidocr", "pypdfium2"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    entries = [asdict(entry) for entry in inputs]
    for entry in entries:
        if entry["reproduce"] is None:
            del entry["reproduce"]
    result = {
        "figrecover": __version__,
        "created_utc": datetime.now(UTC).isoformat(),
        "argv": list(argv),
        "versions": versions,
        "inputs": entries,
    }
    if len(inputs) == 1 and inputs[0].reproduce:
        result["reproduce"] = inputs[0].reproduce
    path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
