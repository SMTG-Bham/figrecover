"""Measure how well figrecover finds and traces several series by colour.

Run from the repository root with ``uv run --with matplotlib
examples/benchmark_colours.py``, which uses the project's environment and
adds matplotlib to draw the plots. It draws plots of several series whose
curves it knows, each a variation on one default plot, finds their colours
with ``find_colours`` and traces them with ``trace_curves``, as ``-n`` does,
and prints a Markdown table: for each case, the series found out of those
drawn, how many were matched to the wrong curve, and the errors of the worst
series, as fractions of the y range, by the median, the 95th percentile and
the maximum. Each case is drawn the same way every time.
"""

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from figrecover import Calibration, Image, find_colours, trace_curves

mpl.use("Agg")

type Curve = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

X = np.linspace(0, 10, 2000)
SEED = 2026  # the phases of the curves
TOP = 3.5  # the top of the y-axis
PALETTES = {
    "tab10": [f"C{index}" for index in range(10)],
    "tab20": [mpl.colormaps["tab20"](index) for index in range(20)],
    "Set1": [mpl.colormaps["Set1"](index) for index in range(9)],
    "viridis": [mpl.colormaps["viridis"](value) for value in np.linspace(0, 0.9, 10)],
    # Office's default chart colours, and ggplot's hues for six series.
    "Excel": ["#4472c4", "#ed7d31", "#a5a5a5", "#ffc000", "#5b9bd5", "#70ad47"],
    "ggplot": ["#f8766d", "#b79f00", "#00ba38", "#00bfc4", "#619cff", "#f564e3"],
    "greys": [str(level) for level in (0, 0.15, 0.3, 0.45, 0.6, 0.7)],
}


@dataclass(frozen=True)
class Case:
    """A plot to draw: the default, six tab10 series of 1.5 pt at 100 dpi."""

    name: str = "default"
    series: int = 6
    palette: str = "tab10"
    width: float = 1.5
    dpi: int = 100
    quality: int | None = None  # JPEG quality, or None for PNG
    grid: bool = False
    legend: str | None = None  # "inside" or "outside" the axes
    crossing: bool = True


def curves(case: Case) -> list[Curve]:
    """Return the curve of each series: waves that cross, or that never do."""
    random = np.random.default_rng(SEED)
    phases = random.uniform(0, 2 * np.pi, case.series)
    step = (TOP - 0.6) / case.series
    if case.crossing:  # waves of different heights and lengths, which cross
        heights = random.uniform(0.6, 1.4, case.series)
        lengths = random.uniform(0.4, 1.2, case.series)
        return [
            lambda x, a=a, f=f, p=p: 1.75 + a * np.sin(f * x + p)
            for a, f, p in zip(heights, lengths, phases, strict=True)
        ]
    return [
        lambda x, k=k, p=p: 0.4 + step * (k + 0.5) + 0.3 * step * np.sin(x + p)
        for k, p in enumerate(phases)
    ]


def draw(case: Case) -> tuple[Image, Calibration, list[Curve]]:
    """Draw a case's plot, and return it, its calibration and its curves."""
    colours = PALETTES[case.palette]
    functions = curves(case)
    figure, axes = plt.subplots(figsize=(6.4, 4), dpi=case.dpi)
    for index, function in enumerate(functions):
        axes.plot(
            X,
            function(X),
            color=colours[index % len(colours)],
            linewidth=case.width,
            label=f"series {index + 1}",
        )
    axes.set(xlim=(0, 10), ylim=(0, TOP))
    if case.grid:
        axes.grid(visible=True)
    if case.legend == "inside":
        axes.legend(loc="upper right", fontsize=7, ncols=2)
    elif case.legend == "outside":
        figure.legend(loc="outside right center", fontsize=7)
        figure.set_layout_engine("constrained")
    figure.canvas.draw()
    corners = [(0, 0), (10, 0), (0, TOP)]
    pixels = [tuple(axes.transData.transform(corner)) for corner in corners]
    buffer = io.BytesIO()
    if case.quality is None:
        figure.savefig(buffer, format="png", dpi=case.dpi)
    else:
        figure.savefig(
            buffer, format="jpg", dpi=case.dpi, pil_kwargs={"quality": case.quality}
        )
    plt.close(figure)
    data = np.frombuffer(buffer.getvalue(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return image, Calibration.from_points(corners, pixels), functions


def score(case: Case) -> str:
    """Return a row of the table for a case."""
    image, calibration, functions = draw(case)
    origin = calibration.origin
    try:
        colours = find_colours(image, case.series, origin)
    except ValueError as error:
        found = str(error).split(",")[0]
        return f"| {case.name} | {found} | | | | |"
    traced = trace_curves(image, colours, origin)
    drawn = [
        np.array(mpl.colors.to_rgb(colour)) * 255 for colour in PALETTES[case.palette]
    ]
    errors, wrong, matched = [], 0, set()
    for colour, pixels in zip(colours, traced, strict=True):
        data = calibration.to_data(pixels)
        medians = [np.median(np.abs(data[:, 1] - f(data[:, 0]))) for f in functions]
        best = int(np.argmin(medians))
        matched.add(best)
        nearest = min(
            range(len(functions)),
            key=lambda k: np.linalg.norm(drawn[k % len(drawn)] - colour),
        )
        wrong += nearest != best
        errors.append(np.abs(data[:, 1] - functions[best](data[:, 0])) / TOP)
    worst = max(errors, key=lambda error: float(np.percentile(error, 95)))
    numbers = (np.median(worst), np.percentile(worst, 95), worst.max())
    median, p95, top = (f"{number:.3f}" for number in numbers)
    found = f"{len(matched)} of {case.series}"
    return f"| {case.name} | {found} | {wrong} | {median} | {p95} | {top} |"


def cases() -> Sequence[Case]:
    """Return the cases: the default, and variations on it one at a time."""
    default = Case()
    varied = [default]
    varied += [replace(default, name=f"{n} series", series=n) for n in (2, 4, 8, 10)]
    varied += [
        replace(default, name="12 series, tab10 wraps", series=12),
        replace(default, name="tab20", palette="tab20", series=10),
        replace(default, name="Set1", palette="Set1"),
        replace(default, name="viridis", palette="viridis", series=10),
        replace(default, name="Excel", palette="Excel"),
        replace(default, name="ggplot", palette="ggplot"),
        replace(default, name="greys", palette="greys", crossing=False),
    ]
    varied += [
        replace(default, name=f"{width} pt", width=width) for width in (0.5, 0.75, 3)
    ]
    varied += [replace(default, name=f"{dpi} dpi", dpi=dpi) for dpi in (50, 72, 200)]
    varied += [
        replace(default, name=f"JPEG {quality}", quality=quality)
        for quality in (50, 75, 95)
    ]
    varied += [
        replace(default, name="grid", grid=True),
        replace(default, name="legend inside", legend="inside"),
        replace(default, name="legend outside", legend="outside"),
        replace(default, name="not crossing", crossing=False),
    ]
    return varied


def main() -> None:
    """Print the table of results."""
    print("| Case | Series found | Wrong | Median | 95th percentile | Max |")
    print("| --- | --- | --- | --- | --- | --- |")
    for case in cases():
        print(score(case), flush=True)


if __name__ == "__main__":
    main()
