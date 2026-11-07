# /// script
# requires-python = ">=3.14"
# dependencies = ["matplotlib>=3.10.7", "numpy>=2.3.3"]
# ///
"""Draw the example plots, whose curves the tests know exactly.

Run from the repository root with ``uv run examples/make_examples.py``. It
prints the pixel position of each reference point, for figrecover's -l option,
or in a PDF its position in points.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend

type Curve = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

HERE = Path(__file__).parent
ALSO_PDF = {"labelled", "logarithmic", "markers", "series"}  # also saved as PDFs


def spectrum(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return two peaks on a flat baseline."""
    return 0.5 + 2 * np.exp(-2 * (x - 3) ** 2) + np.exp(-((x - 7) ** 2) / 2)


def damped(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a damped oscillation."""
    return 1.2 + np.exp(-x / 4) * np.cos(2 * x)


def ringing(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a damped oscillation about zero."""
    return np.exp(-x / 4) * np.cos(2 * x)


def decay(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return an exponential decay."""
    return 2.5 * np.exp(-x / 3)


def growth(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return an exponential approach to a limit, crossing ``decay``."""
    return 2.5 * (1 - np.exp(-x / 3))


def wave(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a sine wave that crosses ``decay`` and ``growth``."""
    return 1.25 + np.sin(x) / 2


def trend(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a straight line that crosses the other series."""
    return 0.4 + 0.2 * x


def lowpass(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the power through a low-pass filter, falling a hundredfold a decade."""
    return 1e3 / (1 + (x / 100) ** 2)


def quadratic(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a quarter of the square, rising fourfold each time x doubles."""
    return x**2 / 4


def draw(
    name: str,
    curves: dict[str, Curve],
    ylim: float | None,
    *,
    grid: bool,
    titles: tuple[str, str] = ("x", "y"),
) -> None:
    """Draw curves, keyed by colour, over x from 0 to 10, and print the points.

    Without ylim, matplotlib leaves a margin around the curves, so the axes do
    not start at a labelled tick.
    """
    figure, axes = plt.subplots(figsize=(6.4, 4), dpi=100)
    x = np.linspace(0, 10, 2000)
    for colour, curve in curves.items():
        axes.plot(x, curve(x), color=colour, linewidth=1.5)
    axes.set(xlabel=titles[0], ylabel=titles[1])
    if ylim is not None:
        axes.set(xlim=(0, 10), ylim=(0, ylim))
    if grid:
        axes.grid(color="0.25", linewidth=0.8)
    else:
        axes.spines[["top", "right"]].set_visible(False)
    save(name, figure, axes, [(0, 0), (10, 0), (0, 1)])


def draw_log(
    name: str,
    curve: Curve,
    titles: tuple[str, str],
    points: list[tuple[float, float]],
    base: int = 10,
) -> None:
    """Draw a curve on logarithmic axes over x from 1 to base^4, and print the points.

    matplotlib labels the ticks with powers of the base, and leaves a margin
    around the curve, so the axes do not start at a labelled tick. The layout
    makes room for the taller tick labels.
    """
    figure, axes = plt.subplots(figsize=(6.4, 4), dpi=100, layout="constrained")
    x = np.logspace(0, 4, 2000, base=base)
    axes.loglog(x, curve(x), color="black", linewidth=1.5, base=base)
    axes.set(xlabel=titles[0], ylabel=titles[1])
    axes.spines[["top", "right"]].set_visible(False)
    save(name, figure, axes, points)


def draw_rasterised(name: str, curves: dict[str, Curve]) -> None:
    """Draw curves, keyed by colour, as an image in a PDF, and print the points.

    The curves are rasterised at 100 dots per inch, as plots of many points
    often are to keep their PDFs small, while the axes and text stay vector.
    """
    figure, axes = plt.subplots(figsize=(6.4, 4), dpi=100)
    x = np.linspace(0, 10, 2000)
    for colour, curve in curves.items():
        axes.plot(x, curve(x), color=colour, linewidth=1.5, rasterized=True)
    axes.set(xlabel="Time (s)", ylabel="Voltage (mV)", xlim=(0, 10), ylim=(0, 3))
    axes.spines[["top", "right"]].set_visible(False)
    save_pdf(name, figure, axes, [(0, 0), (10, 0), (0, 1)])
    plt.close(figure)


def draw_scatter(name: str, errors: Curve, scattered: Curve) -> None:
    """Draw two series of markers in a PDF, and print the points.

    The markers lie every half unit of x from 0.5 to 9.5: blue circles with
    error bars, whose caps are markers too, and orange squares, as a scatter
    plot draws them. The legend lies inside the axes.
    """
    figure, axes = plt.subplots(figsize=(6.4, 4), dpi=100)
    x = np.arange(0.5, 10, 0.5)
    axes.errorbar(
        x, errors(x), yerr=0.1, fmt="o", capsize=3, label=errors.__name__.capitalize()
    )
    axes.scatter(
        x,
        scattered(x),
        color="tab:orange",
        marker="s",
        label=scattered.__name__.capitalize(),
    )
    axes.set(xlabel="Time (s)", ylabel="Voltage (mV)", xlim=(0, 10), ylim=(0, 3))
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="center right")
    save_pdf(name, figure, axes, [(0, 0), (10, 0), (0, 1)])
    plt.close(figure)


def draw_titled(name: str, curve: Curve, titles: tuple[str, str]) -> None:
    """Draw a curve over x from 0 to 10 with titles of two lines, and print the points.

    The layout makes room for the titles.
    """
    figure, axes = plt.subplots(figsize=(6.4, 4), dpi=100, layout="constrained")
    x = np.linspace(0, 10, 2000)
    axes.plot(x, curve(x), color="black", linewidth=1.5)
    axes.set(xlabel=titles[0], ylabel=titles[1], xlim=(0, 10), ylim=(0, 3))
    axes.spines[["top", "right"]].set_visible(False)
    save(name, figure, axes, [(0, 0), (10, 0), (0, 1)])


def draw_slanted(name: str, curve: Curve, titles: tuple[str, str]) -> None:
    """Draw a curve over x from 0 to 10 with x tick labels at 45°, and print the points.

    The layout makes room for the slanted labels.
    """
    figure, axes = plt.subplots(figsize=(6.4, 4), dpi=100, layout="constrained")
    x = np.linspace(0, 10, 2000)
    axes.plot(x, curve(x), color="black", linewidth=1.5)
    axes.set(xlabel=titles[0], ylabel=titles[1], xlim=(0, 10), ylim=(0, 2.5))
    axes.tick_params(axis="x", labelrotation=45)
    axes.spines[["top", "right"]].set_visible(False)
    save(name, figure, axes, [(0, 0), (10, 0), (0, 1)])


def draw_markers(name: str, curves: dict[str, Curve]) -> None:
    """Draw black lines through markers, keyed by marker, and print the points.

    The markers lie every half unit of x from 0.5 to 9.5, and the legend lies
    below the axes, as spreadsheets draw it. The pixel positions of the
    legend's markers are printed too, for figrecover's -m option.
    """
    figure, axes = plt.subplots(figsize=(6.4, 4), dpi=100, layout="constrained")
    x = np.arange(0.5, 10, 0.5)
    for marker, curve in curves.items():
        style = {"marker": marker, "label": curve.__name__.capitalize()}
        if marker == "o":
            style["markerfacecolor"] = "white"
        axes.plot(x, curve(x), color="black", linewidth=1, **style)
    axes.set(xlabel="x", ylabel="y", xlim=(0, 10), ylim=(0, 3))
    axes.spines[["top", "right"]].set_visible(False)
    legend = axes.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=len(curves)
    )
    save(name, figure, axes, [(0, 0), (10, 0), (0, 1)], legend)


def draw_panels(name: str) -> None:
    """Draw four plots in a figure, labelled (a) to (d), and print their points.

    Each plot has axes of its own, framed as matplotlib draws them by
    default, over x from 0 to 10, but the last has logarithmic axes, over x
    from 1 to 10^4, with a margin around the curve. The labels are titles
    aligned with the left of the axes, and the layout makes room for them.
    """
    figure, grid = plt.subplots(2, 2, figsize=(8, 6), dpi=100, layout="constrained")
    linear = [
        (spectrum, ("Energy (eV)", "Intensity"), (0, 3)),
        (ringing, ("Time (s)", "Voltage (mV)"), (-1, 1)),
        (growth, ("Depth (m)", "Temperature (K)"), (0, 3)),
    ]
    x = np.linspace(0, 10, 2000)
    for axes, (curve, titles, ylim) in zip(grid.flat, linear, strict=False):
        axes.plot(x, curve(x), color="black", linewidth=1.5)
        axes.set(xlabel=titles[0], ylabel=titles[1], xlim=(0, 10), ylim=ylim)
    x = np.logspace(0, 4, 2000)
    grid[1, 1].loglog(x, lowpass(x), color="black", linewidth=1.5)
    grid[1, 1].set(xlabel="Frequency (Hz)", ylabel="Power (mW)")
    for label, axes in zip("abcd", grid.flat, strict=True):
        axes.set_title(f"({label})", loc="left")
    figure.savefig(HERE / f"{name}.png", facecolor="white")

    points = [[(0, 0), (10, 0), (0, 1)]] * 3 + [[(1, 0.1), (1e4, 0.1), (1, 1e3)]]
    for label, axes, plot_points in zip("abcd", grid.flat, points, strict=True):
        pixels = axes.transData.transform(plot_points).round().astype(int)
        print(f"{name} ({label})", " ".join(f"-l {x},{y}" for x, y in pixels))
    plt.close(figure)


def draw_shared(name: str) -> None:
    """Draw four plots sharing their axes, labelled (a) to (d), and print their points.

    The plots share their x-axes down each column and their y-axes along
    each row, over x from 0 to 10 and y from 0 to 3, so matplotlib labels the
    ticks only on the outer plots: the x-axes on the lower row and the
    y-axes on the left column. The labels are titles aligned with the left
    of the axes, and the layout makes room for them.
    """
    figure, grid = plt.subplots(
        2, 2, figsize=(8, 6), dpi=100, layout="constrained", sharex=True, sharey=True
    )
    x = np.linspace(0, 10, 2000)
    for axes, curve in zip(grid.flat, (spectrum, damped, growth, decay), strict=True):
        axes.plot(x, curve(x), color="black", linewidth=1.5)
    grid[0, 0].set(xlim=(0, 10), ylim=(0, 3))
    for axes in grid[1]:
        axes.set(xlabel="Time (s)")
    for axes in grid[:, 0]:
        axes.set(ylabel="Voltage (mV)")
    for label, axes in zip("abcd", grid.flat, strict=True):
        axes.set_title(f"({label})", loc="left")
    figure.savefig(HERE / f"{name}.png", facecolor="white")

    for label, axes in zip("abcd", grid.flat, strict=True):
        pixels = axes.transData.transform([(0, 0), (10, 0), (0, 1)]).round().astype(int)
        print(f"{name} ({label})", " ".join(f"-l {x},{y}" for x, y in pixels))
    plt.close(figure)


def save(
    name: str,
    figure: Figure,
    axes: Axes,
    points: list[tuple[float, float]],
    legend: Legend | None = None,
) -> None:
    """Save a plot, and print the pixel positions of reference points in it.

    With a legend, also print the pixel positions of its markers. Plots named
    in ``ALSO_PDF`` are saved as PDFs too.
    """
    figure.savefig(HERE / f"{name}.png", facecolor="white")
    # Display coordinates are pixels from the bottom-left corner, as the
    # image was drawn, before the PDF is.
    pixels = axes.transData.transform(points).round().astype(int)
    options = [f"-l {x},{y}" for x, y in pixels]
    options += [f"-m {x:.0f},{y:.0f}" for x, y in keys(legend)]
    print(name, " ".join(options))
    if name in ALSO_PDF:
        save_pdf(name, figure, axes, points, legend)
    plt.close(figure)


def save_pdf(
    name: str,
    figure: Figure,
    axes: Axes,
    points: list[tuple[float, float]],
    legend: Legend | None = None,
) -> None:
    """Save a plot as a PDF, and print the positions of reference points in points.

    With a legend, also print the positions of its markers. The PDF's date is
    left out, so that it is drawn the same each time.
    """
    figure.savefig(
        HERE / f"{name}.pdf", facecolor="white", metadata={"CreationDate": None}
    )
    # Display coordinates are pixels, of which there are dpi to 72 points,
    # but the legend keeps those of the PDF's drawing, at 72 to the inch.
    positions = axes.transData.transform(points) * 72 / figure.dpi
    options = [f"-l {x:.2f},{y:.2f}" for x, y in positions]
    options += [f"-m {x:.2f},{y:.2f}" for x, y in keys(legend)]
    print(f"{name}.pdf", " ".join(options))


def keys(legend: Legend | None) -> npt.NDArray[np.float64]:
    """Return the display coordinates of the markers in a legend's lines."""
    found = [
        # A legend line with one marker has three points, marked in the middle.
        handle.get_transform().transform(handle.get_xydata()[1])
        for handle in (legend.get_lines() if legend else [])
    ]
    return np.array(found, dtype=np.float64).reshape(-1, 2)


draw("spectrum", {"black": spectrum}, 3, grid=False)
draw("grid", {"black": damped}, 2.5, grid=True)
series = {"tab:blue": decay, "tab:orange": growth, "tab:green": wave, "black": trend}
draw("series", series, 3, grid=False)
titles = ("Time (s)", "Voltage (mV)")
draw("labelled", {"black": ringing}, None, grid=False, titles=titles)
titles = ("Frequency (Hz)", "Power (mW)")
draw_log("logarithmic", lowpass, titles, [(1, 0.1), (1e4, 0.1), (1, 1e3)])
titles = ("Size (KiB)", "Time (ms)")
draw_log("binary", quadratic, titles, [(1, 0.25), (16, 0.25), (1, 64)], base=2)
draw_titled("titled", spectrum, ("Photon energy\n(eV)", "Absorbance\n(arb. units)"))
draw_markers("markers", {"D": decay, "s": growth, "o": wave, "x": trend})
draw_slanted("slanted", damped, ("Time (s)", "Voltage (mV)"))
draw_panels("panels")
draw_shared("shared")
draw_rasterised("rasterised", {"tab:blue": decay, "tab:orange": growth})
draw_scatter("scatter", decay, growth)
