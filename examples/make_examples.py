# /// script
# requires-python = ">=3.14"
# dependencies = ["matplotlib>=3.10.7", "numpy>=2.3.3"]
# ///
"""Draw the example plots, whose curves the tests know exactly.

Run from the repository root with ``uv run examples/make_examples.py``. It
prints the pixel position of each reference point, for figrecover's -l option.
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


def save(
    name: str,
    figure: Figure,
    axes: Axes,
    points: list[tuple[float, float]],
    legend: Legend | None = None,
) -> None:
    """Save a plot, and print the pixel positions of reference points in it.

    With a legend, also print the pixel positions of its markers.
    """
    figure.savefig(HERE / f"{name}.png", facecolor="white")

    # Display coordinates are pixels from the bottom-left corner.
    pixels = axes.transData.transform(points).round().astype(int)
    options = [f"-l {x},{y}" for x, y in pixels]
    for handle in legend.get_lines() if legend else []:
        # A legend line with one marker has three points, marked in the middle.
        x, y = handle.get_transform().transform(handle.get_xydata()[1])
        options.append(f"-m {x:.0f},{y:.0f}")
    print(name, " ".join(options))
    plt.close(figure)


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
