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
    figure.savefig(HERE / f"{name}.png", facecolor="white")

    # Display coordinates are pixels from the bottom-left corner.
    pixels = axes.transData.transform([(0, 0), (10, 0), (0, 1)]).round().astype(int)
    print(name, " ".join(f"-l {x},{y}" for x, y in pixels))
    plt.close(figure)


draw("spectrum", {"black": spectrum}, 3, grid=False)
draw("grid", {"black": damped}, 2.5, grid=True)
series = {"tab:blue": decay, "tab:orange": growth, "tab:green": wave, "black": trend}
draw("series", series, 3, grid=False)
titles = ("Time (s)", "Voltage (mV)")
draw("labelled", {"black": ringing}, None, grid=False, titles=titles)
