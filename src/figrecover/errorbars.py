"""Read error bars: the bounds drawn through the points of a plot."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ErrorBars:
    """The error bars drawn through the points of a series.

    Positions are those of the points: pixels of an image, or points on a
    PDF's page, from its bottom-left corner. ``calibration.to_data`` turns
    each array into data; on a reversed axis, the lower or left end is the
    larger value.

    Attributes:
        points: An ``(n, 2)`` array of the ``(x, y)`` position of each point.
        low: An ``(n, 2)`` array of the x of the left end of each point's
            horizontal bar and the y of the lower end of its vertical bar,
            NaN where it has no such bar or the end cannot be seen.
        high: The same for the right and upper ends.
    """

    points: npt.NDArray[np.float64]
    low: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]
