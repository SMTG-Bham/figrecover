"""A window for clicking on points in an image."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import cv2
import numpy as np

from .calibration import Point
from .digitize import Image

WINDOW = "figrecover"
_QUIT_KEYS = {ord("q"), 27}  # q and Escape


def pick_points(image: Image, labels: Sequence[str]) -> list[Point]:
    """Ask the user to click on one point per label, in order.

    Args:
        image: The image to show.
        labels: A short description of each point, such as ``"0,0"``.

    Returns:
        The pixel coordinates of the clicks, measured from the bottom-left
        corner.

    Raises:
        KeyboardInterrupt: If the user presses q or Escape or closes the
            window first.
    """
    height = image.shape[0]
    clicks: list[Point] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and y < height:  # not on the caption
            clicks.append((x, height - y))

    with _window(on_mouse):
        while len(clicks) < len(labels):
            label = labels[len(clicks)]
            left = len(labels) - len(clicks)
            caption = _caption(image, f"Click on {label} ({left} left)")
            cv2.imshow(WINDOW, cv2.cvtColor(caption, cv2.COLOR_RGB2BGR))
            if not _keep_open():
                raise KeyboardInterrupt
    return clicks


@contextmanager
def _window(on_mouse: Callable[[int, int, int, int, object], None]) -> Iterator[None]:
    """Open the window with a mouse callback, and close it afterwards."""
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        yield
    finally:
        cv2.destroyWindow(WINDOW)


def _keep_open() -> bool:
    """Process window events and report whether the user wants to carry on."""
    key = cv2.waitKey(20) & 0xFF
    visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    return visible and key not in _QUIT_KEYS


def _caption(image: Image, text: str) -> Image:
    """Return the image with a line of text in a black band beneath it."""
    band = np.zeros((30, image.shape[1], 3), dtype=np.uint8)
    white = (255, 255, 255)
    cv2.putText(
        band, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA
    )
    return np.vstack((image, band))
