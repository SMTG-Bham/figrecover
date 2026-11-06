"""Tests for the click windows, with OpenCV's GUI replaced by a fake."""

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from figrecover import Image, interactive

type MouseCallback = Callable[[int, int, int, int, object], None]

IMAGE = np.full((100, 200, 3), 255, dtype=np.uint8)


class FakeWindow:
    """Stand in for an OpenCV window, clicking at each of ``clicks`` in turn."""

    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []
        self.key = -1  # pressed once the clicks run out
        self.visible = 1.0
        self.shown: list[Image] = []
        self.on_mouse: MouseCallback | None = None
        self.destroyed = False

    def set_mouse_callback(self, _name: str, on_mouse: MouseCallback) -> None:
        self.on_mouse = on_mouse

    def wait_key(self, _delay: int) -> int:
        assert self.on_mouse is not None
        if not self.clicks:
            return self.key
        x, y = self.clicks.pop(0)
        self.on_mouse(cv2.EVENT_MOUSEMOVE, x, y, 0, None)
        self.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        return -1

    def destroy_window(self, _name: str) -> None:
        self.destroyed = True


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> FakeWindow:
    """Replace OpenCV's window functions with a fake window."""
    fake = FakeWindow()
    monkeypatch.setattr(cv2, "namedWindow", lambda _name: None)
    monkeypatch.setattr(cv2, "setMouseCallback", fake.set_mouse_callback)
    monkeypatch.setattr(cv2, "imshow", lambda _name, image: fake.shown.append(image))
    monkeypatch.setattr(cv2, "waitKey", fake.wait_key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda _name, _prop: fake.visible)
    monkeypatch.setattr(cv2, "destroyWindow", fake.destroy_window)
    return fake


def test_pick_points_returns_clicks_from_the_bottom_left(window: FakeWindow) -> None:
    window.clicks = [(10, 90), (150, 90), (10, 20)]

    points = interactive.pick_points(IMAGE, ["0,0", "1,0", "0,1"])

    assert points == [(10, 10), (150, 10), (10, 80)]
    assert window.destroyed
    assert window.shown[0].shape == (130, 200, 3)  # the image with a caption


def test_pick_points_stops_when_q_is_pressed(window: FakeWindow) -> None:
    window.clicks = [(10, 90)]
    window.key = ord("q")

    with pytest.raises(KeyboardInterrupt):
        interactive.pick_points(IMAGE, ["0,0", "1,0", "0,1"])
    assert window.destroyed


def test_pick_points_stops_when_the_window_closes(window: FakeWindow) -> None:
    window.visible = 0.0

    with pytest.raises(KeyboardInterrupt):
        interactive.pick_points(IMAGE, ["0,0"])
    assert window.destroyed
