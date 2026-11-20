"""Recover local grey levels from a periodic dot screen in a scanned plot."""

import math

import cv2
import numpy as np
import numpy.typing as npt

_PATCH = 256  # enough samples to distinguish neighbouring screen frequencies
_MIN_PATCH = 128
_PEAK_SHARE = 0.0012  # fraction of high-frequency power in each of two peaks
_PEAK_CONTRAST = 3.5  # a screen peak compared with the power in its neighbourhood
_MIN_FREQUENCY, _MAX_FREQUENCY = 0.11, 0.45  # screen cycles per scan pixel
_HALF_CELL = 0.5  # cell edges in cycles from the centre
_SAMPLES = 16  # samples per pixel side to integrate the rotated screen cell


def recover_greys(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Average a dot screen over its measured repeat, in its own orientation."""
    side = min(_PATCH, *image.shape[:2]) // 2 * 2
    grey = image[:, :, 0]
    if side < _MIN_PATCH or not (
        np.array_equal(grey, image[:, :, 1]) and np.array_equal(grey, image[:, :, 2])
    ):
        return image
    # Blank margins carry little information about the screen. Use a textured
    # patch, with overlapping candidates so a curve can cross their borders.
    patches = (
        grey[r : r + side, c : c + side]
        for r in range(0, grey.shape[0] - side + 1, side // 2)
        for c in range(0, grey.shape[1] - side + 1, side // 2)
    )
    patch = max(patches, key=lambda p: float(p.var()))
    frequency = _frequency(patch)
    if frequency is None:
        return image
    kernel = _cell(*frequency)
    filtered = cv2.filter2D(grey, -1, kernel)
    return np.stack([filtered, filtered, filtered], axis=2).astype(np.uint8)


def _frequency(patch: npt.NDArray[np.uint8]) -> tuple[float, float] | None:
    """Find two narrow perpendicular spectral peaks at the same distance from zero."""
    side = len(patch)
    levels = patch.astype(np.float32)
    detail = levels - cv2.GaussianBlur(levels, (0, 0), 1.5)
    window = np.hanning(side)
    amplitude = np.fft.fftshift(np.fft.fft2(detail * window[:, None] * window[None, :]))
    power = abs(amplitude) ** 2
    peaks = cv2.GaussianBlur(power, (0, 0), 1.3)
    neighbourhood = cv2.GaussianBlur(power, (0, 0), 6)
    contrast = peaks / np.maximum(1, neighbourhood)
    nearby = np.fft.fftshift(np.fft.fftfreq(side))
    x, y = np.meshgrid(nearby, nearby)
    radius = np.hypot(x, y)
    band = (radius > _MIN_FREQUENCY) & (radius < _MAX_FREQUENCY) & (y >= 0)
    # Rolling the rotated array keeps the zero-frequency bin at its centre.
    pair = np.minimum(peaks, np.roll(np.rot90(peaks), 1, axis=0))
    strong = (
        np.minimum(contrast, np.roll(np.rot90(contrast), 1, axis=0)) >= _PEAK_CONTRAST
    )
    paired = np.where(band & strong, pair, 0)
    row, col = np.unravel_index(np.argmax(paired), paired.shape)
    if paired[row, col] <= peaks[band].sum() * _PEAK_SHARE * (_PATCH / side) ** 2:
        return None
    # The peak can lie between Fourier bins. A weighted local centre avoids
    # moving the integration window by a bin on either side of the peak.
    area = np.s_[row - 2 : row + 3, col - 2 : col + 3]
    weights = pair[area]
    return float(np.average(x[area], weights=weights)), float(
        np.average(y[area], weights=weights)
    )


def _cell(fx: float, fy: float) -> npt.NDArray[np.float32]:
    """Integrate the square repeating cell to an area-weighted pixel kernel."""
    radius = math.ceil(1 / math.hypot(fx, fy))
    width = 2 * radius + 1
    coordinates = (np.arange(width * _SAMPLES) + 0.5) / _SAMPLES - radius - 0.5
    x, y = np.meshgrid(coordinates, coordinates)
    inside = (abs(x * fx + y * fy) <= _HALF_CELL) & (abs(y * fx - x * fy) <= _HALF_CELL)
    weights = (
        inside.reshape(width, _SAMPLES, width, _SAMPLES)
        .sum(axis=(1, 3))
        .astype(np.float32)
    )
    return weights / weights.sum()
