"""Tests for levelling images of plots whose axes are turned, as scans can be."""

from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from figrecover import (
    Image,
    Levelling,
    Point,
    find_skew,
    interactive,
    level_image,
    load_image,
)
from figrecover.cli import main
from figrecover.level import LEVEL_THRESHOLD, _tilt

type Curve = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

EXAMPLES = Path(__file__).parents[1] / "examples"
ANGLES = [0.25, -0.25, 0.5, -0.5, 1, -1, 2, -2, 5, -5]
POINTS = ["-p", "0,0", "-p", "10,0", "-p", "0,1"]
# The pixel positions of POINTS that examples/make_examples.py prints: in
# spectrum.png, a plot with only a left and a bottom axis line, and in
# skewed.png, a framed plot turned 1 degree anticlockwise.
OPEN_LOCATIONS = [(80, 44), (576, 44), (80, 147)]
SKEWED_LOCATIONS = [(82.8, 39.8), (578.7, 48.5), (81.0, 142.5)]
SKEWED_ANGLE = 1  # degrees by which make_examples.py turns skewed.png


def spectrum(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in spectrum.png."""
    return 0.5 + 2 * np.exp(-2 * (x - 3) ** 2) + np.exp(-((x - 7) ** 2) / 2)


def damped(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in skewed.png."""
    return 1.2 + np.exp(-x / 4) * np.cos(2 * x)


def ringing(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the curve that examples/make_examples.py draws in labelled.png."""
    return np.exp(-x / 4) * np.cos(2 * x)


def turned(
    image: Image, angle: float, points: Sequence[Point] = ()
) -> tuple[Image, list[Point]]:
    """Turn an image anticlockwise about its middle, as a scan can be turned.

    The image is turned with cubic interpolation, keeping its size, and its
    corners filled in white, as in issue #33's reproduction. Pixel
    positions, measured from the bottom-left corner, are turned with it.
    """
    height, width = image.shape[:2]
    turn = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    picture = cv2.warpAffine(
        image, turn, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255,) * 3
    )
    # OpenCV counts columns and rows from the middle of the top-left pixel.
    places = np.array(points, dtype=np.float64).reshape(-1, 2)
    columns, rows = places[:, 0] - 0.5, height - places[:, 1] - 0.5
    moved = np.column_stack((columns, rows, np.ones(len(places)))) @ turn.T
    positions = [(x + 0.5, height - y - 0.5) for x, y in moved]
    return np.asarray(picture, dtype=np.uint8), positions


def plot_turned(framed: bool, angle: float) -> tuple[Image, list[Point], Curve]:
    """Return an example plot turned anticlockwise, its turned points, and its curve.

    The open plot is spectrum.png turned by the angle, and the framed plot is
    skewed.png, which is already turned, turned on to the angle.
    """
    if framed:
        image = load_image(EXAMPLES / "skewed.png")
        if angle == SKEWED_ANGLE:
            return image, SKEWED_LOCATIONS, damped
        return *turned(image, angle - SKEWED_ANGLE, SKEWED_LOCATIONS), damped
    image = load_image(EXAMPLES / "spectrum.png")
    return *turned(image, angle, OPEN_LOCATIONS), spectrum


def saved(tmp_path: Path, image: Image, name: str = "turned.png") -> str:
    """Save an RGB image as a file, and return its path."""
    path = tmp_path / name
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return str(path)


def read_csv(text: str, header: str = "x,y") -> npt.NDArray[np.float64]:
    """Parse CSV output, checking its header."""
    first, _, body = text.partition("\n")
    assert first == header
    return np.loadtxt(body.splitlines(), delimiter=",", ndmin=2)


def levelled_message(angle: float) -> str:
    """Return the message that figrecover prints when it levels an image."""
    turn = "clockwise" if angle > 0 else "anticlockwise"
    return f"figrecover: turned the image {abs(angle):.2f}° {turn} to level its axes"


def assert_close(
    data: npt.NDArray[np.float64], curve: Curve, span: float, count: int
) -> None:
    """Check that points follow a curve across the plot, as an unturned plot's do.

    The points lie within 0.003 of the y-axis's span of the curve by the
    median, 0.006 by the 95th percentile and 0.01 at most, as the axes read
    from an unturned plot give, and there are about as many as the plot's
    columns, ``count``, as there would be more if an axis line or a frame
    were traced as part of the curve.
    """
    errors = np.abs(data[:, 1] - curve(data[:, 0])) / span
    assert data[:, 0].min() < 0.1
    assert data[:, 0].max() > 9.9
    assert abs(len(data) - count) <= 0.02 * count
    assert np.median(errors) <= 0.003
    assert np.percentile(errors, 95) <= 0.006
    assert errors.max() <= 0.01


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("framed", [False, True], ids=["open", "framed"])
def test_find_skew_measures_the_turn_of_a_plot(framed: bool, angle: float) -> None:
    image, _, _ = plot_turned(framed, angle)
    assert find_skew(image) == pytest.approx(angle, abs=0.02)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("framed", [False, True], ids=["open", "framed"])
def test_traces_turned_plots_from_reference_points_where_they_lie(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], framed: bool, angle: float
) -> None:
    image, locations, curve = plot_turned(framed, angle)
    options = [option for x, y in locations for option in ("-l", f"{x:.2f},{y:.2f}")]

    assert main([saved(tmp_path, image), *POINTS, *options]) == 0

    captured = capsys.readouterr()
    assert captured.err == f"{levelled_message(angle)}\n"
    assert_close(read_csv(captured.out), curve, 3, 490 if framed else 493)


def labelled_span() -> float:
    """Return the span of labelled.png's y-axis: its curve's, and 5% each way."""
    return 1.1 * float(np.ptp(ringing(np.linspace(0, 10, 2000))))


@pytest.mark.parametrize(
    ("framed", "angle"),
    [(False, 0.25), (False, -2), (False, 5), (True, -0.25), (True, 1), (True, -5)],
)
def test_reads_the_axes_of_turned_plots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], framed: bool, angle: float
) -> None:
    if framed:
        image, _, curve = plot_turned(framed, angle)
        y_axis = "0, 0.5, 1, 1.5, 2, 2.5, 3"
        span, count = 3.0, 490
    else:
        image, _ = turned(load_image(EXAMPLES / "labelled.png"), angle)
        curve, y_axis = ringing, "-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1"
        span, count = labelled_span(), 450
    preview = tmp_path / "preview.png"

    assert main([saved(tmp_path, image), "--preview", str(preview)]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        f"{levelled_message(angle)}\n"
        'figrecover: x-axis "Time (s)" with ticks at 0, 2, 4, 6, 8, 10\n'
        f'figrecover: y-axis "Voltage (mV)" with ticks at {y_axis}\n'
    )
    header = "Time (s),Voltage (mV)"
    assert_close(read_csv(captured.out, header), curve, span, count)
    # The preview is drawn on the image as given.
    drawn = cv2.imread(str(preview))
    assert drawn is not None
    assert drawn.shape == image.shape


@pytest.mark.parametrize(
    "path", sorted(path for path in EXAMPLES.glob("*.png") if "." not in path.stem)
)
def test_leaves_level_images_as_they_are(path: Path) -> None:
    image = load_image(path)

    levelled, levelling = level_image(image)

    if path.stem == "skewed":
        assert levelling.angle == pytest.approx(SKEWED_ANGLE, abs=0.02)
    else:
        assert levelled is image
        assert levelling.angle == 0
        assert np.array_equal(levelling.matrix, np.eye(2, 3))
        points = np.array([[80.5, 44.25], [0, 0]])
        assert np.array_equal(levelling.points(points), points)
        assert np.array_equal(levelling.back(points), points)


def test_level_image_turns_by_a_given_angle() -> None:
    image = load_image(EXAMPLES / "spectrum.png")

    levelled, levelling = level_image(image, LEVEL_THRESHOLD / 2)

    # Turned by less than it would be when it finds the angle, onto an image
    # wider and taller by a pixel, to hold it all.
    assert levelling.angle == LEVEL_THRESHOLD / 2
    assert levelled.shape == (401, 641, 3)
    assert level_image(image, 0)[0] is image


def test_levelling_maps_points_into_the_levelled_image_and_back() -> None:
    image = load_image(EXAMPLES / "skewed.png")

    levelled, levelling = level_image(image)

    # The reference points lie on the axes, so once levelled, the first two
    # lie level and the first and third one above the other.
    (x0, y0), (_, y1), (x2, _) = levelling.points(SKEWED_LOCATIONS)
    assert y1 == pytest.approx(y0, abs=0.25)
    assert x2 == pytest.approx(x0, abs=0.25)
    np.testing.assert_allclose(
        levelling.back(levelling.points(SKEWED_LOCATIONS)), SKEWED_LOCATIONS
    )
    # The image is larger, to hold all of the turned image, and its corners
    # are filled in the background colour.
    assert levelled.shape[0] > image.shape[0]
    assert levelled.shape[1] > image.shape[1]
    assert np.all(levelled[0, 0] == 255)
    assert isinstance(levelling, Levelling)


def test_levelled_lines_run_along_the_middles_of_pixels() -> None:
    # The axis lines are a pixel thick, along the row whose middle is at
    # y = 43.5 and the column whose middle is at x = 80.5.
    image, corner = turned(load_image(EXAMPLES / "spectrum.png"), 2, [(80.5, 43.5)])

    levelled, levelling = level_image(image)

    # Once levelled, they lie along a row and a column again, dark all along
    # them, rather than blurred over two, each about half as dark.
    ((x, y),) = levelling.points(corner)
    assert x % 1 == pytest.approx(0.5, abs=0.05)
    assert y % 1 == pytest.approx(0.5, abs=0.05)
    row, column = round(levelled.shape[0] - y - 0.5), round(x - 0.5)
    assert np.all(levelled[row, column + 10 : column + 490] < 96)
    assert np.all(levelled[row - 290 : row - 10, column] < 96)


def lines(horizontal: bool, vertical: bool, angle: float) -> Image:
    """Draw long lines across a blank image, turned anticlockwise by an angle."""
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    if horizontal:
        cv2.line(image, (40, 250), (360, 250), (0, 0, 0), 1)
    if vertical:
        cv2.line(image, (40, 250), (40, 30), (0, 0, 0), 1)
    image[100:110, 100:105] = 0  # a speck, so the image has two colours
    return turned(image, angle)[0]


@pytest.mark.parametrize(
    ("horizontal", "vertical", "angle"),
    [(True, True, 1.5), (True, False, 0), (False, True, 0), (False, False, 0)],
)
def test_find_skew_needs_horizontal_and_vertical_lines_that_agree(
    horizontal: bool, vertical: bool, angle: float
) -> None:
    # The stretches of a curve along one line, close to horizontal, say,
    # can lie straight, but seldom lie square with others close to vertical.
    found = find_skew(lines(horizontal, vertical, 1.5))
    assert found == pytest.approx(angle, abs=0.02)


def test_find_skew_finds_no_lines_in_tiny_images() -> None:
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    image[2:6, 2:6] = 0
    assert find_skew(image) == 0


def test_a_line_must_be_measured_along_most_of_its_length() -> None:
    columns = np.arange(200)
    levels = np.zeros((40, 200))
    levels[20, :40] = 1  # ink across only a fifth of the line
    assert _tilt(levels, (0, 20), (199, 20)) is None
    # A wave, whose middles lie off a straight line through them.
    levels[:] = 0
    levels[np.round(20 + 3 * np.sin(columns / 10)).astype(np.intp), columns] = 1
    assert _tilt(levels, (0, 20), (199, 20)) is None
    levels[:] = 0
    levels[20, :] = 1
    tilt = _tilt(levels, (0, 20), (199, 20))
    assert tilt is not None
    angle, length, middle = tilt
    assert angle == pytest.approx(0, abs=1e-9)
    assert length == 200
    assert middle == pytest.approx((99.5, 20))


def test_leaves_turned_images_as_they_are_when_told_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image, locations, _ = plot_turned(True, 1)
    options = [option for x, y in locations for option in ("-l", f"{x},{y}")]

    assert main([saved(tmp_path, image), *POINTS, *options, "--no-level"]) == 0

    # The frame is traced as part of the curve, as it is not level.
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(read_csv(captured.out)) > 1000


def test_levels_images_by_a_given_angle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image, locations, curve = plot_turned(True, -2)
    options = [option for x, y in locations for option in ("-l", f"{x},{y}")]

    assert main([saved(tmp_path, image), *POINTS, *options, "--level", "-2"]) == 0

    captured = capsys.readouterr()
    assert captured.err == f"{levelled_message(-2)}\n"
    assert_close(read_csv(captured.out), curve, 3, 490)


def test_asks_for_locations_on_the_image_as_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image, locations, curve = plot_turned(True, 1)
    shown: list[Image] = []

    def pick_points(picture: Image, _labels: Sequence[str]) -> list[Point]:
        shown.append(picture)
        return locations

    monkeypatch.setattr(interactive, "pick_points", pick_points)

    assert main([saved(tmp_path, image), *POINTS]) == 0

    assert shown[0].shape == image.shape
    captured = capsys.readouterr()
    clicked = " ".join(f"-l {x:g},{y:g}" for x, y in locations)
    assert captured.err == (
        f"figrecover: locations: {clicked}\n{levelled_message(1)}\n"
    )
    assert_close(read_csv(captured.out), curve, 3, 490)


def test_finds_markers_by_their_keys_where_they_lie(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The pixel positions of the reference points and the legend's markers in
    # markers.png that examples/make_examples.py prints.
    locations = [(55, 83), (627, 83), (55, 185)]
    keys = [(148, 18), (260, 18), (377, 18), (483, 18)]
    image, places = turned(load_image(EXAMPLES / "markers.png"), -2, locations + keys)
    options = [option for x, y in places[:3] for option in ("-l", f"{x},{y}")]
    options += [option for x, y in places[3:] for option in ("-m", f"{x},{y}")]

    assert main([saved(tmp_path, image), *POINTS, *options]) == 0

    data = read_csv(capsys.readouterr().out, "series,x,y")
    curves: list[Curve] = [
        lambda x: 2.5 * np.exp(-x / 3),
        lambda x: 2.5 * (1 - np.exp(-x / 3)),
        lambda x: 1.25 + np.sin(x) / 2,
        lambda x: 0.4 + 0.2 * x,
    ]
    for series, curve in enumerate(curves, start=1):
        x, y = data[data[:, 0] == series, 1:].T
        # Some of the markers every half unit from 0.5 to 9.5 are hidden
        # where the series cross. Turning the image and levelling it blurs
        # thin markers, such as crosses, so a few more can be missed, and
        # their middles found a pixel or two further off, about 0.03.
        assert len(x) >= 15
        np.testing.assert_allclose(x, np.round(2 * x) / 2, atol=0.05)
        np.testing.assert_allclose(y, curve(x), atol=0.05)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["plot.png", "--level", "x"], "expected an angle in degrees, got 'x'"),
        (["plot.png", "--level", "inf"], "expected an angle in degrees, got 'inf'"),
        (["plot.png", "--level", "1", "--no-level"], "not allowed with argument"),
        (["plot.pdf", "--no-level"], "--level and --no-level level images, not PDF"),
    ],
)
def test_usage_errors_exit_with_status_2(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
