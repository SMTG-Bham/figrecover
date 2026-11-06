# figrecover

Recover data from images of plots. figrecover finds the curve in a picture of
a plot, such as a scanned figure or a screenshot, and writes its points as CSV
in the plot's own units.

It suits batches of similar plots: calibrate once, then digitize every image
the same way from the command line.

## Installation

figrecover runs on Linux only, and needs Python 3.14 (a standard build;
OpenCV does not support the free-threaded build yet). Install the command
with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/SMTG-Bham/figrecover
```

or with pip:

```bash
python -m pip install git+https://github.com/SMTG-Bham/figrecover
```

## Usage

### 1. Prepare the image

Crop the image to the axes and the curve. figrecover ignores everything left
of the y-axis and below the x-axis, then traces the most common dark grey level
in what remains. Legends, other curves and the top and right frame lines are
traced too if they are the same grey, so crop or paint them out. Images with a
dark background are inverted automatically.

### 2. Choose reference points

figrecover maps pixels to data with at least three reference points: usually
the origin, a labelled tick on the x-axis and one on the y-axis. Give their
data coordinates with `-p` and their pixel positions with `-l`, in the same
order. Pixel positions are measured from the bottom-left corner of the image.

If you leave out `-l`, figrecover opens the image and asks you to click on each
reference point in turn (press q to cancel). It then prints the matching `-l`
options, ready to reuse for the rest of a batch of similar plots:

```text
figrecover: locations: -l 80,44 -l 576,44 -l 80,147
```

### 3. Digitize

```bash
figrecover examples/spectrum.png \
    -p 0,0 -p 10,0 -p 0,1 \
    -l 80,44 -l 576,44 -l 80,147 \
    -o spectrum.csv --preview spectrum.preview.png
```

The CSV file has a header and one row per image column the curve crosses:

```text
x,y
0.0604839,0.504854
0.0806452,0.504854
...
```

Without `-o`, the CSV goes to standard output. The preview draws the recovered
curve in red and the reference points in blue, so you can check the result at a
glance:

![The input plot](examples/spectrum.png)
![The recovered curve drawn over the plot](examples/spectrum.preview.png)

| Option | Meaning |
| --- | --- |
| `-p`, `--point X,Y` | Data coordinates of a reference point. Repeat for at least three. |
| `-l`, `--location X,Y` | Pixel position of each reference point, in the same order. Default: click on them. |
| `-o`, `--output FILE` | CSV file to write. Default: standard output. |
| `--preview IMAGE` | Also save the image with the recovered curve drawn on it. |
| `--remove-grid` | Paint over grid lines before tracing. |

### Grids

A light grid needs no special handling, because the curve is darker than the
grid. A dark grid has to be removed first:

```bash
figrecover examples/grid.png \
    -p 0,0 -p 10,0 -p 0,1 -l 80,44 -l 576,44 -l 80,167 \
    --remove-grid --preview grid.preview.png
```

![A damped oscillation recovered after removing its grid](examples/grid.preview.png)

### Python

The same steps are available as a library:

```python
from figrecover import Calibration, load_image, trace_curve

image = load_image("examples/spectrum.png")
calibration = Calibration.from_points(
    data=[(0, 0), (10, 0), (0, 1)],
    pixels=[(80, 44), (576, 44), (80, 147)],
)
data = calibration.to_data(trace_curve(image, calibration.origin))
```

`remove_grid(image)` paints over grid lines first, if needed.

## How it works

1. The image is read as greyscale and stretched to the full 0–255 range.
   Transparent pixels become white, and a mostly dark image is inverted.
2. A least-squares fit through the reference points gives a linear map from
   pixels to data for each axis. The leftmost and lowest reference points mark
   the axes.
3. Pixels left of and below the axes are ignored, which drops the axes, their
   ticks and labels.
4. The curve is the most common grey level that is less than half as bright
   as the background and has enough pixels to span the plot.
5. Each image column the curve crosses gives one point, at the median height
   of the curve's pixels in that column.

On the examples, the recovered values are within 0.03 of the true curves, about
three pixels at the steepest point.

## Limitations

- One curve per image. Colour images are converted to grey, so curves are told
  apart by brightness only.
- The axes must be horizontal and vertical, with linear scales.
- Each column gives one point, so steep peaks come out slightly lower, and
  vertical lines in the plot area become single points.
- `--remove-grid` also removes straight horizontal or vertical stretches of the
  curve that are 120 pixels or longer.

## Development

The project uses [uv](https://docs.astral.sh/uv/) 0.13 or newer, which
installs the Python version pinned in `.python-version` if it is missing.
Create the environment and install the Git hooks once:

```bash
uv sync
uv run prek install
```

The hooks run Ruff, lockfile, workflow security and file hygiene checks before
each commit. Run the complete local checks with:

```bash
uv run prek run --all-files
uv run mypy
uv run pytest
uv build
```

Tests must keep full branch coverage. The tests replace OpenCV's windows with a
fake, so they run without a display. They check the recovered data against the
curves that `examples/make_examples.py` draws; after changing that script,
redraw the examples with `uv run examples/make_examples.py`, which also prints
their `-l` options.

`pyproject.toml` declares the oldest supported dependency versions and
`uv.lock` pins the versions used in development and CI. Dependabot refreshes
the lockfile, hooks and actions monthly.

Every push to `main` and every pull request runs the hooks and mypy, the tests
with the locked and the oldest allowed dependencies, and a build with smoke
tests of the sdist and wheel. Require the **All checks passed** job in branch
protection.

### Releasing

1. Add the repository as a
   [trusted publisher](https://docs.pypi.org/trusted-publishers/) on PyPI, with
   workflow `release.yml` and environment `pypi`, and create a `pypi`
   environment in the repository settings.
2. Bump the version, for example `uv version --bump minor`, and update
   `CHANGELOG.md`.
3. Publish a GitHub release tagged with the version prefixed by `v`, such as
   `v0.1.0`. The release workflow checks the tag, builds the distributions and
   uploads them to PyPI.

## Project layout

```text
.
├── .github/                 # CI, PyPI release and Dependabot
├── examples/                # example plots, their previews and the script that draws them
├── src/figrecover/
│   ├── calibration.py       # pixel-to-data mapping
│   ├── digitize.py          # image loading, grid removal and curve tracing
│   ├── interactive.py       # window for clicking on reference points
│   └── cli.py               # the figrecover command
└── tests/
```

## License

figrecover is free software under the GNU Lesser General Public License,
version 3 or later: see [`LICENSE`](LICENSE), which supplements the GNU
General Public License in [`COPYING`](COPYING). It is a modified version of
[PlotDigitizer](https://github.com/dilawar/PlotDigitizer) by Dilawar Singh,
rewritten in 2026.
