# figrecover

Recover data from images of plots. figrecover finds the curves in a picture of
a plot, such as a scanned figure or a screenshot, and writes their points as
CSV in the plot's own units. Curves are told apart by colour.

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

To have figrecover read the axes for you (see
[Reading the axes](#reading-the-axes)), install the `ocr` extra instead. It
adds the [RapidOCR](https://github.com/RapidAI/RapidOCR) text reader and its
models, so it needs no network access when it runs:

```bash
uv tool install "figrecover[ocr] @ git+https://github.com/SMTG-Bham/figrecover"
```

## Usage

### 1. Prepare the image

Crop the image to the axes and the curves. figrecover ignores everything left
of the y-axis and below the x-axis, then traces the most common colour that
stands out from the background in what remains. Legends and the top and right
frame lines are traced too if they are the same colour as a curve, so crop or
paint them out. Dark backgrounds work as well as light ones. To have
figrecover read the axes, keep their tick labels and titles in the image.

### 2. Choose reference points

figrecover maps pixels to data with at least three reference points: usually
the origin, a labelled tick on the x-axis and one on the y-axis. Give their
data coordinates with `-p` and their pixel positions with `-l`, in the same
order. Pixel positions are measured from the bottom-left corner of the image.
With the `ocr` extra, you can leave out both and let figrecover read the tick
labels instead; see [Reading the axes](#reading-the-axes).

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
0.0604839,0.5
0.0806452,0.5
...
```

Without `-o`, the CSV goes to standard output. The preview draws the recovered
curve in its own colour over a faded copy of the plot, and the reference points
in blue, so you can check the result at a glance:

![The input plot](examples/spectrum.png)
![The recovered curve drawn over the plot](examples/spectrum.preview.png)

| Option | Meaning |
| --- | --- |
| `-p`, `--point X,Y` | Data coordinates of a reference point. Repeat for at least three. Default: [read the axes](#reading-the-axes). |
| `-l`, `--location X,Y` | Pixel position of each reference point, in the same order. Default: click on them. |
| `-n`, `--series N` | Trace the N most common curve colours, and print the matching `-c` options. Default: 1. |
| `-c`, `--colour RRGGBB` | Hexadecimal colour of a curve to trace, such as `1f77b4`. Repeat for several curves. |
| `-o`, `--output FILE` | CSV file to write. Default: standard output. |
| `--preview IMAGE` | Also save a faded copy of the image with the recovered curves drawn on it. |
| `--remove-grid` | Paint over grid lines before tracing. |

### Several curves

Give the number of curves with `-n`:

```bash
figrecover examples/series.png \
    -p 0,0 -p 10,0 -p 0,1 -l 80,44 -l 576,44 -l 80,147 \
    -n 4 --preview series.preview.png
```

figrecover traces the four most common colours that stand out from the
background, longest curve first, and prints their colours as `-c` options:

```text
figrecover: colours: -c 2ea12e -c ff7f0e -c 1f77b4 -c 040404
```

The CSV gains a `series` column that numbers the curves in that order:

```text
series,x,y
1,0.0604839,1.28155
1,0.0806452,1.29126
...
```

![Four crossing curves, each recovered in its own colour](examples/series.preview.png)

Which curve is longest can change from one plot to the next, so for a batch,
pass the printed `-c` options instead of `-n`. They fix the order of the
series, and each picks the curve whose colour is nearest, so they also work on
similar plots whose colours differ slightly, such as JPEG files. Give only the
`-c` options of the curves you want. The colours of plotting libraries work
too, such as `-c 1f77b4` for matplotlib's first colour.

`-n` skips light curves, such as pale blue or light grey ones, so that it does
not mistake grid lines for curves; trace them with `-c` instead.

### Grids

A light grid needs no special handling, because the curves stand out more
from the background than the grid does. A dark grid has to be removed first:

```bash
figrecover examples/grid.png \
    -p 0,0 -p 10,0 -p 0,1 -l 80,44 -l 576,44 -l 80,167 \
    --remove-grid --preview grid.preview.png
```

![A damped oscillation recovered after removing its grid](examples/grid.preview.png)

### Reading the axes

With the `ocr` extra installed, leave out `-p` and `-l`, and figrecover reads
the tick labels and axis titles itself:

```bash
figrecover examples/labelled.png --preview labelled.preview.png
```

It prints what it read, so you can check it:

```text
figrecover: x-axis "Time (s)" with ticks at 0, 2, 4, 6, 8, 10
figrecover: y-axis "Voltage (mV)" with ticks at -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1
```

and names the CSV columns after the axis titles:

```text
Time (s),Voltage (mV)
-0.00419701,0.999744
...
```

The preview circles each tick it read, on its axis line:

![A damped oscillation recovered from a plot whose axes were read](examples/labelled.preview.png)

Each image is calibrated from its own labels, so the plots in a batch can have
different axes. The other options, such as `-n`, `-c` and `--remove-grid`,
work as before.

- The x-axis tick labels are the lowest row of three or more numbers, and the
  y-axis tick labels the leftmost column of three or more numbers above them.
  Labels whose values are out of line with the rest, such as misread ones, are
  dropped. An axis with fewer than three labels in line cannot be read, so
  neither can a logarithmic axis.
- A scale factor printed beside an axis, such as `1e-5` or `×10³`, multiplies
  its tick values. An axis offset, such as `+1e3`, cannot be read.
- figrecover traces the curves from the axis lines, not from the first ticks,
  so the margins that plotting libraries leave inside the axes are kept. If
  there are no axis lines, it starts at the first ticks.
- The titles are the nearest text below the x-axis tick labels and left of the
  y-axis ones. A column without a title is named `x` or `y`. Titles of a single
  letter are easily misread, such as `x` as `X`, or missed.

When a plot's labels cannot be read, give `-p` as before.

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

or, with the `ocr` extra, reading the axes:

```python
from figrecover import load_image, read_axes, trace_curve

image = load_image("examples/labelled.png")
axes = read_axes(image)
data = axes.calibration.to_data(trace_curve(image, axes.calibration.origin))
print(axes.x_title, axes.y_title)  # Time (s) Voltage (mV)
```

For several curves, `find_colours` returns the colours of the most common
curves as `(red, green, blue)` tuples, and `trace_curves` traces one curve per
colour:

```python
from figrecover import find_colours, trace_curves

colours = find_colours(image, 4, calibration.origin)
curves = trace_curves(image, colours, calibration.origin)
data = [calibration.to_data(curve) for curve in curves]
```

`load_image` returns an RGB array, and `remove_grid(image)` paints over grid
lines first, if needed.

## How it works

1. The image is read in colour and stretched to the full 0–255 range.
   Transparent pixels become white.
2. A least-squares fit through the reference points gives a linear map from
   pixels to data for each axis. The leftmost and lowest reference points mark
   the axes.
3. Pixels left of and below the axes are ignored, which drops the axes, their
   ticks and labels.
4. A coarse colour histogram gives the common colours: those with enough pixels
   to span the plot, merging shades less than 48 levels apart. The most common
   is the background. Antialiasing blends the edges of each curve with the
   background, and a blend can be common too, so a colour whose pixels mostly
   touch those of a colour further from the background is dropped.
5. The curves are the common colours that differ from the background by at
   least 128 levels in some channel, most pixels first. `-c` instead picks the
   common colour nearest to each given colour, up to 96 levels away.
6. Each pixel belongs to the common colour that it best matches when blended
   with the background, if it is at least half-way there, so a curve's
   antialiased edges count towards it. Specks of fewer than four pixels, such
   as where two other curves cross, are ignored.
7. Each image column a curve crosses gives one point, at the median height of
   the curve's pixels in that column.

On the examples, the recovered values are within 0.03 of the true curves, about
three pixels at the steepest point.

## Limitations

- Curves are told apart by colour only, so curves of the same colour are traced
  as one, and line styles and markers are ignored.
- A curve needs enough pixels to span the plot, so short curves and sparse
  dotted lines may not be found.
- Where curves cross, the one drawn underneath loses a few pixels, and its
  points there can be a pixel or two out.
- The axes must be horizontal and vertical, with linear scales.
- Each column gives one point, so steep peaks come out slightly lower, and
  vertical lines in the plot area become single points.
- `--remove-grid` also removes straight horizontal or vertical stretches of
  curves that are 120 pixels or longer.

## Development

The project uses [uv](https://docs.astral.sh/uv/) 0.13 or newer, which
installs the Python version pinned in `.python-version` if it is missing.
uv prefers its own Python builds, so if the only uv-managed Python 3.14 is the
free-threaded `3.14t`, it picks that one and tries to compile OpenCV from
source; run `uv python install 3.14` once to add the standard build, which uv
then prefers. Create the environment and install the Git hooks once:

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
│   ├── digitize.py          # image loading, grid removal, curve colours and tracing
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
