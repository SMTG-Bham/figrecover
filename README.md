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
of the y-axis and below the x-axis, and inside a frame around the axes, as
matplotlib draws, everything beyond the frame too. It then traces the most
common colour that stands out from the background in what remains, skipping
filled areas and grid lines. Legends are traced too if they are the same
colour as a curve, so crop or paint them out, unless the series are told apart
by their [markers](#markers). Dark backgrounds work as well as light ones. To have figrecover read the axes, keep their tick labels and
titles in the image.

### 2. Choose reference points

figrecover maps pixels to data with at least three reference points: usually
the origin, a labelled tick on the x-axis and one on the y-axis. Give their
data coordinates with `-p` and their pixel positions with `-l`, in the same
order. Pixel positions are measured from the bottom-left corner of the image,
so the centre of the bottom-left pixel is at 0.5,0.5.
With the `ocr` extra, you can leave out both and let figrecover read the tick
labels instead; see [Reading the axes](#reading-the-axes). For logarithmic
axes, see [Logarithmic axes](#logarithmic-axes).

figrecover traces the part of the image above and right of the lowest and
leftmost reference points, so take them on the axis lines. A reversed axis,
whose values fall from left to right or from bottom to top, starts at its
largest value, so take the corner where the axes cross rather than the
origin, and ticks on the axis lines.

If you leave out `-l`, figrecover opens the image and asks you to click on each
reference point in turn (press q to cancel). It then prints the matching `-l`
options, ready to reuse for the rest of a batch of similar plots:

```text
figrecover: locations: -l 80.5,43.5 -l 576.5,43.5 -l 80.5,147.5
```

### 3. Digitize

```bash
figrecover examples/spectrum.png \
    -p 0,0 -p 10,0 -p 0,1 \
    -l 80,44 -l 576,44 -l 80,147 \
    -o spectrum.csv --preview spectrum.preview.png
```

The CSV file has a header and a row for each image column the curve crosses,
or more than one where it crosses a column more than once, as a loop does:

```text
x,y
0.0705645,0.495146
0.0907258,0.495146
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
| `--log AXES` | The axes of the `-p` reference points that are logarithmic: `x`, `y` or `xy`. Default: neither. |
| `-n`, `--series N` | Trace the N most common curve colours, and print the matching `-c` options. Default: 1. |
| `-c`, `--colour RRGGBB` | Hexadecimal colour of a curve to trace, such as `1f77b4`. Repeat for several curves. |
| `-m`, `--marker X,Y` | Pixel position of the marker in a series' legend key, to find the series by its markers' shape. Repeat for each series. See [Markers](#markers). |
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
1,0.0705645,1.2767
1,0.0907258,1.28641
...
```

![Four crossing curves, each recovered in its own colour](examples/series.preview.png)

Which curve is longest can change from one plot to the next, so for a batch,
pass the printed `-c` options instead of `-n`. They fix the order of the
series, and each picks the curve whose colour is nearest, so they also work on
similar plots whose colours differ slightly, such as JPEG files. Give only the
`-c` options of the curves you want. The colours of plotting libraries work
too, such as `-c 1f77b4` for matplotlib's first colour.

`-n` skips colours that fill areas, those drawn mostly as straight lines
across the plot, such as grid lines, and very pale ones, less than 64 levels
from the background; trace a pale curve with `-c` instead. Curves in
neighbouring colours of a colour map, such as viridis, are told apart, and two
`-c` colours that match the same curve are an error.

### Markers

When the series share a colour and differ in their markers, as in many
spreadsheet charts, give the pixel position of the marker in each series'
legend key with `-m`, measured from the bottom-left corner like `-l`:

```bash
figrecover examples/markers.png \
    -p 0,0 -p 10,0 -p 0,1 -l 55,83 -l 627,83 -l 55,185 \
    -m 148,18 -m 260,18 -m 377,18 -m 483,18 --preview markers.preview.png
```

figrecover cuts each key's marker out of the legend, leaving out the line
through it, and finds the markers of that shape in the plot. The CSV has a
point at the middle of each marker, numbered by series in the order of the
`-m` options, and figrecover prints how many markers each series has:

```text
figrecover: markers found in each series: 18, 19, 18, 18
```

The preview rings each marker, and each key, in a colour for its series:

![Four series told apart by their markers](examples/markers.preview.png)

- The markers can be filled or open, such as squares, diamonds, triangles,
  circles and crosses, but each series needs a shape of its own. Use `-c`
  instead for series in different colours.
- Lines and error bars that cross a marker are allowed for, but a marker
  covered by another, as where series cross or crowd together, is missed or
  taken for the other's. Three markers are missed where the series above
  cross. Check the preview.
- The legend must be in the image, but may be in the plot area: its keys are
  not taken for markers.

### Grids

Grids need no special handling, light or dark, because grid lines run
straight across the plot and curves do not:

```bash
figrecover examples/grid.png \
    -p 0,0 -p 10,0 -p 0,1 -l 80,44 -l 576,44 -l 80,167 \
    --preview grid.preview.png
```

![A damped oscillation recovered from a plot with a dark grid](examples/grid.preview.png)

A grid drawn in the same colour as a curve has to be removed first, with
`--remove-grid`.

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
0.00688239,0.996743
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
  dropped. On a linear axis, labels must also take the even steps that
  plotting libraries label at: at low resolution OCR misreads digits as
  others, such as 750 as 758, close enough to their ticks to look in line,
  and those are dropped too. An axis with fewer than three labels in line
  cannot be read, so neither can a linear axis labelled at uneven values,
  such as 0, 273 and 373; give `-p` instead.
  Logarithmic axes are read too; see [Logarithmic axes](#logarithmic-axes).
  So are reversed axes, whose values fall from left to right or from bottom
  to top, as depth often does.
- Numbers can have thousands separators (`2,000` or `2 000`), a percent sign
  (`50%`, read in percent) or an exponent (`2.5e-3`). A tick mark that OCR
  reads as a dash beside a label, as it can where no axis line joins them, is
  ignored.
- Decimal commas work too, with points or spaces as thousands separators
  (`0,5` and `2.000`), as many European plots write them. A plot's labels
  decide: a label such as `0,5` reads as a number only with decimal commas,
  `0.5` only with decimal points, and `2.000` either way, so figrecover takes
  the way more labels read only one way, decimal points if neither wins. A
  single label misread with a comma does not outvote the rest.
- OCR can mistake letters for digits, and dashes for minus signs. In a label
  with a digit, O and o are read as 0, and l, I and | as 1 before a digit or
  a point, so `1O` and `l.5` read as 10 and 1.5; after a digit, they may be a
  unit, such as `5 l`, and are not read. Any dash before a number, such as
  an en or em dash, is a minus sign.
- A scale factor printed where plotting libraries put one, such as `1e-5` or
  `×10³` above the y-axis or below the x-axis, multiplies that axis's tick
  values. Text elsewhere that looks like one, such as a legend entry, does
  not. An axis offset, such as `+1e3`, cannot be read.
- Each tick lies at the middle of its label, or, for labels that end at their
  tick marks rather than centring on them, as rotated labels do, at the mark.
  Labels written upwards, turned a quarter turn, are read as well as those
  written across. OCR often misreads x-axis labels at a slant, such as 45°,
  so if it misreads one and finds another's box tilted, figrecover reads the
  labels again with the image turned 45° each way, which takes about a
  second more, and takes their texts from the turn that reads the most of
  them as numbers. Labels at 60° can still be missed, and those misread are
  dropped.
- figrecover traces the curves from the axis lines, not from the first ticks,
  so the margins that plotting libraries leave inside the axes are kept. If
  there are no axis lines, it starts just beyond the tick labels and their
  tick marks.
- A plot with a second y-axis on the right, with other values than the left
  one, cannot be read: crop the right axis out, or give `-p`.
- The titles are the nearest text below the x-axis tick labels and left of the
  y-axis ones, centred anywhere along their axes, so a title aligned with
  either end of its axis is read too. A title can run over several lines,
  each within a line's height of the one before and lined up with it, so a
  caption further below is left out. A y-axis title written upwards, as most
  are, is read again with the text left of the tick labels turned upright,
  which OCR reads more surely; one written across is read as it is. A column
  without a title is named `x` or `y`. Titles of a single letter are easily
  misread, such as `x` as `X`, or missed.
- Titles keep their Greek letters, symbols, superscripts and subscripts, such
  as `ΔE (%)`, `Wavelength (μm)` and `CO₂ (mg L⁻¹)`, and the CSV header
  writes them in UTF-8. OCR reads some symbols as either of two that look
  alike, so each is written one way: the increment sign as Greek Δ, the micro
  sign as Greek μ, and full-width signs, such as `％`, as plain ones. It can
  also read the raised minus sign of a superscript full size, so a minus sign
  before a superscript digit is raised, and `10−³` reads as 10⁻³, in titles
  and tick labels alike. OCR reads symbols less surely than letters, and has
  read `ΔE` written upwards as `ΔΔE`, so check the printed titles.

The titles of `examples/titled.png` run over two lines:

```bash
figrecover examples/titled.png
```

```text
figrecover: x-axis "Photon energy (eV)" with ticks at 0, 2, 4, 6, 8, 10
figrecover: y-axis "Absorbance (arb. units)" with ticks at 0, 0.5, 1, 1.5, 2, 2.5, 3
```

The x-axis tick labels of `examples/slanted.png` are written at 45°, which
OCR misreads as drawn, reading 2 as a Chinese character and 6 as 9, but reads
turned level:

```bash
figrecover examples/slanted.png
```

```text
figrecover: x-axis "Time (s)" with ticks at 0, 2, 4, 6, 8, 10
figrecover: y-axis "Voltage (mV)" with ticks at 0, 0.5, 1, 1.5, 2, 2.5
```

When a plot's labels cannot be read, give `-p` as before.

### Logarithmic axes

When figrecover reads the axes, it finds logarithmic ones itself: an axis is
logarithmic if more of its tick labels fall in line on a logarithmic scale
than on a linear one. It says so as it prints the ticks:

```bash
figrecover examples/logarithmic.png --preview logarithmic.preview.png
```

```text
figrecover: logarithmic x-axis "Frequency (Hz)" with ticks at 1, 10, 100, 1000, 10000
figrecover: logarithmic y-axis "Power (mW)" with ticks at 0.1, 1, 10, 100, 1000
```

![A low-pass filter's response recovered from a plot with logarithmic axes](examples/logarithmic.preview.png)

The tick labels can be decimals, such as `0.01` and `1000`, or powers with
superscript exponents, such as 10⁻², 2×10³ and 2⁵, as matplotlib draws them.
OCR often misses a superscript, reading 10² as `102` and 2⁵ as `25`, so
figrecover checks how a label of digits is drawn: if its last digit sits
raised above the others, the raised digits are an exponent and the full-size
ones its base, and a raised minus sign before the exponent makes it negative.
Each axis takes the base that most of its powers have. A label that reads as
that base followed by digits, but is not drawn that way, is still a power if
most of the others on its axis are, so an axis whose labels all read as
`100`, `101`, `102` is logarithmic when they are drawn as 10⁰, 10¹, 10²,
while labels really drawn as 100, 101, 102 stay linear, even if one is
misread as a power. As on a linear axis, three labelled ticks are needed, so
an axis labelled only at 10⁰ and 10¹ cannot be read.

The plot in `examples/binary.png` is labelled with powers of two, most of
whose superscripts OCR misses:

```bash
figrecover examples/binary.png
```

```text
figrecover: logarithmic x-axis "Size (KiB)" with ticks at 1, 2, 4, 8, 16
figrecover: logarithmic y-axis "Time (ms)" with ticks at 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64
```

Superscripts are small, so at low resolution OCR can misread them as other
digits, such as 2¹⁶ as `226`, and an axis whose exponents are all misread
alike is read wrongly. Check the printed ticks.

With `-p`, name the logarithmic axes with `--log x`, `--log y` or `--log xy`.
Reference points on a logarithmic axis must be positive, so take the lowest
labelled tick as the origin, rather than zero:

```bash
figrecover examples/logarithmic.png \
    -p 1,0.1 -p 1e4,0.1 -p 1,1e3 -l 93,66 -l 610,66 -l 93,380 --log xy
```

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

For logarithmic axes, pass `x_log=True` or `y_log=True` to
`Calibration.from_points`; `read_axes` sets them itself.

For several curves, `find_colours` returns the colours of the most common
curves as `(red, green, blue)` tuples, and `trace_curves` traces one curve per
colour. Pixel positions are those of pixel centres, and a curve that crosses a
column more than once gives a point for each crossing, sorted by x, then y:

```python
from figrecover import find_colours, trace_curves

colours = find_colours(image, 4, calibration.origin)
curves = trace_curves(image, colours, calibration.origin)
data = [calibration.to_data(curve) for curve in curves]
```

For series told apart by their markers, `find_markers` takes the pixel
position of each series' legend key, and returns the middle of each marker of
that shape:

```python
from figrecover import Calibration, find_markers, load_image

image = load_image("examples/markers.png")
calibration = Calibration.from_points(
    data=[(0, 0), (10, 0), (0, 1)],
    pixels=[(55, 83), (627, 83), (55, 185)],
)
keys = [(148, 18), (260, 18), (377, 18), (483, 18)]
markers = find_markers(image, keys, calibration.origin)
data = [calibration.to_data(series) for series in markers]
```

`load_image` returns an RGB array, and `remove_grid(image)` paints over grid
lines first, if needed.

## How it works

1. The image is read in colour and stretched to the full 0–255 range.
   Transparent pixels become white.
2. A least-squares fit through the reference points gives a linear map from
   pixels to data for each axis, or to the logarithm of the data for a
   logarithmic axis. The leftmost and lowest reference points mark the axes.
3. Pixels left of and below the axes are ignored, which drops the axes, their
   ticks and labels. A frame closing the axes, a thin line along the top from
   the y-axis that meets one down the right, is dropped with everything
   beyond it.
4. A coarse colour histogram gives the common colours: those with enough pixels
   to span the plot, merging shades less than 48 levels apart. Each is then
   counted again in a finer histogram, and split if it holds two or more
   colours with enough pixels, such as neighbouring colours of a colour map.
   The most common is the background, unless the colour along the top of the
   plot differs from it by 64 levels and has at least a third as many pixels:
   then the most common colour is a filled area. Antialiasing blends the edges
   of each curve with the background, and a blend can be common too, so a
   colour whose pixels mostly touch those of a colour further from the
   background is dropped.
5. The curves are the common colours that differ from the background by at
   least 64 levels in some channel, most pixels first, except those whose
   pixels lie mostly more than 4 pixels inside their shapes, which fill areas,
   and those drawn mostly as straight lines across 60% of the plot, as grid
   lines are. `-c` instead picks the common colour nearest to each given
   colour, up to 96 levels away.
6. Each pixel belongs to the common colour that it best matches when blended
   with the background, if it is at least half-way there, so a curve's
   antialiased edges count towards it. Specks of fewer than four pixels, such
   as where two other curves cross, are ignored.
7. In each image column, a curve's pixels form runs, split by gaps of more
   than 6 pixels, and runs that overlap one to one in neighbouring columns
   form branches. Each run of a branch across at least 5% of the plot gives a
   point at its median height, so both sides of a loop are traced, and shorter
   pieces beside a longer branch, such as the caps of error bars, are ignored.
   A column without such a run gives one point at the median height of all
   its pixels.

With `-m`, steps 4 to 7 give way to these:

1. Each key's marker is the piece of ink, 64 levels or more from the
   background, nearest the key's position. A key at least twice as wide as
   tall has a line through its marker, along the rows with ink across half
   the key or more; these rows are left out of the marker, as lines cross the
   markers in the plot at other angles.
2. A marker lies where a key's marker correlates with the plot by 0.6 or
   more, and the plot has at least 60% of the key's ink. Where the markers of
   several keys match less than three quarters of a marker's width apart, the
   best match wins. Each marker's middle is placed to a fraction of a pixel.

On the examples, the recovered values are within 0.03 of the true curves, about
three pixels at the steepest point, and the markers within 0.02 of the points
they mark, about two pixels.

## Limitations

- Curves are told apart by colour, or by the shapes of their markers, so
  curves of the same colour without markers are traced as one, and line
  styles are ignored. Series told apart by their markers give only the
  markers that can be seen, not the lines between them.
- A curve needs enough pixels to span the plot, so short curves and sparse
  dotted lines may not be found.
- Where curves cross, the one drawn underneath loses a few pixels, and its
  points there can be a pixel or two out.
- The axes must be horizontal and vertical, with linear or logarithmic scales,
  though either may be reversed. To read the axes, the x-axis must lie below
  the plot and the y-axis left of it.
- Each column gives one point per branch of a curve, so steep peaks come out
  slightly lower, vertical lines in the plot area become single points, and
  the two sides of a loop merge into one point where they come within 6
  pixels of each other.
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
│   ├── axes.py              # reading the axes from their tick labels
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
