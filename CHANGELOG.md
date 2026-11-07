# Changelog

## 0.1.0 (unreleased)

The first release.

- The `figrecover` command writes the curves in an image of a plot as CSV, in
  the plot's own units, from three or more reference points.
- Curves are told apart by colour, even neighbouring colours of a colour map.
  `-n` traces the most common curve colours and prints the matching `-c`
  options to reuse for similar plots, skipping filled areas and grid lines of
  any shade, and `-c` traces the curves nearest to given colours. With several
  curves, the CSV has a `series` column.
- A curve that crosses a column more than once, such as a loop, gives a point
  on each side, and short marks beside a curve, such as the caps of error
  bars, are ignored. A frame around the axes, as matplotlib draws, is left
  out.
- `-m` tells series apart by the shapes of their markers, such as squares,
  diamonds, open circles and crosses, cut out of the legend, for plots whose
  series share a colour, as many spreadsheet charts do. It gives a point at
  the middle of each marker, allowing for lines and error bars across them.
- Reference points can be clicked on, after which figrecover prints the
  matching `-l` options to reuse for similar plots.
- With the `ocr` extra, figrecover reads the axes when `-p` is left out: it
  calibrates each image from its tick labels, including thousands separators,
  percentages and exponents, applying scale factors such as `1e-5` printed
  beside an axis, traces from where the axis lines cross, and names the CSV
  columns after the axis titles. It places the ticks of rotated labels at
  their tick marks, reads plots without axis lines, and refuses a second
  y-axis on the right.
- Logarithmic axes work too. figrecover finds them itself when it reads the
  axes, including tick labels written as powers, such as 10⁻² or 2⁵, even
  where OCR misses the superscripts; and `--log` names them for `-p`.
- `--preview` saves a faded copy of the image with each recovered curve drawn
  in its colour, and `--remove-grid` paints over grid lines before tracing.
- Transparent pixels are treated as white, and plots with a dark background
  work as well as light ones. Pixel positions are those of pixel centres.
- Reference points must be finite numbers, and `-l` positions must lie on the
  image. `-o` and `--preview` refuse to overwrite the image, or each other.
- A typed Python API: `Calibration`, with linear or logarithmic axes,
  `load_image`, `remove_grid`, `find_colours`, `trace_curves`, `trace_curve`,
  `find_markers` and, with the `ocr` extra, `read_axes`.
- Requires Python 3.14 on Linux, with NumPy and OpenCV.
