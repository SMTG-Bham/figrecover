# Changelog

## 0.1.0 (unreleased)

The first release.

- The `figrecover` command writes the curves in an image of a plot as CSV, in
  the plot's own units, from three or more reference points.
- Curves are told apart by colour. `-n` traces the most common curve colours
  and prints the matching `-c` options to reuse for similar plots, and `-c`
  traces the curves nearest to given colours. With several curves, the CSV has
  a `series` column.
- Reference points can be clicked on, after which figrecover prints the
  matching `-l` options to reuse for similar plots.
- `--preview` saves a faded copy of the image with each recovered curve drawn
  in its colour, and `--remove-grid` paints over grid lines before tracing.
- Transparent pixels are treated as white, and plots with a dark background
  work as well as light ones.
- A typed Python API: `Calibration`, `load_image`, `remove_grid`,
  `find_colours`, `trace_curves` and `trace_curve`.
- Requires Python 3.14 on Linux, with NumPy and OpenCV.
