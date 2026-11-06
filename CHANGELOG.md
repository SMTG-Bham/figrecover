# Changelog

## 0.1.0 (unreleased)

The first release.

- The `figrecover` command writes the curve in an image of a plot as CSV, in
  the plot's own units, from three or more reference points.
- Reference points can be clicked on, after which figrecover prints the
  matching `-l` options to reuse for similar plots.
- `--preview` saves the image with the recovered curve drawn on it, and
  `--remove-grid` paints over grid lines before tracing.
- Transparent pixels are treated as white, and images with a dark background
  are inverted automatically.
- A typed Python API: `Calibration`, `load_image`, `remove_grid` and
  `trace_curve`.
- Requires Python 3.14 on Linux, with NumPy and OpenCV.
