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
  bars, are ignored, as is ink of other colours, such as the black arrow of
  an annotation among coloured curves. A frame around the axes, as
  matplotlib draws, is left out. Insets and legends inside a frame of their
  own are painted out with all they hold, so an inset's curve or a legend's
  key in a curve's colour is not traced as part of that curve, and an
  inset's markers are not taken for a series' with `-m`; step plots, bars,
  shaded spans, grid lines and boxes drawn around part of a curve are kept.
- `-m` tells series apart by the shapes of their markers, such as squares,
  diamonds, open circles and crosses, cut out of the legend, for plots whose
  series share a colour, as many spreadsheet charts do. It gives a point at
  the middle of each marker, allowing for lines and error bars across them.
- `-m` tells series apart by the colours of their markers too, so series of
  one shape in several colours, or that mix shapes and colours, are found
  apart. Markers that look alike at the image's size, such as circles,
  hexagons and octagons 8 pixels across, are fitted to a fraction of a pixel
  and told apart, with a warning of the keys alike; markers partly hidden
  under others are found by what shows of them; and figrecover prints how
  many markers of each series are ambiguous or probably hidden, and where,
  and warns of a series without markers. Legends in small type, whose lines
  are short, and JPEG images, down to quality 25, work too.
  `find_marker_series` gives these details.
- Reference points can be clicked on, after which figrecover prints the
  matching `-l` options to reuse for similar plots.
- With the `ocr` extra, figrecover reads the axes when `-p` is left out: it
  calibrates each image from its tick labels, including thousands separators,
  decimal commas, percentages and exponents, applying scale factors such as
  `1e-5` printed beside an axis or `×10⁻³` written in its title, traces from
  where the axis lines cross, and names the CSV columns after the axis titles.
  Titles can run over several lines, as `\n` breaks them, and be aligned with
  either end of their axes; y-axis titles written upwards are read turned
  upright, so OCR no longer splits or misreads them; and the second lines of
  tick labels broken over lines, such as units, are not taken for the x-axis
  title. It reads tick labels written upwards, reads y-axis tick labels
  written upwards or downwards, which OCR runs together and misreads, again
  turned a quarter turn whichever way it reads them more surely, reads those
  written at a slant again turned level if OCR misread them, places the ticks
  of rotated labels at their tick marks, whether the labels end at them or
  lie beside them, reads plots without axis lines, and refuses a second
  y-axis on the right. Tick labels in which OCR mistook letters for digits,
  such as `1O` or `l.5`, or dashes for minus signs, are read as they were
  written, and those misread as other digits, off the even steps of a linear
  axis, such as 758 among 250 and 500, are dropped. Titles keep their Greek
  letters, symbols, superscripts and subscripts, such as Δ, μm, CO₂ and mg
  L⁻¹, each written one way however OCR reads it.
- With the `ocr` extra, a figure of several plots, such as panels labelled
  (a), (b), (c), is read panel by panel: each is found by its axis lines,
  read from its own part of the image, and named after its label, taking the
  letters in reading order where OCR misreads or misses some, such as (b) as
  (p). `-o` writes a CSV for each panel, such as `out_a.csv`, and standard
  output gets one with a panel column. Each label is painted over before
  tracing, and is not read as another panel's title. Panels that share an
  axis, labelled only on the outer panels, as matplotlib's `sharex` and
  `sharey` draw them, take the axis and its title from the panel beside them
  that lines up with them. Panels that cannot be read, such as colour bars,
  are left out with a warning. An inset inside a frame in a panel's axes is
  read as a panel of its own, with its own axes, from the text around it,
  while the panel is traced with it painted out: `-o out.csv` writes a plot
  with an inset to `out.csv` and `out_inset.csv`, and an inset of panel (b)
  to `out_b_inset.csv`. Frames whose axes cannot be read, such as legends,
  stay painted out, without a warning. `read_panels` gives each inset after
  its panel, as its `inset_of`. Scale factors are read only where plotting
  libraries print them, so that of a panel below is not taken for one of the
  x-axis above.
- Logarithmic axes work too. figrecover finds them itself when it reads the
  axes, including tick labels written as powers, such as 10⁻² or 2⁵, even
  where OCR misses the superscripts; and `--log` names them for `-p`.
- Reversed axes, whose values fall from left to right or from bottom to top,
  work too, when figrecover reads the axes or with `-p` reference points on
  the axis lines.
- With the `pdf` extra, figrecover reads plots in PDF files exactly, from what
  they draw rather than from pixels: the axes from the PDF's text, with its
  superscripts and subscripts, and their ticks at their tick marks; and the
  curves as the points their paths pass through, told apart by colour, without
  axis lines, tick marks, grid lines, legend keys, frames, filled areas or
  points hidden beyond the axes. Markers, as in scatter plots, give the points
  they mark, exactly, told apart by colour, or by their shapes with `-m`,
  given positions in points; markers in legends and the caps of error bars are
  left out. Curves drawn as images in the PDF are traced as in an image and
  placed on the page. `--page` and `--box` choose the page and the part of it
  to read, such as one figure on a page of a paper, and positions on the page
  are in points.
- `--preview` saves a faded copy of the image with each recovered curve drawn
  in its colour, and `--remove-grid` paints over grid lines before tracing.
- Transparent pixels are treated as white, and plots with a dark background
  work as well as light ones. Pixel positions are those of pixel centres.
- Reference points must be finite numbers, and `-l` positions must lie on the
  image. `-o` and `--preview` refuse to overwrite the image, or each other.
- A typed Python API: `Calibration`, with linear or logarithmic axes,
  `load_image`, `remove_grid`, `find_colours`, `trace_curves`, `trace_curve`,
  `find_markers` and, with the `ocr` extra, `read_axes`, and `read_panels`,
  which returns a `Panel` for each plot in a figure; and, with the `pdf`
  extra, `load_pdf`, `read_pdf_axes`, `find_pdf_colours`, `trace_pdf_curves`,
  `find_pdf_markers`, `render_pdf` and `remove_pdf_grid`.
- Requires Python 3.14 on Linux, with NumPy and OpenCV.
