# Accuracy on public line charts

The 202 charts below are the paired line and dot-line items in the published PlotPick results. The test measures the path from the rendered image through axis calibration, series selection and CSV output. Each automatic invocation receives the reference series count with `-n`. The PlotQA hinted run supplies three independent axis tick positions with `-p` and `-l`. The markers run also supplies the centres of the annotated legend keys with `-m`. The benchmark never uses data-point bounding boxes to calibrate an image.

The ChartX sample contains 50 validation items in each of two line categories. The PlotQA sample contains the 102 dot-line items among the 529 scored test charts. Bars, bubble charts and other figure types fall outside these tables. The released paired results contain no separate scatter category. The ChartX annotations lack independent axis geometry for a hinted run. Blank cells indicate an unavailable comparison. [ChartX dataset and Apache 2.0 licence](https://huggingface.co/datasets/InternScience/ChartX), [PlotQA dataset and CC BY 4.0 licence](https://github.com/NiteshMethani/PlotQA), [PlotPick per-item scores](https://huggingface.co/datasets/tommycarstensen/plotpick-validation).

## Results

| System | ChartX line (50) | ChartX numeric line (50) | PlotQA dot-line (102) | All 202 |
| --- | ---: | ---: | ---: | ---: |
| automatic | 0.423 | 0.409 | 0.230 | 0.322 |
| hinted | — | — | 0.553 | — |
| markers | — | — | 0.047 | — |
| DePlot | 0.860 | 0.908 | 0.896 | 0.890 |
| Gemini 3.1 Flash-Lite | 0.956 | 0.978 | 0.931 | 0.949 |
| Gemini 3 Flash | 0.963 | 0.980 | 0.947 | 0.959 |
| GPT-5.4 mini | 0.962 | 0.969 | 0.898 | 0.931 |
| GPT-5.4 nano | 0.946 | 0.968 | 0.771 | 0.863 |
| Claude Haiku 4.5 | 0.923 | 0.958 | 0.829 | 0.884 |
| Mistral Large 3 | 0.790 | 0.923 | — | — |
| Mistral Medium 3.5 | 0.922 | 0.966 | — | — |
| Mistral Small 4 | 0.851 | 0.946 | — | — |
| Claude Sonnet 4.6 | 0.944 | 0.974 | 0.941 | 0.950 |

| Run | Chart type | Scored y positions | Median absolute error / range | p95 |
| --- | --- | ---: | ---: | ---: |
| automatic | ChartX line_chart | 281 | 0.0008 | 0.4924 |
| automatic | ChartX line_chart_num | 323 | 0.0016 | 0.9953 |
| automatic | PlotQA dot-line | 309 | 0.0013 | 0.0225 |
| hinted | PlotQA dot-line | 630 | 0.0013 | 0.1260 |
| markers | PlotQA dot-line | 14 | 0.0051 | 0.0521 |

| First observable outcome | Automatic | Axis hints | Axis and marker hints |
| --- | ---: | ---: | ---: |
| axis reading or calibration | 82 | 0 | 0 |
| complete numeric match | 25 | 5 | 0 |
| line and markers selected in the same ink | 8 | 2 | 0 |
| no markers from any legend key | 0 | 0 | 65 |
| no sampled x in trace domain | 0 | 2 | 0 |
| non-numeric x labels | 42 | 0 | 0 |
| numeric mismatch at sampled positions | 22 | 0 | 0 |
| one or more empty marker series | 0 | 0 | 34 |
| partial coverage of labelled x positions | 22 | 54 | 3 |
| selected ink absent from annotation palette | 1 | 8 | 0 |
| series detection or count | 0 | 31 | 0 |

## How the scores are calculated

Numeric F1 is the mean of per-chart scores. Failed runs and unaddressable category coordinates receive zero. Within each chart, unique values are matched greedily one-to-one at 5% relative tolerance. A ground-truth zero matches absolute values below 0.01. ChartX scores the whole table, including numeric x ticks actually read within the query domain. PlotQA scores the first annotated series against the best predicted column, following the reference. F1 ignores series labels and repeated values. The overall column averages the three groups by their actual number of charts. [Reference scoring implementation](https://github.com/tommycarstensen/plotpick-validation/blob/89693d4e52577cd4e334e99f22834019d98e47fe/benchmarks/shared.py), [PlotQA evaluation protocol](https://github.com/tommycarstensen/plotpick-validation/blob/89693d4e52577cd4e334e99f22834019d98e47fe/benchmarks/score_plotqa.py).

The scorer averages y values at duplicate output x coordinates and linearly interpolates at the table's x positions, strictly within the observed trace. For PlotQA, automatic extraction is queried at the printed numeric labels, while manual calibration uses annotated category ordinals. This matters for charts with equally spaced but nonconsecutive years. The scorer assigns series one-to-one to minimise absolute error, charging one range for a missing position during assignment. It reports y errors only for available positions on outputs with the expected series count. For PlotQA the denominator is the span of the labelled y ticks. For ChartX it is the span of the ground-truth y data, since tick geometry is unavailable. The median and p95 pool individual positions. Failures and missing positions reduce coverage, so compare the counts as well as the percentiles.

Each run has one observable outcome. Categorical labels take precedence over an axis-reading failure. After extraction, the bins distinguish selecting markers and a line in the same ink, a colour over 60 RGB levels in Euclidean distance from all annotated PlotQA inks, missing x coverage, and residual numeric disagreement. The category describes a symptom and can coexist with other errors. Individual item IDs, scores, outcomes and counts are in [items.csv](items.csv). The largest gaps have dedicated issues for [axis reading](https://github.com/SMTG-Bham/figrecover/issues/52), [marker keys](https://github.com/SMTG-Bham/figrecover/issues/53), [category labels](https://github.com/SMTG-Bham/figrecover/issues/54), and [series detection and coverage](https://github.com/SMTG-Bham/figrecover/issues/55).

## Reproduce

The measured extractor is commit `45db93d`, using the project's locked test dependencies on Linux and Python 3.14.6. The scripts download about 900 MB and verify the SHA-256 of every input, including the original PlotQA test annotations. Their revisions and digests are in [datasets.py](datasets.py). Only the 202 required PNGs are extracted. Downloads and intermediate files stay under the two directories supplied on the command line. Redistributing the chart images is unnecessary.

To reproduce the published result from the repository root, keep the benchmark scripts from this checkout and run them with the measured extractor in a separate worktree.

```sh
git worktree add --detach /tmp/figrecover-measured 45db93d
uv sync --project /tmp/figrecover-measured --locked --group test
scripts="$(pwd)/benchmarks"
uv run --project /tmp/figrecover-measured --locked --with pyarrow==26.0.1 python "$scripts/run.py" /tmp/figrecover-data /tmp/figrecover-results
uv run --project /tmp/figrecover-measured --locked --with pyarrow==26.0.1 python "$scripts/score.py" /tmp/figrecover-data /tmp/figrecover-results
```

The results directory retains each CLI invocation's CSV, diagnostics, metadata and exit status. The score command also creates `items.csv`. Reuse the data directory for other extractor versions and choose a fresh results directory for each version.
