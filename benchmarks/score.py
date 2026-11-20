"""Score cached CLI results and regenerate the paired accuracy tables."""

import argparse
import csv
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import reference

KINDS = ("ChartX line_chart", "ChartX line_chart_num", "PlotQA dot-line")
TOLERANCE = 0.05
ZERO_TOLERANCE = 0.01
INK_DISTANCE = 60  # diagnostic distance in 8-bit RGB


def number(cell):
    """Read the numeric units in the public tables, including percentages."""
    return float(str(cell).strip().replace(",", "").removesuffix("%"))


def numeric_f1(predicted, truth):
    """Greedy one-to-one matching of unique numbers, as in plotpick-validation."""
    values = sorted(set(predicted))
    targets = sorted(set(truth))
    denominator = len(values) + len(targets)
    hits = 0
    for target in targets:
        for index, value in enumerate(values):
            if (
                abs(value) < ZERO_TOLERANCE
                if target == 0
                else abs(value - target) <= abs(target) * TOLERANCE * (1 + 1e-9)
            ):
                values.pop(index)
                hits += 1
                break
    return 2 * hits / denominator if denominator else 1.0


def sampled(points, queries):
    """Average duplicate x values and interpolate inside the observed domain."""
    xs, ys = np.array(points).T
    grid, index = np.unique(xs, return_inverse=True)
    means = np.bincount(index, weights=ys) / np.bincount(index)
    return np.interp(queries, grid, means, left=np.nan, right=np.nan)


def samples_from_csv(path, queries):
    """Read only the numeric output columns. Keep the panel and series identity."""
    series = defaultdict(list)
    with path.open() as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            series[tuple(row[:-2])].append((float(row[-2]), float(row[-1])))
    return [sampled(points, queries) for points in series.values()]


def diagnose(message):
    """Classify a failed run by the first stage reported by the CLI."""
    if "marker" in message:
        return "marker key or shape"
    if any(
        s in message for s in ("tick", "axes", "axis", "calibrat", "reference point")
    ):
        return "axis reading or calibration"
    if "curve" in message:
        return "series detection or count"
    return "other CLI error"


def paired_errors(answers, truth, span):
    """Match series by positional error, charging one span per missing value."""
    truth = np.array(truth)
    if span <= 0:
        return []
    distance = np.array(
        [
            [
                np.mean(np.nan_to_num(abs(row - target) / span, nan=1.0))
                for row in answers
            ]
            for target in truth
        ]
    )
    # The matched public items have one to four series. Penalise missing
    # positions during assignment, but report error only where a trace exists.
    pairs = min(
        itertools.permutations(range(len(answers)), len(truth)),
        key=lambda perm: sum(distance[i, j] for i, j in enumerate(perm)),
    )
    errors = np.abs(truth - np.stack([answers[i] for i in pairs])) / span
    return errors[np.isfinite(errors)].tolist()


def score(case, results, mode):
    """Score all attempted charts, with zero F1 for an unreadable chart."""
    path = results / mode / case["id"]
    status = int(path.with_suffix(".status").read_text())
    reason = ""
    try:
        xs = (
            [number(v) for v in case["labels"]]
            if mode == "automatic"
            else case["ordinal"]
        )
    except ValueError:
        reason = "non-numeric x labels"
        xs = []
    if status and not reason:
        reason = diagnose(path.with_suffix(".log").read_text().lower())
    if reason:
        return 0.0, [], reason
    rows = samples_from_csv(path.with_suffix(".csv"), xs)
    plots = json.loads(path.with_suffix(".json").read_bytes())["inputs"][0]["plots"]
    if not rows:
        return (
            0.0,
            [],
            "no markers from any legend key" if mode == "markers" else "empty output",
        )
    predictions = [r[np.isfinite(r)].tolist() for r in rows]
    truth = [[number(v) for v in col] for col in case["ys"]]
    if case["kind"] == KINDS[-1]:
        # The reference grades the first annotated series against its best output.
        f1 = max(numeric_f1(candidate, case["truth"]) for candidate in predictions)
    else:
        # Numeric x values enter the predicted table only when actually read on
        # the axis and within the table's query domain. Do not copy truth labels.
        ticks = {
            value
            for plot in plots
            for value, _ in plot["calibration"]["x"]["ticks"]
            if min(xs) <= value <= max(xs)
        }
        f1 = numeric_f1(ticks.union(*predictions), case["truth"])
    span = case.get("span", np.ptp(truth))
    if len(rows) != len(truth):
        cause = (
            "one or more empty marker series"
            if mode == "markers"
            else "output series count differs"
        )
        return f1, [], cause
    errors = paired_errors(rows, truth, span)
    if all(not p for p in predictions):
        reason = "no sampled x in trace domain"
    elif f1 < 1:
        reason = mismatch(case, plots, predictions, xs)
    else:
        reason = "complete numeric match"
    return f1, errors, reason


def mismatch(case, plots, predictions, xs):
    """Identify observable failure signatures after traces have been returned."""
    series = [s for plot in plots for s in plot["series"] if s["colour"]]
    for row in series:
        if row["style"] == "markers" and any(
            other["colour"] == row["colour"] and other["style"] != "markers"
            for other in series
        ):
            return "line and markers selected in the same ink"
    if series and "inks" in case:
        actual = np.array([list(bytes.fromhex(c.lstrip("#"))) for c in case["inks"]])
        if any(
            np.linalg.norm(actual - list(bytes.fromhex(s["colour"])), axis=1).min()
            > INK_DISTANCE
            for s in series
        ):
            return "selected ink absent from annotation palette"
    if any(len(values) < len(xs) for values in predictions):
        return "partial coverage of labelled x positions"
    return "numeric mismatch at sampled positions"


def report(cache, results):
    """Render Markdown and retain per-item scores for auditing."""
    cases = json.loads((results / "cases.json").read_bytes())
    scores = defaultdict(lambda: defaultdict(list))
    errors = defaultdict(list)
    counts = Counter()
    items = []
    for case in cases:
        for mode in (
            ("automatic", "hinted", "markers") if "hints" in case else ("automatic",)
        ):
            value, differences, cause = score(case, results, mode)
            scores[mode][case["kind"]].append(value)
            errors[mode, case["kind"]].extend(differences)
            counts[mode, cause] += 1
            items.append(
                (case["id"], case["kind"], mode, value, cause, len(differences))
            )
    for dataset in ("chartx", "plotqa"):
        for row in reference(cache, dataset):
            kind = "ChartX " + row["chart_type"] if dataset == "chartx" else KINDS[-1]
            scores[row["system"]][kind].append(
                row["numeric_f1"] if dataset == "chartx" else row["best_series_f1"]
            )
    with (results / "items.csv").open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("item", "kind", "mode", "numeric_f1", "outcome", "scored_y"))
        writer.writerows(items)
    print(
        "| System | ChartX line (50) | ChartX numeric line (50) "
        "| PlotQA dot-line (102) | All 202 |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")
    for system, kinds in scores.items():
        columns = [
            f"{statistics.mean(kinds[k]):.3f}" if kinds[k] else "—" for k in KINDS
        ]
        combined = list(itertools.chain.from_iterable(kinds.values()))
        overall = (
            f"{statistics.mean(combined):.3f}" if len(combined) == len(cases) else "—"
        )
        print("|", " | ".join([system, *columns, overall]), "|")
    print(
        "\n| Run | Chart type | Scored y positions "
        "| Median absolute error / range | p95 |"
    )
    print("| --- | --- | ---: | ---: | ---: |")
    for (mode, kind), values in errors.items():
        med, p95 = np.quantile(values, [0.5, 0.95]) if values else (np.nan, np.nan)
        print(f"| {mode} | {kind} | {len(values)} | {med:.4f} | {p95:.4f} |")
    print(
        "\n| First observable outcome | Automatic "
        "| Axis hints | Axis and marker hints |"
    )
    print("| --- | ---: | ---: | ---: |")
    for cause in sorted({cause for _, cause in counts}):
        print(
            f"| {cause} | {counts['automatic', cause]} "
            f"| {counts['hinted', cause]} | {counts['markers', cause]} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    report(args.cache, args.results)
