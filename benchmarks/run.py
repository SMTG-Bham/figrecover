"""Run the ordinary CLI on the paired charts, preserving its diagnostics and CSV."""

import argparse
import contextlib
import io
import json
from pathlib import Path

from datasets import cases

try:
    from plottracer.cli import main as plottracer
except ModuleNotFoundError:  # extractors from before the rename, such as 45db93d
    from figrecover.cli import main as plottracer


def extract(cache, results, case, mode):
    """Capture one CLI invocation. Cache per-item outputs for scoring and inspection."""
    path = results / mode / case["id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(".status").exists():
        return
    args = [str(cache / "images" / (case["id"] + ".png"))]
    args.extend(case["markers"] if mode == "markers" else ["-n", str(case["n"])])
    if mode != "automatic":
        args.extend(case["hints"])
    args.extend(["--metadata", str(path.with_suffix(".json"))])
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            status = plottracer(args)
        except SystemExit as exc:
            status = exc.code
    path.with_suffix(".csv").write_text(stdout.getvalue())
    path.with_suffix(".log").write_text(stderr.getvalue())
    path.with_suffix(".status").write_text(str(status))


def main():
    """Download the matched inputs and run all applicable modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    items = cases(args.cache)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "cases.json").write_text(json.dumps(items))
    for case in items:
        modes = ["automatic", "hinted", "markers"] if "hints" in case else ["automatic"]
        for mode in modes:
            extract(args.cache, args.results, case, mode)


if __name__ == "__main__":
    main()
