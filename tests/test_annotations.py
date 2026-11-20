"""Regression plots for exclusions in images and page coordinates."""

import csv
import io
import json
import shlex
from pathlib import Path

import numpy as np
import pytest

from figrecover.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("vector", [False, True])
def test_repeatable_rectangles_exclude_intervals(
    vector: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factor = 0.72 if vector else 1
    image = EXAMPLES / ("series.pdf" if vector else "series.png")
    args = [str(image), "-p", "0,0", "-p", "10,0", "-p", "0,1", "-c", "1f77b4"]
    for x, y in [(80, 44), (576, 44), (80, 147)]:
        args.extend(("-l", f"{x * factor},{y * factor}"))
    for left, right in [(200, 230), (300, 350), (800, 830)]:
        args.extend(("--mask", f"{left * factor},0,{right * factor},{400 * factor}"))
    sidecar = tmp_path / "masked.json"
    assert main([*args, "--metadata", str(sidecar)]) == 0
    original = capsys.readouterr().out
    points = np.array(list(csv.reader(io.StringIO(original)))[1:], dtype=float)
    pixels = points[:, 0] * 49.6 + 80
    assert len(points) > 20
    for left, right in [(200, 230), (300, 350)]:
        assert not ((pixels > left + 0.1) & (pixels < right - 0.1)).any()
    replay = shlex.split(json.loads(sidecar.read_text())["reproduce"])
    assert main(replay[1:]) == 0
    assert capsys.readouterr().out == original
