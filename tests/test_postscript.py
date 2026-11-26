"""Tests for Illustrator compatibility and PostScript conversion failures."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from matplotlib.figure import Figure

from plottracer.postscript import load_postscript


def test_illustrator_placeholder(tmp_path: Path) -> None:
    source = tmp_path / "no PDF artwork.ai"
    figure = Figure()
    figure.text(0.1, 0.7, "This is an Adobe Illustrator File that was")
    figure.text(0.1, 0.6, "saved without PDF Content.")
    figure.savefig(source, format="pdf")
    with pytest.raises(ValueError, match="Create PDF Compatible File"):
        load_postscript(source)


@pytest.mark.parametrize(
    ("suffix", "error", "diagnostic"),
    [
        (".eps", subprocess.TimeoutExpired("gs", 60), "timed out"),
        (".ps", subprocess.CalledProcessError(1, "gs"), "could not convert"),
        (".ai", subprocess.CalledProcessError(1, "gs"), "Create PDF Compatible File"),
    ],
)
def test_conversion_failure(
    tmp_path: Path, suffix: str, error: subprocess.SubprocessError, diagnostic: str
) -> None:
    source = tmp_path / f"@file with %d{suffix}"
    contents = b"%!PS-Adobe-3.0\n"
    source.write_bytes(contents)
    with (
        patch("plottracer.postscript.shutil.which", return_value="/usr/bin/gs"),
        patch("plottracer.postscript.subprocess.run", side_effect=error) as run,
        pytest.raises(ValueError, match=diagnostic),
    ):
        load_postscript(source)
    assert not run.call_args.kwargs["cwd"].exists()
    assert source.read_bytes() == contents
