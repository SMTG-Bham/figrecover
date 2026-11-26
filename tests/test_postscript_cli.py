"""Exercise PostScript and Illustrator inputs through the CLI and real Ghostscript."""

import csv
import hashlib
import io
import json
import shlex
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from plottracer.cli import main

EXAMPLES = Path(__file__).parents[1] / "examples"
# Fractional page coordinates make raster tracing an observable regression.
VERTICES = ((81.25, 72.5), (160.5, 113.75), (261.75, 211.25), (338.25, 246.25))
EXPECTED = np.array(
    ((0.2125, 1.25), (1.005, 5.375), (2.0175, 15.125), (2.7825, 18.625))
)
CALIBRATION = [
    "-p",
    "0,0",
    "-p",
    "3,0",
    "-p",
    "0,20",
    "-l",
    "60,60",
    "-l",
    "360,60",
    "-l",
    "60,260",
]


@pytest.fixture
def ghostscript() -> None:
    """Run integration cases only where the actual converter is installed."""
    if not any(shutil.which(name) for name in ("gs", "gswin64c", "gswin32c")):
        pytest.skip("PostScript integration tests require Ghostscript")


def _plot(*, descending: bool = False) -> str:
    """Draw labelled axes and one polyline using ordinary PostScript operators."""
    parts = [
        "/Helvetica findfont 10 scalefont setfont",
        "0 setgray 1 setlinewidth",
        "newpath 60 60 moveto 360 60 lineto stroke",
        "newpath 60 60 moveto 60 260 lineto stroke",
    ]
    for x, label in ((60, "0"), (160, "1"), (260, "2"), (360, "3")):
        parts += [
            f"newpath {x} 56 moveto {x} 60 lineto stroke",
            f"{x - 2.78:g} 45 moveto ({label}) show",
        ]
    for y, label in ((60, "0"), (160, "10"), (260, "20")):
        parts += [
            f"newpath 56 {y} moveto 60 {y} lineto stroke",
            f"{53 - 5.56 * len(label):g} {y - 3.6:g} moveto ({label}) show",
        ]
    parts += ["1 0 0 setrgbcolor", "newpath"]
    for index, (curve_x, curve_y) in enumerate(VERTICES):
        ordinate = 320 - curve_y if descending else curve_y
        operator = "moveto" if index == 0 else "lineto"
        parts.append(f"{curve_x:g} {ordinate:g} {operator}")
    return "\n".join([*parts, "stroke"])


def _write_postscript(
    path: Path, *, pages: int = 1, bounds: str = "0 0 400 300"
) -> Path:
    """Write a compact vector fixture, with a different curve on its second page."""
    eps = " EPSF-3.0" if path.suffix.lower() == ".eps" else ""
    parts = [
        f"%!PS-Adobe-3.0{eps}",
        "%%Creator: Adobe Illustrator(R) 8.0",
        f"%%BoundingBox: {bounds}",
        f"%%Pages: {pages}",
        "%%EndComments",
        "<< /PageSize [400 300] >> setpagedevice",
    ]
    for page in range(1, pages + 1):
        parts += [f"%%Page: {page} {page}", _plot(descending=page == 2), "showpage"]
    path.write_text("\n".join([*parts, "%%EOF", ""]), encoding="ascii")
    return path


def _read_csv(output: str) -> npt.NDArray[np.float64]:
    """Return the numeric rows after checking the two calibrated column names."""
    rows = list(csv.reader(io.StringIO(output)))
    assert rows[0] == ["x", "y"]
    return np.array([[float(value) for value in row] for row in rows[1:]])


@pytest.mark.usefixtures("ghostscript", "without_ocr")
@pytest.mark.parametrize("suffix", [".eps", ".ps", ".ai", ".EPS"])
def test_recovers_original_vector_vertices_and_text(
    suffix: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_postscript(tmp_path / f"curve with spaces{suffix}")
    preview = tmp_path / "preview.png"
    sidecar = tmp_path / "metadata.json"

    assert (
        main(
            [
                str(source),
                "-c",
                "ff0000",
                "--preview",
                str(preview),
                "--metadata",
                str(sidecar),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    np.testing.assert_allclose(_read_csv(captured.out), EXPECTED, atol=1e-5)
    assert "ticks at 0, 1, 2, 3" in captured.err
    assert "ticks at 0, 10, 20" in captured.err
    rendered = cv2.imread(str(preview))
    assert rendered is not None
    assert rendered.shape == (600, 800, 3)
    document = json.loads(sidecar.read_text())
    (entry,) = document["inputs"]
    assert entry["path"] == str(source)
    assert entry["kind"] == suffix[1:].lower()
    assert entry["page"] == 1
    assert entry["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert entry["error"] is None
    (record,) = entry["plots"]
    assert record["calibration"]["source"] == "ticks"
    assert record["series"][0]["points"] == len(VERTICES)
    # Reproduction must use the original document, after temporary PDFs are gone.
    assert main(shlex.split(document["reproduce"])[1:]) == 0
    assert capsys.readouterr().out == captured.out


@pytest.mark.usefixtures("ghostscript", "without_ocr")
def test_reads_eps_with_a_nonzero_bounding_box(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_postscript(tmp_path / "offset.eps", bounds="10 20 390 290")

    assert main([str(source), "-c", "ff0000"]) == 0

    np.testing.assert_allclose(_read_csv(capsys.readouterr().out), EXPECTED, atol=1e-5)


@pytest.mark.usefixtures("ghostscript", "without_ocr")
def test_reads_dos_eps_with_a_binary_preview_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_postscript(tmp_path / "binary.eps")
    postscript = source.read_bytes()
    header = struct.pack(
        "<4s6IH", b"\xc5\xd0\xd3\xc6", 30, len(postscript), 0, 0, 0, 0, 0xFFFF
    )
    source.write_bytes(header + postscript)

    assert main([str(source), "-c", "ff0000"]) == 0

    np.testing.assert_allclose(_read_csv(capsys.readouterr().out), EXPECTED, atol=1e-5)


@pytest.mark.usefixtures("ghostscript", "without_ocr")
@pytest.mark.parametrize("page", [1, 2])
def test_selects_a_postscript_page_and_box_with_manual_calibration(
    page: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_postscript(tmp_path / "two pages.ps", pages=2)

    assert (
        main(
            [
                str(source),
                *CALIBRATION,
                "--page",
                str(page),
                "--box",
                "40,30,380,280",
                "-c",
                "ff0000",
            ]
        )
        == 0
    )

    expected = EXPECTED.copy()
    if page == 2:
        expected[:, 1] = 20 - expected[:, 1]
    np.testing.assert_allclose(_read_csv(capsys.readouterr().out), expected, atol=1e-5)


@pytest.mark.usefixtures("without_ocr")
def test_reads_pdf_compatible_illustrator_without_ghostscript(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "modern.ai"
    source.write_bytes((EXAMPLES / "labelled.pdf").read_bytes())
    monkeypatch.setenv("PATH", "")

    assert main([str(source)]) == 0

    rows = list(csv.reader(io.StringIO(capsys.readouterr().out)))
    assert rows[0] == ["Time (s)", "Voltage (mV)"]
    data = np.array([[float(value) for value in row] for row in rows[1:]])
    assert len(data) > 100
    np.testing.assert_allclose(
        data[:, 1], np.exp(-data[:, 0] / 4) * np.cos(2 * data[:, 0]), atol=1e-5
    )


@pytest.mark.usefixtures("ghostscript")
def test_reports_a_page_outside_the_converted_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_postscript(tmp_path / "one page.ps")

    assert main([str(source), "--page", "2"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "has no page 2" in captured.err
    assert str(source) in captured.err


@pytest.mark.usefixtures("ghostscript")
def test_reports_invalid_postscript_without_writing_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "broken.ps"
    source.write_text("%!PS-Adobe-3.0\nnot_a_postscript_operator\n", encoding="ascii")
    output = tmp_path / "curve.csv"

    assert main([str(source), "-o", str(output)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "plottracer: error:" in captured.err
    assert "Ghostscript" in captured.err
    assert not output.exists()


def test_explains_how_to_install_the_missing_converter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_postscript(tmp_path / "plot.eps")
    monkeypatch.setenv("PATH", "")

    assert main([str(source)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Ghostscript" in captured.err
    assert "install" in captured.err.lower()


def test_explains_how_to_export_unsupported_illustrator_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "private.ai"
    source.write_bytes(b"unsupported private Illustrator data")

    assert main([str(source)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Create PDF Compatible File" in captured.err
