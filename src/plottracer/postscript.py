"""Read vector plots from PostScript, EPS and Adobe Illustrator artwork."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .pdf import Box, Drawing, _load_pdf, load_pdf

_DOS_EPS = b"\xc5\xd0\xd3\xc6"
_AI_HINT = 'save the artwork in Illustrator with "Create PDF Compatible File" enabled'
_TIMEOUT = 60  # seconds per conversion


def load_postscript(
    path: str | os.PathLike[str],
    page: int = 1,
    box: Box | None = None,
    *,
    figure: str | None = None,
) -> Drawing:
    """Read a PostScript, EPS or Illustrator drawing, in points like a PDF.

    PDF-compatible Illustrator files need only the pdf extra. For PostScript,
    including self-contained legacy Illustrator artwork, Ghostscript must be
    on PATH. It writes a vector PDF for the same page and figure selection
    as ``load_pdf``. EPS coordinates start at the cropped bounding box.
    """
    source = Path(path)
    if source.suffix.lower() == ".ai":
        with source.open("rb") as stream:
            header = stream.read(1024).lstrip(b" \t\r\n\x04")
        if not header.startswith((b"%!", _DOS_EPS)):
            if b"%PDF-" not in header:
                raise ValueError(
                    f"cannot read Illustrator artwork {source!s}: {_AI_HINT}"
                )
            drawing = load_pdf(path, page, box, figure=figure)
            text = "".join(line.plain.casefold() for line in drawing.texts)
            if "illustrator" in text and "savedwithoutpdfcontent" in text:
                raise ValueError(
                    f"Illustrator file contains no PDF artwork: {_AI_HINT}"
                )
            return drawing

    pdf = _convert(source)
    return _load_pdf(os.fspath(path), page, box, figure, pdf)


def _convert(source: Path) -> bytes:
    """Convert with pdfwrite and keep the source name out of the interpreter."""
    gs = shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c")
    if gs is None:
        msg = "reading PostScript artwork needs Ghostscript; install gs on PATH"
        raise ValueError(msg)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Ghostscript has its own filename and option syntax. Give it fixed
        # names in a private directory, even for inputs named like options.
        shutil.copyfile(source, root / "input.ps")
        try:
            subprocess.run(  # noqa: S603
                [
                    gs,
                    "-q",
                    "-dSAFER",
                    "-dBATCH",
                    "-dNOPAUSE",
                    "-sDEVICE=pdfwrite",
                    "-dAutoRotatePages=/None",
                    "-dEPSCrop",
                    "-sOutputFile=output.pdf",
                    "-f",
                    "input.ps",
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_TIMEOUT,
                check=True,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"Ghostscript timed out reading {source!s}") from error
        except subprocess.CalledProcessError as error:
            msg = f"Ghostscript could not convert {source!s} (exit {error.returncode})"
            if source.suffix.lower() == ".ai":
                msg += f"; {_AI_HINT}"
            raise ValueError(msg) from error
        return (root / "output.pdf").read_bytes()
