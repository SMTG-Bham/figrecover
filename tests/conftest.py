"""Fixtures shared by the tests."""

import sys
from collections.abc import Iterator

import pytest

from figrecover.axes import _engine
from figrecover.pdf import _pdfium


@pytest.fixture
def without_ocr(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make the OCR engine of the ocr extra impossible to import."""
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    _engine.cache_clear()
    yield
    _engine.cache_clear()


@pytest.fixture
def without_pdf(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make the PDF library of the pdf extra impossible to import."""
    monkeypatch.setitem(sys.modules, "pypdfium2", None)
    _pdfium.cache_clear()
    yield
    _pdfium.cache_clear()
