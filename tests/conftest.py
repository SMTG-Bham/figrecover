"""Fixtures shared by the tests."""

import sys
from collections.abc import Iterator

import pytest

from figrecover.axes import _engine


@pytest.fixture
def without_ocr(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make the OCR engine of the ocr extra impossible to import."""
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    _engine.cache_clear()
    yield
    _engine.cache_clear()
