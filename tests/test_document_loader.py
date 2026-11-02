
from io import BytesIO

import pytest
from docx import Document

from hallucination_guard.document_loader import DocumentLoader


def test_load_txt():
    loader = DocumentLoader()

    result = loader.load(
        "sample.txt",
        b"Microsoft was founded in 1975.",
    )

    assert result == "Microsoft was founded in 1975."


def test_load_docx():
    document = Document()
    document.add_paragraph("Microsoft was founded in 1975.")

    buffer = BytesIO()
    document.save(buffer)

    loader = DocumentLoader()

    result = loader.load(
        "sample.docx",
        buffer.getvalue(),
    )

    assert "Microsoft was founded in 1975." in result


def test_empty_file():
    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load("empty.txt", b"")


def test_unsupported_format():
    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load("sample.mp4", b"some content")


def test_empty_document():
    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load("empty.txt", b"   ")
