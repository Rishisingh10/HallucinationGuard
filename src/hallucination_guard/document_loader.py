
from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


class DocumentLoader:

    SUPPORTED_FORMATS = {".pdf", ".txt", ".docx"}

    def load(self, file_name: str, file_bytes: bytes) -> str:
        extension = Path(file_name).suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        if not file_bytes:
            raise ValueError("The uploaded file is empty.")

        if extension == ".pdf":
            text = self._load_pdf(file_bytes)

        elif extension == ".docx":
            text = self._load_docx(file_bytes)

        else:
            text = self._load_txt(file_bytes)

        text = text.strip()

        if not text:
            raise ValueError(
                "No readable text was found in the document."
            )

        return text

    def _load_pdf(self, file_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(file_bytes))

        if reader.is_encrypted:
            raise ValueError(
                "Password-protected PDFs are not supported."
            )

        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages.append(page_text)

        return "\n".join(pages)

    def _load_docx(self, file_bytes: bytes) -> str:
        document = Document(BytesIO(file_bytes))

        content = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        for table in document.tables:
            for row in table.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]

                if cells:
                    content.append(" | ".join(cells))

        return "\n".join(content)

    def _load_txt(self, file_bytes: bytes) -> str:
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(
                "The TXT file must use UTF-8 encoding."
            ) from error
