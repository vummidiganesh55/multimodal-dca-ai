from pathlib import Path
from importlib import import_module


try:
    convert_from_path = import_module("pdf2image").convert_from_path
except ImportError as exc:
    raise ImportError(
        "pdf2image is required. Install it with: pip install pdf2image"
    ) from exc


class PDFProcessor:
    """
    Converts PDF pages into PIL images.
    """

    POPPLER_PATH = (
        r"E:\Asus Data\Downloads\Release-26.09.0-0"
        r"\poppler-26.09.0\Library\bin"
    )

    def __init__(self, dpi=200):
        self.dpi = dpi

    def convert_to_images(self, pdf_path):
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Input file must be a PDF."
            )

        images = convert_from_path(
            str(pdf_path),
            dpi=self.dpi,
            poppler_path=self.POPPLER_PATH
        )

        if not images:
            raise ValueError(
                "No pages found in PDF."
            )

        return images

    def get_page_count(self, pdf_path):
        images = self.convert_to_images(pdf_path)
        return len(images)