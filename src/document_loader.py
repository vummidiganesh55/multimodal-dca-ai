from pathlib import Path
import pymupdf
from PIL import Image


SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


def load_document(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    # PDF → list of PIL images
    if path.suffix.lower() == ".pdf":
        images = []

        document = pymupdf.open(path)

        for page in document:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )
            images.append(image)

        document.close()

        return images

    # JPG / PNG → single image
    image = Image.open(path).convert("RGB")

    return [image]