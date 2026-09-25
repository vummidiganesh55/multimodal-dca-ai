import importlib


try:
    pytesseract = importlib.import_module("pytesseract")
except ImportError as exc:
    raise ImportError(
        "pytesseract is required for OCR processing. "
        "Install it with: pip install pytesseract"
    ) from exc

from PIL import Image


class OCRProcessor:

    def __init__(self):
        self.engine = "tesseract"

        self.tesseract_path = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        pytesseract.pytesseract.tesseract_cmd = (
            self.tesseract_path
        )

    def process_image(self, image):

        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT
        )

        texts = []
        confidences = []

        for text, confidence in zip(
            data["text"],
            data["conf"]
        ):

            text = text.strip()

            try:
                confidence = float(confidence)
            except ValueError:
                continue

            if text and confidence >= 0:
                texts.append(text)
                confidences.append(confidence)

        extracted_text = " ".join(texts)

        if confidences:
            average_confidence = (
                sum(confidences) /
                len(confidences)
            )
        else:
            average_confidence = 0.0

        return {
            "text": extracted_text,
            "ocr_confidence": (
                average_confidence / 100.0
            ),
            "word_count": len(texts)
        }

    def process_file(self, image_path):

        image = Image.open(image_path)

        return self.process_image(image)