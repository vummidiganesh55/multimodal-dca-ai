from src.ocr.paddle_ocr import PaddleOCRProcessor


class MultilingualOCR:

    def __init__(self, lang="en"):
        self.lang = lang
        self.processor = PaddleOCRProcessor(lang=lang)

    def process(self, image_path):

        results = self.processor.process(image_path)

        return {
            "language": self.lang,
            "text": results
        }


if __name__ == "__main__":

    ocr = MultilingualOCR(lang="en")

    result = ocr.process(
        "data/samples/invoice_table.png"
    )

    print("\nMULTILINGUAL OCR")
    print("================")

    print("Language:", result["language"])

    for item in result["text"]:
        print(
            item["text"],
            "| confidence:",
            round(item["confidence"], 4)
        )