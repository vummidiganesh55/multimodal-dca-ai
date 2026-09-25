import os

# ============================================================
# PaddlePaddle CPU configuration
# ============================================================

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

class PaddleOCRProcessor:
    """
    PaddleOCR wrapper for the Document AI pipeline.

    Input:
        Image file path

    Output:
        List of OCR words with:
        - text
        - bounding box
        - confidence
    """

    def __init__(self, lang="en"):

        try:
            
            from importlib import import_module

            PaddleOCR = import_module("paddleocr").PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is required. Install it with 'pip install paddleocr'."
            ) from exc

        self.ocr = PaddleOCR(
            lang=lang,
            enable_mkldnn=False
        )

    # ========================================================
    # OCR PROCESSING
    # ========================================================

    def process(self, image_path):

        if not image_path:
            raise ValueError(
                "image_path cannot be empty."
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        results = self.ocr.predict(
            image_path
        )

        extracted = []

        # ----------------------------------------------------
        # PaddleOCR result processing
        # ----------------------------------------------------

        for result in results:

            texts = result["rec_texts"]
            scores = result["rec_scores"]
            boxes = result["rec_boxes"]

            for text, score, box in zip(
                texts,
                scores,
                boxes
            ):

                # Ignore empty OCR results
                if not text or not text.strip():
                    continue

                extracted.append(
                    {
                        "text": text.strip(),

                        "bbox": box.tolist(),

                        "confidence": float(score)
                    }
                )

        return extracted

    # ========================================================
    # OCR SUMMARY
    # ========================================================

    def get_summary(self, ocr_results):

        if not ocr_results:

            return {
                "word_count": 0,
                "average_confidence": 0.0
            }

        confidences = []

        for item in ocr_results:

            confidence = item.get(
                "confidence"
            )

            if confidence is not None:

                confidences.append(
                    float(confidence)
                )

        average_confidence = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        return {
            "word_count": len(ocr_results),

            "average_confidence":
                round(
                    average_confidence,
                    4
                )
        }