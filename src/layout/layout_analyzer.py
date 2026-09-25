from transformers import LayoutLMv3Processor, LayoutLMv3Model
from PIL import Image
import torch


class LayoutAnalyzer:

    def __init__(self):

        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False
        )

        self.model = LayoutLMv3Model.from_pretrained(
            "microsoft/layoutlmv3-base"
        )

        self.model.eval()

    def analyze(self, image, ocr_results):

        # --------------------------------------------------
        # 1. Ensure image is PIL RGB
        # --------------------------------------------------

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        # --------------------------------------------------
        # 2. Extract OCR words and bounding boxes
        # --------------------------------------------------

        words = [
            item["text"]
            for item in ocr_results
        ]

        boxes = [
            item["bbox"]
            for item in ocr_results
        ]

        # --------------------------------------------------
        # 3. Get image dimensions
        # --------------------------------------------------

        width, height = image.size

        # --------------------------------------------------
        # 4. Convert pixel coordinates
        #    to LayoutLM 0-1000 coordinates
        # --------------------------------------------------

        normalized_boxes = []

        for box in boxes:

            x1, y1, x2, y2 = box

            normalized_box = [
                max(
                    0,
                    min(
                        1000,
                        int(x1 / width * 1000)
                    )
                ),

                max(
                    0,
                    min(
                        1000,
                        int(y1 / height * 1000)
                    )
                ),

                max(
                    0,
                    min(
                        1000,
                        int(x2 / width * 1000)
                    )
                ),

                max(
                    0,
                    min(
                        1000,
                        int(y2 / height * 1000)
                    )
                )
            ]

            normalized_boxes.append(
                normalized_box
            )

        # --------------------------------------------------
        # 5. LayoutLMv3 Processor
        # --------------------------------------------------

        encoding = self.processor(
            images=image,
            text=words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True
        )

        # --------------------------------------------------
        # 6. LayoutLMv3 inference
        # --------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                **encoding
            )

        # --------------------------------------------------
        # 7. Return layout representation
        # --------------------------------------------------

        return {
            "last_hidden_state": outputs.last_hidden_state,
            "num_tokens": outputs.last_hidden_state.shape[1]
        }