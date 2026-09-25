from PIL import Image
import torch
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection
)


class TableExtractor:

    def __init__(self):

        self.model_name = "microsoft/table-transformer-detection"

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = TableTransformerForObjectDetection.from_pretrained(
            self.model_name
        )

        self.model.eval()

    def detect(self, image):

        # -----------------------------------------
        # Ensure image is RGB
        # -----------------------------------------

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        # -----------------------------------------
        # Preprocess image
        # -----------------------------------------

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # -----------------------------------------
        # Model inference
        # -----------------------------------------

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        # -----------------------------------------
        # Post-processing
        # -----------------------------------------

        target_sizes = torch.tensor(
            [image.size[::-1]]
        )

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=0.7,
            target_sizes=target_sizes
        )[0]

        # -----------------------------------------
        # Extract detected tables
        # -----------------------------------------

        tables = []

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):

            tables.append({
                "label": self.model.config.id2label[
                    label.item()
                ],

                "confidence": float(score),

                "bbox": [
                    round(float(x), 2)
                    for x in box.tolist()
                ]
            })

        return tables

    # ---------------------------------------------
    # Backward-compatible method
    # ---------------------------------------------

    def detect_tables(self, image):

        return self.detect(image)