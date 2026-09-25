from PIL import Image
import torch
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection
)


class TableStructureExtractor:

    def __init__(self):

        self.model_name = (
            "microsoft/table-transformer-structure-recognition"
        )

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = TableTransformerForObjectDetection.from_pretrained(
            self.model_name
        )

        self.model.eval()

    def extract(self, table_image):

        # -----------------------------------------
        # Ensure image is PIL RGB
        # -----------------------------------------

        if not isinstance(table_image, Image.Image):
            table_image = Image.fromarray(table_image)

        table_image = table_image.convert("RGB")

        # -----------------------------------------
        # Preprocess
        # -----------------------------------------

        inputs = self.processor(
            images=table_image,
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
            [table_image.size[::-1]]
        )

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=target_sizes
        )[0]

        # -----------------------------------------
        # Build structure results
        # -----------------------------------------

        structures = []

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):

            structures.append({
                "label": self.model.config.id2label[
                    label.item()
                ],

                "confidence": float(score),

                "bbox": [
                    round(float(x), 2)
                    for x in box.tolist()
                ]
            })

        return structures

    # -----------------------------------------
    # Backward-compatible method
    # -----------------------------------------

    def extract_structure(self, table_image):

        return self.extract(table_image)