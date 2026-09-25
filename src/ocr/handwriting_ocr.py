from PIL import Image
import torch

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


class HandwritingOCR:

    def __init__(self):

        model_name = "microsoft/trocr-base-handwritten"

        print("Loading TrOCR model...")

        self.processor = TrOCRProcessor.from_pretrained(
            model_name,
            use_fast=False
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name
        )

        self.model.eval()

        print("TrOCR loaded successfully!")

    def process(self, image_path):

        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values

        with torch.no_grad():

            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128
            )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return {
            "text": text,
            "confidence": None
        }


if __name__ == "__main__":

    ocr = HandwritingOCR()

    result = ocr.process(
        "data/samples/handwriting.png"
    )

    print("\nHANDWRITING OCR")
    print("================")
    print("Text:", result["text"])