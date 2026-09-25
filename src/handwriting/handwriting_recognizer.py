from PIL import Image
import torch

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class HandwritingRecognizer:
    """
    Handwriting recognition using Microsoft's TrOCR model.

    Model:
        microsoft/trocr-base-handwritten

    Input:
        PIL image containing handwritten text.

    Output:
        Recognized handwritten text.
    """

    def __init__(
        self,
        model_name="microsoft/trocr-base-handwritten"
    ):
        self.model_name = model_name

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("=" * 50)
        print("LOADING HANDWRITING RECOGNITION MODEL")
        print("=" * 50)

        self.processor = TrOCRProcessor.from_pretrained(
            self.model_name
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            f"TrOCR loaded successfully | "
            f"device={self.device}"
        )

    def recognize(self, image):
        """
        Recognize handwritten text from an image.
        """

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values

        pixel_values = pixel_values.to(
            self.device
        )

        with torch.no_grad():

            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128
            )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        text = text.strip()

        return {
            "text": text,
            "model": self.model_name,
            "device": self.device
        }

