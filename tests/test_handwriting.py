from PIL import Image

from src.handwriting.handwriting_recognizer import (
    HandwritingRecognizer
)


image = Image.open(
    "data/handwritten.png"
)

recognizer = HandwritingRecognizer()

result = recognizer.recognize(image)

print("\n" + "=" * 50)
print("HANDWRITING RESULT")
print("=" * 50)

print("Text:", result["text"])
print("Model:", result["model"])
print("Device:", result["device"])