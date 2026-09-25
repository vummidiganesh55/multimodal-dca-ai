from src.document_loader import load_document
from src.ocr.paddle_ocr import PaddleOCRProcessor
from src.layout.layout_analyzer import LayoutAnalyzer


image_path = "data/samples/1.png"

# Load document
images = load_document(image_path)
image = images[0]

# OCR
ocr = PaddleOCRProcessor()
ocr_results = ocr.process(image_path)

print("OCR words:", len(ocr_results))

# LayoutLMv3
layout = LayoutAnalyzer()

result = layout.analyze(
    image,
    ocr_results
)

print("LayoutLMv3 successful!")
print("Number of tokens:", result["num_tokens"])
print(
    "Embedding shape:",
    result["last_hidden_state"].shape
)