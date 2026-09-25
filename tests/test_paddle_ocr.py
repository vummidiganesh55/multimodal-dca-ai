from src.ocr.paddle_ocr import PaddleOCRProcessor


image_path = "data/samples/1.png"

ocr = PaddleOCRProcessor()

result = ocr.process(image_path)

print("\nOCR RESULTS\n")

for item in result:
    print("Text:", item["text"])
    print("Bounding Box:", item["bbox"])
    print("Confidence:", round(item["confidence"], 4))
    print("-" * 50)