from src.ocr.paddle_ocr import PaddleOCRProcessor


image_path = "data/samples/invoice_table.png"

ocr = PaddleOCRProcessor(lang="en")

results = ocr.process(image_path)

print("\nTABLE OCR RESULTS")
print("=================")

for item in results:
    print(f"Text: {item['text']}")
    print(f"BBOX: {item['bbox']}")
    print(f"Confidence: {item['confidence']:.4f}")
    print("-" * 50)