from src.extraction.information_extractor import InformationExtractor


ocr_results = [
    {"text": "Invoice No: INV-1024", "bbox": [10, 10, 200, 40], "confidence": 0.98},
    {"text": "Date: 20/09/2026", "bbox": [10, 50, 200, 80], "confidence": 0.97},
    {"text": "Vendor: ABC Technologies", "bbox": [10, 90, 300, 120], "confidence": 0.96},
    {"text": "Subtotal: 10000", "bbox": [10, 130, 250, 160], "confidence": 0.95},
    {"text": "Tax: 1800", "bbox": [10, 170, 200, 200], "confidence": 0.95},
    {"text": "Total: 11800", "bbox": [10, 210, 200, 240], "confidence": 0.96},
]


extractor = InformationExtractor()

result = extractor.extract(ocr_results)

print("\nINFORMATION EXTRACTION")
print("======================")

for key, value in result.items():
    print(f"{key}: {value}")