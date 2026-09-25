from PIL import Image

from src.tables.table_structure import (
    TableStructureExtractor
)


image_path = "data/samples/invoice_table.png"

image = Image.open(image_path).convert("RGB")

extractor = TableStructureExtractor()

structures = extractor.extract_structure(image)

print("\nTABLE STRUCTURE")
print("================")

for item in structures:

    print(f"Label: {item['label']}")
    print(f"Confidence: {item['confidence']:.4f}")
    print(f"BBOX: {item['bbox']}")
    print("-" * 40)