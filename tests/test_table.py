from PIL import Image

from src.tables.table_extractor import TableExtractor


image_path = "data/samples/invoice_table.png"

image = Image.open(image_path).convert("RGB")

extractor = TableExtractor()

tables = extractor.detect_tables(image)

print("\nTABLE DETECTION")
print("================")

if not tables:
    print("No tables detected.")

else:
    for table in tables:

        print(f"Label: {table['label']}")
        print(f"Confidence: {table['confidence']:.4f}")
        print(f"BBOX: {table['bbox']}")
        print("-" * 40)