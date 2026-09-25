from src.ocr.paddle_ocr import PaddleOCRProcessor
from src.tables.table_json_builder import TableJSONBuilder


image_path = "data/samples/invoice_table.png"


# -------------------------
# OCR
# -------------------------

ocr = PaddleOCRProcessor(lang="en")

ocr_results = ocr.process(image_path)


# -------------------------
# Table JSON
# -------------------------

builder = TableJSONBuilder()

table = builder.build(ocr_results)


print("\nTABLE JSON")
print("================")

print("Columns:")

for column in table["columns"]:
    print(f"  - {column}")


print("\nRows:")

for row in table["rows"]:

    print(row)