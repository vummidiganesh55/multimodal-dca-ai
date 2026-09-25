from src.ocr.paddle_ocr import PaddleOCRProcessor
from src.tables.table_json_builder import TableJSONBuilder
from src.validation.validator import DocumentValidator


image_path = "data/samples/invoice_table.png"


# OCR
ocr = PaddleOCRProcessor(lang="en")
ocr_results = ocr.process(image_path)


# Table JSON
builder = TableJSONBuilder()
table = builder.build(ocr_results)


# Validation
validator = DocumentValidator()
results = validator.validate_table(table)


print("\nTABLE VALIDATION")
print("================")

for result in results:

    if result["valid"]:

        print(
            f"Row {result['row_number']}: "
            "VALID"
        )

    else:

        print(
            f"Row {result['row_number']}: "
            "INVALID"
        )

        print(
            f"Data: {result['data']}"
        )

        for issue in result["issues"]:

            print(f"  ❌ {issue}")

        print("-" * 60)