from src.ocr.paddle_ocr import PaddleOCRProcessor
from src.tables.table_json_builder import TableJSONBuilder
from src.validation.validator import DocumentValidator
from src.anomaly.rules import RuleBasedRiskDetector


image_path = "data/samples/invoice_table.png"


# OCR
ocr = PaddleOCRProcessor(lang="en")
ocr_results = ocr.process(image_path)


# Table extraction
builder = TableJSONBuilder()
table = builder.build(ocr_results)


# Validation
validator = DocumentValidator()
validation_results = validator.validate_table(table)


# Risk detection
risk_detector = RuleBasedRiskDetector()

risk = risk_detector.analyze(
    validation_results
)


print("\nDOCUMENT RISK ANALYSIS")
print("======================")

print(
    f"Risk Score: {risk['risk_score']}/100"
)

print(
    f"Risk Level: {risk['risk_level']}"
)

print(
    f"Total Issues: {len(risk['issues'])}"
)

print("\nIssues:")

for issue in risk["issues"][:10]:

    print(
        f"Row {issue['row']}: "
        f"{issue['issue']}"
    )