# tests/test_document_analyzer.py

from src.document_analyzer import DocumentAnalyzer


# ============================================================
# Training documents
# ============================================================

training_documents = [

    {
        "ocr_confidence": 0.95,
        "text": "Invoice Number 1001 Total Amount 5000",
        "page_count": 1,
        "table_count": 1,
        "image_count": 0,
        "layout_complexity": 0.30,
        "fields": {
            "invoice_number": "1001",
            "total_amount": "5000"
        }
    },

    {
        "ocr_confidence": 0.93,
        "text": "Invoice Number 1002 Total Amount 5200",
        "page_count": 1,
        "table_count": 1,
        "image_count": 0,
        "layout_complexity": 0.32,
        "fields": {
            "invoice_number": "1002",
            "total_amount": "5200"
        }
    },

    {
        "ocr_confidence": 0.94,
        "text": "Invoice Number 1003 Total Amount 4800",
        "page_count": 1,
        "table_count": 1,
        "image_count": 0,
        "layout_complexity": 0.31,
        "fields": {
            "invoice_number": "1003",
            "total_amount": "4800"
        }
    },

    {
        "ocr_confidence": 0.92,
        "text": "Invoice Number 1004 Total Amount 5100",
        "page_count": 1,
        "table_count": 1,
        "image_count": 0,
        "layout_complexity": 0.35,
        "fields": {
            "invoice_number": "1004",
            "total_amount": "5100"
        }
    },

    {
        "ocr_confidence": 0.96,
        "text": "Invoice Number 1005 Total Amount 4950",
        "page_count": 1,
        "table_count": 1,
        "image_count": 0,
        "layout_complexity": 0.29,
        "fields": {
            "invoice_number": "1005",
            "total_amount": "4950"
        }
    }
]


# ============================================================
# Create analyzer
# ============================================================

analyzer = DocumentAnalyzer(
    contamination=0.1
)


# ============================================================
# Train
# ============================================================

training_result = analyzer.train(
    training_documents
)

print("\nTraining Result:")
print(training_result)


# ============================================================
# New document
# ============================================================

new_document = {

    "ocr_confidence": 0.25,

    "text": (
        "UNKNOWN DOCUMENT "
        "WITH UNUSUAL CONTENT"
    ),

    "page_count": 7,

    "table_count": 10,

    "image_count": 8,

    "layout_complexity": 0.95,

    "fields": {
        "unknown": "data"
    }
}


# ============================================================
# Analyze
# ============================================================

result = analyzer.analyze(
    new_document
)


print("\nDocument Analysis:")
print(result)


# ============================================================
# Display anomaly information
# ============================================================

print("\nAnomaly Status:")
print(
    result["anomaly"]["status"]
)

print("\nAnomaly Score:")
print(
    result["anomaly"]["anomaly_score"]
)

print("\nDecision Score:")
print(
    result["anomaly"]["decision_score"]
)


# ============================================================
# Analyzer status
# ============================================================

print("\nAnalyzer Status:")
print(
    analyzer.get_status()
)