# tests/test_anomaly.py

import numpy as np

from multi_doc.src.anomaly_detector import DocumentAnomalyDetector
from src.feature_extractor import DocumentFeatureExtractor
from src.anomaly_pipeline import AnomalyPipeline


# ============================================================
# Test 1: Feature Extractor
# ============================================================

def test_feature_extractor():

    document = {
        "ocr_confidence": 0.95,
        "text": "Invoice Number 12345 Total Amount 5000",
        "page_count": 1,
        "table_count": 1,
        "image_count": 0,
        "layout_complexity": 0.30,
        "fields": {
            "invoice_number": "12345",
            "total_amount": "5000"
        }
    }

    extractor = DocumentFeatureExtractor()

    features = extractor.extract(document)

    assert isinstance(
        features,
        np.ndarray
    )

    assert features.shape == (8,)

    print("✓ Feature extractor test passed")


# ============================================================
# Test 2: Anomaly Detector
# ============================================================

def test_anomaly_detector():

    training_features = np.array([
        [0.95, 100, 20, 1, 1, 0, 0.30, 5],
        [0.92, 110, 22, 1, 1, 0, 0.35, 5],
        [0.94, 105, 21, 1, 1, 0, 0.32, 5],
        [0.91, 115, 23, 1, 1, 0, 0.40, 6],
        [0.96, 98, 19, 1, 1, 0, 0.28, 5],
        [0.93, 108, 21, 1, 1, 0, 0.33, 5],
        [0.95, 102, 20, 1, 1, 0, 0.31, 5],
        [0.90, 120, 24, 1, 1, 0, 0.42, 6],
    ])

    detector = DocumentAnomalyDetector(
        contamination=0.1
    )

    detector.fit(
        training_features
    )

    test_features = np.array([
        [0.94, 106, 21, 1, 1, 0, 0.32, 5],
        [0.20, 900, 180, 8, 15, 10, 0.95, 2],
    ])

    predictions = detector.predict(
        test_features
    )

    assert len(predictions) == 2

    assert predictions[0] in [1, -1]
    assert predictions[1] in [1, -1]

    print("✓ Anomaly detector test passed")

    results = detector.analyze(
        test_features
    )

    print("\nAnomaly Results:")

    for result in results:
        print(result)


# ============================================================
# Test 3: Complete Pipeline
# ============================================================

def test_anomaly_pipeline():

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

    pipeline = AnomalyPipeline(
        contamination=0.1
    )

    pipeline.fit(
        training_documents
    )

    print("\n✓ Pipeline training completed")

    test_document = {
        "ocr_confidence": 0.25,
        "text": "UNKNOWN DOCUMENT WITH UNUSUAL CONTENT",
        "page_count": 7,
        "table_count": 10,
        "image_count": 8,
        "layout_complexity": 0.95,
        "fields": {
            "unknown": "data"
        }
    }

    result = pipeline.predict_single(
        test_document
    )

    print("\nSingle Document Result:")
    print(result)

    assert "status" in result
    assert "prediction" in result
    assert "anomaly_score" in result
    assert "decision_score" in result

    print("\n✓ Complete anomaly pipeline test passed")


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DCA AI - ANOMALY DETECTION TESTS")
    print("=" * 60)

    test_feature_extractor()

    test_anomaly_detector()

    test_anomaly_pipeline()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)