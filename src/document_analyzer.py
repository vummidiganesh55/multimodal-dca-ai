# src/document_analyzer.py

from src.feature_extractor import DocumentFeatureExtractor
from multi_doc.src.anomaly_detector import DocumentAnomalyDetector


class DocumentAnalyzer:
    """
    Main document analysis component.

    Flow:
        OCR / CV / NLP results
                ↓
        Feature Extraction
                ↓
        Anomaly Detection
                ↓
        Final Document Analysis
    """

    def __init__(
        self,
        n_estimators=100,
        contamination=0.1,
        random_state=42
    ):

        self.feature_extractor = (
            DocumentFeatureExtractor()
        )

        self.anomaly_detector = (
            DocumentAnomalyDetector(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state
            )
        )

        self.is_fitted = False

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self, documents):
        """
        Train anomaly detector using historical documents.
        """

        if not documents:
            raise ValueError(
                "Training documents cannot be empty."
            )

        features = (
            self.feature_extractor.extract_batch(
                documents
            )
        )

        self.anomaly_detector.fit(
            features
        )

        self.is_fitted = True

        return {
            "status": "success",
            "documents_used": len(documents),
            "features": features.shape[1]
        }

    # ========================================================
    # ANALYZE ONE DOCUMENT
    # ========================================================

    def analyze(self, document):
        """
        Analyze a single document.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Analyzer is not trained. "
                "Call train() first."
            )

        # --------------------------------------------
        # Extract numerical features
        # --------------------------------------------

        features = (
            self.feature_extractor.extract(
                document
            )
        )

        # --------------------------------------------
        # Anomaly analysis
        # --------------------------------------------

        anomaly_result = (
            self.anomaly_detector.analyze(
                [features]
            )[0]
        )

        # --------------------------------------------
        # Build final result
        # --------------------------------------------

        result = {
            "document": document,
            "features": features.tolist(),
            "anomaly": anomaly_result
        }

        return result

    # ========================================================
    # ANALYZE MULTIPLE DOCUMENTS
    # ========================================================

    def analyze_batch(self, documents):
        """
        Analyze multiple documents.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Analyzer is not trained. "
                "Call train() first."
            )

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        results = []

        for document in documents:

            result = self.analyze(
                document
            )

            results.append(result)

        return results

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        return {
            "trained": self.is_fitted,
            "feature_count": len(
                self.feature_extractor
                .get_feature_names()
            ),
            "feature_names": (
                self.feature_extractor
                .get_feature_names()
            )
        }