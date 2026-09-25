# src/anomaly_pipeline.py

import numpy as np

from src.feature_extractor import DocumentFeatureExtractor
from multi_doc.src.anomaly_detector import DocumentAnomalyDetector


class AnomalyPipeline:
    """
    End-to-end document anomaly detection pipeline.

    Flow:
        Document
            ↓
        Feature Extraction
            ↓
        Isolation Forest
            ↓
        Anomaly Result
    """

    def __init__(
        self,
        n_estimators=100,
        contamination=0.1,
        random_state=42
    ):
        self.feature_extractor = DocumentFeatureExtractor()

        self.detector = DocumentAnomalyDetector(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

        self.is_fitted = False

    # ---------------------------------------------------------
    # Prepare features
    # ---------------------------------------------------------
    def extract_features(self, documents):
        """
        Convert documents into feature vectors.
        """

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        return self.feature_extractor.extract_batch(
            documents
        )

    # ---------------------------------------------------------
    # Train pipeline
    # ---------------------------------------------------------
    def fit(self, documents):
        """
        Train anomaly detection using documents.
        """

        features = self.extract_features(
            documents
        )

        self.detector.fit(
            features
        )

        self.is_fitted = True

        return self

    # ---------------------------------------------------------
    # Predict multiple documents
    # ---------------------------------------------------------
    def predict(self, documents):
        """
        Predict normal/anomaly for multiple documents.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline is not fitted. "
                "Call fit() first."
            )

        features = self.extract_features(
            documents
        )

        results = self.detector.analyze(
            features
        )

        return results

    # ---------------------------------------------------------
    # Predict one document
    # ---------------------------------------------------------
    def predict_single(self, document):
        """
        Analyze one document.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline is not fitted. "
                "Call fit() first."
            )

        features = self.feature_extractor.extract(
            document
        )

        features = np.asarray(
            features,
            dtype=float
        ).reshape(1, -1)

        result = self.detector.analyze(
            features
        )

        return result[0]

    # ---------------------------------------------------------
    # Get feature vector
    # ---------------------------------------------------------
    def get_features(self, document):
        """
        Return the numerical features of one document.
        """

        features = self.feature_extractor.extract(
            document
        )

        return features

    # ---------------------------------------------------------
    # Pipeline status
    # ---------------------------------------------------------
    def get_status(self):
        """
        Return pipeline status.
        """

        return {
            "pipeline_fitted": self.is_fitted,
            "detector": self.detector.get_status(),
            "features": self.feature_extractor.get_feature_names()
        }