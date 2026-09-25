import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DocumentAnomalyDetector:
    """
    Isolation Forest based document anomaly detector.

    Designed for document-level anomaly detection.

    Important:
        An ML anomaly does NOT mean fraud.
        It only indicates that the document is statistically
        different from the documents used during training.
    """

    def __init__(
        self,
        n_estimators=200,
        contamination="auto",
        random_state=42
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

        self.scaler = StandardScaler()

        self.is_fitted = False

    # ---------------------------------------------------------
    # VALIDATE INPUT
    # ---------------------------------------------------------

    def _validate_features(self, feature_vectors):

        if feature_vectors is None:
            raise ValueError(
                "Feature vectors cannot be None."
            )

        feature_vectors = np.asarray(
            feature_vectors,
            dtype=float
        )

        if feature_vectors.size == 0:
            raise ValueError(
                "Feature vectors cannot be empty."
            )

        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        if feature_vectors.ndim != 2:
            raise ValueError(
                "Feature vectors must be a 1D or 2D array."
            )

        if not np.isfinite(feature_vectors).all():
            raise ValueError(
                "Feature vectors contain NaN or infinite values."
            )

        return feature_vectors

    # ---------------------------------------------------------
    # FIT
    # ---------------------------------------------------------

    def fit(self, feature_vectors):
        """
        Train the Isolation Forest model.

        Features are standardized before training.
        """

        feature_vectors = self._validate_features(
            feature_vectors
        )

        if len(feature_vectors) < 2:
            raise ValueError(
                "At least 2 documents are required for training."
            )

        # Standardize document features
        scaled_features = self.scaler.fit_transform(
            feature_vectors
        )

        self.model.fit(
            scaled_features
        )

        self.is_fitted = True

        return self

    # ---------------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------------

    def _transform(self, feature_vectors):

        feature_vectors = self._validate_features(
            feature_vectors
        )

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )

        return self.scaler.transform(
            feature_vectors
        )

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    def predict(self, feature_vectors):
        """
        Predict normal/anomaly.

        1  = normal
        -1 = anomaly
        """

        scaled_features = self._transform(
            feature_vectors
        )

        return self.model.predict(
            scaled_features
        )

    # ---------------------------------------------------------
    # SCORE SAMPLES
    # ---------------------------------------------------------

    def score_samples(self, feature_vectors):
        """
        Return Isolation Forest scores.

        Higher = more normal
        Lower  = more anomalous
        """

        scaled_features = self._transform(
            feature_vectors
        )

        return self.model.score_samples(
            scaled_features
        )

    # ---------------------------------------------------------
    # DECISION FUNCTION
    # ---------------------------------------------------------

    def decision_function(self, feature_vectors):
        """
        Return decision scores.

        Positive = normal
        Negative = anomaly
        """

        scaled_features = self._transform(
            feature_vectors
        )

        return self.model.decision_function(
            scaled_features
        )

    # ---------------------------------------------------------
    # ANALYZE
    # ---------------------------------------------------------

    def analyze(self, feature_vectors):
        """
        Return detailed anomaly results.
        """

        predictions = self.predict(
            feature_vectors
        )

        anomaly_scores = self.score_samples(
            feature_vectors
        )

        decision_scores = self.decision_function(
            feature_vectors
        )

        results = []

        for prediction, anomaly_score, decision_score in zip(
            predictions,
            anomaly_scores,
            decision_scores
        ):

            status = (
                "anomaly"
                if prediction == -1
                else "normal"
            )

            results.append(
                {
                    "prediction": int(prediction),
                    "status": status,
                    "anomaly_score": float(
                        anomaly_score
                    ),
                    "decision_score": float(
                        decision_score
                    )
                }
            )

        return results

    # ---------------------------------------------------------
    # SINGLE DOCUMENT CHECK
    # ---------------------------------------------------------

    def is_anomaly(self, feature_vector):
        """
        Check whether a single document is anomalous.
        """

        prediction = self.predict(
            feature_vector
        )

        return bool(
            prediction[0] == -1
        )

    # ---------------------------------------------------------
    # ANOMALY SCORE
    # ---------------------------------------------------------

    def anomaly_score(self, feature_vector):
        """
        Return a 0-1 anomaly-oriented score.

        Higher = more anomalous.

        This is NOT a calibrated probability.
        """

        decision_score = self.decision_function(
            feature_vector
        )

        score = 1.0 / (
            1.0 + np.exp(decision_score)
        )

        return float(
            score[0]
        )

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def get_status(self):

        return {
            "is_fitted": self.is_fitted,
            "model": "IsolationForest",
            "n_estimators": self.model.n_estimators,
            "contamination": self.model.contamination,
            "scaling": "StandardScaler"
        }