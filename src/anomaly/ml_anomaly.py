import numpy as np
from sklearn.ensemble import IsolationForest


class DocumentAnomalyDetector:
    """
    Machine-learning based document anomaly detector.

    Uses Isolation Forest to detect documents that differ
    from the normal document feature distribution.

    Prediction:
        1  = normal
       -1  = anomaly

    anomaly_score:
        Higher value = more anomalous.

    Note:
        anomaly_score is NOT a probability.
    """

    def __init__(
        self,
        n_estimators=100,
        contamination=0.1,
        random_state=42
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

        self.is_fitted = False

    def _validate_features(self, feature_vectors):

        features = np.asarray(
            feature_vectors,
            dtype=float
        )

        # Single feature vector
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2:
            raise ValueError(
                "Feature vectors must be a 2D array."
            )

        if features.shape[0] == 0:
            raise ValueError(
                "Feature vectors cannot be empty."
            )

        if features.shape[1] == 0:
            raise ValueError(
                "Feature vectors must contain features."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "Feature vectors contain NaN or infinite values."
            )

        return features

    def fit(self, feature_vectors):

        features = self._validate_features(
            feature_vectors
        )

        self.model.fit(features)

        self.is_fitted = True

        return self

    def predict(self, feature_vector):

        if not self.is_fitted:
            raise RuntimeError(
                "Anomaly detector must be fitted before prediction."
            )

        features = self._validate_features(
            feature_vector
        )

        predictions = self.model.predict(
            features
        )

        decision_scores = (
            self.model.decision_function(
                features
            )
        )

        raw_scores = (
            self.model.score_samples(
                features
            )
        )

        results = []

        for prediction, decision_score, raw_score in zip(
            predictions,
            decision_scores,
            raw_scores
        ):

            prediction = int(
                prediction
            )

            decision_score = float(
                decision_score
            )

            raw_score = float(
                raw_score
            )

            # Isolation Forest:
            # lower raw score = more anomalous
            anomaly_score = float(
                -raw_score
            )

            if prediction == -1:
                status = "anomaly"
            else:
                status = "normal"

            results.append(
                {
                    "prediction": prediction,
                    "status": status,
                    "anomaly_score": anomaly_score,
                    "decision_score": decision_score
                }
            )

        # Return a dictionary for one document
        if len(results) == 1:
            return results[0]

        return results

    def score_samples(self, feature_vectors):

        if not self.is_fitted:
            raise RuntimeError(
                "Anomaly detector must be fitted before scoring."
            )

        features = self._validate_features(
            feature_vectors
        )

        return self.model.score_samples(
            features
        )

    def decision_function(self, feature_vectors):

        if not self.is_fitted:
            raise RuntimeError(
                "Anomaly detector must be fitted before calculating decision scores."
            )

        features = self._validate_features(
            feature_vectors
        )

        return self.model.decision_function(
            features
        )

    def analyze(self, feature_vector):

        return self.predict(
            feature_vector
        )

    def is_anomaly(self, feature_vector):

        result = self.predict(
            feature_vector
        )

        if isinstance(result, list):

            return [
                item["prediction"] == -1
                for item in result
            ]

        return result["prediction"] == -1

    def anomaly_score(self, feature_vector):

        result = self.predict(
            feature_vector
        )

        if isinstance(result, list):

            return [
                item["anomaly_score"]
                for item in result
            ]

        return result["anomaly_score"]

    def get_status(self):

        return {
            "trained": self.is_fitted,
            "model": "IsolationForest",
            "n_estimators": self.model.n_estimators,
            "contamination": self.model.contamination
        }