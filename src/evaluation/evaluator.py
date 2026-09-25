import time
import re


class DocumentAIEvaluator:
    """
    Evaluation utilities for the Multimodal DCA AI project.

    Supported metrics:

    1. OCR
       - Character Error Rate (CER)
       - Word Error Rate (WER)

    2. Information Extraction
       - Precision
       - Recall
       - F1

    3. Table Extraction
       - Precision
       - Recall
       - F1

    4. Anomaly Detection
       - Precision
       - Recall
       - F1
       - Anomaly Rate

    5. Performance
       - Total processing time
       - Average page time
       - Pages per second
    """

    # ========================================================
    # BASIC HELPERS
    # ========================================================

    @staticmethod
    def _safe_divide(numerator, denominator):
        if denominator == 0:
            return 0.0

        return float(numerator) / float(denominator)

    # ========================================================
    # LEVENSHTEIN DISTANCE
    # ========================================================

    @staticmethod
    def _levenshtein_distance(reference, prediction):
        """
        Calculate Levenshtein edit distance.
        """

        reference = str(reference)
        prediction = str(prediction)

        rows = len(reference)
        cols = len(prediction)

        previous = list(range(cols + 1))

        for i in range(1, rows + 1):

            current = [i]

            for j in range(1, cols + 1):

                insertion = (
                    current[j - 1] + 1
                )

                deletion = (
                    previous[j] + 1
                )

                substitution = (
                    previous[j - 1]
                    + (
                        0
                        if reference[i - 1]
                        == prediction[j - 1]
                        else 1
                    )
                )

                current.append(
                    min(
                        insertion,
                        deletion,
                        substitution
                    )
                )

            previous = current

        return previous[-1]

    # ========================================================
    # CER
    # ========================================================

    def character_error_rate(
        self,
        reference,
        prediction
    ):
        """
        Character Error Rate:

            CER = edit_distance / reference_length
        """

        reference = str(reference)
        prediction = str(prediction)

        if len(reference) == 0:

            return 0.0 if len(prediction) == 0 else 1.0

        distance = self._levenshtein_distance(
            reference,
            prediction
        )

        return self._safe_divide(
            distance,
            len(reference)
        )

    # ========================================================
    # WER
    # ========================================================

    def word_error_rate(
        self,
        reference,
        prediction
    ):
        """
        Word Error Rate.
        """

        reference_words = str(
            reference
        ).split()

        prediction_words = str(
            prediction
        ).split()

        if len(reference_words) == 0:

            return (
                0.0
                if len(prediction_words) == 0
                else 1.0
            )

        distance = self._levenshtein_distance(
            reference_words,
            prediction_words
        )

        return self._safe_divide(
            distance,
            len(reference_words)
        )

    # ========================================================
    # OCR EVALUATION
    # ========================================================

    def evaluate_ocr(
        self,
        reference,
        prediction
    ):
        """
        Evaluate OCR using CER and WER.

        Lower CER/WER is better.
        """

        cer = self.character_error_rate(
            reference,
            prediction
        )

        wer = self.word_error_rate(
            reference,
            prediction
        )

        return {
            "character_error_rate":
                round(cer, 4),

            "word_error_rate":
                round(wer, 4)
        }

    # ========================================================
    # GENERIC CLASSIFICATION METRICS
    # ========================================================

    def classification_metrics(
        self,
        true_values,
        predicted_values,
        positive_value=1
    ):
        """
        Calculate Precision, Recall and F1.

        Works with binary labels.
        """

        true_values = list(
            true_values
        )

        predicted_values = list(
            predicted_values
        )

        if len(true_values) != len(
            predicted_values
        ):
            raise ValueError(
                "true_values and "
                "predicted_values must "
                "have the same length."
            )

        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        for actual, predicted in zip(
            true_values,
            predicted_values
        ):

            actual_positive = (
                actual == positive_value
            )

            predicted_positive = (
                predicted == positive_value
            )

            if (
                actual_positive
                and predicted_positive
            ):

                true_positive += 1

            elif (
                not actual_positive
                and predicted_positive
            ):

                false_positive += 1

            elif (
                actual_positive
                and not predicted_positive
            ):

                false_negative += 1

            else:

                true_negative += 1

        precision = self._safe_divide(
            true_positive,
            true_positive
            + false_positive
        )

        recall = self._safe_divide(
            true_positive,
            true_positive
            + false_negative
        )

        f1 = self._safe_divide(
            2 * precision * recall,
            precision + recall
        )

        return {

            "precision":
                round(precision, 4),

            "recall":
                round(recall, 4),

            "f1_score":
                round(f1, 4),

            "true_positive":
                true_positive,

            "false_positive":
                false_positive,

            "false_negative":
                false_negative,

            "true_negative":
                true_negative
        }

    # ========================================================
    # INFORMATION EXTRACTION
    # ========================================================

    def evaluate_information_extraction(
        self,
        expected,
        predicted
    ):
        """
        Field-level evaluation.

        expected and predicted should be dictionaries.

        Example:

            expected = {
                "invoice_number": "INV001",
                "total": "500"
            }

            predicted = {
                "invoice_number": "INV001",
                "total": "500"
            }
        """

        expected = expected or {}
        predicted = predicted or {}

        expected_fields = set(
            expected.keys()
        )

        predicted_fields = set(
            predicted.keys()
        )

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for field in predicted_fields:

            predicted_value = predicted.get(
                field
            )

            if field not in expected_fields:

                false_positive += 1
                continue

            expected_value = expected.get(
                field
            )

            if self._normalize_value(
                expected_value
            ) == self._normalize_value(
                predicted_value
            ):

                true_positive += 1

            else:

                false_positive += 1
                false_negative += 1

        for field in (
            expected_fields
            - predicted_fields
        ):

            false_negative += 1

        precision = self._safe_divide(
            true_positive,
            true_positive + false_positive
        )

        recall = self._safe_divide(
            true_positive,
            true_positive + false_negative
        )

        f1 = self._safe_divide(
            2 * precision * recall,
            precision + recall
        )

        return {

            "precision":
                round(precision, 4),

            "recall":
                round(recall, 4),

            "f1_score":
                round(f1, 4),

            "correct_fields":
                true_positive,

            "predicted_fields":
                len(predicted_fields),

            "expected_fields":
                len(expected_fields)
        }

    # ========================================================
    # NORMALIZE VALUE
    # ========================================================

    @staticmethod
    def _normalize_value(value):

        if value is None:
            return ""

        value = str(value).lower().strip()

        value = re.sub(
            r"\s+",
            " ",
            value
        )

        value = value.replace(
            ",",
            ""
        )

        return value

    # ========================================================
    # TABLE EVALUATION
    # ========================================================

    def evaluate_table(
        self,
        expected_rows,
        predicted_rows
    ):
        """
        Basic table row-level evaluation.

        Rows are compared after normalization.
        """

        expected_rows = (
            expected_rows or []
        )

        predicted_rows = (
            predicted_rows or []
        )

        expected_normalized = [
            self._normalize_row(row)
            for row in expected_rows
        ]

        predicted_normalized = [
            self._normalize_row(row)
            for row in predicted_rows
        ]

        matched = 0
        used = set()

        for predicted_row in predicted_normalized:

            for index, expected_row in enumerate(
                expected_normalized
            ):

                if index in used:
                    continue

                if predicted_row == expected_row:

                    matched += 1

                    used.add(index)

                    break

        false_positive = (
            len(predicted_rows)
            - matched
        )

        false_negative = (
            len(expected_rows)
            - matched
        )

        precision = self._safe_divide(
            matched,
            matched + false_positive
        )

        recall = self._safe_divide(
            matched,
            matched + false_negative
        )

        f1 = self._safe_divide(
            2 * precision * recall,
            precision + recall
        )

        return {

            "precision":
                round(precision, 4),

            "recall":
                round(recall, 4),

            "f1_score":
                round(f1, 4),

            "expected_rows":
                len(expected_rows),

            "predicted_rows":
                len(predicted_rows),

            "matched_rows":
                matched
        }

    # ========================================================
    # NORMALIZE TABLE ROW
    # ========================================================

    def _normalize_row(self, row):

        if not isinstance(
            row,
            dict
        ):
            return str(row)

        normalized = {}

        for key, value in row.items():

            if key == "confidence":
                continue

            normalized[
                str(key).lower().strip()
            ] = self._normalize_value(
                value
            )

        return normalized

    # ========================================================
    # ANOMALY EVALUATION
    # ========================================================

    def evaluate_anomaly(
        self,
        true_labels,
        predicted_labels
    ):
        """
        Evaluate anomaly detection.

        Convention:

            1 = anomaly
            0 = normal
        """

        metrics = self.classification_metrics(
            true_labels,
            predicted_labels,
            positive_value=1
        )

        total = len(true_labels)

        anomaly_count = sum(
            1
            for value in predicted_labels
            if value == 1
        )

        anomaly_rate = self._safe_divide(
            anomaly_count,
            total
        )

        metrics["anomaly_rate"] = round(
            anomaly_rate,
            4
        )

        return metrics

    # ========================================================
    # PERFORMANCE EVALUATION
    # ========================================================

    def evaluate_performance(
        self,
        total_time,
        total_pages
    ):
        """
        Evaluate processing performance.
        """

        total_pages = int(
            total_pages
        )

        total_time = float(
            total_time
        )

        average_page_time = (
            self._safe_divide(
                total_time,
                total_pages
            )
        )

        pages_per_second = (
            self._safe_divide(
                total_pages,
                total_time
            )
        )

        return {

            "total_processing_time_seconds":
                round(total_time, 2),

            "total_pages":
                total_pages,

            "average_page_time_seconds":
                round(
                    average_page_time,
                    2
                ),

            "pages_per_second":
                round(
                    pages_per_second,
                    4
                )
        }

    # ========================================================
    # COMPLETE EVALUATION REPORT
    # ========================================================

    def build_report(
        self,
        ocr=None,
        information_extraction=None,
        table=None,
        anomaly=None,
        performance=None
    ):
        """
        Build a unified evaluation report.
        """

        return {

            "evaluation": {

                "ocr":
                    ocr or {},

                "information_extraction":
                    information_extraction or {},

                "table":
                    table or {},

                "anomaly_detection":
                    anomaly or {},

                "performance":
                    performance or {}
            }
        }
