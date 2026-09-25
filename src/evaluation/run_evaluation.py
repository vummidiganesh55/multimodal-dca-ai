import json
from pathlib import Path

from src.evaluation.evaluator import DocumentAIEvaluator


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PIPELINE_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "pipeline_results.json"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth.json"
)

EVALUATION_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)

EVALUATION_REPORT_PATH = (
    EVALUATION_OUTPUT_DIR
    / "evaluation_report.json"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(path):
    """
    Load JSON file.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_json(data, path):
    """
    Save JSON with readable formatting.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# RECORD EXTRACTION HELPERS
# ============================================================

def get_records(data):
    """
    Support multiple possible JSON structures.

    Pipeline:
        {
            "pages": [...]
        }

    Ground truth:
        {
            "documents": [...]
        }

    Original draft ground truth:
        {
            "pages": [...]
        }
    """

    if not isinstance(data, dict):
        return []

    for key in (
        "pages",
        "documents",
        "results"
    ):

        value = data.get(key)

        if isinstance(value, list):
            return value

    return []


def get_invoice_number(record):
    """
    Extract invoice number from a pipeline or
    ground-truth record.
    """

    if not isinstance(record, dict):
        return None

    # Direct field
    invoice_number = record.get(
        "invoice_number"
    )

    if invoice_number:
        return str(invoice_number).strip()

    # Extraction object
    extraction = record.get(
        "extraction"
    )

    if isinstance(extraction, dict):

        invoice_number = extraction.get(
            "invoice_number"
        )

        if invoice_number:
            return str(
                invoice_number
            ).strip()

    # Original ground-truth format
    information_extraction = record.get(
        "information_extraction"
    )

    if isinstance(
        information_extraction,
        dict
    ):

        invoice_number = (
            information_extraction.get(
                "invoice_number"
            )
        )

        if invoice_number:
            return str(
                invoice_number
            ).strip()

    return None


def get_extraction(record):
    """
    Return the information-extraction dictionary.

    Supports:

        extraction

    and:

        information_extraction
    """

    if not isinstance(record, dict):
        return {}

    extraction = record.get(
        "extraction"
    )

    if isinstance(extraction, dict):
        return extraction

    extraction = record.get(
        "information_extraction"
    )

    if isinstance(extraction, dict):
        return extraction

    return {}


def get_table(record):
    """
    Return table information.

    Supports:

        table_json

    and:

        table

    Returns an empty dictionary when unavailable.
    """

    if not isinstance(record, dict):
        return {}

    table = record.get(
        "table_json"
    )

    if isinstance(table, dict):
        return table

    table = record.get(
        "table"
    )

    if isinstance(table, dict):
        return table

    return {}


def get_table_rows(record):
    """
    Return table rows.
    """

    table = get_table(record)

    rows = table.get(
        "rows",
        []
    )

    if isinstance(rows, list):
        return rows

    return []


def get_ocr_text(record):
    """
    Extract OCR text from ground truth.

    Supports:

        ocr_text

    and:

        text
    """

    if not isinstance(record, dict):
        return ""

    text = record.get(
        "ocr_text"
    )

    if isinstance(text, str):
        return text

    text = record.get(
        "text"
    )

    if isinstance(text, str):
        return text

    return ""


def get_predicted_ocr_text(record):
    """
    Convert pipeline OCR list into plain text.
    """

    if not isinstance(record, dict):
        return ""

    ocr = record.get(
        "ocr"
    )

    if not isinstance(ocr, list):
        return ""

    words = []

    for item in ocr:

        if not isinstance(item, dict):
            continue

        text = item.get(
            "text"
        )

        if text:
            words.append(
                str(text)
            )

    return " ".join(words)


# ============================================================
# VERIFICATION HELPERS
# ============================================================

def is_ground_truth_verified(ground_truth):
    """
    Determine whether the ground truth explicitly says
    that it has been human verified.

    The current project ground truth uses:

        verification_required: true

    for the original draft.

    The replacement ground truth using "documents"
    does not contain that flag, so it is treated as usable
    for information-extraction evaluation.

    Table/anomaly evaluation still requires actual labels.
    """

    if not isinstance(
        ground_truth,
        dict
    ):
        return False

    verification_required = (
        ground_truth.get(
            "verification_required"
        )
    )

    if verification_required is True:
        return False

    verification_status = (
        ground_truth.get(
            "verification_status"
        )
    )

    if verification_status in {
        "DRAFT_NOT_HUMAN_VERIFIED",
        "UNVERIFIED",
        "DRAFT"
    }:
        return False

    return True


# ============================================================
# RECORD MATCHING
# ============================================================

def build_invoice_index(records):
    """
    Create:

        invoice_number -> record
    """

    index = {}

    for record in records:

        invoice_number = (
            get_invoice_number(
                record
            )
        )

        if invoice_number:

            index[
                invoice_number
            ] = record

    return index


# ============================================================
# INFORMATION EXTRACTION
# ============================================================

def evaluate_information_extraction(
    evaluator,
    pipeline_by_invoice,
    ground_truth_by_invoice,
    common_invoices
):
    """
    Evaluate information extraction one invoice at a time.

    evaluator.evaluate_information_extraction()
    expects dictionaries, NOT lists.
    """

    true_positive = 0
    false_positive = 0
    false_negative = 0

    evaluated_documents = 0

    for invoice_number in common_invoices:

        pipeline_record = (
            pipeline_by_invoice[
                invoice_number
            ]
        )

        ground_truth_record = (
            ground_truth_by_invoice[
                invoice_number
            ]
        )

        expected = get_extraction(
            ground_truth_record
        )

        predicted = get_extraction(
            pipeline_record
        )

        if not expected:
            continue

        result = (
            evaluator.evaluate_information_extraction(
                expected,
                predicted
            )
        )

        correct_fields = int(
            result.get(
                "correct_fields",
                0
            )
        )

        predicted_fields = int(
            result.get(
                "predicted_fields",
                0
            )
        )

        expected_fields = int(
            result.get(
                "expected_fields",
                0
            )
        )

        true_positive += (
            correct_fields
        )

        false_positive += (
            predicted_fields
            - correct_fields
        )

        false_negative += (
            expected_fields
            - correct_fields
        )

        # ----------------------------------------------------
        # DEBUG: SHOW FIELD MISMATCHES
        # ----------------------------------------------------

        for field in expected.keys():

            expected_value = expected.get(field)
            predicted_value = predicted.get(field)

            normalized_expected = evaluator._normalize_value(
                expected_value
            )

            normalized_predicted = evaluator._normalize_value(
                predicted_value
            )

            if normalized_expected != normalized_predicted:

                print(
                    "\n[MISMATCH]"
                )

                print(
                    f"Invoice: {invoice_number}"
                )

                print(
                    f"Field: {field}"
                )

                print(
                    f"Expected: {expected_value}"
                )

                print(
                    f"Predicted: {predicted_value}"
                )

        evaluated_documents += 1

    precision = evaluator._safe_divide(
        true_positive,
        true_positive
        + false_positive
    )

    recall = evaluator._safe_divide(
        true_positive,
        true_positive
        + false_negative
    )

    f1_score = evaluator._safe_divide(
        2
        * precision
        * recall,
        precision + recall
    )

    return {
        "precision": round(
            precision,
            4
        ),

        "recall": round(
            recall,
            4
        ),

        "f1_score": round(
            f1_score,
            4
        ),

        "correct_fields": (
            true_positive
        ),

        "predicted_fields": (
            true_positive
            + false_positive
        ),

        "expected_fields": (
            true_positive
            + false_negative
        ),

        "documents_evaluated": (
            evaluated_documents
        )
    }


# ============================================================
# TABLE EXTRACTION
# ============================================================

def evaluate_table_extraction(
    evaluator,
    pipeline_by_invoice,
    ground_truth_by_invoice,
    common_invoices
):
    """
    Evaluate tables only when verified/available
    ground-truth rows actually exist.

    Never return misleading 0.0 metrics when
    table ground truth is absent.
    """

    total_expected = 0
    total_predicted = 0
    total_matched = 0

    invoices_with_ground_truth = 0

    for invoice_number in common_invoices:

        pipeline_record = (
            pipeline_by_invoice[
                invoice_number
            ]
        )

        ground_truth_record = (
            ground_truth_by_invoice[
                invoice_number
            ]
        )

        expected_rows = (
            get_table_rows(
                ground_truth_record
            )
        )

        predicted_rows = (
            get_table_rows(
                pipeline_record
            )
        )

        # No verified/available GT rows.
        if not expected_rows:
            continue

        invoices_with_ground_truth += 1

        result = (
            evaluator.evaluate_table(
                expected_rows,
                predicted_rows
            )
        )

        total_expected += int(
            result.get(
                "expected_rows",
                0
            )
        )

        total_predicted += int(
            result.get(
                "predicted_rows",
                0
            )
        )

        total_matched += int(
            result.get(
                "matched_rows",
                0
            )
        )

    if total_expected == 0:

        return {
            "status": "not_evaluated",

            "reason": (
                "No verified table "
                "ground-truth labels "
                "are available."
            )
        }

    precision = evaluator._safe_divide(
        total_matched,
        total_predicted
    )

    recall = evaluator._safe_divide(
        total_matched,
        total_expected
    )

    f1_score = evaluator._safe_divide(
        2
        * precision
        * recall,
        precision + recall
    )

    return {
        "status": "evaluated",

        "precision": round(
            precision,
            4
        ),

        "recall": round(
            recall,
            4
        ),

        "f1_score": round(
            f1_score,
            4
        ),

        "expected_rows": (
            total_expected
        ),

        "predicted_rows": (
            total_predicted
        ),

        "matched_rows": (
            total_matched
        ),

        "documents_with_table_ground_truth": (
            invoices_with_ground_truth
        )
    }


# ============================================================
# TABLE QUALITY EVALUATION
# ============================================================

def evaluate_table_quality(
    pipeline_records
):
    """
    Evaluate internal table consistency when verified
    table ground-truth labels are unavailable.

    This does NOT calculate table Precision/Recall/F1.

    Metrics:

        - pages containing table JSON
        - total extracted rows
        - row arithmetic accuracy
        - subtotal consistency
        - table validation rate
        - average table confidence
    """

    pages_with_tables = 0
    total_rows = 0

    valid_formula_rows = 0
    total_formula_rows = 0

    subtotal_matches = 0
    subtotal_checks = 0

    valid_table_pages = 0
    confidence_values = []

    for record in pipeline_records:

        if not isinstance(
            record,
            dict
        ):
            continue

        # ----------------------------------------------------
        # TABLE JSON
        # ----------------------------------------------------

        table = get_table(
            record
        )

        if not table:
            continue

        rows = table.get(
            "rows",
            []
        )

        if not isinstance(
            rows,
            list
        ):
            rows = []

        if rows:

            pages_with_tables += 1

            total_rows += len(
                rows
            )

        # ----------------------------------------------------
        # ROW ARITHMETIC
        # Qty * Unit Price == Amount
        # ----------------------------------------------------

        for row in rows:

            if not isinstance(
                row,
                dict
            ):
                continue

            try:

                qty = float(
                    row.get(
                        "Qty"
                    )
                )

                unit_price = float(
                    row.get(
                        "Unit Price"
                    )
                )

                amount = float(
                    row.get(
                        "Amount"
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            total_formula_rows += 1

            calculated_amount = (
                qty
                * unit_price
            )

            if abs(
                calculated_amount
                - amount
            ) < 0.01:

                valid_formula_rows += 1

        # ----------------------------------------------------
        # SUBTOTAL CONSISTENCY
        # Sum(table Amount) == extracted subtotal
        # ----------------------------------------------------

        extraction = get_extraction(
            record
        )

        subtotal = extraction.get(
            "subtotal"
        )

        if (
            subtotal is not None
            and rows
        ):

            try:

                expected_subtotal = float(
                    subtotal
                )

                calculated_subtotal = sum(
                    float(
                        row.get(
                            "Amount",
                            0
                        )
                    )
                    for row in rows
                    if isinstance(
                        row,
                        dict
                    )
                )

                subtotal_checks += 1

                if abs(
                    calculated_subtotal
                    - expected_subtotal
                ) < 0.01:

                    subtotal_matches += 1

            except (
                TypeError,
                ValueError
            ):

                pass

        # ----------------------------------------------------
        # TABLE VALIDATION
        # ----------------------------------------------------

        validation = record.get(
            "validation",
            {}
        )

        if isinstance(
            validation,
            dict
        ):

            if validation.get(
                "valid"
            ) is True:

                valid_table_pages += 1

            confidence = validation.get(
                "confidence"
            )

            if confidence is not None:

                try:

                    confidence_values.append(
                        float(
                            confidence
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    pass

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    row_arithmetic_accuracy = (
        valid_formula_rows
        / total_formula_rows
        if total_formula_rows > 0
        else 0.0
    )

    subtotal_consistency_rate = (
        subtotal_matches
        / subtotal_checks
        if subtotal_checks > 0
        else 0.0
    )

    table_validation_rate = (
        valid_table_pages
        / pages_with_tables
        if pages_with_tables > 0
        else 0.0
    )

    average_confidence = (
        sum(
            confidence_values
        )
        / len(
            confidence_values
        )
        if confidence_values
        else 0.0
    )

    return {

        "status": "quality_evaluated",

        "pages_with_tables": (
            pages_with_tables
        ),

        "total_rows": (
            total_rows
        ),

        "row_arithmetic_accuracy": round(
            row_arithmetic_accuracy,
            4
        ),

        "valid_formula_rows": (
            valid_formula_rows
        ),

        "total_formula_rows": (
            total_formula_rows
        ),

        "subtotal_consistency_rate": round(
            subtotal_consistency_rate,
            4
        ),

        "subtotal_matches": (
            subtotal_matches
        ),

        "subtotal_checks": (
            subtotal_checks
        ),

        "table_validation_rate": round(
            table_validation_rate,
            4
        ),

        "valid_table_pages": (
            valid_table_pages
        ),

        "average_table_confidence": round(
            average_confidence,
            4
        )
    }


# ============================================================
# ANOMALY DETECTION
# ============================================================

def evaluate_anomaly_detection(
    evaluator,
    pipeline_by_invoice,
    ground_truth_by_invoice,
    common_invoices
):
    """
    Evaluate anomaly detection only when ground truth
    contains explicit anomaly labels.

    Supported ground-truth keys:

        anomaly
        is_anomaly
        anomaly_label
        label

    Values:

        1 / 0
        true / false
        anomaly / normal
    """

    true_labels = []
    predicted_labels = []

    found_ground_truth = False

    for invoice_number in common_invoices:

        pipeline_record = (
            pipeline_by_invoice[
                invoice_number
            ]
        )

        ground_truth_record = (
            ground_truth_by_invoice[
                invoice_number
            ]
        )

        # ----------------------------------------------------
        # FIND GROUND-TRUTH ANOMALY LABEL
        # ----------------------------------------------------

        true_value = None

        for key in (
            "anomaly",
            "is_anomaly",
            "anomaly_label",
            "label"
        ):

            if key in ground_truth_record:

                true_value = (
                    ground_truth_record[
                        key
                    ]
                )

                break

        if true_value is None:
            continue

        # ----------------------------------------------------
        # NORMALIZE TRUE LABEL
        # ----------------------------------------------------

        if isinstance(
            true_value,
            bool
        ):

            true_label = (
                1
                if true_value
                else 0
            )

        elif isinstance(
            true_value,
            (int, float)
        ):

            true_label = (
                1
                if int(true_value) == 1
                else 0
            )

        elif isinstance(
            true_value,
            str
        ):

            normalized = (
                true_value
                .strip()
                .lower()
            )

            if normalized in {
                "anomaly",
                "anomalous",
                "fraud",
                "suspicious",
                "1",
                "true"
            }:

                true_label = 1

            elif normalized in {
                "normal",
                "0",
                "false"
            }:

                true_label = 0

            else:
                continue

        else:
            continue

        # ----------------------------------------------------
        # PIPELINE PREDICTION
        # ----------------------------------------------------

        predicted_label = 0

        ml_anomaly = (
            pipeline_record.get(
                "ml_anomaly",
                {}
            )
        )

        if isinstance(
            ml_anomaly,
            dict
        ):

            status = (
                ml_anomaly.get(
                    "status",
                    ""
                )
            )

            prediction = (
                ml_anomaly.get(
                    "prediction"
                )
            )

            if (
                str(status).lower()
                == "anomaly"
            ):

                predicted_label = 1

            elif prediction == -1:

                predicted_label = 1

        true_labels.append(
            true_label
        )

        predicted_labels.append(
            predicted_label
        )

        found_ground_truth = True

    if not found_ground_truth:

        return {
            "status": "not_evaluated",

            "reason": (
                "No verified anomaly "
                "ground-truth labels found."
            )
        }

    result = (
        evaluator.evaluate_anomaly(
            true_labels,
            predicted_labels
        )
    )

    result["status"] = "evaluated"

    return result


# ============================================================
# ANOMALY PIPELINE SUMMARY
# ============================================================

def evaluate_anomaly_pipeline_summary(
    pipeline_records
):
    """
    Descriptive anomaly statistics from the pipeline.

    IMPORTANT:

    These are NOT accuracy metrics.

    They are reported because the pipeline currently
    produces anomaly predictions but there is no verified
    anomaly ground truth.

    Metrics:

        - ML anomaly count
        - ML normal count
        - ML anomaly rate
        - average ML anomaly score
        - fraud anomaly count
        - fraud normal count
        - fraud anomaly rate
        - average fraud anomaly score
    """

    ml_anomaly_count = 0
    ml_normal_count = 0

    fraud_anomaly_count = 0
    fraud_normal_count = 0

    ml_scores = []
    fraud_scores = []

    for record in pipeline_records:

        if not isinstance(
            record,
            dict
        ):
            continue

        # ----------------------------------------------------
        # ML ANOMALY
        # ----------------------------------------------------

        ml_anomaly = record.get(
            "ml_anomaly",
            {}
        )

        if isinstance(
            ml_anomaly,
            dict
        ):

            ml_status = str(
                ml_anomaly.get(
                    "status",
                    ""
                )
            ).strip().lower()

            ml_prediction = (
                ml_anomaly.get(
                    "prediction"
                )
            )

            if (
                ml_status == "anomaly"
                or ml_prediction == -1
            ):

                ml_anomaly_count += 1

            elif ml_status == "normal":

                ml_normal_count += 1

            ml_score = (
                ml_anomaly.get(
                    "score"
                )
            )

            if ml_score is not None:

                try:

                    ml_scores.append(
                        float(
                            ml_score
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    pass

        # ----------------------------------------------------
        # FRAUD ANOMALY
        # ----------------------------------------------------

        fraud_anomaly = record.get(
            "fraud_anomaly",
            {}
        )

        if isinstance(
            fraud_anomaly,
            dict
        ):

            fraud_status = str(
                fraud_anomaly.get(
                    "status",
                    ""
                )
            ).strip().lower()

            if fraud_status in {
                "anomaly",
                "fraud",
                "suspicious"
            }:

                fraud_anomaly_count += 1

            elif fraud_status == "normal":

                fraud_normal_count += 1

            fraud_score = (
                fraud_anomaly.get(
                    "score"
                )
            )

            if fraud_score is not None:

                try:

                    fraud_scores.append(
                        float(
                            fraud_score
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    pass

    total_ml_predictions = (
        ml_anomaly_count
        + ml_normal_count
    )

    total_fraud_predictions = (
        fraud_anomaly_count
        + fraud_normal_count
    )

    ml_anomaly_rate = (
        ml_anomaly_count
        / total_ml_predictions
        if total_ml_predictions > 0
        else 0.0
    )

    fraud_anomaly_rate = (
        fraud_anomaly_count
        / total_fraud_predictions
        if total_fraud_predictions > 0
        else 0.0
    )

    average_ml_score = (
        sum(ml_scores)
        / len(ml_scores)
        if ml_scores
        else 0.0
    )

    average_fraud_score = (
        sum(fraud_scores)
        / len(fraud_scores)
        if fraud_scores
        else 0.0
    )

    return {

        "status": "descriptive_only",

        "reason": (
            "No verified anomaly ground truth "
            "is available, so anomaly accuracy "
            "metrics are not calculated."
        ),

        "ml_anomaly_count": (
            ml_anomaly_count
        ),

        "ml_normal_count": (
            ml_normal_count
        ),

        "ml_anomaly_rate": round(
            ml_anomaly_rate,
            4
        ),

        "average_ml_anomaly_score": round(
            average_ml_score,
            4
        ),

        "fraud_anomaly_count": (
            fraud_anomaly_count
        ),

        "fraud_normal_count": (
            fraud_normal_count
        ),

        "fraud_anomaly_rate": round(
            fraud_anomaly_rate,
            4
        ),

        "average_fraud_anomaly_score": round(
            average_fraud_score,
            4
        )
    }


# ============================================================
# OCR EVALUATION
# ============================================================

def evaluate_ocr(
    evaluator,
    pipeline_by_invoice,
    ground_truth_by_invoice,
    common_invoices,
    ground_truth_verified
):
    """
    OCR evaluation requires human-verified reference text.

    The project's original ground_truth.json explicitly marks
    its OCR data as draft/unverified.

    Therefore OCR is skipped when the ground truth is not
    verified.
    """

    if not ground_truth_verified:

        return {
            "status": "not_evaluated",

            "reason": (
                "OCR ground truth is marked "
                "as not human verified."
            )
        }

    cer_values = []
    wer_values = []

    for invoice_number in common_invoices:

        pipeline_record = (
            pipeline_by_invoice[
                invoice_number
            ]
        )

        ground_truth_record = (
            ground_truth_by_invoice[
                invoice_number
            ]
        )

        reference = get_ocr_text(
            ground_truth_record
        )

        prediction = (
            get_predicted_ocr_text(
                pipeline_record
            )
        )

        if not reference:
            continue

        cer = (
            evaluator.character_error_rate(
                reference,
                prediction
            )
        )

        wer = (
            evaluator.word_error_rate(
                reference,
                prediction
            )
        )

        cer_values.append(
            float(cer)
        )

        wer_values.append(
            float(wer)
        )

    if not cer_values:

        return {
            "status": "not_evaluated",

            "reason": (
                "No verified OCR "
                "reference text available."
            )
        }

    average_cer = (
        sum(cer_values)
        / len(cer_values)
    )

    average_wer = (
        sum(wer_values)
        / len(wer_values)
    )

    return {
        "status": "evaluated",

        "average_cer": round(
            average_cer,
            4
        ),

        "average_wer": round(
            average_wer,
            4
        ),

        "documents_evaluated": (
            len(cer_values)
        )
    }


# ============================================================
# PERFORMANCE
# ============================================================

def get_pipeline_total_time(
    pipeline_results,
    pipeline_records
):
    """
    Prefer the pipeline summary processing time.

    Fall back to summing page processing times.
    """

    summary = pipeline_results.get(
        "summary",
        {}
    )

    if isinstance(
        summary,
        dict
    ):

        total_time = (
            summary.get(
                "processing_time_seconds"
            )
        )

        if total_time is not None:

            return float(
                total_time
            )

    total_time = 0.0

    for record in pipeline_records:

        value = record.get(
            "processing_time_seconds"
        )

        if value is None:
            continue

        try:

            total_time += float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            continue

    return total_time


def evaluate_performance(
    evaluator,
    pipeline_results,
    pipeline_records
):
    """
    Evaluate pipeline processing performance.
    """

    total_pages = len(
        pipeline_records
    )

    total_time = (
        get_pipeline_total_time(
            pipeline_results,
            pipeline_records
        )
    )

    return (
        evaluator.evaluate_performance(
            total_time,
            total_pages
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MULTIMODAL DCA AI - EVALUATION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    evaluator = (
        DocumentAIEvaluator()
    )

    # --------------------------------------------------------
    # LOAD PIPELINE RESULTS
    # --------------------------------------------------------

    print(
        "\nLoading pipeline results..."
    )

    if not PIPELINE_RESULTS_PATH.exists():

        raise FileNotFoundError(
            "Pipeline results not found:\n"
            f"{PIPELINE_RESULTS_PATH}"
        )

    pipeline_results = load_json(
        PIPELINE_RESULTS_PATH
    )

    print(
        "Loaded: "
        f"{PIPELINE_RESULTS_PATH}"
    )

    # --------------------------------------------------------
    # LOAD GROUND TRUTH
    # --------------------------------------------------------

    print(
        "\nLoading ground truth..."
    )

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            "Ground truth not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    ground_truth = load_json(
        GROUND_TRUTH_PATH
    )

    print(
        "Loaded: "
        f"{GROUND_TRUTH_PATH}"
    )

    # --------------------------------------------------------
    # GET RECORDS
    # --------------------------------------------------------

    pipeline_records = get_records(
        pipeline_results
    )

    ground_truth_records = get_records(
        ground_truth
    )

    print(
        "\nPipeline pages: "
        f"{len(pipeline_records)}"
    )

    print(
        "Ground-truth records: "
        f"{len(ground_truth_records)}"
    )

    # --------------------------------------------------------
    # BUILD INDICES
    # --------------------------------------------------------

    pipeline_by_invoice = (
        build_invoice_index(
            pipeline_records
        )
    )

    ground_truth_by_invoice = (
        build_invoice_index(
            ground_truth_records
        )
    )

    common_invoices = sorted(
        set(
            pipeline_by_invoice.keys()
        )
        &
        set(
            ground_truth_by_invoice.keys()
        )
    )

    print(
        "Matched invoices: "
        f"{len(common_invoices)}"
    )

    if not common_invoices:

        raise RuntimeError(
            "No matching invoice numbers "
            "were found between pipeline "
            "results and ground truth."
        )

    # --------------------------------------------------------
    # GROUND-TRUTH STATUS
    # --------------------------------------------------------

    ground_truth_verified = (
        is_ground_truth_verified(
            ground_truth
        )
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "OCR EVALUATION"
    )

    print(
        "=" * 70
    )

    ocr_result = evaluate_ocr(
        evaluator,
        pipeline_by_invoice,
        ground_truth_by_invoice,
        common_invoices,
        ground_truth_verified
    )

    if (
        ocr_result.get("status")
        == "not_evaluated"
    ):

        print(
            "SKIPPED: "
            + ocr_result.get(
                "reason",
                "No verified OCR ground truth."
            )
        )

    else:

        print(
            json.dumps(
                ocr_result,
                indent=4
            )
        )

    # --------------------------------------------------------
    # INFORMATION EXTRACTION
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INFORMATION EXTRACTION"
    )

    print(
        "=" * 70
    )

    extraction_result = (
        evaluate_information_extraction(
            evaluator,
            pipeline_by_invoice,
            ground_truth_by_invoice,
            common_invoices
        )
    )

    print(
        f"Precision: "
        f"{extraction_result['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{extraction_result['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{extraction_result['f1_score']:.4f}"
    )

    print(
        f"Correct fields: "
        f"{extraction_result['correct_fields']}"
    )

    print(
        f"Predicted fields: "
        f"{extraction_result['predicted_fields']}"
    )

    print(
        f"Expected fields: "
        f"{extraction_result['expected_fields']}"
    )

    print(
        f"Documents evaluated: "
        f"{extraction_result['documents_evaluated']}"
    )

    # --------------------------------------------------------
    # TABLE EXTRACTION
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TABLE EXTRACTION"
    )

    print(
        "=" * 70
    )

    table_result = (
        evaluate_table_extraction(
            evaluator,
            pipeline_by_invoice,
            ground_truth_by_invoice,
            common_invoices
        )
    )

    if (
        table_result.get("status")
        == "not_evaluated"
    ):

        print(
            "SKIPPED: "
            + table_result.get(
                "reason",
                "No verified table ground truth."
            )
        )

    else:

        print(
            json.dumps(
                table_result,
                indent=4
            )
        )

    # --------------------------------------------------------
    # TABLE QUALITY
    # --------------------------------------------------------

    table_quality_result = (
        evaluate_table_quality(
            pipeline_records
        )
    )

    print(
        "\nTable Quality Evaluation:"
    )

    print(
        f"Pages with tables: "
        f"{table_quality_result['pages_with_tables']}/"
        f"{len(pipeline_records)}"
    )

    print(
        f"Total rows: "
        f"{table_quality_result['total_rows']}"
    )

    print(
        f"Row arithmetic accuracy: "
        f"{table_quality_result['row_arithmetic_accuracy']:.4f}"
    )

    print(
        f"Subtotal consistency: "
        f"{table_quality_result['subtotal_consistency_rate']:.4f}"
    )

    print(
        f"Table validation rate: "
        f"{table_quality_result['table_validation_rate']:.4f}"
    )

    print(
        f"Average table confidence: "
        f"{table_quality_result['average_table_confidence']:.4f}"
    )

    # --------------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANOMALY DETECTION"
    )

    print(
        "=" * 70
    )

    anomaly_result = (
        evaluate_anomaly_detection(
            evaluator,
            pipeline_by_invoice,
            ground_truth_by_invoice,
            common_invoices
        )
    )

    if (
        anomaly_result.get("status")
        == "not_evaluated"
    ):

        print(
            "SKIPPED: "
            + anomaly_result.get(
                "reason",
                "No verified anomaly ground truth."
            )
        )

    else:

        print(
            json.dumps(
                anomaly_result,
                indent=4
            )
        )

    # --------------------------------------------------------
    # ANOMALY PIPELINE SUMMARY
    # --------------------------------------------------------

    anomaly_summary = (
        evaluate_anomaly_pipeline_summary(
            pipeline_records
        )
    )

    print(
        "\nAnomaly Pipeline Summary:"
    )

    print(
        f"ML anomalies: "
        f"{anomaly_summary['ml_anomaly_count']}"
    )

    print(
        f"ML normal: "
        f"{anomaly_summary['ml_normal_count']}"
    )

    print(
        f"ML anomaly rate: "
        f"{anomaly_summary['ml_anomaly_rate']:.4f}"
    )

    print(
        f"Average ML anomaly score: "
        f"{anomaly_summary['average_ml_anomaly_score']:.4f}"
    )

    print(
        f"Fraud anomalies: "
        f"{anomaly_summary['fraud_anomaly_count']}"
    )

    print(
        f"Fraud normal: "
        f"{anomaly_summary['fraud_normal_count']}"
    )

    print(
        f"Fraud anomaly rate: "
        f"{anomaly_summary['fraud_anomaly_rate']:.4f}"
    )

    print(
        f"Average fraud anomaly score: "
        f"{anomaly_summary['average_fraud_anomaly_score']:.4f}"
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PERFORMANCE"
    )

    print(
        "=" * 70
    )

    performance_result = (
        evaluate_performance(
            evaluator,
            pipeline_results,
            pipeline_records
        )
    )

    print(
        json.dumps(
            performance_result,
            indent=4
        )
    )

    # --------------------------------------------------------
    # BUILD FINAL REPORT
    # --------------------------------------------------------

    report = (
        evaluator.build_report(
            ocr=ocr_result,
            information_extraction=(
                extraction_result
            ),
            table=table_result,
            anomaly=anomaly_result,
            performance=(
                performance_result
            )
        )
    )

    # --------------------------------------------------------
    # ADD PROJECT SUMMARY
    # --------------------------------------------------------

    report[
        "evaluation_summary"
    ] = {

        "documents_evaluated": (
            len(common_invoices)
        ),

        "information_extraction_f1": (
            extraction_result.get(
                "f1_score",
                0.0
            )
        ),

        "table_extraction": (
            "NOT EVALUATED"
            if table_result.get(
                "status"
            ) == "not_evaluated"
            else table_result.get(
                "f1_score",
                0.0
            )
        ),

        "table_quality": (
            table_quality_result
        ),

        "anomaly_detection": (
            "NOT EVALUATED"
            if anomaly_result.get(
                "status"
            ) == "not_evaluated"
            else anomaly_result
        ),

        "anomaly_pipeline_summary": (
            anomaly_summary
        ),

        "average_page_time_seconds": (
            performance_result.get(
                "average_page_time_seconds",
                0.0
            )
        )
    }

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    save_json(
        report,
        EVALUATION_REPORT_PATH
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Documents evaluated: "
        f"{len(common_invoices)}"
    )

    print(
        "Information Extraction F1: "
        f"{extraction_result.get('f1_score', 0.0):.4f}"
    )

    if (
        table_result.get("status")
        == "not_evaluated"
    ):

        print(
            "Table Extraction F1: "
            "NOT EVALUATED"
        )

    else:

        print(
            "Table Extraction F1: "
            f"{table_result.get('f1_score', 0.0):.4f}"
        )

    print(
        "Table Row Arithmetic Accuracy: "
        f"{table_quality_result.get('row_arithmetic_accuracy', 0.0):.4f}"
    )

    print(
        "Table Subtotal Consistency: "
        f"{table_quality_result.get('subtotal_consistency_rate', 0.0):.4f}"
    )

    print(
        "Table Validation Rate: "
        f"{table_quality_result.get('table_validation_rate', 0.0):.4f}"
    )

    if (
        anomaly_result.get("status")
        == "not_evaluated"
    ):

        print(
            "Anomaly Detection: "
            "NOT EVALUATED"
        )

    else:

        print(
            "Anomaly Detection: "
            "EVALUATED"
        )

    print(
        "ML Anomalies Detected: "
        f"{anomaly_summary.get('ml_anomaly_count', 0)}"
    )

    print(
        "Fraud Anomalies Detected: "
        f"{anomaly_summary.get('fraud_anomaly_count', 0)}"
    )

    print(
        "Average page time: "
        f"{performance_result.get('average_page_time_seconds', 0.0):.4f}s"
    )

    print(
        "\nReport saved to:"
    )

    print(
        EVALUATION_REPORT_PATH
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()