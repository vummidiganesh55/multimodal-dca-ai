import json
import os
import time
import numpy as np

from src.document_loader import load_document
from src.preprocessing.image_processor import preprocess_image
from src.ocr.paddle_ocr import PaddleOCRProcessor
from src.layout.layout_analyzer import LayoutAnalyzer
from src.extraction.information_extractor import InformationExtractor

from src.tables.table_extractor import TableExtractor
from src.tables.table_structure import TableStructureExtractor
from src.tables.table_json_builder import TableJSONBuilder

from src.validation.document_validator import DocumentValidator

from src.anomaly.rules import RuleBasedRiskDetector
from src.anomaly.ml_anomaly import DocumentAnomalyDetector

from src.handwriting.handwriting_recognizer import HandwritingRecognizer
from src.multilingual import LanguageDetector
from src.chart_analysis.chart_analyzer import ChartAnalyzer


# ============================================================
# FRAUD / ANOMALY SIGNAL ANALYZER
# ============================================================


class FraudAnomalyDetector:
    """
    Combines:

        1. Validation issues
        2. Rule-based risk
        3. ML anomaly detection

    IMPORTANT:
    An ML anomaly alone does NOT mean fraud.

    ML anomaly is treated as a statistical signal.
    A document becomes suspicious only when there is
    supporting validation or rule-based risk evidence.
    """

    def __init__(self):
        pass

    def analyze(
        self,
        validation_result,
        risk_result,
        anomaly_result
    ):
        score = 0
        signals = []

        # ----------------------------------------------------
        # VALIDATION ISSUES
        # ----------------------------------------------------

        validation_issues = 0

        if isinstance(validation_result, dict):

            if "issue_count" in validation_result:
                try:
                    validation_issues = int(
                        validation_result["issue_count"]
                    )
                except (TypeError, ValueError):
                    validation_issues = 0

            elif "error_count" in validation_result:
                try:
                    validation_issues = int(
                        validation_result["error_count"]
                    )
                except (TypeError, ValueError):
                    validation_issues = 0

            elif "issues" in validation_result:
                issues = validation_result["issues"]

                if isinstance(issues, list):
                    validation_issues = len(issues)

            elif "errors" in validation_result:
                errors = validation_result["errors"]

                if isinstance(errors, list):
                    validation_issues = len(errors)

            elif "warnings" in validation_result:
                warnings = validation_result["warnings"]

                if isinstance(warnings, list):
                    validation_issues = len(warnings)

        elif isinstance(validation_result, list):

            validation_issues = len(validation_result)

        # ----------------------------------------------------
        # VALIDATION SCORE
        # ----------------------------------------------------

        if validation_issues > 0:

            validation_score = min(
                validation_issues * 15,
                40
            )

            score += validation_score

            signals.append(
                f"{validation_issues} validation issue(s)"
            )

        # ----------------------------------------------------
        # RULE-BASED RISK
        # ----------------------------------------------------

        risk_level = "LOW"
        risk_score = 0.0

        if isinstance(risk_result, dict):

            risk_level = str(
                risk_result.get(
                    "risk_level",
                    risk_result.get(
                        "level",
                        "LOW"
                    )
                )
            ).upper()

            try:

                risk_score = float(
                    risk_result.get(
                        "risk_score",
                        risk_result.get(
                            "score",
                            0
                        )
                    )
                )

            except (TypeError, ValueError):

                risk_score = 0.0

        # ----------------------------------------------------
        # RISK LEVEL SCORE
        # ----------------------------------------------------

        if risk_level == "HIGH":

            score += 35

            signals.append(
                "High rule-based risk"
            )

        elif risk_level == "MEDIUM":

            score += 20

            signals.append(
                "Medium rule-based risk"
            )

        # Numerical risk score
        if risk_score > 0:

            score += min(
                risk_score,
                20
            )

        # ----------------------------------------------------
        # ML ANOMALY
        # ----------------------------------------------------

        ml_status = "normal"
        ml_anomaly_score = 0.0

        if isinstance(anomaly_result, dict):

            ml_status = str(
                anomaly_result.get(
                    "status",
                    "normal"
                )
            ).lower()

            try:

                ml_anomaly_score = float(
                    anomaly_result.get(
                        "anomaly_score",
                        0.0
                    )
                )

            except (TypeError, ValueError):

                ml_anomaly_score = 0.0

        # ----------------------------------------------------
        # ML ANOMALY SIGNAL
        # ----------------------------------------------------

        ml_anomaly_detected = (
            ml_status == "anomaly"
        )

        if ml_anomaly_detected:

            signals.append(
                "ML anomaly detected"
            )

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        # ML anomaly by itself does NOT increase the
        # suspicious score.
        #
        # It becomes meaningful when combined with:
        #   - validation issues
        #   - medium/high rule-based risk

        supporting_evidence = (
            validation_issues > 0
            or risk_level in {"MEDIUM", "HIGH"}
            or risk_score > 0
        )

        if ml_anomaly_detected and supporting_evidence:

            signals.append(
                "ML anomaly supported by validation/risk evidence"
            )

            score += 30

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        score = min(
            score,
            100
        )

        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        if score >= 60:

            status = "highly_suspicious"

        elif score >= 30:

            status = "suspicious"

        else:

            status = "normal"

        # ----------------------------------------------------
        # REVIEW DECISION
        # ----------------------------------------------------

        review_required = (
            status != "normal"
        )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {

            "status":
                status,

            "score":
                round(
                    float(score),
                    2
                ),

            "signals":
                signals,

            "validation_issues":
                validation_issues,

            "risk_level":
                risk_level,

            "risk_score":
                round(
                    float(risk_score),
                    2
                ),

            "ml_status":
                ml_status,

            "ml_anomaly_score":
                round(
                    float(ml_anomaly_score),
                    4
                ),

            "ml_anomaly_detected":
                ml_anomaly_detected,

            "supporting_evidence":
                supporting_evidence,

            "review_required":
                review_required
        }



# ============================================================
# DOCUMENT AI PIPELINE
# ============================================================

class DocumentAIPipeline:

    def __init__(self):

        print("\n")
        print("=" * 60)
        print("INITIALIZING MULTIMODAL DCA AI PIPELINE")
        print("=" * 60)

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        self.ocr = PaddleOCRProcessor()

        # ----------------------------------------------------
        # LAYOUT
        # ----------------------------------------------------

        self.layout_analyzer = LayoutAnalyzer()

        # ----------------------------------------------------
        # INFORMATION EXTRACTION
        # ----------------------------------------------------

        self.extractor = InformationExtractor()

        # ----------------------------------------------------
        # TABLE DETECTION
        # ----------------------------------------------------

        self.table_extractor = TableExtractor()

        # ----------------------------------------------------
        # TABLE STRUCTURE
        # ----------------------------------------------------

        self.table_structure = (
            TableStructureExtractor()
        )

        # ----------------------------------------------------
        # TABLE JSON
        # ----------------------------------------------------

        self.table_json_builder = (
            TableJSONBuilder()
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        self.validator = (
            DocumentValidator()
        )

        # ----------------------------------------------------
        # RULE BASED RISK
        # ----------------------------------------------------

        self.risk_detector = (
            RuleBasedRiskDetector()
        )

        # ----------------------------------------------------
        # ML ANOMALY
        # ----------------------------------------------------

        self.anomaly_detector = (
            DocumentAnomalyDetector()
        )

        # ----------------------------------------------------
        # FRAUD / ANOMALY SIGNAL
        # ----------------------------------------------------

        self.fraud_detector = (
            FraudAnomalyDetector()
        )

        # ----------------------------------------------------
        # HANDWRITING
        # ----------------------------------------------------

        self.handwriting_recognizer = (
            HandwritingRecognizer()
        )

        # ----------------------------------------------------
        # MULTILINGUAL LANGUAGE DETECTION
        # ----------------------------------------------------

        self.language_detector = LanguageDetector()

        # ----------------------------------------------------
        # CHART ANALYSIS
        # ----------------------------------------------------

        self.chart_analyzer = ChartAnalyzer()

        print("=" * 60)
        print("PIPELINE INITIALIZED")
        print("=" * 60)


    # ========================================================
    # HANDWRITING PROCESSING
    # ========================================================

    def _process_handwriting(
        self,
        handwriting_image
    ):

        print("\n")
        print("=" * 50)
        print("HANDWRITING RECOGNITION")
        print("=" * 50)

        if handwriting_image is None:

            print(
                "No handwriting region "
                "provided/detected."
            )

            return {

                "detected": False,

                "text": "",

                "model":
                    self.handwriting_recognizer.model_name,

                "device":
                    self.handwriting_recognizer.device
            }

        try:

            result = (
                self.handwriting_recognizer.recognize(
                    handwriting_image
                )
            )

            print(
                f"Recognized text: "
                f"{result['text']}"
            )

            return {

                "detected": True,

                "text":
                    result.get(
                        "text",
                        ""
                    ),

                "model":
                    result.get(
                        "model",
                        ""
                    ),

                "device":
                    result.get(
                        "device",
                        ""
                    )
            }

        except Exception as e:

            print(
                "Handwriting recognition "
                f"failed: {e}"
            )

            return {

                "detected": False,

                "text": "",

                "error": str(e)
            }


    # ========================================================
    # CREATE ANOMALY FEATURES
    # ========================================================

    def _create_feature_vector(
        self,
        extraction,
        table_json,
        validation,
        ocr_result,
        processing_time
    ):

        # ----------------------------------------------------
        # TOTAL AMOUNT
        # ----------------------------------------------------

        total_amount = 0.0

        if isinstance(extraction, dict):

            possible_keys = [
                "total",
                "total_amount",
                "grand_total",
                "amount"
            ]

            for key in possible_keys:

                if key in extraction:

                    try:

                        value = extraction[key]

                        if value is not None:

                            total_amount = float(
                                str(value)
                                .replace(",", "")
                                .replace("₹", "")
                                .replace("$", "")
                                .strip()
                            )

                            break

                    except (
                        TypeError,
                        ValueError
                    ):

                        pass

        # ----------------------------------------------------
        # TABLE ROW COUNT
        # ----------------------------------------------------

        row_count = 0

        if isinstance(table_json, dict):

            rows = table_json.get(
                "rows",
                []
            )

            if isinstance(rows, list):

                row_count = len(rows)

        elif isinstance(table_json, list):

            row_count = len(
                table_json
            )

        # ----------------------------------------------------
        # OCR CONFIDENCE
        # ----------------------------------------------------

        avg_ocr_confidence = 0.0

        confidences = []

        if isinstance(
            ocr_result,
            dict
        ):

            possible_confidence_keys = [
                "confidence",
                "confidences",
                "ocr_confidence"
            ]

            for key in possible_confidence_keys:

                if key in ocr_result:

                    value = ocr_result[key]

                    if isinstance(
                        value,
                        (list, tuple)
                    ):

                        for item in value:

                            try:

                                confidences.append(
                                    float(item)
                                )

                            except (
                                TypeError,
                                ValueError
                            ):

                                pass

                    else:

                        try:

                            confidences.append(
                                float(value)
                            )

                        except (
                            TypeError,
                            ValueError
                        ):

                            pass

        if len(confidences) > 0:

            avg_ocr_confidence = float(
                np.mean(confidences)
            )

        # ----------------------------------------------------
        # VALIDATION ISSUE COUNT
        # ----------------------------------------------------

        validation_issue_count = 0

        if isinstance(
            validation,
            dict
        ):

            if "issue_count" in validation:

                try:

                    validation_issue_count = int(
                        validation["issue_count"]
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    validation_issue_count = 0

            elif "error_count" in validation:

                try:

                    validation_issue_count = int(
                        validation["error_count"]
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    validation_issue_count = 0

            elif isinstance(
                validation.get("issues"),
                list
            ):

                validation_issue_count = len(
                    validation["issues"]
                )

            elif isinstance(
                validation.get("errors"),
                list
            ):

                validation_issue_count = len(
                    validation["errors"]
                )

        elif isinstance(
            validation,
            list
        ):

            validation_issue_count = len(
                validation
            )

        # ----------------------------------------------------
        # FEATURE VECTOR
        # ----------------------------------------------------

        feature_vector = np.array(
            [
                total_amount,
                row_count,
                avg_ocr_confidence,
                validation_issue_count,
                processing_time
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # SAFETY
        # ----------------------------------------------------

        feature_vector = np.nan_to_num(
            feature_vector,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return feature_vector


    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================

    def process(
        self,
        document_path,
        handwriting_image=None
    ):

        pipeline_start = time.time()

        print("\n")
        print("=" * 60)
        print("DOCUMENT PROCESSING STARTED")
        print("=" * 60)

        print(
            f"Input document: "
            f"{document_path}"
        )

        # ----------------------------------------------------
        # LOAD DOCUMENT
        # ----------------------------------------------------

        print("\nLoading document...")

        pages = load_document(
            document_path
        )

        print(
            f"Total pages: "
            f"{len(pages)}"
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        results = []

        feature_vectors = []

        page_processing_times = []

        # ====================================================
        # PAGE PROCESSING
        # ====================================================

        for page_index, page in enumerate(
            pages
        ):

            page_number = page_index + 1

            print("\n")
            print("=" * 40)
            print(
                f"PAGE {page_number}/"
                f"{len(pages)}"
            )
            print("=" * 40)

            page_start = time.time()

            # ------------------------------------------------
            # PREPROCESSING
            # ------------------------------------------------

            print("Preprocessing...")

            processed_image = (
                preprocess_image(page)
            )

            # ------------------------------------------------
            # SAVE PROCESSED IMAGE
            # ------------------------------------------------

            os.makedirs(
                "outputs",
                exist_ok=True
            )

            processed_image_path = os.path.join(
                "outputs",
                f"processed_page_"
                f"{page_number}.png"
            )

            if hasattr(
                processed_image,
                "save"
            ):

                processed_image.save(
                    processed_image_path
                )

            else:

                from PIL import Image

                Image.fromarray(
                    processed_image
                ).save(
                    processed_image_path
                )

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            print("OCR...")

            ocr_result = (
                self.ocr.process(
                    processed_image_path
                )
            )
            
            # ------------------------------------------------
            # OCR DEBUG
            # ------------------------------------------------

            if isinstance(
                ocr_result,
                dict
            ):

                words = ocr_result.get(
                    "words",
                    []
                )

                if isinstance(
                    words,
                    list
                ):

                    print(
                        f"OCR words: "
                        f"{len(words)}"
                    )

            # ------------------------------------------------
            # MULTILINGUAL LANGUAGE DETECTION
            # ------------------------------------------------

            print("Language detection...")

            ocr_text_parts = []

            if isinstance(ocr_result, list):

                for word in ocr_result:

                    if isinstance(word, dict):

                        text = word.get(
                            "text",
                            ""
                        )

                        if text:
                            ocr_text_parts.append(
                                str(text)
                            )

                    elif isinstance(word, str):

                        ocr_text_parts.append(
                            word
                        )

            elif isinstance(ocr_result, dict):

                words = ocr_result.get(
                    "words",
                    []
                )

                if isinstance(words, list):

                    for word in words:

                        if isinstance(word, dict):

                            text = word.get(
                                "text",
                                ""
                            )

                            if text:
                                ocr_text_parts.append(
                                    str(text)
                                )

                        elif isinstance(word, str):

                            ocr_text_parts.append(
                                word
                            )

            ocr_text = " ".join(
                ocr_text_parts
            )

            language_result = (
                self.language_detector.detect(
                    ocr_text
                )
            )

            print(
                f"Detected language: "
                f"{language_result.get('language', 'unknown')} "
                f"| confidence="
                f"{language_result.get('confidence', 0.0):.4f}"
            )


            # ------------------------------------------------
            # LAYOUT
            # ------------------------------------------------

            print("Layout...")

            layout_result = (
                self.layout_analyzer.analyze(
                    processed_image,
                    ocr_result
                )
            )

            # ------------------------------------------------
            # CHART ANALYSIS
            # ------------------------------------------------

            print("Chart analysis...")

            chart_result = (
                self.chart_analyzer.analyze(
                    processed_image,
                    ocr_result
                )
            )

            # ------------------------------------------------
            # INFORMATION EXTRACTION
            # ------------------------------------------------

            print(
                "Information extraction..."
            )

            extraction = (
                self.extractor.extract(
                    ocr_result
                )
            )

            # ------------------------------------------------
            # TABLE DETECTION
            # ------------------------------------------------

            print("Table detection...")

            table_result = (
                self.table_extractor.detect(
                    processed_image
                )
            )

            # ------------------------------------------------
            # TABLE STRUCTURE
            #
            # IMPORTANT:
            # extract() accepts ONLY table_result.
            # Do NOT pass processed_image here.
            # ------------------------------------------------

            print("Table structure...")

            table_structure = (
                self.table_structure.extract(
                    processed_image
                )
            )

            # ------------------------------------------------
            # TABLE JSON
            # ------------------------------------------------

            print("Table JSON...")

            table_json = (
                self.table_json_builder.build(
                    ocr_result,
                    table_structure
                )
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            print("Validation...")

            validation = (
                self.validator.validate(
                    extraction,
                    table_json
                )
            )

            # ------------------------------------------------
            # RULE BASED RISK
            # ------------------------------------------------

            print("Risk analysis...")

            risk = (
                self.risk_detector.analyze(
                    validation
                )
            )

            # ------------------------------------------------
            # PAGE PROCESSING TIME
            # ------------------------------------------------

            page_time = (
                time.time()
                - page_start
            )

            page_processing_times.append(
                page_time
            )

            # ------------------------------------------------
            # ML FEATURE VECTOR
            # ------------------------------------------------

            feature_vector = (
                self._create_feature_vector(
                    extraction,
                    table_json,
                    validation,
                    ocr_result,
                    page_time
                )
            )

            feature_vectors.append(
                feature_vector
            )

            # ------------------------------------------------
            # PAGE RESULT
            # ------------------------------------------------

            page_result = {

                "page":
                    page_number,

                "language":
                    language_result,

                "ocr":
                    ocr_result,

                "layout":
                    layout_result,

                "chart_analysis":
                    chart_result,

                "extraction":
                    extraction,

                "table_detection":
                    table_result,

                "table_structure":
                    table_structure,

                "table_json":
                    table_json,

                "validation":
                    validation,

                "risk":
                    risk,

                "processing_time_seconds":
                    round(
                        page_time,
                        3
                    )
            }

            results.append(
                page_result
            )

            print(
                f"Page {page_number} "
                f"completed in "
                f"{page_time:.2f}s"
            )

        # ====================================================
        # ML ANOMALY DETECTION
        # ====================================================

        print("\n")
        print("=" * 50)
        print("ML ANOMALY DETECTION")
        print("=" * 50)

        if len(feature_vectors) > 0:

            feature_matrix = np.vstack(
                feature_vectors
            )

            # ----------------------------------------------
            # TRAIN
            # ----------------------------------------------

            print(
                "Training Isolation Forest..."
            )

            self.anomaly_detector.fit(
                feature_matrix
            )

            # ----------------------------------------------
            # PREDICTION
            # ----------------------------------------------

            for index, feature_vector in enumerate(
                feature_vectors
            ):

                anomaly = (
                    self.anomaly_detector.predict(
                        feature_vector
                    )
                )

                results[index][
                    "ml_anomaly"
                ] = anomaly

                # ------------------------------------------
                # FRAUD / ANOMALY SIGNAL
                # ------------------------------------------

                fraud_result = (
                    self.fraud_detector.analyze(
                        results[index].get(
                            "validation",
                            {}
                        ),

                        results[index].get(
                            "risk",
                            {}
                        ),

                        anomaly
                    )
                )

                results[index][
                    "fraud_anomaly"
                ] = fraud_result

                # ------------------------------------------
                # DISPLAY
                # ------------------------------------------

                print(
                    f"Page "
                    f"{index + 1} "
                    f"ML anomaly: "
                    f"{anomaly['status']} "
                    f"| score="
                    f"{anomaly['anomaly_score']:.4f}"
                )

                print(
                    f"Page "
                    f"{index + 1} "
                    f"Fraud signal: "
                    f"{fraud_result['status']} "
                    f"| score="
                    f"{fraud_result['score']:.2f}"
                )

        # ====================================================
        # HANDWRITING RECOGNITION
        # ====================================================

        handwriting_result = (
            self._process_handwriting(
                handwriting_image
            )
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        total_processing_time = (
            time.time()
            - pipeline_start
        )

        # ----------------------------------------------------
        # ML ANOMALY COUNT
        # ----------------------------------------------------

        ml_anomalous_pages = 0

        for result in results:

            anomaly = result.get(
                "ml_anomaly"
            )

            if (
                isinstance(anomaly, dict)
                and anomaly.get(
                    "status"
                ) == "anomaly"
            ):

                ml_anomalous_pages += 1

        # ----------------------------------------------------
        # SUSPICIOUS COUNTS
        # ----------------------------------------------------

        suspicious_pages = 0
        highly_suspicious_pages = 0
        normal_signal_pages = 0

        for result in results:

            fraud = result.get(
                "fraud_anomaly"
            )

            if not isinstance(
                fraud,
                dict
            ):
                continue

            status = fraud.get(
                "status",
                "normal"
            )

            if status == "highly_suspicious":

                highly_suspicious_pages += 1

            elif status == "suspicious":

                suspicious_pages += 1

            else:

                normal_signal_pages += 1

        # ----------------------------------------------------
        # RISK COUNTS
        # ----------------------------------------------------

        high_risk_pages = 0
        medium_risk_pages = 0
        low_risk_pages = 0

        for result in results:

            risk = result.get(
                "risk"
            )

            if not isinstance(
                risk,
                dict
            ):
                continue

            risk_level = str(
                risk.get(
                    "risk_level",
                    risk.get(
                        "level",
                        "LOW"
                    )
                )
            ).upper()

            if risk_level == "HIGH":

                high_risk_pages += 1

            elif risk_level == "MEDIUM":

                medium_risk_pages += 1

            else:

                low_risk_pages += 1

        # ====================================================
        # FINAL RESULT
        # ====================================================

        final_result = {

            "document":
                document_path,

            "pages":
                results,

            "handwriting":
                handwriting_result,

            "summary": {

                "total_pages":
                    len(results),

                "processing_time_seconds":
                    round(
                        total_processing_time,
                        2
                    ),

                "ml_anomalous_pages":
                    ml_anomalous_pages,

                "suspicious_pages":
                    suspicious_pages,

                "highly_suspicious_pages":
                    highly_suspicious_pages,

                "normal_signal_pages":
                    normal_signal_pages,

                "high_risk_pages":
                    high_risk_pages,

                "medium_risk_pages":
                    medium_risk_pages,

                "low_risk_pages":
                    low_risk_pages
            }
        }

        # ====================================================
        # FINAL CONSOLE OUTPUT
        # ====================================================

        print("\n")
        print("=" * 50)
        print("PAGE ANOMALY SUMMARY")
        print("=" * 50)

        for result in results:

            page_number = result.get(
                "page"
            )

            anomaly = result.get(
                "ml_anomaly",
                {}
            )

            fraud = result.get(
                "fraud_anomaly",
                {}
            )

            if isinstance(
                anomaly,
                dict
            ):

                print(
                    f"Page {page_number} "
                    f"ML anomaly: "
                    f"{anomaly.get('status', 'unknown')} "
                    f"| score="
                    f"{anomaly.get('anomaly_score', 0):.4f}"
                )

            if isinstance(
                fraud,
                dict
            ):

                print(
                    f"Page {page_number} "
                    f"Fraud signal: "
                    f"{fraud.get('status', 'unknown')} "
                    f"| score="
                    f"{fraud.get('score', 0):.2f}"
                )

        # ====================================================
        # HANDWRITING SUMMARY
        # ====================================================

        print("\n")
        print("=" * 50)
        print("HANDWRITING SUMMARY")
        print("=" * 50)

        if handwriting_result.get(
            "detected",
            False
        ):

            print(
                "Detected: YES"
            )

            print(
                "Text: "
                f"{handwriting_result.get('text', '')}"
            )

        else:

            print(
                "Detected: NO"
            )

            if handwriting_result.get(
                "error"
            ):

                print(
                    "Error: "
                    f"{handwriting_result['error']}"
                )

            else:

                print(
                    "No handwriting region "
                    "provided/detected."
                )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print("\n")
        print("=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)

        print(
            f"Pages: "
            f"{len(results)}"
        )

        print(
            f"Processing time: "
            f"{total_processing_time:.2f}s"
        )

        print(
            f"ML anomalous pages: "
            f"{ml_anomalous_pages}"
        )

        print(
            f"Suspicious pages: "
            f"{suspicious_pages}"
        )

        print(
            f"Highly suspicious pages: "
            f"{highly_suspicious_pages}"
        )

        print(
            f"Normal signal pages: "
            f"{normal_signal_pages}"
        )

        print(
            f"High-risk pages: "
            f"{high_risk_pages}"
        )

        print(
            f"Medium-risk pages: "
            f"{medium_risk_pages}"
        )

        print(
            f"Low-risk pages: "
            f"{low_risk_pages}"
        )

        print(
            "Results generated successfully."
        )

        return final_result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    pipeline = (
        DocumentAIPipeline()
    )

    result = pipeline.process(
        "data/invoice.pdf"
    )

    with open(
        "outputs/pipeline_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    # ========================================================
    # SAVE ACTUAL PIPELINE RESULTS
    # ========================================================

    output_directory = "outputs"

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    pipeline_results_path = os.path.join(
        output_directory,
        "pipeline_results.json"
    )

    with open(
        pipeline_results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    print("\n" + "=" * 50)
    print("PIPELINE RESULTS SAVED")
    print("=" * 50)

    print(
        f"File: {pipeline_results_path}"
    )

    print("=" * 50)
