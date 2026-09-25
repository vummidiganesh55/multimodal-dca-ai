import re
import cv2
import numpy as np
from PIL import Image


class ChartAnalyzer:

    def __init__(self):

        self.model_name = "OpenCV Chart Analyzer"

    # ============================================================
    # IMAGE PREPARATION
    # ============================================================

    def _to_numpy(self, image):

        if isinstance(image, Image.Image):

            image = image.convert("RGB")

            return np.array(image)

        return np.asarray(image)

    # ============================================================
    # TEXT EXTRACTION FROM OCR
    # ============================================================

    def _extract_text(self, ocr_result):

        texts = []

        if isinstance(ocr_result, list):

            for item in ocr_result:

                if isinstance(item, dict):

                    text = item.get("text", "")

                    if text:
                        texts.append(str(text))

                elif isinstance(item, str):

                    texts.append(item)

        elif isinstance(ocr_result, dict):

            words = ocr_result.get("words", [])

            if isinstance(words, list):

                for item in words:

                    if isinstance(item, dict):

                        text = item.get("text", "")

                        if text:
                            texts.append(str(text))

                    elif isinstance(item, str):

                        texts.append(item)

        return texts

    # ============================================================
    # NUMERIC VALUE EXTRACTION
    # ============================================================

    def _extract_numeric_values(self, texts):

        values = []

        for text in texts:

            matches = re.findall(
                r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?",
                text
            )

            for match in matches:

                try:

                    value = float(
                        match.replace(",", "")
                    )

                    values.append(value)

                except ValueError:

                    pass

        return values

    # ============================================================
    # CHART REGION DETECTION
    # ============================================================

    def _detect_chart_region(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        # Edge detection
        edges = cv2.Canny(
            gray,
            50,
            150
        )

        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        height, width = gray.shape

        image_area = height * width

        candidates = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            area = w * h

            if area < image_area * 0.05:
                continue

            if w < 100 or h < 100:
                continue

            aspect_ratio = w / max(h, 1)

            if 0.3 <= aspect_ratio <= 5.0:

                candidates.append(
                    {
                        "bbox": [
                            int(x),
                            int(y),
                            int(x + w),
                            int(y + h)
                        ],
                        "area": int(area),
                        "width": int(w),
                        "height": int(h)
                    }
                )

        if not candidates:

            return None

        # Largest suitable region
        candidates.sort(
            key=lambda item: item["area"],
            reverse=True
        )

        return candidates[0]

    # ============================================================
    # CHART TYPE ESTIMATION
    # ============================================================

    def _estimate_chart_type(
        self,
        image,
        region
    ):

        if region is None:

            return "unknown"

        x1, y1, x2, y2 = region["bbox"]

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:

            return "unknown"

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2GRAY
        )

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        vertical_shapes = 0
        horizontal_shapes = 0

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < 5 or h < 5:
                continue

            if h > w * 1.5:

                vertical_shapes += 1

            elif w > h * 1.5:

                horizontal_shapes += 1

        # Bar-chart heuristic
        if vertical_shapes >= 3:

            return "bar"

        if horizontal_shapes >= 3:

            return "bar"

        # Detect circular structure
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=max(
                20,
                min(crop.shape[:2]) // 2
            )
        )

        if circles is not None:

            return "pie"

        # Default chart type
        return "chart"

    # ============================================================
    # CHART SUMMARY
    # ============================================================

    def _create_summary(
        self,
        chart_type,
        values
    ):

        if not values:

            return (
                f"{chart_type.capitalize()} chart detected "
                "but no numeric values were confidently extracted."
            )

        if len(values) == 1:

            return (
                f"{chart_type.capitalize()} chart containing "
                f"one detected numeric value: {values[0]:g}."
            )

        minimum = min(values)
        maximum = max(values)

        max_index = values.index(maximum)
        min_index = values.index(minimum)

        if max_index > min_index:

            trend = (
                "The detected values generally increase "
                "from the minimum toward the maximum."
            )

        else:

            trend = (
                "The detected values contain a decrease "
                "from the maximum toward the minimum."
            )

        return (
            f"{chart_type.capitalize()} chart with "
            f"{len(values)} detected numeric values. "
            f"Minimum: {minimum:g}. "
            f"Maximum: {maximum:g}. "
            f"{trend}"
        )

    # ============================================================
    # MAIN ANALYSIS
    # ============================================================

    def analyze(
        self,
        image,
        ocr_result
    ):

        image = self._to_numpy(image)

        if image.ndim != 3:

            return {
                "detected": False,
                "chart_type": "unknown",
                "confidence": 0.0,
                "text": [],
                "numeric_values": [],
                "bbox": None,
                "summary": "Invalid image format."
            }

        # Ensure RGB
        if image.shape[2] == 4:

            image = image[:, :, :3]

        # --------------------------------------------------------
        # OCR TEXT
        # --------------------------------------------------------

        texts = self._extract_text(
            ocr_result
        )

        numeric_values = (
            self._extract_numeric_values(
                texts
            )
        )

        # --------------------------------------------------------
        # CHART REGION
        # --------------------------------------------------------

        region = self._detect_chart_region(
            image
        )

        if region is None:

            return {
                "detected": False,
                "chart_type": "none",
                "confidence": 0.0,
                "text": texts,
                "numeric_values": numeric_values,
                "bbox": None,
                "summary": "No chart region detected.",
                "model": self.model_name
            }

        # --------------------------------------------------------
        # CHART TYPE
        # --------------------------------------------------------

        chart_type = self._estimate_chart_type(
            image,
            region
        )

        # --------------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------------

        confidence = 0.60

        if chart_type != "unknown":

            confidence += 0.10

        if len(texts) >= 3:

            confidence += 0.10

        if len(numeric_values) >= 2:

            confidence += 0.10

        confidence = min(
            confidence,
            0.95
        )

        # --------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------

        summary = self._create_summary(
            chart_type,
            numeric_values
        )

        # --------------------------------------------------------
        # RESULT
        # --------------------------------------------------------

        return {

            "detected": True,

            "chart_type":
                chart_type,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "bbox":
                region["bbox"],

            "text":
                texts,

            "numeric_values":
                numeric_values,

            "summary":
                summary,

            "model":
                self.model_name
        }