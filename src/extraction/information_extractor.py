import re
from typing import Any, Dict, List, Optional


class InformationExtractor:

    def __init__(self):
        pass

    # ============================================================
    # MAIN EXTRACTION
    # ============================================================

    def extract(self, ocr_result: Any) -> Dict[str, Any]:

        ocr_items = self._get_ocr_items(ocr_result)

        texts = [
            str(item.get("text", "")).strip()
            for item in ocr_items
            if str(item.get("text", "")).strip()
        ]

        full_text = " ".join(texts)

        normalized_text = re.sub(
            r"\s+",
            " ",
            full_text
        ).strip()

        return {
            "invoice_number":
                self._extract_invoice_number(
                    normalized_text
                ),

            "invoice_date":
                self._extract_invoice_date(
                    normalized_text
                ),

            "vendor_name":
                self._extract_vendor(
                    ocr_items
                ),

            "customer_name":
                self._extract_customer(
                    ocr_items
                ),

            "subtotal":
                self._extract_subtotal(
                    normalized_text
                ),

            "tax":
                self._extract_tax(
                    normalized_text
                ),

            "total":
                self._extract_total(
                    normalized_text,
                    ocr_items
                ),

            "payment_status":
                self._extract_payment_status(
                    normalized_text
                )
        }

    # ============================================================
    # OCR NORMALIZATION
    # ============================================================

    def _get_ocr_items(
        self,
        ocr_result: Any
    ) -> List[Dict[str, Any]]:

        if isinstance(ocr_result, dict):

            if "items" in ocr_result:
                items = ocr_result["items"]

                if isinstance(items, list):
                    return [
                        item
                        for item in items
                        if isinstance(item, dict)
                    ]

            if "words" in ocr_result:
                items = ocr_result["words"]

                if isinstance(items, list):
                    return [
                        item
                        for item in items
                        if isinstance(item, dict)
                    ]

        if isinstance(ocr_result, list):
            return [
                item
                for item in ocr_result
                if isinstance(item, dict)
            ]

        return []

    # ============================================================
    # BOUNDING BOX
    # ============================================================

    def _get_bbox(
        self,
        item: Dict[str, Any]
    ) -> Optional[List[float]]:

        bbox = item.get("bbox")

        if bbox is None:
            bbox = item.get("bounding_box")

        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
        ):
            try:
                return [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3])
                ]
            except Exception:
                return None

        return None

    # ============================================================
    # SPATIAL ENTITY EXTRACTION
    # ============================================================

    def _extract_entity_spatial(
        self,
        ocr_items: List[Dict[str, Any]],
        label_patterns: List[str]
    ) -> Optional[str]:

        labels = []

        # --------------------------------------------------------
        # Find Seller / Customer labels
        # --------------------------------------------------------

        for item in ocr_items:

            text = str(
                item.get("text", "")
            ).strip()

            if not text:
                continue

            lower_text = text.lower()

            for pattern in label_patterns:

                if re.search(
                    rf"\b{re.escape(pattern)}\b",
                    lower_text
                ):

                    bbox = self._get_bbox(item)

                    if not bbox:
                        continue

                    labels.append({
                        "item": item,
                        "bbox": bbox,
                        "pattern": pattern
                    })

                    inline = self._extract_inline_value(
                        text,
                        pattern
                    )

                    if inline:
                        return inline

                    break

        # --------------------------------------------------------
        # Find nearest text BELOW label
        # --------------------------------------------------------

        for label in labels:

            lx1, ly1, lx2, ly2 = label["bbox"]

            label_width = max(
                1.0,
                lx2 - lx1
            )

            best_candidate = None
            best_score = float("inf")

            for item in ocr_items:

                text = str(
                    item.get("text", "")
                ).strip()

                if not text:
                    continue

                bbox = self._get_bbox(item)

                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox

                if item is label["item"]:
                    continue

                # Candidate must be below label
                vertical_distance = y1 - ly2

                if vertical_distance < -5:
                    continue

                if vertical_distance > 100:
                    continue

                # Horizontal overlap
                horizontal_overlap = max(
                    0,
                    min(lx2, x2) -
                    max(lx1, x1)
                )

                candidate_width = max(
                    1.0,
                    x2 - x1
                )

                overlap_ratio = (
                    horizontal_overlap /
                    min(
                        label_width,
                        candidate_width
                    )
                )

                center_label = (
                    lx1 + lx2
                ) / 2

                center_candidate = (
                    x1 + x2
                ) / 2

                horizontal_distance = abs(
                    center_label -
                    center_candidate
                )

                # Candidate should belong to same column
                if (
                    overlap_ratio <= 0
                    and horizontal_distance > 150
                ):
                    continue

                if self._is_noise_entity_text(text):
                    continue

                score = (
                    vertical_distance
                    + horizontal_distance * 0.15
                    - overlap_ratio * 30
                )

                if score < best_score:

                    best_score = score

                    best_candidate = text

            if best_candidate:
                return best_candidate

        # --------------------------------------------------------
        # Line-level fallback
        # --------------------------------------------------------

        for item in ocr_items:

            text = str(
                item.get("text", "")
            ).strip()

            if not text:
                continue

            for pattern in label_patterns:

                match = re.search(
                    rf"{re.escape(pattern)}\s*[:\-]\s*(.+)",
                    text,
                    re.IGNORECASE
                )

                if not match:
                    continue

                value = match.group(1).strip(" :-")

                if value:
                    return value

        return None

    # ============================================================
    # INLINE VALUE
    # ============================================================

    def _extract_inline_value(
        self,
        text: str,
        pattern: str
    ) -> Optional[str]:

        match = re.search(
            rf"\b{re.escape(pattern)}\b\s*[:\-]?\s*(.*)",
            text,
            re.IGNORECASE
        )

        if not match:
            return None

        value = match.group(1).strip(" :-")

        if not value:
            return None

        return value

    # ============================================================
    # ENTITY NOISE
    # ============================================================

    def _is_noise_entity_text(
        self,
        text: str
    ) -> bool:

        lower = text.lower().strip()

        noise = {
            "seller",
            "seller:",
            "customer",
            "customer:",
            "vendor",
            "vendor:",
            "supplier",
            "supplier:",
            "buyer",
            "buyer:",
            "bill to",
            "ship to",
            "item",
            "quantity",
            "qty",
            "price",
            "unit price",
            "amount",
            "subtotal",
            "tax",
            "gst",
            "total",
            "total amount",
            "payment",
            "payment status",
            "paid",
            "pending",
            "unpaid"
        }

        return lower in noise

    # ============================================================
    # VENDOR
    # ============================================================

    def _extract_vendor(
        self,
        ocr_items: List[Dict[str, Any]]
    ) -> Optional[str]:

        return self._extract_entity_spatial(
            ocr_items,
            [
                "seller",
                "vendor",
                "supplier",
                "from"
            ]
        )

    # ============================================================
    # CUSTOMER
    # ============================================================

    def _extract_customer(
        self,
        ocr_items: List[Dict[str, Any]]
    ) -> Optional[str]:

        return self._extract_entity_spatial(
            ocr_items,
            [
                "customer",
                "buyer",
                "bill to",
                "billed to",
                "ship to"
            ]
        )

    # ============================================================
    # INVOICE NUMBER
    # ============================================================

    def _extract_invoice_number(
        self,
        text: str
    ) -> Optional[str]:

        patterns = [
            r"invoice\s*(?:number|no|#)?\s*[:\-]?\s*"
            r"(INV[-/]?\d+)",

            r"\b(INV[-/]?\d+)\b"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:
                return match.group(1)

        return None

    # ============================================================
    # DATE
    # ============================================================

    def _extract_invoice_date(
        self,
        text: str
    ) -> Optional[str]:

        patterns = [
            r"(?:invoice\s*)?date\s*[:\-]?\s*"
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",

            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:
                return match.group(1)

        return None

    # ============================================================
    # MONEY
    # ============================================================

    def _money_to_float(
        self,
        value: str
    ) -> Optional[float]:

        if value is None:
            return None

        value = str(value).strip()

        value = re.sub(
            r"[₹$€£]",
            "",
            value
        )

        value = re.sub(
            r"\b(?:rs\.?|inr)\b",
            "",
            value,
            flags=re.IGNORECASE
        )

        value = re.sub(
            r"[^0-9.,\-]",
            "",
            value
        )

        if not value:
            return None

        try:

            # Invoice numbers use Indian comma grouping:
            # 4,07,800
            if "," in value:

                parts = value.split(",")

                if (
                    len(parts[-1]) == 2
                    and all(
                        p.isdigit()
                        for p in parts
                    )
                ):
                    value = (
                        "".join(parts[:-1])
                        + "."
                        + parts[-1]
                    )

                else:
                    value = "".join(parts)

            return float(value)

        except Exception:
            return None

    # ============================================================
    # SUBTOTAL
    # ============================================================

    def _extract_subtotal(
        self,
        text: str
    ) -> Optional[float]:

        match = re.search(
            r"\bsubtotal\b\s*[:\-]?\s*"
            r"(?:rs\.?|inr|₹|\$)?\s*"
            r"([\d,]+(?:\.\d+)?)",
            text,
            re.IGNORECASE
        )

        if match:
            return self._money_to_float(
                match.group(1)
            )

        return None

    # ============================================================
    # TAX / GST
    # ============================================================

    def _extract_tax(
        self,
        text: str
    ) -> Optional[float]:

        patterns = [

            r"\bgst\b\s*"
            r"(?:\([^)]*\))?\s*"
            r"[:\-]\s*"
            r"(?:rs\.?|inr|₹|\$)?\s*"
            r"([\d,]+(?:\.\d+)?)",

            r"\btax\b\s*"
            r"(?:amount\s*)?"
            r"(?:\([^)]*\))?\s*"
            r"[:\-]\s*"
            r"(?:rs\.?|inr|₹|\$)?\s*"
            r"([\d,]+(?:\.\d+)?)"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                value = self._money_to_float(
                    match.group(1)
                )

                if value is not None:
                    return value

        return None

    # ============================================================
    # TOTAL
    # ============================================================

    def _extract_total(
        self,
        text: str,
        ocr_items: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[float]:

        # --------------------------------------------------------
        # 1. Normal same-line text patterns
        # --------------------------------------------------------

        patterns = [
            r"\btotal\s+amount\b\s*[:\-]?\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+(?:\.\d+)?)",

            r"\bgrand\s+total\b\s*[:\-]?\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+(?:\.\d+)?)",

            r"(?<!sub)\btotal\b\s*[:\-]?\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+(?:\.\d+)?)",

            r"\bamount\s+due\b\s*[:\-]?\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+(?:\.\d+)?)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                value = self._money_to_float(
                    match.group(1)
                )

                if value is not None:
                    return value

        # --------------------------------------------------------
        # 2. Spatial OCR fallback
        #
        # Handles cases where OCR order is:
        #
        # Rs. 4,81,204
        # Total Amount:
        #
        # instead of:
        #
        # Total Amount: Rs. 4,81,204
        # --------------------------------------------------------

        if ocr_items:

            total_labels = []

            for item in ocr_items:

                item_text = str(
                    item.get("text", "")
                ).strip()

                if not item_text:
                    continue

                lower = item_text.lower()

                is_total_label = (
                    "total amount" in lower
                    or "grand total" in lower
                    or "amount due" in lower
                    or re.fullmatch(
                        r"total\s*:?",
                        lower
                    ) is not None
                )

                if not is_total_label:
                    continue

                bbox = self._get_bbox(item)

                if bbox:
                    total_labels.append(
                        {
                            "text": item_text,
                            "bbox": bbox
                        }
                    )

            # ----------------------------------------------------
            # Search monetary OCR items near total label
            # ----------------------------------------------------

            best_value = None
            best_score = float("inf")

            for label in total_labels:

                lx1, ly1, lx2, ly2 = label["bbox"]

                label_center_x = (
                    lx1 + lx2
                ) / 2

                label_center_y = (
                    ly1 + ly2
                ) / 2

                for item in ocr_items:

                    candidate_text = str(
                        item.get("text", "")
                    ).strip()

                    if not candidate_text:
                        continue

                    # Never use percentages such as GST (18%)
                    if "%" in candidate_text:
                        continue

                    bbox = self._get_bbox(item)

                    if not bbox:
                        continue

                    x1, y1, x2, y2 = bbox

                    candidate_center_x = (
                        x1 + x2
                    ) / 2

                    candidate_center_y = (
                        y1 + y2
                    ) / 2

                    # ------------------------------------------------
                    # Candidate must contain a currency marker.
                    #
                    # This prevents values such as "18" from
                    # GST (18%) being selected.
                    # ------------------------------------------------

                    has_currency = bool(
                        re.search(
                            r"(?:₹|rs\.?|inr|\$|€|£)",
                            candidate_text,
                            re.IGNORECASE
                        )
                    )

                    if not has_currency:
                        continue

                    # Extract numeric money value
                    money_match = re.search(
                        r"[\d][\d,]*(?:\.\d+)?",
                        candidate_text
                    )

                    if not money_match:
                        continue

                    value = self._money_to_float(
                        money_match.group(0)
                    )

                    if value is None:
                        continue

                    # Ignore very small monetary values
                    # that are unlikely to be invoice totals.
                    if value < 1000:
                        continue

                    horizontal_distance = abs(
                        candidate_center_x -
                        label_center_x
                    )

                    vertical_distance = abs(
                        candidate_center_y -
                        label_center_y
                    )

                    # Candidate should be reasonably close
                    # to the total label.
                    if horizontal_distance > 600:
                        continue

                    if vertical_distance > 200:
                        continue

                    # Prefer candidates on the same horizontal
                    # line and close to the label.
                    score = (
                        horizontal_distance
                        + vertical_distance * 1.5
                    )

                    if score < best_score:

                        best_score = score
                        best_value = value

            if best_value is not None:
                return best_value

        # --------------------------------------------------------
        # 3. Token-based fallback
        # --------------------------------------------------------

        tokens = text.split()

        currency_tokens = {
            "rs",
            "rs.",
            "inr",
            "₹",
            "$"
        }

        for i, token in enumerate(tokens):

            token_clean = token.strip().lower()

            # Grand Total
            if (
                token_clean == "grand"
                and i + 1 < len(tokens)
            ):

                if tokens[i + 1].lower().startswith(
                    "total"
                ):

                    j = i + 2

                    while j < len(tokens):

                        candidate = tokens[j].strip(
                            ":,-"
                        )

                        if (
                            candidate.lower()
                            in currency_tokens
                        ):
                            j += 1
                            continue

                        value = self._money_to_float(
                            candidate
                        )

                        if value is not None:
                            return value

                        j += 1

            # Total
            if token_clean.startswith("total"):

                j = i + 1

                while j < len(tokens):

                    candidate = tokens[j].strip(
                        ":-,"
                    )

                    if (
                        candidate.lower()
                        in currency_tokens
                    ):
                        j += 1
                        continue

                    value = self._money_to_float(
                        candidate
                    )

                    if value is not None:
                        return value

                    j += 1

        return None

    # ============================================================
    # PAYMENT STATUS
    # ============================================================

    def _extract_payment_status(
        self,
        text: str
    ) -> Optional[str]:

        lower = text.lower()

        if re.search(
            r"\bunpaid\b",
            lower
        ):
            return "unpaid"

        if re.search(
            r"\bpending\b",
            lower
        ):
            return "pending"

        if re.search(
            r"\bpaid\b",
            lower
        ):
            return "paid"

        return None
