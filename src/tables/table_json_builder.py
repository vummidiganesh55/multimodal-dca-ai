
import re


class TableJSONBuilder:

    def __init__(self):
        pass

    # ==========================================================
    # NUMBER HELPERS
    # ==========================================================

    def _clean_number_text(self, text):

        if text is None:
            return ""

        text = str(text).strip()

        text = re.sub(
            r"^(?:Rs\.?|₹|INR|\$|€|£)\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        return text.replace(",", "").strip()

    def _is_number(self, text):

        cleaned = self._clean_number_text(text)

        return bool(
            re.fullmatch(
                r"\d+(?:\.\d+)?",
                cleaned
            )
        )

    def _to_number(self, text):

        cleaned = self._clean_number_text(text)

        try:
            return float(cleaned)
        except Exception:
            return None

    # ==========================================================
    # QUANTITY CORRECTION
    # ==========================================================

    def _correct_quantity(
        self,
        qty,
        unit_price,
        amount
    ):

        if (
            qty is None
            or unit_price is None
            or amount is None
        ):
            return qty

        try:
            qty = float(qty)
            unit_price = float(unit_price)
            amount = float(amount)
        except Exception:
            return qty

        # Original value already correct
        if abs(
            qty * unit_price - amount
        ) <= 0.01:
            return qty

        # Common OCR:
        # 51 -> 5
        # 11 -> 1
        # 21 -> 2

        if qty.is_integer():

            qty_text = str(int(qty))

            if len(qty_text) >= 2:

                candidate_text = qty_text[:-1]

                try:
                    candidate = float(
                        candidate_text
                    )
                except Exception:
                    candidate = None

                if (
                    candidate is not None
                    and candidate > 0
                    and abs(
                        candidate * unit_price
                        - amount
                    ) <= 0.01
                ):
                    return candidate

        return qty

    # ==========================================================
    # BOUNDING BOX
    # ==========================================================

    def _get_center_x(self, bbox):

        return (
            bbox[0] +
            bbox[2]
        ) / 2

    def _get_center_y(self, bbox):

        return (
            bbox[1] +
            bbox[3]
        ) / 2

    # ==========================================================
    # MAIN BUILDER
    # ==========================================================

    def build(
        self,
        ocr_results,
        structure_results=None,
        table_offset=(0, 0)
    ):

        rows = []

        words = []

        # ------------------------------------------------------
        # 1. Normalize OCR words
        # ------------------------------------------------------

        for item in ocr_results:

            text = str(
                item.get("text", "")
            ).strip()

            if not text:
                continue

            bbox = item.get("bbox")

            if not bbox or len(bbox) != 4:
                continue

            confidence = float(
                item.get(
                    "confidence",
                    0.0
                )
            )

            words.append({
                "text": text,
                "bbox": bbox,
                "cx": self._get_center_x(bbox),
                "cy": self._get_center_y(bbox),
                "confidence": confidence
            })

        # ------------------------------------------------------
        # 2. Find actual table header
        # ------------------------------------------------------

        header_y = None

        for word in words:

            lower = word["text"].lower().strip()

            if lower in {
                "item",
                "quantity",
                "qty",
                "unit price",
                "price",
                "amount"
            }:

                if header_y is None:
                    header_y = word["cy"]
                else:
                    header_y = min(
                        header_y,
                        word["cy"]
                    )

        # If a table header exists, only use text below it.
        # This prevents Seller/Customer/address from becoming
        # table rows.

        if header_y is not None:

            words = [
                word
                for word in words
                if word["cy"] > header_y + 10
            ]

        # ------------------------------------------------------
        # 3. Remove document-level labels
        # ------------------------------------------------------

        ignored_exact = {
            "item",
            "quantity",
            "qty",
            "price",
            "unit price",
            "amount",
            "invoice",
            "invoice no",
            "invoice number",
            "seller",
            "customer",
            "vendor",
            "supplier",
            "subtotal",
            "tax",
            "gst",
            "total",
            "total amount",
            "grand total",
            "payment",
            "payment status",
            "paid",
            "pending",
            "unpaid"
        }

        data_words = []

        for word in words:

            lower = (
                word["text"]
                .strip()
                .lower()
            )

            if lower in ignored_exact:
                continue

            data_words.append(word)

        # ------------------------------------------------------
        # 4. Group words into rows
        # ------------------------------------------------------

        data_words.sort(
            key=lambda x: x["cy"]
        )

        grouped_rows = []

        y_threshold = 12

        for word in data_words:

            placed = False

            for row in grouped_rows:

                avg_y = (
                    sum(
                        w["cy"]
                        for w in row
                    )
                    / len(row)
                )

                if abs(
                    word["cy"] - avg_y
                ) <= y_threshold:

                    row.append(word)
                    placed = True
                    break

            if not placed:
                grouped_rows.append(
                    [word]
                )

        # ------------------------------------------------------
        # 5. Sort each row left -> right
        # ------------------------------------------------------

        for row in grouped_rows:

            row.sort(
                key=lambda x: x["cx"]
            )

        # ------------------------------------------------------
        # 6. Convert rows to columns
        # ------------------------------------------------------

        for row in grouped_rows:

            row_text = " ".join(
                word["text"]
                for word in row
            ).lower()

            if any(
                label in row_text
                for label in [
                    "subtotal",
                    "gst",
                    "tax",
                    "total amount",
                    "grand total",
                    "payment status"
                ]
            ):
                continue

            item_text = []

            qty = None
            unit_price = None
            amount = None

            confidences = []

            for word in row:

                text = word["text"]
                cx = word["cx"]

                confidences.append(
                    word["confidence"]
                )

                # ITEM
                if cx < 500:

                    if not self._is_number(text):
                        item_text.append(text)

                # QTY
                elif 500 <= cx < 710:

                    if self._is_number(text):
                        qty = self._to_number(text)

                # UNIT PRICE
                elif 710 <= cx < 1000:

                    if self._is_number(text):
                        unit_price = self._to_number(text)

                # AMOUNT
                elif cx >= 1000:

                    if self._is_number(text):
                        amount = self._to_number(text)

            # --------------------------------------------------
            # Quantity correction
            # --------------------------------------------------

            qty = self._correct_quantity(
                qty,
                unit_price,
                amount
            )

            # --------------------------------------------------
            # Only retain actual table rows
            # --------------------------------------------------

            if (
                not item_text
                or qty is None
                or unit_price is None
                or amount is None
            ):
                continue

            confidence = (
                sum(confidences)
                / len(confidences)
                if confidences
                else 0.0
            )

            rows.append({

                "Item":
                    " ".join(
                        item_text
                    ).strip(),

                "Qty":
                    qty,

                "Unit Price":
                    unit_price,

                "Amount":
                    amount,

                "confidence":
                    confidence
            })

        # ------------------------------------------------------
        # 7. Return
        # ------------------------------------------------------

        return {

            "columns": [
                "Item",
                "Qty",
                "Unit Price",
                "Amount"
            ],

            "rows": rows,

            "row_count": len(rows)
        }
