import re


class DocumentValidator:

    def _to_number(self, value):

        if value is None:
            return None

        value = str(value).replace(",", "")
        value = value.replace("$", "")
        value = value.replace("₹", "")
        value = value.replace("€", "")
        value = value.replace("£", "")

        try:
            return float(value)
        except ValueError:
            return None

    def validate_table(self, table):

        validated_rows = []

        for index, row in enumerate(table["rows"], start=1):

            issues = []
            warnings = []

            qty = self._to_number(row["Qty"])
            unit_price = self._to_number(row["Unit Price"])
            amount = self._to_number(row["Amount"])

            # -------------------------
            # Quantity validation
            # -------------------------

            if qty is None:

                issues.append(
                    "Quantity is not numeric"
                )

            elif qty <= 0:

                issues.append(
                    "Quantity must be greater than zero"
                )

            # -------------------------
            # Unit price validation
            # -------------------------

            if unit_price is None:

                warnings.append(
                    "Unit Price is missing or not numeric"
                )

            elif unit_price < 0:

                issues.append(
                    "Unit Price cannot be negative"
                )

            # -------------------------
            # Amount validation
            # -------------------------

            if amount is None:

                issues.append(
                    "Amount is not numeric"
                )

            elif amount < 0:

                issues.append(
                    "Amount cannot be negative"
                )

            # -------------------------
            # Arithmetic validation
            # -------------------------

            if (
                qty is not None
                and unit_price is not None
                and amount is not None
            ):

                expected_amount = qty * unit_price

                if abs(expected_amount - amount) > 0.01:

                    issues.append(
                        f"Amount mismatch: "
                        f"expected {expected_amount:.2f}, "
                        f"found {amount:.2f}"
                    )

            # -------------------------
            # Confidence validation
            # -------------------------

            confidence = float(
                row.get("confidence", 0.0)
            )

            if confidence < 0.80:

                warnings.append(
                    f"Low OCR confidence: {confidence:.2f}"
                )

            # -------------------------
            # Final row status
            # -------------------------

            validated_rows.append({

                "row_number": index,

                "data": row,

                "valid": len(issues) == 0,

                "issues": issues,

                "warnings": warnings,

                "confidence": confidence
            })

        return validated_rows