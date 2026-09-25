import re


class DocumentValidator:

    def _to_number(self, value):
        if value is None:
            return None

        value = str(value).strip()

        value = value.replace(",", "")
        value = value.replace("$", "")
        value = value.replace("₹", "")
        value = value.replace("€", "")
        value = value.replace("£", "")
        value = value.replace("Rs.", "")
        value = value.replace("Rs", "")

        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # ---------------------------------------------------------
    # MAIN VALIDATION
    # ---------------------------------------------------------

    def validate(self, extraction, table):

        issues = []
        warnings = []

        extraction = extraction or {}
        table = table or {}

        # -----------------------------------------------------
        # REQUIRED EXTRACTION FIELDS
        # -----------------------------------------------------

        invoice_number = extraction.get("invoice_number")
        invoice_date = extraction.get("invoice_date")
        vendor_name = extraction.get("vendor_name")
        customer_name = extraction.get("customer_name")

        if not invoice_number:
            warnings.append("Invoice number is missing.")

        if not invoice_date:
            warnings.append("Invoice date is missing.")

        if not vendor_name:
            warnings.append("Vendor name is missing.")

        if not customer_name:
            warnings.append("Customer name is missing.")

        # -----------------------------------------------------
        # FINANCIAL VALUES
        # -----------------------------------------------------

        subtotal = self._to_number(
            extraction.get("subtotal")
        )

        tax = self._to_number(
            extraction.get("tax")
        )

        total = self._to_number(
            extraction.get("total")
        )

        # Missing financial values are WARNINGS,
        # not critical issues.
        if subtotal is None:
            warnings.append("Subtotal is missing or not numeric.")

        if tax is None:
            warnings.append("Tax is missing or not numeric.")

        if total is None:
            warnings.append("Total is missing or not numeric.")

        # -----------------------------------------------------
        # TOTAL CONSISTENCY
        # -----------------------------------------------------

        if (
            subtotal is not None
            and tax is not None
            and total is not None
        ):

            expected_total = subtotal + tax

            if abs(expected_total - total) > 0.01:

                issues.append(
                    "Total mismatch: "
                    f"expected {expected_total:.2f}, "
                    f"found {total:.2f}"
                )

        # -----------------------------------------------------
        # TABLE VALIDATION
        # -----------------------------------------------------

        table_validation = self.validate_table(table)

        for row in table_validation:

            if row["issues"]:
                issues.extend(
                    [
                        f"Row {row['row_number']}: {issue}"
                        for issue in row["issues"]
                    ]
                )

            if row["warnings"]:
                warnings.extend(
                    [
                        f"Row {row['row_number']}: {warning}"
                        for warning in row["warnings"]
                    ]
                )

        # -----------------------------------------------------
        # CONFIDENCE
        # -----------------------------------------------------

        confidences = []

        for row in table_validation:

            confidence = row.get(
                "confidence"
            )

            if confidence is not None:
                confidences.append(
                    float(confidence)
                )

        if confidences:

            confidence = (
                sum(confidences)
                / len(confidences)
            )

        else:

            confidence = 0.80

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "confidence": round(
                confidence,
                4
            ),
            "table_validation": table_validation
        }

    # ---------------------------------------------------------
    # TABLE VALIDATION
    # ---------------------------------------------------------

    def validate_table(self, table):

        validated_rows = []

        rows = table.get(
            "rows",
            []
        )

        for index, row in enumerate(
            rows,
            start=1
        ):

            issues = []
            warnings = []

            qty = self._to_number(
                row.get("Qty")
            )

            unit_price = self._to_number(
                row.get("Unit Price")
            )

            amount = self._to_number(
                row.get("Amount")
            )

            # -------------------------------------------------
            # QUANTITY
            # -------------------------------------------------

            if qty is None:

                warnings.append(
                    "Quantity is missing or not numeric"
                )

            elif qty <= 0:

                issues.append(
                    "Quantity must be greater than zero"
                )

            # -------------------------------------------------
            # UNIT PRICE
            # -------------------------------------------------

            if unit_price is None:

                warnings.append(
                    "Unit Price is missing or not numeric"
                )

            elif unit_price < 0:

                issues.append(
                    "Unit Price cannot be negative"
                )

            # -------------------------------------------------
            # AMOUNT
            # -------------------------------------------------

            if amount is None:

                warnings.append(
                    "Amount is missing or not numeric"
                )

            elif amount < 0:

                issues.append(
                    "Amount cannot be negative"
                )

            # -------------------------------------------------
            # QTY × UNIT PRICE = AMOUNT
            # -------------------------------------------------

            if (
                qty is not None
                and unit_price is not None
                and amount is not None
            ):

                expected_amount = (
                    qty * unit_price
                )

                if abs(
                    expected_amount - amount
                ) > 0.01:

                    issues.append(
                        "Amount mismatch: "
                        f"expected "
                        f"{expected_amount:.2f}, "
                        f"found "
                        f"{amount:.2f}"
                    )

            # -------------------------------------------------
            # OCR CONFIDENCE
            # -------------------------------------------------

            confidence = float(
                row.get(
                    "confidence",
                    0.0
                )
            )

            if confidence < 0.80:

                warnings.append(
                    f"Low OCR confidence: "
                    f"{confidence:.2f}"
                )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            validated_rows.append(
                {
                    "row_number": index,
                    "data": row,
                    "valid": len(issues) == 0,
                    "issues": issues,
                    "warnings": warnings,
                    "confidence": confidence
                }
            )

        return validated_rows