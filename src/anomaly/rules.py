class RuleBasedRiskDetector:

    def analyze(self, validation_results):

        risk_score = 0
        risk_level = "LOW"
        issues = []

        # --------------------------------------------------
        # Normalize input
        # --------------------------------------------------

        if validation_results is None:
            validation_results = []

        # If a single validation dictionary is passed,
        # convert it into a list.
        if isinstance(validation_results, dict):

            # Document-level validation result
            if "row_number" not in validation_results:

                document_issues = validation_results.get(
                    "issues",
                    []
                )

                validation_results = [
                    {
                        "row_number": 0,
                        "issues": document_issues
                    }
                ]

            else:
                validation_results = [
                    validation_results
                ]

        # --------------------------------------------------
        # Analyze validation results
        # --------------------------------------------------

        for result in validation_results:

            # Safety check
            if not isinstance(result, dict):
                continue

            row_number = result.get(
                "row_number",
                0
            )

            row_issues = result.get(
                "issues",
                []
            )

            # Ensure issues is a list
            if isinstance(row_issues, str):
                row_issues = [row_issues]

            for issue in row_issues:

                issue_text = str(issue)

                # ------------------------------------------
                # Numeric field problem
                # ------------------------------------------

                if "not numeric" in issue_text.lower():

                    risk_score += 15

                # ------------------------------------------
                # Arithmetic mismatch
                # ------------------------------------------

                elif "amount mismatch" in issue_text.lower():

                    risk_score += 10

                # ------------------------------------------
                # Negative value
                # ------------------------------------------

                elif "negative" in issue_text.lower():

                    risk_score += 10

                # ------------------------------------------
                # Other validation issue
                # ------------------------------------------

                else:

                    risk_score += 5

                issues.append({
                    "row": row_number,
                    "issue": issue_text
                })

        # --------------------------------------------------
        # Cap risk score
        # --------------------------------------------------

        risk_score = min(
            risk_score,
            100
        )

        # --------------------------------------------------
        # Determine risk level
        # --------------------------------------------------

        if risk_score >= 70:

            risk_level = "HIGH"

        elif risk_score >= 30:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        # --------------------------------------------------
        # Final result
        # --------------------------------------------------

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "issues": issues
        }