
import re


class LanguageDetector:
    """
    Lightweight multilingual language detector.

    Detects common document languages using Unicode ranges.
    This is intended as a fast MVP feature, not a full NLP
    language-identification model.
    """

    LANGUAGE_RANGES = {
        "telugu": r"[\u0C00-\u0C7F]",
        "hindi": r"[\u0900-\u097F]",
        "tamil": r"[\u0B80-\u0BFF]",
        "kannada": r"[\u0C80-\u0CFF]",
        "malayalam": r"[\u0D00-\u0D7F]",
        "bengali": r"[\u0980-\u09FF]",
        "gujarati": r"[\u0A80-\u0AFF]",
        "punjabi": r"[\u0A00-\u0A7F]",
    }

    def detect(self, text):
        if not text or not isinstance(text, str):
            return {
                "language": "unknown",
                "confidence": 0.0
            }

        total_letters = 0
        language_counts = {}

        for language, pattern in self.LANGUAGE_RANGES.items():
            count = len(re.findall(pattern, text))

            if count > 0:
                language_counts[language] = count
                total_letters += count

        english_count = len(re.findall(r"[A-Za-z]", text))

        if english_count > 0:
            language_counts["english"] = english_count
            total_letters += english_count

        if not language_counts or total_letters == 0:
            return {
                "language": "unknown",
                "confidence": 0.0
            }

        detected_language = max(
            language_counts,
            key=language_counts.get
        )

        confidence = (
            language_counts[detected_language] / total_letters
        )

        return {
            "language": detected_language,
            "confidence": round(float(confidence), 4),
            "language_counts": language_counts
        }

    def detect_batch(self, texts):
        return [
            self.detect(text)
            for text in texts
        ]
