
from src.multilingual import LanguageDetector


def main():
    detector = LanguageDetector()

    samples = {
        "English": "Invoice Number Total Amount Payment",
        "Telugu": "ఇది ఒక పరీక్ష పత్రం",
        "Hindi": "यह एक परीक्षण दस्तावेज़ है",
        "Tamil": "இது ஒரு சோதனை ஆவணம்",
    }

    print("=" * 50)
    print("MULTILINGUAL LANGUAGE DETECTION TEST")
    print("=" * 50)

    for name, text in samples.items():
        result = detector.detect(text)

        print(f"\n{name}")
        print(f"Text       : {text}")
        print(f"Language   : {result['language']}")
        print(f"Confidence : {result['confidence']}")


if __name__ == "__main__":
    main()
