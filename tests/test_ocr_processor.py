from src.pdf_processor import PDFProcessor
from src.ocr_processor import OCRProcessor


PDF_PATH = "data/invoice.pdf"


# --------------------------------------------------
# 1. Convert PDF to images
# --------------------------------------------------

pdf_processor = PDFProcessor(dpi=200)

try:
    pages = pdf_processor.convert_to_images(PDF_PATH)

    print("✓ PDF conversion successful")
    print(f"Pages: {len(pages)}")

except Exception as error:
    print(f"PDF processing error: {error}")
    raise


# --------------------------------------------------
# 2. OCR processing
# --------------------------------------------------

ocr_processor = OCRProcessor()

print("\nStarting OCR...")

all_text = []
all_confidences = []

for index, page in enumerate(pages):

    try:
        result = ocr_processor.process_image(page)

        all_text.append(result["text"])
        all_confidences.append(
            result["ocr_confidence"]
        )

        print(
            f"Page {index + 1}: "
            f"words={result['word_count']}, "
            f"confidence={result['ocr_confidence']:.3f}"
        )

    except Exception as error:
        print(
            f"Page {index + 1} OCR error: {error}"
        )


# --------------------------------------------------
# 3. Combine results
# --------------------------------------------------

combined_text = " ".join(all_text)

average_confidence = (
    sum(all_confidences) / len(all_confidences)
    if all_confidences
    else 0.0
)


# --------------------------------------------------
# 4. Final OCR result
# --------------------------------------------------

print("\n" + "=" * 60)
print("OCR SUMMARY")
print("=" * 60)

print(f"Pages processed: {len(pages)}")
print(f"Total text characters: {len(combined_text)}")
print(f"Average OCR confidence: {average_confidence:.3f}")

print("\nExtracted Text Preview:")
print("-" * 60)
print(combined_text[:2000])
print("-" * 60)