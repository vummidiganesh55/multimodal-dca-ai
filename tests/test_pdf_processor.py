from src.pdf_processor import PDFProcessor


PDF_PATH = "data/invoice.pdf"


processor = PDFProcessor(
    dpi=200
)


try:

    images = processor.convert_to_images(
        PDF_PATH
    )

    print(
        f"✓ PDF conversion successful"
    )

    print(
        f"Pages converted: {len(images)}"
    )

    for index, image in enumerate(images):

        print(
            f"Page {index + 1}: "
            f"{image.size}"
        )

except FileNotFoundError as error:

    print(error)

except Exception as error:

    print(
        f"PDF processing error: {error}"
    )