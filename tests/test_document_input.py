from pathlib import Path

from src.document_input import DocumentInput


DATA_DIR = Path("data")


def find_document():

    supported_extensions = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff"
    }

    for file_path in DATA_DIR.iterdir():

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in supported_extensions
        ):
            return file_path

    return None


document_path = find_document()


if document_path is None:

    print(
        "\nNo document found in the data folder."
    )

    print(
        "Add a PDF or image to:"
    )

    print(
        "D:\\dlprojects\\multi_doc\\data\\"
    )

else:

    print(
        f"\nTesting document: {document_path}"
    )

    document = DocumentInput(
        document_path
    )

    try:

        document.validate()

        print(
            "✓ Document validation passed"
        )

        metadata = document.get_metadata()

        print("\nDocument Metadata:")

        for key, value in metadata.items():

            print(
                f"{key}: {value}"
            )

    except (
        FileNotFoundError,
        ValueError
    ) as error:

        print(
            f"\nValidation error: {error}"
        )