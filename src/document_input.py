from pathlib import Path


class DocumentInput:
    """
    Handles basic document input validation and metadata.

    Supported:
        PDF
        PNG
        JPG
        JPEG
        TIFF
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff"
    }

    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def validate(self):
        """
        Validate document path and extension.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {self.file_path}"
            )

        if not self.file_path.is_file():
            raise ValueError(
                f"Path is not a file: {self.file_path}"
            )

        extension = self.file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document format: {extension}"
            )

        return True

    def get_metadata(self):
        """
        Return basic document metadata.
        """

        self.validate()

        return {
            "file_name": self.file_path.name,
            "file_path": str(self.file_path),
            "extension": self.file_path.suffix.lower(),
            "file_size_bytes": self.file_path.stat().st_size
        }