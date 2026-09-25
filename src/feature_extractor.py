import numpy as np
class DocumentFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            "ocr_confidence",
            "text_length",
            "word_count",
            "page_count",
            "table_count",
            "image_count",
            "layout_complexity",
            "field_count"
        ]

    def extract(self, document_data):

        ocr_confidence = float(
            document_data.get("ocr_confidence", 0.0)
        )

        text = document_data.get("text", "")

        text_length = len(text)

        word_count = len(
            text.split()
        )

        page_count = int(
            document_data.get("page_count", 1)
        )

        table_count = int(
            document_data.get("table_count", 0)
        )

        image_count = int(
            document_data.get("image_count", 0)
        )

        layout_complexity = float(
            document_data.get("layout_complexity", 0.0)
        )

        fields = document_data.get(
            "fields",
            {}
        )

        if isinstance(fields, dict):
            field_count = len(fields)
        else:
            field_count = 0

        features = np.array(
            [
                ocr_confidence,
                text_length,
                word_count,
                page_count,
                table_count,
                image_count,
                layout_complexity,
                field_count
            ],
            dtype=float
        )

        return features

    def extract_batch(self, documents):
        """
        Extract features from multiple documents.

        Returns:
            2D numpy array
        """

        feature_vectors = []

        for document in documents:
            features = self.extract(document)
            feature_vectors.append(features)

        return np.asarray(
            feature_vectors,
            dtype=float
        )

    def get_feature_names(self):
        """
        Return feature names.
        """

        return self.feature_names