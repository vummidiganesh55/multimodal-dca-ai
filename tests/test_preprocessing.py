from src.document_loader import load_document
from src.preprocessing.image_processor import preprocess_image


input_path = "data/samples/1.png"

images = load_document(input_path)

processed = preprocess_image(images[0])

processed.save("data/processed/preprocessed.png")

print("Preprocessing successful!")
print("Original size:", images[0].size)
print("Processed size:", processed.size)