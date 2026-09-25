from src.document_loader import load_document


file_path = "data/samples/1.png"

images = load_document(file_path)

print("Number of pages/images:", len(images))
print("Image size:", images[0].size)
print("Image mode:", images[0].mode)