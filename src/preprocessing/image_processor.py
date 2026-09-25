import cv2
import numpy as np
from PIL import Image
def preprocess_image(image: Image.Image) -> Image.Image:
    # PIL → OpenCV
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Resize
    height, width = img.shape[:2]

    target_width = 1600

    if width > target_width:
        scale = target_width / width
        img = cv2.resize(
            img,
            (target_width, int(height * scale)),
            interpolation=cv2.INTER_AREA
        )

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Denoising
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 4. Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    # 5. Adaptive threshold
    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # OpenCV → PIL
    processed_image = Image.fromarray(thresholded)

    return processed_image