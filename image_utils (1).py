import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224), normalize=True, to_rgb=True):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not read: {image_path}")
        return None

    # Resize
    img = cv2.resize(img, target_size)

    # Normalize to 0–1
    if normalize:
        img = img / 255.0

    # Convert to 3 channels (RGB format)
    if to_rgb:
        img = np.stack((img,) * 3, axis=-1)  # Shape: (224, 224, 3)

    return img
