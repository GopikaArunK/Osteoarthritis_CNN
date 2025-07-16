import os
from utils.image_utils import preprocess_image

def load_images_and_labels(base_dir):
    images = []
    labels = []

    for label_name in sorted(os.listdir(base_dir)):
        label_path = os.path.join(base_dir, label_name)

        # Extract the numeric label from folder name (e.g., Grade_2 → 2)
        try:
            label = int(label_name[-1])
        except:
            print(f"⚠️ Skipping folder: {label_name}")
            continue

        for file in os.listdir(label_path):
            image_path = os.path.join(label_path, file)
            img = preprocess_image(image_path)
            if img is not None:
                images.append(img)
                labels.append(label)

    return images, labels
