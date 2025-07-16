# model_utils.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Phase 1: Freeze base initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model

