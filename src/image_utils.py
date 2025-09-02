"""Image loading and preprocessing utilities."""
import numpy as np
import cv2
import tensorflow as tf

class ImageUtils:
    @staticmethod
    def load_image(path, target_size=(224, 224)):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size)
        return img, img_resized

    @staticmethod
    def prepare_input(image, preprocess_fn=None):
        img = image.astype(np.float32)
        if preprocess_fn:
            img = preprocess_fn(img.copy())
        return np.expand_dims(img, axis=0)

    @staticmethod
    def batch_load(paths, target_size=(224, 224)):
        images = []
        for p in paths:
            _, resized = ImageUtils.load_image(p, target_size)
            images.append(resized)
        return np.array(images)
