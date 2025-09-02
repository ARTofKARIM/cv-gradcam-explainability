"""Tests for Grad-CAM module."""
import unittest
import numpy as np
import tensorflow as tf
from src.gradcam import GradCAM

class TestGradCAM(unittest.TestCase):
    def test_heatmap_shape(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 3, activation="relu", input_shape=(32, 32, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        gc = GradCAM(model)
        dummy = np.random.randn(1, 32, 32, 3).astype(np.float32)
        heatmap, cls_idx = gc.compute(dummy, class_idx=0)
        self.assertEqual(heatmap.ndim, 2)
        self.assertGreaterEqual(heatmap.min(), 0)

    def test_heatmap_range(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, activation="relu", input_shape=(16, 16, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation="softmax"),
        ])
        gc = GradCAM(model)
        dummy = np.random.randn(1, 16, 16, 3).astype(np.float32)
        heatmap, _ = gc.compute(dummy)
        self.assertLessEqual(heatmap.max(), 1.0 + 1e-6)

if __name__ == "__main__":
    unittest.main()
