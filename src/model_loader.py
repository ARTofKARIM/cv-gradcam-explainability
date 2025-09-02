"""Pre-trained model loading for Grad-CAM analysis."""
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2

class ModelLoader:
    MODELS = {"VGG16": (VGG16, tf.keras.applications.vgg16.preprocess_input, tf.keras.applications.vgg16.decode_predictions),
              "ResNet50": (ResNet50, tf.keras.applications.resnet50.preprocess_input, tf.keras.applications.resnet50.decode_predictions),
              "InceptionV3": (InceptionV3, tf.keras.applications.inception_v3.preprocess_input, tf.keras.applications.inception_v3.decode_predictions),
              "MobileNetV2": (MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input, tf.keras.applications.mobilenet_v2.decode_predictions)}

    def __init__(self, model_name="VGG16"):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = model_name
        model_class, self.preprocess_fn, self.decode_fn = self.MODELS[model_name]
        self.model = model_class(weights="imagenet")
        print(f"Loaded {model_name}: {self.model.count_params():,} params")

    def predict(self, image):
        preprocessed = self.preprocess_fn(image.copy())
        if preprocessed.ndim == 3:
            preprocessed = tf.expand_dims(preprocessed, 0)
        preds = self.model.predict(preprocessed, verbose=0)
        decoded = self.decode_fn(preds, top=5)
        return preds, decoded[0]

    def get_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None
