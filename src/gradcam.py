"""Grad-CAM and Grad-CAM++ implementations."""
import numpy as np
import tensorflow as tf

class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv()

    def _find_last_conv(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found")

    def compute(self, image, class_idx=None):
        grad_model = tf.keras.Model(self.model.inputs, [self.model.get_layer(self.layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_output)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(conv_output * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        return cam[0].numpy(), int(class_idx)

class GradCAMPlusPlus:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv()

    def _find_last_conv(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found")

    def compute(self, image, class_idx=None):
        grad_model = tf.keras.Model(self.model.inputs, [self.model.get_layer(self.layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            tape.watch(image)
            conv_output, predictions = grad_model(image)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_output)
        grads_2 = grads ** 2
        grads_3 = grads ** 3
        sum_activations = tf.reduce_sum(conv_output, axis=(1, 2), keepdims=True)
        alpha = grads_2 / (2 * grads_2 + sum_activations * grads_3 + 1e-10)
        weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))
        cam = tf.reduce_sum(conv_output * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        return cam[0].numpy(), int(class_idx)
