import json

import tensorflow as tf

with open("class_names.json", "r") as f:
    SKIN_CLASSES = json.load(f)


def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (300, 300))
    return tf.expand_dims(image, axis=0)


def predict(keras_model, image_bytes: bytes) -> tuple[str, float]:
    """Return (class_name, confidence)."""
    tensor = preprocess_image(image_bytes)
    predictions: list[float] = keras_model.predict(tensor)[0].tolist()
    max_idx = predictions.index(max(predictions))
    return SKIN_CLASSES[max_idx], round(predictions[max_idx], 4)
