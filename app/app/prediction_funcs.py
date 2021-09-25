## Functions for creating the prediction in main.py

## Imports

import io

import numpy as np
import tensorflow as tf
from PIL import Image


## Functions

def load_classes(path_classes):
    """
    Load mushroom classes from file.

    :param path_classes: Path to csv file with classes
    :return: List with class names
    """

    # Read class names from csv file
    with open(path_classes, "r") as f:
        classes = [row.rstrip("\n") for row in f]

    # Drop header
    classes = classes[1:]

    return classes


def read_image(img_bytes, target_size=(224, 224)):
    """
    Read an image from bytedata and prepare it for the model.

    :param img_bytes: Image as byte data
    :param target_size: Target size for resizing image. Must match model input size.
    :return: Numpy array with resized and preprocessed image
    """

    # Open the image
    img = Image.open(io.BytesIO(img_bytes))
    # Convert to rgb
    img = img.convert("RGB")
    # Resize to match model input size
    img = img.resize(target_size, Image.NEAREST)

    # Preprocess image (other preprocessing steps are as a layer in the network)
    # To numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Add the "batch" dimension required by the model
    img_array = tf.expand_dims(img_array, 0)

    return img_array


def create_prediction(model, img_array, classes, top_k=3):
    """
    Predict image class with model, return predictions with highest confidence.

    :param model: Mushi-identifier model
    :param img_array: Numpy array with resized and preprocessed image
    :param classes: Mushroom classes
    :param top_k: Number of results to return, ranked by highest confidence score
    :return: Dictionary with class names corresponding to highest confidence scores
    """
    # Make prediction
    prediction = model.predict(img_array)
    # Apply softmax to turn raw prediction values into confidence scores
    prediction_softmax = tf.nn.softmax(prediction)

    # Create ordered prediction indices and confidences. Use tolist(), because
    # FastAPI does not support numpy floats.
    pred_indices = np.flip(np.argsort(prediction_softmax).flatten()).tolist()
    pred_confidences = prediction_softmax.numpy().flatten().tolist()

    # Save top k results to lists
    # TODO: Round confidence scores, if they are only displayed
    pred_classes_top_k = [classes[ix] for ix in pred_indices][:top_k]
    pred_confidences_top_k = [pred_confidences[ix] for ix in pred_indices][:top_k]

    # Create dictionary: map class names to confidence scores
    predictions_top_k = dict(zip(pred_classes_top_k, pred_confidences_top_k))

    return predictions_top_k
