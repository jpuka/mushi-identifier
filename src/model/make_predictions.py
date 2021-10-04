# Create predictions with the trained model

import pathlib

## Imports
import numpy as np
import tensorflow as tf

## Paths
# Model directory
path_model_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/models/"
)

## Load model
model = tf.keras.models.load_model(path_model_dir / "mushi_identifier_v1.keras")

## Load classes

# Could easily do this with pandas, but don't need to import anything this way
path_classes = path_model_dir / "classes_mushi_identifier_v1.csv"
with open(path_classes, "r") as f:
    classes = [row.rstrip("\n") for row in f]

# Drop header
classes = classes[1:]

## Load image
# TODO: make this cleaner, now just for testing and show

# Path to image
img_path = "/home/jpe/Documents/python_projects/mushi-identifier/data/00_external/2018_FGVCx_Fungi/interim_done/cantharellus_cibarius/APE2017-9196936_BkzOiYFVZ.JPG"

# Load, resize image
img = tf.keras.preprocessing.image.load_img(
    img_path, target_size=(224, 224, 3)
)

## Preprocess image (mobilenet preprocessing attached to model though,
# so just PIL -> array -> batches here)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

## Create prediction
prediction = model.predict(img_array)

## Apply softmax to calculate confidences, since not done in model output layer
prediction_softmax = tf.nn.softmax(prediction)

# Create ordered prediction indices and confidences
pred_indices = np.flip(np.argsort(prediction_softmax).flatten()).tolist()
pred_confidences = prediction_softmax.numpy().flatten().tolist()

## Save top 3 results to dictionary
pred_classes_top3 = [classes[ix] for ix in pred_indices][:3]
pred_confidences_top3 = [pred_confidences[ix] for ix in pred_indices][:3]

predictions_top3 = dict(zip(pred_classes_top3, pred_confidences_top3))

## Print top 3 results
print("-Top 3 predictions-", end="\n\n")
# Dictionaries are ordered in python 3.7+, so this is safe
for cls, conf in predictions_top3.items():
    print(f"Prediction: {cls} \nConfidence: {conf:.3f}", end="\n\n")
