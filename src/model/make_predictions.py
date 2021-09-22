# create predictions with the trained model

## imports
import numpy as np
import tensorflow as tf

## load model
model = tf.keras.models.load_model("models/mushi_model_1.keras")

## load classes

# could easily do this with pandas, but don't need to import anything this way
path_classes = "/home/jpe/Documents/python_projects/mushi-identifier/models/classes_mushi_model_1.csv"
with open(path_classes, "r") as f:
    classes = [row.rstrip("\n") for row in f]

# drop header
classes = classes[1:]

## get image
img_path = "/home/jpe/Documents/python_projects/mushi-identifier/data/00_external/2018_FGVCx_Fungi/interim_done/cantharellus_cibarius/APE2017-9196936_BkzOiYFVZ.JPG"

img = tf.keras.preprocessing.image.load_img(
    img_path, target_size=(224, 224, 3)
)

img.show()

## preprocess image (mobilenet preprocessing attached to model though,
# so just PIL -> array -> batches here)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

## make prediction
prediction = model.predict(img_array)

## apply softmax, since not used in model
prediction_softmax = tf.nn.softmax(prediction)

## Calculate result

prediction_args_sorted = np.flip(np.argsort(prediction_softmax).flatten())
prediction_softmax_flat = prediction_softmax.numpy().flatten()

prediction_classes_top_k = [None] * 3
prediction_confidences_top_k = np.empty(3)

## Report result
print("Top 3 predictions:")
for k in range(3):
    prediction_classes_top_k[k] = classes[prediction_args_sorted[k]]
    prediction_confidences_top_k[k] = prediction_softmax_flat[prediction_args_sorted[k]]
    print(
        f"Prediction {k + 1}: {prediction_classes_top_k[k]} (Finnish name) "
        f"Confidence: {prediction_confidences_top_k[k]:.3f}"
    )