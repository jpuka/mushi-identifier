## imports
import io
import pathlib

import numpy as np
import tensorflow as tf
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

## create app
app = FastAPI()

## load model
path_model = pathlib.Path("../model/mushi_identifier_v1.keras")
model = tf.keras.models.load_model(path_model)

## read classes
path_classes = pathlib.Path("../model/classes_mushi_identifier_v1.csv")
with open(path_classes, "r") as f:
    classes = [row.rstrip("\n") for row in f]

# drop header
classes = classes[1:]


## define index
@app.get("/")
async def index():
    return {"message": "This is the mushroom classification API!"}


## get image

@app.post("/predict")
async def create_upload_file(image_file: UploadFile = File(...)):
    # support jpeg and png extensions
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400,
                            detail="Invalid image type. Please submit a .jpeg or .png image.")
    img_bytes = await image_file.read()

    # Read image to array ready for model
    target_size = (224, 224)  # hard-coded for now, since model is made for this
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make prediction
    prediction = model.predict(img_array)

    # Apply softmax to turn raw predictions into confidence scores
    prediction_softmax = tf.nn.softmax(prediction)

    # Create ordered prediction indices and confidences. Use tolist(), because
    # FastAPI does not support numpy floats.
    pred_indices = np.flip(np.argsort(prediction_softmax).flatten()).tolist()
    pred_confidences = prediction_softmax.numpy().flatten().tolist()

    # Save top 3 results to dictionary
    pred_classes_top3 = [classes[ix] for ix in pred_indices][:3]
    pred_confidences_top3 = [pred_confidences[ix] for ix in pred_indices][:3]
    predictions_top3 = dict(zip(pred_classes_top3, pred_confidences_top3))

    return predictions_top3


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
