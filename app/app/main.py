## Imports
import pathlib

import tensorflow as tf
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException

from prediction_funcs import load_classes, read_image, create_prediction

## Define paths
path_model = pathlib.Path("../model/mushi_identifier_v1.keras")
path_classes = pathlib.Path("../model/classes_mushi_identifier_v1.csv")

## Import model and mushroom classes
model = tf.keras.models.load_model(path_model)
classes = load_classes(path_classes)

## Create API

# Initialize the API by creating a FastAPI instance
app = FastAPI()


# Create a simple index
@app.get("/")
async def index():
    return {"message": "This is the mushroom classification API!"}


# Predict class of user-submitted image
@app.post("/predict")
async def predict_image_class(img_file: UploadFile = File(...)):
    # Support jpeg and png images
    if img_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400,
                            detail="Invalid image type. Please submit a .jpeg or .png image.")

    # Read image to bytes
    img_bytes = await img_file.read()

    # Read the image to numpy array. Target size matches model input size.
    img_array = read_image(img_bytes, target_size=(224, 224))

    # Create a dictionary with top k predictions
    predictions_top_k = create_prediction(model, img_array, classes, top_k=5)

    return predictions_top_k


# Run API on (Docker) port 8000, when called with "python main.py"
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
