# Here we build and train the model. There's also a notebook with the same code.

## Colab initialization
# from google.colab import drive
# drive.mount('/content/drive')

## Libraries
import pathlib
import csv

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

## Paths

# Processed data directory
path_processed_dir = pathlib.Path(
    # "drive/MyDrive/colab_data/02_processed/"
    "/home/jpe/Documents/python_projects/mushi-identifier/data/02_processed/"
)

# Model directory
path_model_dir = pathlib.Path(
    # "drive/MyDrive/colab_models/"
    "/home/jpe/Documents/python_projects/mushi-identifier/models/"
)

# Training and validation data
path_train_val_dir = path_processed_dir / "train_and_validation"

# Test data
path_test_dir = path_processed_dir / "test"

## Import data
# TODO: To function
# TODO: Implement k-fold cross validation

# MobileNetV2 max is 224 in Keras Danish Fungi authors used (299, 299) - possible
# performance hit. Also makes the file size of our processed set feel unnecessarily big.
image_size = (224, 224)

# Batch size (TODO: test 64)
batch_size = 32

# This is roughly a 80-10-10 split for the data

train_dataset = image_dataset_from_directory(
    path_train_val_dir,
    validation_split=0.11,
    subset="training",
    seed=666,
    image_size=image_size,
    batch_size=batch_size
)

validation_dataset = image_dataset_from_directory(
    path_train_val_dir,
    validation_split=0.11,
    subset="validation",
    seed=666,
    image_size=image_size,
    batch_size=batch_size
)

test_dataset = image_dataset_from_directory(
    path_test_dir,
    image_size=image_size,
    batch_size=batch_size
)

## Classes

# Print class names
classes = train_dataset.class_names
print(classes)
print(len(classes))

# Set number of classes
num_classes = len(train_dataset.class_names)


# Save classes to file for use in predictions
with open(path_model_dir / "classes_mushi_model_1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["species"])
    writer.writerows(zip(classes))

## View some images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for j in range(4):
        ax = plt.subplot(2, 2, j + 1)
        plt.imshow(images[j].numpy().astype("uint8"))
        plt.title(train_dataset.class_names[labels[j]])
        plt.axis("off")


## Configure for performance (https://www.tensorflow.org/tutorials/load_data/images)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

## Augment images

data_augmentation = tf.keras.Sequential([
    # No vertical flip, mushroom pictures are usually not upside down
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
])

## Look at augmented images

plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
    for j in range(4):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(2, 2, j + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

## Load base model

image_shape = image_size + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                               include_top=False,
                                               weights="imagenet")

## Example batch output

# image_batch, label_batch = next(iter(train_dataset))
# feature_batch = base_model(image_batch)
# print(feature_batch.shape)

## Freeze conv base

base_model.trainable = False

## Build model

inputs = tf.keras.Input(shape=image_shape)
x = data_augmentation(inputs)
# Rescale images for MobileNetV2, which expects image scale [-1, 1]
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)  # False due to the batch norm layers
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# https://github.com/keras-team/keras/issues/8470 - global better for small data!
x = tf.keras.layers.Dropout(0.2)(x)  # TODO: tune
outputs = tf.keras.layers.Dense(num_classes)(x)

model = tf.keras.Model(inputs, outputs)

# use activation "softmax" with from_logits=False
# or no activation and from_logits=True (latter might have more numerical stability:
# https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do
# -in-sparsecategoricalcrossentropy-loss-function)

## Define metrics
# TODO: Implement F1 macro? (not in keras metrics)
# consider: precision, recall, roc curves, top-k accuracy (1, 3)

metrics = [
    keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top_1_acc"),
    keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_acc")
]

## Define callbacks
# TODO: Check keras.callbacks.LearningRateScheduler for dynamically reducing l_rate

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        verbose=1,
        patience=10
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=path_model_dir / "mushi_model_1.keras",
        monitor="val_loss",
        verbose=1,
        save_best_only=True
    )
]

## Compile model

# TODO: tune
learning_rate = 0.0001

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=metrics)


## Colab tensorboard

# import datetime
#
# # Define log dir path (create a timestamped subdirectory)
# log_dir = "drive/MyDrive/colab_models/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#
# # Add callback
# callbacks += [keras.callbacks.TensorBoard(log_dir=log_dir)]
#
# # Launch tensorboard
# %load_ext tensorboard
# %tensorboard --logdir drive/MyDrive/colab_models/logs/

## Train model

model_history = model.fit(train_dataset,
                          epochs=100,
                          callbacks=callbacks,
                          validation_data=validation_dataset)

# TODO: save history


## Plot results

# check the tensorflow imbalanced tutorial for plotting functions
# confusion matrix

# lower accuracy can be random since we don't do k-fold cross valid
# lower accuracy can be bc we have less classes, we use a subset