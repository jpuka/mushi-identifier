# Build and train the baseline model, visualize the results. There's also a notebook
# with the same code.

## Colab initialization
# from google.colab import drive
# drive.mount('/content/drive')

## Libraries

import csv
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.model.data_funcs import load_dataset, create_confusion_matrix, \
    plot_confusion_matrix, plot_loss_accuracy

## Paths (relative to project root)

# Model name
model_name = "mushi_identifier_v11"

# Processed data directory
path_processed_dir = pathlib.Path(
    # "drive/MyDrive/colab_data/02_processed/"
    "data/02_processed/"
)

# Model directory
path_model_dir = pathlib.Path(
    # "drive/MyDrive/colab_models/"
    "models/"
)

# Training and validation data
path_train_val_dir = path_processed_dir / "train_and_validation"

# Test data
path_test_dir = path_processed_dir / "test"

# Training logs directory
path_training_logs = path_model_dir / "training_logs" / (
        model_name + "_training_logs.csv")

# Names of the classes that the model is trained on
path_model_class_names = path_model_dir / (model_name + "_classes.csv")

# Saved model
path_saved_model = path_model_dir / (model_name + ".keras")

## Import data
# TODO: Implement k-fold cross validation

# MobileNetV2 max is 224 in Keras Danish Fungi authors used (299, 299) - possible
# performance hit. Also makes the file size of our processed set feel unnecessarily big.
image_size = (224, 224)

# Batch size
# TODO: test 64
batch_size = 32

# This is roughly a 80-10-10 split for the data. The Danish Fungi article used 10 %
# test data so we are matching validation dataset to that. But this is obviously too
# little, and we should implement k-fold cross-validation to get reliable validation
# scores.
train_ds, validation_ds, test_ds = load_dataset(path_train_val_dir, path_test_dir,
                                                train_val_split=0.11)

## Classes

# Print class names
class_names = train_ds.class_names
print(class_names)
print(len(class_names))

# Save classes to file for use in predictions
# TODO: Save with Finnish name, so they can be shown in predictions - use pandas to
#  merge with class file
with open(path_model_class_names, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["species"])
    writer.writerows(zip(class_names))

## View some images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for j in range(4):
        ax = plt.subplot(2, 2, j + 1)
        plt.imshow(images[j].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[j]])
        plt.axis("off")

## Configure datasets for performance

# .cache() keeps the images in memory after they are loaded off disk during the first epoch.
# .prefetch() overlaps data preprocessing and model execution while training.

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

## Augment images
# TODO: add more augmentation (?)

data_augmentation = keras.Sequential([
    # No vertical flip, mushroom pictures are usually not upside down
    keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.2)
])

## View some augmented images

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for j in range(4):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(2, 2, j + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

## Load the base model used for feature extraction

image_shape = image_size + (3,)
base_model = keras.applications.MobileNetV2(input_shape=image_shape,
                                            include_top=False,
                                            weights="imagenet")

## Freeze the base model so its weights won't be updated during training

base_model.trainable = False

## Build model

# Define number of classes = number of output neurons
num_classes = len(class_names)

inputs = keras.Input(shape=image_shape)
x = data_augmentation(inputs)
# Rescale images for MobileNetV2, which expects image scale [-1, 1]
x = keras.applications.mobilenet_v2.preprocess_input(x)
# Set training to False due to the batch normalization layers
x = base_model(x, training=False)
# Global pooling is better than flatten for small datasets. See:
# https://github.com/keras-team/keras/issues/8470
x = keras.layers.GlobalAveragePooling2D()(x)
# TODO: Tune the dropout
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes)(x)

model = keras.Model(inputs, outputs)

# Use activation "softmax" with from_logits=False
# or no activation and from_logits=True (latter might have more numerical stability:
# https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do
# -in-sparsecategoricalcrossentropy-loss-function)

## Define metrics

# TODO: Implement macro F1 score (?), since it is not in keras metrics
# TODO: Implement precision (and maybe recall), since they are only supported
#  for binary classification https://github.com/tensorflow/addons/issues/1753

metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),  # "top 1 accuracy"
    keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_acc")
    # keras.metrics.Precision(name="prec")
    # keras.metrics.Recall(name="rec")
]

## Define callbacks

# TODO: Use keras.callbacks.LearningRateScheduler for dynamically reducing learning
#  rate

callbacks = [
    # Stop training once validation loss ceases to improve
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        verbose=1,
        patience=10
    ),
    # Save the best model during training
    keras.callbacks.ModelCheckpoint(
        filepath=path_saved_model,
        monitor="val_loss",
        verbose=1,
        save_best_only=True
    ),
    # Log training loss and metrics to a csv file
    keras.callbacks.CSVLogger(
        filename=path_training_logs,
        append=True
    )
]

## Compile model

# TODO: Tune learning rate
learning_rate = 0.001

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
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

model_history = model.fit(train_ds,
                          epochs=100,
                          callbacks=callbacks,
                          validation_data=validation_ds)

# lower accuracy can be random since we don't do k-fold cross valid
# lower accuracy can be bc we have less classes, we use a subset


## Load model
# model = keras.models.load_model(filepath=path_saved_model)

## Plot accuracy and loss

# This looks insane, but it is related to the batch norm and dropout layers.
plot_loss_accuracy(model_history)

## Plot confusion matrix

cm = create_confusion_matrix(model, validation_ds)
plot_confusion_matrix(cm, class_names, normalize=False)
