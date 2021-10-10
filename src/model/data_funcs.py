# Functions for loading and plotting modeling data.

## Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from seaborn import heatmap
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image_dataset_from_directory


## Functions

def load_dataset(path_train_val_dir, path_test_dir, train_val_split, seed=666,
                 image_size=(224, 224), batch_size=32):
    """
    Load train, validation and test datasets from processed directory.

    :param path_train_val_dir: Path to train/validation directory.
    :param path_test_dir: Path to test directory.
    :param train_val_split: Fraction of training data to reserve for validation.
    :param seed: Random seed for train/validation split.
    :param image_size: Size to resize images to after they are read from disk.
    :param batch_size: Size of the batches of data.
    :return: Train, validation and test datasets.
    """
    # TODO: Currently assumes that train and validation data are in the same directory
    #  and split with this function. Once they have been pre-split to processed,
    #  change this functionality to load them from separate directories.

    # Load train dataset
    train_dataset = image_dataset_from_directory(
        path_train_val_dir,
        validation_split=train_val_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    # Load validation dataset
    validation_dataset = image_dataset_from_directory(
        path_train_val_dir,
        validation_split=train_val_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    # Load test dataset
    test_dataset = image_dataset_from_directory(
        path_test_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    return train_dataset, validation_dataset, test_dataset


def plot_loss_accuracy(model_history=None, from_logs=False, path_training_logs=None):
    """
    Plot model loss and accuracy metrics from history object or from training logs.

    :param model_history: Model history object. Required, if from_logs = False.
    :param from_logs: Should the plots be drawn from history or training logs.
    :param path_training_logs: Path to training logs. Required, if from_logs = True.
    """

    # If plots are to be drawn from logs
    if from_logs:
        # Check that path to log file has been defined
        if not path_training_logs:
            raise ValueError("Define path to the training log file")
        # Read training logs
        training_logs = pd.read_csv(path_training_logs)

        # Read loss and metrics
        acc = training_logs["acc"]
        val_acc = training_logs["val_acc"]

        top3_acc = training_logs["top_3_acc"]
        val_top3_acc = training_logs["val_top_3_acc"]

        loss = training_logs["loss"]
        val_loss = training_logs["val_loss"]

    # Otherwise read from the model history object
    else:
        # Check that the history object has been defined
        if not model_history:
            raise ValueError("Define the model history object")

        # Read loss and metrics
        acc = model_history.history["acc"]
        val_acc = model_history.history["val_acc"]

        top3_acc = model_history.history["top_3_acc"]
        val_top3_acc = model_history.history["val_top_3_acc"]

        loss = model_history.history["loss"]
        val_loss = model_history.history["val_loss"]

    # Initialize figure
    plt.figure(figsize=(8, 8))

    # Draw accuracies in first subplot
    plt.subplot(2, 1, 1)
    plt.plot(acc, "C0", label="Training accuracy")
    plt.plot(top3_acc, "C0--", label="Training top-3 accuracy")
    plt.plot(val_acc, "C1", label="Validation accuracy")
    plt.plot(val_top3_acc, "C1--", label="Validation top-3 accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])

    # Draw losses in second subplot
    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.legend(loc="upper right")
    plt.ylabel("Categorical cross-entropy")

    plt.xlabel("epoch")

    # Show the figure
    plt.show()
    # plt.savefig("loss_accuracy.png")


def find_predicted_true(model, ds):
    """
    Get predicted and true labels by looping over batches of the dataset.

    :param model: Trained model used for predictions
    :param ds: Dataset for which to find predicted labels and true labels.
    :return: Predicted and true labels
    """

    # Initialize predicted and true labels
    predicted_labels = np.array([])
    true_labels = np.array([])

    # Find predicted and true labels by looping over batches of data
    for data_batch, label_batch in ds:
        prediction = np.argmax(tf.nn.softmax(model.predict(data_batch)), axis=1)
        predicted_labels = np.append(predicted_labels, prediction)
        true_labels = np.append(true_labels, label_batch)

    return predicted_labels, true_labels


def create_confusion_matrix(model, ds):
    """
    Compute a confusion matrix based on a trained model and a dataset object.

    :param model: Trained model.
    :param ds: Dataset for which to plot predicted labels and true labels.
    :return: Confusion matrix computed for the dataset.
    """

    # Find predicted and true labels from the dataset
    predicted_labels, true_labels = find_predicted_true(model, ds)

    # Compute the confusion matrix
    confusion_matrix = tf.math.confusion_matrix(true_labels, predicted_labels).numpy()

    return confusion_matrix


def create_class_report(model, ds, classes):
    """
    Create a classification report based on a trained model and a dataset object.

    :param model: Trained model.
    :param ds: Dataset for which to create classification report.
    :param classes:
    :return:
    """
    # Find predicted and true labels from the dataset
    predicted_labels, true_labels = find_predicted_true(model, ds)

    # Compute the classification report
    class_report = classification_report(predicted_labels, true_labels,
                                         target_names=classes, zero_division=0)

    return class_report


def plot_confusion_matrix(confusion_matrix, classes, normalize=True):
    """
    Plot the confusion matrix computed with create_confusion_matrix().

    :param confusion_matrix: Confusion matrix from create_confusion_matrix().
    :param classes: Class names corresponding to confusion matrix labels.
    :param normalize: Should the confusion matrix be normalized prior to plotting.
    """

    if normalize:
        # Normalize data to compensate for unbalanced classes
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = np.nan_to_num(confusion_matrix)
        # Format labels as float
        fmt = ".2f"
    else:
        # Format labels as integer
        fmt = "d"

    # Plot the confusion matrix
    plt.figure(figsize=(16, 14))
    heatmap(confusion_matrix, xticklabels=classes, yticklabels=classes, annot=True,
            cmap="Blues", fmt=fmt)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Show the figure
    plt.show()
    # plt.savefig("confusion_matrix.png")
