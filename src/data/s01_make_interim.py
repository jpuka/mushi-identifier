# The purpose of this script is to transfer images to the interim folder. Only
# non-corrupted images will be transferred.
# TODO:
#  -add logic for test data
#  -transfer functions to s01_make_interim_funcs
#  -turn some cells into functions, e.g. transfer
#  -make tensorflow check function prettier, document
#  -add external data for missing/lacking classes
#  -make paths relative to project repository, check best practices

# We are mixing the predefined "train" and "validation" images here,
# since the validation set is way too small. We will do a better split later.

## Import libraries
import os
import pathlib
import shutil

import pandas as pd

from src.data.s00_eda_functions import read_json_file, merge_data

## Set paths
# TODO: make relative to project repository, check best practices

# External data
path_external_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/00_external/")

# Raw data
path_raw_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/00_raw/")

# Interim data
path_interim_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/01_interim/"
)

# Create path for interim image directory
path_interim_data_dir = path_interim_dir / "images_noncorrupt"

# Training and validation metadata
path_json_train = path_raw_dir / "train.json"
path_json_valid = path_raw_dir / "val.json"

# Classes for the recognition task - chosen Evira mushroom species
path_mushroom_classes = path_external_dir / "evira_species.csv"

## Import and setup metadata

# Read training and validation metadata from json files
df_train_ann, df_train_img, df_train_cat, _, _ = read_json_file(path_json_train)
df_valid_ann, df_valid_img, df_valid_cat, _, _ = read_json_file(path_json_valid)

# Combine relevant metadata into single dataframes
df_train = merge_data(df_train_ann, df_train_img, df_train_cat)
df_valid = merge_data(df_valid_ann, df_valid_img, df_train_cat)

# Import classes
df_mushroom_classes = pd.read_csv(path_mushroom_classes)


##
def create_path_class_lists(df, df_classes):
    """
    Create two lists from a metadata dataframe: image file paths and respective
    image classes. This is a pre-step for transferring the files to interim.

    :param df: Dataframe with metadata (annotation, image, category)
    :param df_classes: Dataframe with mushroom classes
    :return: File path and class for each image
    """
    # Create a mask for the classes
    class_mask = df["name"].isin(df_classes["scientific_name"])

    # Select data rows with the chosen classes
    df_classes = df[class_mask]

    # Select filename and class information
    df_filenames = df_classes.loc[:, ["id", "file_name", "name"]]

    # Save filenames to list
    image_paths = df_filenames["file_name"].to_list()

    # Save classes to list. Make lowercase, add an underscore to make
    # names more filesystem friendly
    image_classes = (df_filenames["name"]
                     .str.replace(" ", "_")
                     .str.lower()
                     .to_list())

    return image_paths, image_classes


##
# File paths and classes as lists
train_image_paths, train_image_classes = create_path_class_lists(df_train,
                                                                 df_mushroom_classes)
valid_image_paths, valid_image_classes = create_path_class_lists(df_valid,
                                                                 df_mushroom_classes)

## Create interim folder structure TODO: to function
unique_classes = (df_mushroom_classes["scientific_name"]
                  .str.replace(" ", "_")
                  .str.lower()
                  .to_list())

for class_name in unique_classes:
    os.makedirs(path_interim_data_dir / class_name, exist_ok=True)

##
import tensorflow as tf


# This function is slow, but only reliable way I found to check if images look
# corrupt to tensorflow
def check_not_corrupted(file_path):
    try:
        img_tf = tf.io.read_file(str(file_path))
        tf.image.decode_jpeg(img_tf, channels=3)
        return True
    except ValueError as e:
        print(f"Bad file: {file_path}, \nError: {e}")
        return False


##

def transfer_to_interim(image_paths, image_classes, path_raw_dir, path_interim_data_dir):
    """
    Transfer images from raw directory into interim. During the transfer, each image
    is checked for corruption and only non-corrupted images are transferred.

    :param image_paths: List of raw image paths
    :param image_classes: List of raw image classes
    :param path_raw_dir: Path to raw directory
    :param path_interim_data_dir: Path to interim directory
    """
    # Copy non-corrupted files to interim directory
    for (image_path, image_class) in zip(image_paths, image_classes):
        # Create path for the raw image
        path_raw_image = path_raw_dir / image_path
        # Find filename from the file path
        filename_raw_image = pathlib.Path(image_path).parts[-1]
        # Create path for the interim image
        path_interim_image = path_interim_data_dir / image_class / filename_raw_image

        # Check for files that tensorflow cannot read
        if check_not_corrupted(path_raw_image):
            # Copy valid files to the interim data directory
            shutil.copyfile(src=path_raw_image,
                            dst=path_interim_image)


## Transfer training and validation images to the same folder
# Train and validation images are mixed
transfer_to_interim(train_image_paths, train_image_classes, path_raw_dir, path_interim_data_dir)
transfer_to_interim(valid_image_paths, valid_image_classes, path_raw_dir, path_interim_data_dir)
