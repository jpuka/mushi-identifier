# The purpose of this script is to transfer images to the interim folder. Only
# non-corrupted images will be transferred.
# TODO:
#  -add logic for validation data
#  -turn some cells into functions, e.g. transfer
#  -make tensorflow check function prettier, document
#  -add external data for missing/lacking classes
#  -make paths relative to project repository, check best practices

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
df_validate = merge_data(df_valid_ann, df_valid_img, df_train_cat)

# Import classes
df_mushroom_classes = pd.read_csv(path_mushroom_classes)

## Create dataframe with correct classes and filenames

# Create a mask for the classes
train_class_mask = df_train["name"].isin(df_mushroom_classes["scientific_name"])

# Select data rows with the chosen classes
df_train_classes = df_train[train_class_mask]

# Select relevant information
df_train_filenames = df_train_classes.loc[:, ["id", "file_name", "name"]]

## Select filenames and class names

# Save filenames to list
train_file_paths = df_train_filenames["file_name"].to_list()

# Save class names to list. There is no data for all the (Evira) classes,
# so we will work with what we have. Make lowercase, add an underscore to make
# names more filesystem friendly
train_class_names = (df_train_filenames["name"]
                     .str.replace(" ", "_")
                     .str.lower()
                     .to_list())

## Create interim folder structure
unique_classes = (df_mushroom_classes["scientific_name"]
                  .str.replace(" ", "_")
                  .str.lower()
                  .to_list())

for class_name in unique_classes:
    os.makedirs(path_interim_data_dir / class_name, exist_ok=True)

## Transfer files

# The directory numbers don't correspond to any metadata

# Copy non-corrupted files to interim directory
for (file_path, class_name) in zip(train_file_paths, train_class_names):
    # Create path for the raw image
    path_raw_image = path_raw_dir / file_path
    # Find filename from the file path
    filename_raw_image = pathlib.Path(file_path).parts[-1]
    # Create path for the interim image
    path_interim_image = path_interim_data_dir / class_name / filename_raw_image

    # Check for files that tensorflow cannot read
    if check_not_corrupted(path_raw_image):
        # Copy valid files to the interim data directory
        shutil.copyfile(src=path_raw_image,
                        dst=path_interim_image)

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
