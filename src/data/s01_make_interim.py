# This script transfers the images from raw and external to interim creating a
# subdirectory for each class. The images are mixed so they can later be easily split
# into train/test/validate sets. The images are checked for corruption before the
# transfer.


## Import libraries
import pathlib

import pandas as pd

# For terminal from project root
import s01_make_interim_funcs as funcs

# For pycharm
# import src.data.s01_make_interim_funcs as funcs

# TODO: Find a common approach for funcs imports without tweaking PATH

## Set paths

# External data directory
path_external_dir = pathlib.Path(
    "data/00_external/")

# Raw data directory
path_raw_dir = pathlib.Path(
    "data/00_raw/")

# Interim data directory
path_interim_dir = pathlib.Path(
    "data/01_interim/"
)

# Raw image directory
path_raw_image_dir = path_raw_dir / "DF20"

# Interim image directory
path_interim_image_dir = path_interim_dir / "images_per_class"

# Training and validation image metadata
path_meta_train_val = path_raw_dir / "DF20-train_metadata_PROD.csv"

# Test image metadata
path_meta_test = path_raw_dir / "DF20-public_test_metadata_PROD.csv"

# Mushroom classes for the image recognition task
path_mushroom_classes = path_external_dir / "mushroom_classes.csv"

## Load metadata and classes

# Load training and validation image metadata
df_meta_train_val_raw = pd.read_csv(path_meta_train_val)

# Load test image metadata
df_meta_test_raw = pd.read_csv(path_meta_test)

# Load mushroom classes
df_mushroom_classes = pd.read_csv(path_mushroom_classes)

## Find filenames for the classes of interest

df_meta_train_val = funcs.filter_path_class_metadata(df_meta_train_val_raw,
                                                     df_mushroom_classes)
df_meta_test = funcs.filter_path_class_metadata(df_meta_test_raw, df_mushroom_classes)

## Create interim folder structure
funcs.create_interim_folders(df_mushroom_classes, path_interim_image_dir)

## Transfer raw data to interim

# Train and validation data
funcs.transfer_raw_to_interim(df_meta_train_val, path_raw_image_dir,
                              path_interim_image_dir)

# TODO: Mix test data in interim with the train/validation set. For now test is
#  kept separate so we can compare test set performance to the DF2020 article.
# Test data
path_interim_test_image_dir = path_interim_dir / "test"
funcs.create_interim_folders(df_mushroom_classes, path_interim_test_image_dir)
funcs.transfer_raw_to_interim(df_meta_test, path_raw_image_dir,
                              path_interim_test_image_dir)

## TODO: Transfer external data to interim
