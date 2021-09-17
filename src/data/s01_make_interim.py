# The purpose of this script is to transfer images to the interim folder. Only
# non-corrupted images will be transferred.

# The script combines the external and raw images to interim, where there is a
# a folder for each species.

# Interim folder is used, since external data will be easy to add there
# and split to train/validation/test.

# TODO:
#  -add logic for test data - possibly own directory in interim?
#  -transfer functions to s01_make_interim_funcs
#  -add external data for missing/lacking classes
#  -make paths relative to project repository, check best practices

## Import libraries
import os
import pathlib
import shutil

import pandas as pd
from src.data.s01_make_interim_funcs import filter_path_class_metadata, \
    create_interim_folders, transfer_raw_to_interim

## Set paths
# TODO: make relative to project repository, check best practices

# External data directory
path_external_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/00_external/")

# Raw data directory
path_raw_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/00_raw/")

# Interim data directory
path_interim_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/01_interim/"
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

df_meta_train_val = filter_path_class_metadata(df_meta_train_val_raw, df_mushroom_classes)
df_meta_test = filter_path_class_metadata(df_meta_test_raw, df_mushroom_classes)

## Create interim folder structure
create_interim_folders(df_mushroom_classes, path_interim_image_dir)

## Transfer raw and external data to interim
transfer_raw_to_interim(df_meta_train_val, path_raw_image_dir, path_interim_image_dir)
# transfer_raw_to_interim(df_meta_test)
