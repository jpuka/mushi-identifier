# Takes the data from interim and transfers it to processed with a folder structure
# that tf.keras.preprocessing is able to read

# The main difference between interim and processed is that there are no empty
# folders in processed, where as in interim some folders are kept empty for future data.
# Furthermore, in processed the data is split to train/validation/test.

## Libraries
import pathlib

import pandas as pd

# For terminal from project root
import s02_make_processed_funcs as funcs

# For pycharm
# import src.data.s02_make_processed_funcs as funcs

# TODO: Find a common approach for funcs imports without tweaking PATH

## Paths

# External data directory
path_external_dir = pathlib.Path(
    "data/00_external/")

# Interim data directory
path_interim_dir = pathlib.Path(
    "data/01_interim/"
)

# Processed data directory
path_processed_dir = pathlib.Path(
    "data/02_processed/"
)

# Interim data image directory
path_interim_image_dir = path_interim_dir / "images_per_class"

# Mushroom classes for the image recognition task
path_mushroom_classes = path_external_dir / "mushroom_classes.csv"

## Define classes

# Load mushroom classes
df_mushroom_classes = pd.read_csv(path_mushroom_classes)

# Take class names as series, make filesystem-friendly
sr_classes = (df_mushroom_classes["species"]
              .str.replace(" ", "_")
              .str.lower())

## Transfer interim data to processed
# TODO: Make a function for splitting data into subsets according to the distribution
#  of each class. Use this to create train/validation/test in processed. Check the
#  permutation idea from manning. But basically shuffle + split + copy

# Train and validation data
funcs.transfer_interim_to_processed(sr_classes, "train_and_validation",
                                    path_interim_image_dir,
                                    path_processed_dir)

# Test data
# TODO: Remove this once test is mixed with the rest of the data in interim and
#  the split function is implemented.
path_interim_test_image_dir = path_interim_dir / "test"
funcs.transfer_interim_to_processed(sr_classes, "test", path_interim_test_image_dir,
                                    path_processed_dir)
