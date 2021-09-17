# Takes the data from interim and transfers it to processed with a folder structure
# that tf.keras.preprocessing is able to read

# The main difference between interim and processes is that there are no empty
# folders in processed, where as in interim some folders are kept empty for future data.

## Libraries
import os
import pathlib
import shutil

import pandas as pd

## Paths

# External data directory
path_external_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/00_external/")

# Processed data directory
path_processed_image_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/02_processed/"
)

# Interim image directory
path_interim_image_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/01_interim/images_per_class"
)

# Mushroom classes for the image recognition task
path_mushroom_classes = path_external_dir / "mushroom_classes.csv"

## Load

# Load mushroom classes
df_mushroom_classes = pd.read_csv(path_mushroom_classes)

# Take class names as list, make filesystem-friendly
list_classes = (df_mushroom_classes["species"]
                .str.replace(" ", "_")
                .str.lower()
                .to_list())


## transfer_interim_to_processed
subset = "train_and_validation"

for image_class in list_classes:
    orig_folder = path_interim_image_dir / image_class
    dest_folder = path_processed_image_dir / subset / image_class
    orig_fnames = os.listdir(orig_folder)
    # if orig folder is not empty:
    if len(orig_fnames) > 10:

        os.makedirs(dest_folder, exist_ok=True)

        dest_fnames = [f"{image_class}_{j}.jpg" for j in range(0, len(orig_fnames))]

        for orig_fname, dest_fname in zip(orig_fnames, dest_fnames):
            shutil.copyfile(src=orig_folder / orig_fname,
                            dst=dest_folder / dest_fname)

        print(f"Transfer interim -> processed complete "
              f"for {image_class} ({len(orig_fnames)} files).")
    else:
        print(f"Skip transfer for {image_class}, "
              f"too little data ({len(orig_fnames)} files).")
