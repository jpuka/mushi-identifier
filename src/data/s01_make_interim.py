# The purpose of this script is to transfer images to the interim folder. Only
# non-corrupted images will be transferred.

# The script combines the external and raw images to interim, where there is a
# a folder for each species

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

## Set paths
# TODO: make relative to project repository, check best practices

# External data path
path_external_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/00_external/")

# Raw data path
path_raw_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/00_raw/")

# Interim data path
path_interim_dir = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/01_interim/"
)

# Create path to raw image directory
path_raw_image_dir = path_raw_dir / "DF20"

# Create path to interim image directory
path_interim_image_dir = path_interim_dir / "images_per_class"

# Training and validation image metadata
path_meta_train_val = path_raw_dir / "DF20-train_metadata_PROD.csv"

# Test image metadata
path_meta_test = path_raw_dir / "DF20-public_test_metadata_PROD.csv"

# Classes for the recognition task - chosen Evira mushroom species
path_mushroom_classes = path_external_dir / "mushroom_classes.csv"

## Load metadata and classes

# Load training and validation image metadata
df_meta_train_val_raw = pd.read_csv(path_meta_train_val)

# Load test image metadata
df_meta_test_raw = pd.read_csv(path_meta_test)

# Import mushroom classes
df_mushroom_classes = pd.read_csv(path_mushroom_classes)


def filter_path_class_metadata(df_meta_raw, df_classes):
    """
    Search raw metadata for image filenames of classes of interest.
    This is a pre-step for transferring the images to the interim folder.

    :param df_meta_raw: Dataframe with raw metadata
    :param df_classes: Dataframe with mushroom classes
    :return: Dataframe with filename and class of each image
    """

    # Choose columns of interest. There is a lot of interesting metadata, but
    # we are only interested in species and image paths.
    df_meta = df_meta_raw.loc[:, ["genus", "specificEpithet", "image_path"]]

    # Rename image_path, since it contains filenames
    df_meta.rename(columns={"image_path": "image_filename"}, inplace=True)

    # Construct the species name (in the raw metadata "scientificName" has additional
    # characters and "species" has NaNs)
    df_meta["species"] = df_meta["genus"] + " " + df_meta["specificEpithet"]

    # Drop the columns used for constructing the species name
    df_meta = df_meta.drop(columns=["genus", "specificEpithet"])

    # Choose rows with species that correspond to our classes
    mask_classes = df_meta["species"].isin(df_classes["species"])
    df_meta = df_meta[mask_classes]

    # Make the species names more file-system friendly, since they will be
    # used to match folders later
    df_meta["species"] = (df_meta["species"]
                          .str.replace(" ", "_")
                          .str.lower())

    return df_meta


##
df_meta_train_val = filter_path_class_metadata(df_meta_train_val_raw, df_mushroom_classes)
df_meta_test = filter_path_class_metadata(df_meta_test_raw, df_mushroom_classes)


## Create interim folder structure
def create_interim_folders(df_classes, path_interim_image_dir):
    """
    Create interim folder structure: each class has a folder

    :param df_classes: Dataframe with mushroom classes
    :param path_interim_image_dir: Path to interim image directory
    """

    # Create a filesystem-friendly folder name for each class
    interim_folder_names = (df_classes["species"]
                            .str.replace(" ", "_")
                            .str.lower())

    # Create the folder structure
    for name in interim_folder_names:
        path_new_dir = path_interim_image_dir / name
        # If the folder structure does not yet exist
        if not path_new_dir.is_dir():
            print(f"Creating interim directory: {name}")
            os.makedirs(path_new_dir)


create_interim_folders(df_mushroom_classes, path_interim_image_dir)

##
# import tensorflow as tf
# from tensorflow.python.framework.errors_impl import InvalidArgumentError


# This function is slow, but only reliable way I found to check if images look
# corrupt to tensorflow
def verify_not_corrupted(image_path):
    """
    Check that the input image can be imported by tensorflow.

    :param image_path: Path to image to be checked
    :return: True if image is ok, False if image is corrupted
    """

    # TODO: Rethink - currently not used, possibly leaks memory
    # Attempt to read the image file to a tensor. This is not the fastest or
    # cleanest approach, but in practice I have found it more reliable at
    # finding bizarre corruptions than other methods (e.g. PIL, cv2). Note, that
    # this does not catch the "Corrupt JPEG data" message, since it is technically
    # not an error (see https://github.com/tensorflow/models/issues/2194).
    try:
        img_tf = tf.io.read_file(str(image_path))
        tf.image.decode_jpeg(img_tf, channels=3)
        return True
    except (ValueError, InvalidArgumentError) as e:
        print(f"Bad file: {image_path}, \nError: {e}")
        return False


##

def transfer_raw_to_interim(df_meta, path_raw_image_dir, path_interim_image_dir):
    """
    Transfer images from raw directory into interim. During the transfer, each image
    is checked for corruption and only non-corrupted images are transferred.

    :param df_meta: Dataframe with filename and class of each image
    :param path_raw_dir: Path to raw directory
    :param path_interim_image_dir: Path to interim directory
    """
    # Copy image files to interim directory
    for j, (image_name, image_class) in enumerate(
            zip(df_meta["image_filename"], df_meta["species"])):
        # Create path for the raw image
        path_raw_image = path_raw_image_dir / image_name
        # Create path for the interim image
        path_interim_image = path_interim_image_dir / image_class / image_name

        # TODO: Remove or rethink
        # Check for files that tensorflow cannot read
        # if verify_not_corrupted(path_raw_image):

        # Print progress
        if ((j + 1) % 500 == 0):
            print(f"Transferring file {j + 1} / {len(df_meta)}")

        # Copy files to the interim data directory
        shutil.copyfile(src=path_raw_image,
                        dst=path_interim_image)


##
transfer_raw_to_interim(df_meta_train_val, path_raw_image_dir, path_interim_image_dir)
# transfer_raw_to_interim(df_meta_test)
