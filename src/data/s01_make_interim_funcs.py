# Functions for creating interim subdirectories and transferring data from raw and
# external. Used in s01_make_interim.py.


## Import libraries
import os
import shutil


##
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


##
def transfer_raw_to_interim(df_meta, path_raw_image_dir, path_interim_image_dir):
    """
    Transfer images from raw directory into interim. During the transfer, each image
    is checked for corruption and only non-corrupted images are transferred.

    :param df_meta: Dataframe with filename and class of each image
    :param path_raw_dir: Path to raw directory
    :param path_interim_image_dir: Path to interim image directory
    """
    # Copy image files to interim directory
    for j, (image_name, image_class) in enumerate(
            zip(df_meta["image_filename"], df_meta["species"])):
        # Create path for the raw image
        path_raw_image = path_raw_image_dir / image_name
        # Create path for the interim image
        path_interim_image = path_interim_image_dir / image_class / image_name

        # TODO: implement a robust corruption check (even though raw dataset
        #  is known-good). PIL should be good enough.
        # Check for corrupted images
        # if verify_not_corrupted(path_raw_image):

        # Print progress
        if ((j + 1) % 500 == 0):
            print(f"Transferring file {j + 1} / {len(df_meta)}")
        elif (j + 1) == len(df_meta):
            print("File transfer raw -> interim complete.")

        # Copy files to the interim data directory
        shutil.copyfile(src=path_raw_image,
                        dst=path_interim_image)


##
# This function is slow, but the most reliable way to check if images look
# corrupt to tensorflow
# TODO: Rethink - currently not used since clunky and unclean.

# import tensorflow as tf
# from tensorflow.python.framework.errors_impl import InvalidArgumentError

# def verify_not_corrupted(image_path):
#     """
#     Check that the input image can be imported by tensorflow.
#
#     :param image_path: Path to image to be checked
#     :return: True if image is ok, False if image is corrupted
#     """
#
#
#     # Attempt to read the image file to a tensor. This is not the fastest or
#     # cleanest approach, but in practice I have found it more reliable at
#     # finding bizarre corruptions than other methods (e.g. PIL, cv2). Note, that
#     # this does not catch the "Corrupt JPEG data" message, since it is technically
#     # not an error (see https://github.com/tensorflow/models/issues/2194).
#     try:
#         img_tf = tf.io.read_file(str(image_path))
#         tf.image.decode_jpeg(img_tf, channels=3)
#         return True
#     except (ValueError, InvalidArgumentError) as e:
#         print(f"Bad file: {image_path}, \nError: {e}")
#         return False
