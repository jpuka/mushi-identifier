# Functions for creating processed subdirectories and transferring data from interim.
# Used in s02_make_processed.py.

## Import libraries
import os
import shutil


##
def transfer_interim_to_processed(sr_classes, subset, path_interim_image_dir,
                                  path_processed_dir):
    """
    Transfer images from interim directory into processed. During the transfer,
    each image is renamed to the {image_class}_{j}.jpg format.

    :param sr_classes: Series with mushroom classes
    :param subset: Subset to be transferred: "train", "validation" or "test" (not enforced)
    :param path_interim_image_dir: Path to interim image directory
    :param path_processed_dir: Path to processed directory
    """
    # Loop over classes
    for image_class in sr_classes:

        # Define origin path for class
        orig_folder = path_interim_image_dir / image_class
        # Define destination path for class
        dest_folder = path_processed_dir / subset / image_class

        # List files in origin class directory
        orig_fnames = os.listdir(orig_folder)
        # If the class has a reasonable number of files
        if len(orig_fnames) > 5:

            # Create the destination folder
            os.makedirs(dest_folder, exist_ok=True)
            # Build the destination filename
            dest_fnames = [f"{image_class}_{j}.jpg" for j in range(0, len(orig_fnames))]

            # Copy files from origin to destination, rename
            for orig_fname, dest_fname in zip(orig_fnames, dest_fnames):
                shutil.copyfile(src=orig_folder / orig_fname,
                                dst=dest_folder / dest_fname)

            # Report transfer progress
            print(f"Transfer interim -> processed complete "
                  f"for {image_class} ({len(orig_fnames)} files).")

        # Report skipped classes
        else:
            print(f"Skip transfer for {image_class}, "
                  f"too little data ({len(orig_fnames)} files).")
