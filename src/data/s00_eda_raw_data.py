##
import json

import pandas as pd


##
# TODO: Make this work for test json
def read_json_file(path_json):
    """
    Read mushroom json files into pandas dataframes.

    :param path_json: Path to json file
    :return: Tuple of pandas dataframes with json contents
    """
    with open(path_json) as json_file:
        json_data = json.load(json_file)

    df_annotations = pd.DataFrame(json_data["annotations"])
    df_images = pd.DataFrame(json_data["images"])
    df_categories = pd.DataFrame(json_data["categories"])
    df_info = pd.DataFrame.from_dict(json_data["info"], orient="index")
    df_licenses = pd.DataFrame(json_data["licenses"])

    return df_annotations, df_images, df_categories, df_info, df_licenses


def merge_data(df_annotations, df_images, df_categories):
    """
    Merge relevant dataframes created by read_json_file() into a single dataframe.

    :param df_annotations: Dataframe with annotation data
    :param df_images: Dataframe with image data
    :param df_categories: Dataframe with category data
    :return: Merged dataframe with relevant contents
    """
    df_merged = pd.merge(df_annotations, df_images,
                         how="left", left_on="image_id", right_on="id")
    df_merged = pd.merge(df_merged, df_categories,
                         how="left", left_on="category_id", right_on="id")
    df_merged = (df_merged
                 .drop(columns=["id", "id_y"])
                 .rename(columns={"id_x": "id"}))

    return df_merged


##
def create_evira_class_dataframe(df_merged, df_mushi_classes):
    """
    Create dataframe with Evira's recommended mushroom species and the counts
    of each species.

    :param df_merged: Dataframe with annotation, image and category data
    :param df_mushi_classes: Dataframe with Evira mushroom species names
    :return: A dataframe with Evira mushroom names and counts.
    """
    # Create a dataframe with the scientific name and count of every species
    df_all_classes = (df_merged["name"]
                      .value_counts()
                      .to_frame()
                      .reset_index()
                      .rename(columns={"index": "scientific_name", "name": "count"}))

    # Modify into a dataframe with only Evira species
    df_evira_mushi_classes = df_mushi_classes.merge(df_all_classes,
                                                    how="left", on="scientific_name")

    return df_all_classes, df_evira_mushi_classes


def search_mushi(df_classes, string):
    """
    Search image class dataframe for mushroom names.

    :param df_classes: Dataframe with raw data image classes
    :param string: String to be found
    :return: Number of found mushroom names
    """
    # Create mask
    name_mask = df_classes["scientific_name"].str.lower().str.contains(
        string)
    # Use mask to find entries
    search_results = df_classes[name_mask]

    return search_results

##
# TODO:
#  -draw bar chart, are our categories balanced
#  -find file list for the classes that are in data
#  -put all functions to s00_eda_functions
