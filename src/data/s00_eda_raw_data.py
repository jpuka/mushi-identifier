##
import json
import pathlib

import pandas as pd

## Set paths
path_raw_data = pathlib.Path(
    "/home/jpe/Documents/python_projects/mushi-identifier/data/00_raw/")

path_json_train = path_raw_data / "train.json"
path_json_validate = path_raw_data / "val.json"
path_json_test = path_raw_data / "test.json"


# this will fail, since the json structure is not that simple
# pd.read_json(path_json_train)

##
def read_json_file(path_json):
    """
    Read mushroom json files into pandas dataframes.

    :return: Tuple of pandas dataframes with json contents
    :param path_json: Path to json file
    """
    with open(path_json) as json_file:
        json_data = json.load(json_file)

    df_annotations = pd.DataFrame(json_data["annotations"])
    df_images = pd.DataFrame(json_data["images"])
    df_categories = pd.DataFrame(json_data["categories"])
    df_info = pd.DataFrame.from_dict(json_data["info"], orient="index")
    df_licenses = pd.DataFrame(json_data["licenses"])

    return df_annotations, df_images, df_categories, df_info, df_licenses


##
df_train_ann, df_train_img, df_train_cat, _, _ = read_json_file(path_json_train)

##
# Turn this cell into a function
df_train = pd.merge(df_train_ann, df_train_img,
                    how="left", left_on="image_id", right_on="id")
df_train = pd.merge(df_train, df_train_cat,
                    how="left", left_on="category_id", right_on="id")
df_train = (df_train
            .drop(columns=["id", "id_y"])
            .rename(columns={"id_x": "id"}))

##
# Now all the data is combined
df_train
