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

## TODO: turn this cell into a function
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

##
# check na's
df_train.isnull().sum()

# no values marked as NA

## TODO: order in cleaner way, tatti - hapero - rousku
# define our classes
# suppilovahvero has wrong scientific name in evira!!

mushi_classes = {"scientific_name":
                     ["Cantharellus cibarius",
                      "Craterellus cornucopioides",
                      "Craterellus tubaeformis",  # wrong in evira, "cantharellus"
                      "Hydnum repandum",
                      "Suillus luteus",
                      "Lactarius rufus",
                      "Lactarius torminosus",
                      "Russula paludosa",
                      "Russula claroflava",
                      "Russula vinosa",
                      "Suillus variegatus",
                      "Hygrophorus camarophyllus",
                      "Cortinarius caperatus",
                      "Albatrellus ovinus",
                      "Morchella spp",
                      "Tricholoma matsutake",
                      "Boletus edulis",
                      "Boletus pinophilus",
                      "Boletus reticulatus",
                      "Leccinum versipelle",
                      "Leccinum aurantiacum",
                      "Leccinum vulpinum",
                      "Lactarius trivialis",
                      "Lactarius utilis",
                      "Lactarius deterrimus",
                      "Lactarius deliciosus",
                      "Russula decolorans"
                      ],
                 "finnish_name":
                     ["keltavahvero",
                      "mustatorvisieni",
                      "suppilovahvero",
                      "vaaleaorakas",
                      "voitatti",
                      "kangasrousku",
                      "karvarousku",
                      "isohapero",
                      "keltahapero",
                      "viinihapero",
                      "kangastatti",
                      "mustavahakas",
                      "kehnäsieni",
                      "lampaankääpä",
                      "huhtasieni",
                      "tuoksuvalmuska",
                      "herkkutatti",
                      "männynherkkutatti",
                      "tammenherkkutatti",
                      "koivunpunikkitatti",
                      "haavanpunikkitatti",
                      "männynpunikkitatti",
                      "haaparousku",
                      "kalvashaaparousku",
                      "kuusenleppärousku",
                      "männynleppärousku",
                      "kangashapero"
                      ]}

df_mushi_classes = pd.DataFrame(mushi_classes)

## TODO: turn into func, in train, mushi, out n_mushi_classes
# create a dataframe with scientific names and number of each species
df_train_classes = (df_train["name"]
                    .value_counts()
                    .to_frame()
                    .reset_index()
                    .rename(columns={"index": "scientific_name", "name": "number"}))

# merge into our classes to see how many classes there are
df_n_mushi_classes = df_mushi_classes.merge(df_train_classes,
                                            how="left", on="scientific_name")

##
df_n_mushi_classes


## try to find missing mushrooms manually, so no typos etc

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


# Hygrophorus camarophyllus, mustavahakas
search_mushi(df_train_classes, "hygrophorus")
# not in images

# Albatrellus ovinus, lampaankääpä
search_mushi(df_train_classes, "albatrellus")
# not in images

# Morchella spp, huhtasieni
search_mushi(df_train_classes, "morchella")
# not in images

# Leccinum aurantiacum, haavanpunikkitatti
search_mushi(df_train_classes, "leccinum")
# not in images

# Lactarius utilis, kalvashaaparousku
search_mushi(df_train_classes, "lactarius")
# not in images

## plotly (will only work in notebook)
import plotly.express as px

df_plot_n_mushi_classes = df_n_mushi_classes.dropna()

fig = px.bar(df_plot_n_mushi_classes, x="finnish_name", y="number")
fig.show()

##
# TODO:
#  -draw bar chart, are our categories balanced
#  -find file list for the classes that are in data
#  -put all functions to s00_eda_functions
