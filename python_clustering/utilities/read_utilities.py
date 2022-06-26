import arff
from scipy.io import arff as scipy_arff
import json
import pandas as pd
import preprocessing_utilities

def read_arff(filepath: str, mode: str = "scipy_arff"):
    """
    Reading aiff format.
    By default scipy.io arff reader is used (mode: scipy_arff).

    Args:
        filepath - str: full path to aiff file
        mode - str: scipy_arff | arff

    Return:
        arff reade format
    """
    if mode == "scipy_arff":
        return scipy_arff.loadarff(filepath)
    elif mode == "arff":
        return arff.load(open(filepath, "r"))
    return None


def get_dataset_info(name):
    path = "./../datasets_all/info.json"
    with open(path, "r") as f:
        data = json.load(f)
        if name in data:
            return data[name]
        else:
            print("Dataset not found")


def load(name, load_description=False):
    dataset_info = get_dataset_info(name)
    path = dataset_info["filepath"]
    try:
        data = read_arff(f"{path}")
        df = pd.DataFrame(data[0])
        df.columns = df.columns.str.lower()
        description = data[1]
    except NotImplementedError:
        data = read_arff(f"{path}", mode="arff")
        df = pd.DataFrame(
            data["data"], columns=[x[0].lower() for x in data["attributes"]]
        )
        description = data["description"]
    df = preprocessing_utilities.preprocessing(df)

    if load_description:
        return df, description
    return df
