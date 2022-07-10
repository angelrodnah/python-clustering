from typing import List
import arff
import os
from scipy.io import arff as scipy_arff
import json
import pandas as pd
import requests
from . import preprocessing_utilities


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
    path = "python_clustering/jsons/dataset_info.json"
    with open(path, "r") as f:
        data = json.load(f)
        if name in data:
            return data[name]
        else:
            print("Dataset not found")


def load(name, load_description=False):
    dataset_info = get_dataset_info(name)
    path = dataset_info["local_filepath"]
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
    return df, None


def download(datasets: str or List[str], overwrite=False):
    if isinstance(datasets, str):
        datasets = [datasets]
    status_datasets = {
        "Dataset_not_found_in_catalogue": [],
        "Download_success": [],
        "Filepath_not_valid": [],
    }
    dataset_info = json.load(open("./python_clustering/jsons/dataset_info.json"))
    for dataset in datasets:
        if dataset not in dataset_info:
            status_datasets["Dataset_not_found_in_catalogue"].append(dataset)
        else:
            github_path = dataset_info[dataset]["github_filepath"]
            r = requests.get(github_path, allow_redirects=True)
            if r.status_code != 200:
                status_datasets["Filepath_not_valid"].append(dataset)
            open(
                f'./python_clustering/datasets/{dataset_info[dataset]["name"]}.{dataset_info[dataset]["filetype"]}',
                "w",
            ).write(r.text)
            status_datasets["Download_success"].append(dataset)

    for status in status_datasets:
        if status_datasets[status]:
            print(f"{status}: {status_datasets[status]}")


def list_local_datasets():
    catalogue_info = json.load(open("./python_clustering/jsons/catalogue_info.json"))
    dataset_info = json.load(open("./python_clustering/jsons/dataset_info.json"))
    local_filepath_dict = {
        dataset_info[filename]["local_filepath"]: filename for filename in dataset_info
    }
    filenames = [
        local_filepath_dict[f'{catalogue_info["PATH_TO_LOCAL"]}/{x}']
        if f'{catalogue_info["PATH_TO_LOCAL"]}/{x}' in local_filepath_dict
        else x
        for x in os.listdir(catalogue_info["PATH_TO_LOCAL"])
    ]
    print(filenames)


if __name__ == "__main__":
    pass
