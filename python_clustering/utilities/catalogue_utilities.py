import os
import pandas as pd
import json
from pytablewriter import MarkdownTableWriter
from dataset_utilities import Dataset


def append_catalogue(file, path):
    dataset = Dataset(file, path=path)
    with open(f"{PATH_TO_ALL_DATASETS_FOLDER}/info.json", "r") as f:
        catalogue = json.load(f)
    catalogue[file.split(".")[-2]] = collect_calculated_data(
        dataset, filepath=f"{path}/{file}"
    )
    with open(f"{PATH_TO_ALL_DATASETS_FOLDER}/info.json", "w") as json_file:
        json.dump(catalogue, json_file)


def create_catalogue():
    catalogue = {}
    for path in PATHS_TO_DATA:
        for file in os.listdir(path):
            dataset = Dataset(file, path=path)
            catalogue[file.split(".")[-2]] = collect_calculated_data(
                dataset, filepath=f"{path}/{file}"
            )
    create_markdown_table(catalogue)
    with open(f"{PATH_TO_ALL_DATASETS_FOLDER}/info.json", "w") as json_file:
        json.dump(catalogue, json_file)


def create_markdown_table(data):
    value_matrix = [[]] * len(data)
    for idx_file, file in enumerate(data):
        row = [None] * len(COLUMN_NAMES)
        for idx_col, col in enumerate(COLUMN_NAMES):
            row[idx_col] = data[file][col]
        value_matrix[idx_file] = row
    writer = MarkdownTableWriter(
        table_name="Dataset Description",
        headers=COLUMN_NAMES,
        value_matrix=value_matrix,
    )
    with open(f"{PATH_TO_ALL_DATASETS_FOLDER}/README.md", "w") as f:
        writer.stream = f
        writer.write_table()


def collect_calculated_data(dataset, filepath=None):
    return {
        "name": dataset.name,
        "origin": dataset.origin,
        "filepath": filepath,
        "description": str(dataset.description),
        "is_imbalanced": dataset.is_imbalanced,
        "number_of_clusters": dataset.number_of_clusters,
        "number_of_datapoint": dataset.number_of_datapoint,
        "number_of_dimensions": dataset.number_of_dimensions,
        "weak_outlier_proportion": dataset.weak_outlier_proportion,
        "strong_outlier_proportion": dataset.strong_outlier_proportion,
        "estimated_max_overlap": dataset.estimate_max_overlap,
        "estimated_mean_overlap": dataset.estimate_mean_overlap,
        "missing_values_proportion": dataset.missing_values_proportion,
    }


if __name__ == "__main__":
    PATH_TO_ALL_DATASETS_FOLDER = f"./../datasets_all"
    PATHS_TO_DATA = [
        f"{PATH_TO_ALL_DATASETS_FOLDER}/real",
        f"{PATH_TO_ALL_DATASETS_FOLDER}/artificial",
    ]
    COLUMN_NAMES = [
        "name",
        "origin",
        "is_imbalanced",
        "number_of_clusters",
        "number_of_datapoint",
        "number_of_dimensions",
        "weak_outlier_proportion",
        "strong_outlier_proportion",
        "estimated_max_overlap",
        "estimated_mean_overlap",
        "missing_values_proportion",
    ]
    create_catalogue()
