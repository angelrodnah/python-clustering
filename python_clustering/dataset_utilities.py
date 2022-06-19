from scipy import stats
import arff
from scipy.io import arff as scipy_arff
from statistics import NormalDist
import os
import pandas as pd
import numpy as np
import json
from pytablewriter import MarkdownTableWriter

def read_aiff(filename: str, mode: str = "scipy_arff") -> pd.DataFrame:
    if mode == "scipy_arff":
        return scipy_arff.loadarff(filename)
    elif mode == "arff":
        return arff.load(open(filename, "r"))
    return None


def string_handling(df, handle_binary_columns=True, thresh_dummies=5):
    for col in df.columns:
        if df[col].dtype == "object":
            _col = df[col].str.decode("ascii")
            if not _col.isnull().all():
                df[col] = _col
            if (
                handle_binary_columns
                and col != "class"
                and df[col].nunique() < thresh_dummies
            ):
                _dummies = pd.get_dummies(df[col])
                _dummies.columns = [f"{col}_{value}" for value in _dummies.columns]
                df = pd.concat([df, _dummies], axis=1).drop([col], axis=1)
    return df

def append_catalogue(file, path):
    dataset = Dataset(file, path=path)
    with open('../datasets_all/info.json', 'r') as f:
        catalogue = json.load(f)
    catalogue[file[:-5]] = collect_calculated_data(dataset, filepath=f'{path}/{file}')
    with open("./../datasets_all/info.json", "w") as json_file:
        json.dump(catalogue, json_file)

def create_catalogue():
    catalogue = {}
    for path in PATHS_TO_DATA:
        for file in os.listdir(path):
            dataset = Dataset(file, path=path)
            catalogue[file[:-5]] = collect_calculated_data(dataset, filepath=f'{path}/{file}')
    create_markdown_table(catalogue)
    with open("./../datasets_all/info.json", "w") as json_file:
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
    with open("./../datasets_all/README.md", "w") as f:
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

def get_dataset_info(name):
    with open('../datasets_all/info.json', 'r') as f:
        data = json.load(f)
        if name in data:
            return data[name]
        else:
            print("Dataset not found")

def load(name, load_description=False):
    dataset_info = get_dataset_info(name)
    path = dataset_info['filepath']
    try:
        data = read_aiff(f"{path}")
        df = pd.DataFrame(data[0])
        df.columns = df.columns.str.lower()
        description = data[1]
    except NotImplementedError:
        data = read_aiff(f"{path}", mode="arff")
        df = pd.DataFrame(
            data["data"], columns=[x[0].lower() for x in data["attributes"]]
        )
        description = data["description"]
    df = string_handling(df)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_series = numeric_df.isnull().all()

    numeric_df = numeric_df[numeric_series[~numeric_series].index]
    if "class" in df:
        if "class" not in numeric_df.columns:
            df = pd.concat([numeric_df, df["class"]], axis=1)
        else:
            df = numeric_df
        df["class"].fillna(-1, inplace=True)
    else:
        df = numeric_df

    if load_description:
        return df, description
    return df

class Dataset:
    def __init__(self, dataset_name, path) -> None:
        self.name = dataset_name[:-5]
        self.pd_dataframe, self.description = load(self.name, load_description=True)
        self.origin = path.split("/")[-1]
        self.classes = (
            self.pd_dataframe["class"].unique()
            if "class" in self.pd_dataframe.columns
            else []
        )
        self.number_of_clusters = len(self.classes)
        if self.number_of_clusters == 1 and "class" in self.pd_dataframe.columns:
            self.pd_dataframe.drop("class", inplace=True, axis=1)
        self.dimensions = [x for x in self.pd_dataframe if x != "class"]
        self.std_dictionary, self.mean_dictionary = self.calculate_std_mean()
        self.number_of_datapoint = self.calculate_dataframe_shape(cols=False)
        self.number_of_dimensions = self.calculate_dataframe_shape(rows=False) - int(
            "class" in self.pd_dataframe.columns
        )
        self.missing_values_proportion = np.round(self.calculate_number_of_nan(), 4)
        self.is_imbalanced = None
        self.estimate_max_overlap, self.estimate_mean_overlap = None, None
        if self.number_of_clusters > 1:
            self.is_imbalanced = bool(self.calculate_is_imbalanced())
            (
                self.estimate_max_overlap,
                self.estimate_mean_overlap,
            ) = self.calculate_estimate_overlap()
        self.strong_outlier_proportion = np.round(
            self.calculate_number_of_outliers(number_of_std=3), 4
        )
        self.weak_outlier_proportion = np.round(
            (
                self.calculate_number_of_outliers(number_of_std=2)
                - self.strong_outlier_proportion
            ),
            4,
        )

    def calculate_number_of_nan(self):
        return (
            self.pd_dataframe.isnull().sum().sum()
            / self.pd_dataframe.shape[0]
            / self.pd_dataframe.shape[1]
        )

    def calculate_std_mean(self):
        if self.number_of_clusters > 1:
            return self.pd_dataframe.groupby("class").std().to_dict(
                "index"
            ), self.pd_dataframe.groupby("class").mean().to_dict("index")
        else:
            return self.pd_dataframe.std().to_dict(), self.pd_dataframe.mean().to_dict()

    def calculate_estimate_overlap(self):
        overlap_list = []
        for idx, cluster1 in enumerate(self.classes):
            for cluster2 in self.classes[idx + 1 :]:
                for dimension in self.dimensions:
                    overlap = NormalDist(
                                mu=self.mean_dictionary[cluster1][dimension],
                                sigma=np.nan_to_num(self.std_dictionary[cluster1][dimension]) + 1e-10,
                            ).overlap(
                                NormalDist(
                                    mu=self.mean_dictionary[cluster2][dimension],
                                    sigma=np.nan_to_num(self.std_dictionary[cluster2][dimension])
                                    + 1e-10,
                                )
                            )
                overlap_list.append(np.prod(overlap))
        return np.round(np.max(overlap_list), 4), np.round(np.mean(overlap_list), 4)

    def calculate_dataframe_shape(self, rows=True, cols=True):
        if rows:
            if cols:
                return self.pd_dataframe.shape
            return self.pd_dataframe.shape[0]
        elif cols:
            return self.pd_dataframe.shape[1]
        return None

    def calculate_is_imbalanced(
        self, shapiro_alpha: float = 0.05, binary_alpha=0.2
    ) -> bool:
        if self.number_of_clusters == 2:
            return (
                self.pd_dataframe["class"].value_counts().min()
                / self.number_of_datapoint
                > binary_alpha
            )
        _, p = stats.shapiro(self.pd_dataframe["class"].value_counts().to_list())
        return p < shapiro_alpha

    def calculate_number_of_outliers(self, number_of_std=2):
        if self.number_of_clusters > 1:
            return (
                self.pd_dataframe.groupby("class")
                .apply(lambda x: np.abs(stats.zscore(x) > number_of_std))
                .any(axis=1)
                .sum()
                / self.number_of_datapoint
            )
        else:
            return (
                self.pd_dataframe.apply(lambda x: np.abs(stats.zscore(x) > 2))
                .any(axis=1)
                .sum()
                / self.number_of_datapoint
            )


def main():
    create_catalogue()


if __name__ == "__main__":
    PATHS_TO_DATA = ["../datasets_all/real", "../datasets_all/artificial"]
    COLUMN_NAMES = ["name", "origin", "is_imbalanced", "number_of_clusters", "number_of_datapoint", "number_of_dimensions", "weak_outlier_proportion", "strong_outlier_proportion", "estimated_max_overlap", "estimated_mean_overlap", "missing_values_proportion"]
    main()
