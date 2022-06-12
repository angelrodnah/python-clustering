from scipy import stats
from scipy.io import arff
import os
import pandas as pd
import numpy as np


def read_aiff(filename: str) -> pd.DataFrame:
    data = arff.loadarff(filename)
    return data


# check column consistency
# create clustering stats


class Dataset:
    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
        self.pd_dataframe = self.load_dataset()
        self.is_class_present = True if "class" in self.pd_dataframe.columns else False
        self.number_of_datapoint = self.calculate_dataframe_shape(cols=False)
        self.number_of_dimensions = self.calculate_dataframe_shape(rows=False) - int(
            self.is_class_present
        )
        self.number_of_clusters, self.is_imbalanced = None, None
        if self.is_class_present:
            self.number_of_clusters = self.calculate_number_of_clusters()
            self.is_imbalanced = self.calculate_is_imbalanced()
        self.strong_outlier_proportion = self.calculate_number_of_outliers(number_of_std = 3)
        self.weak_outlier_proportion = self.calculate_number_of_outliers(number_of_std = 2) - self.strong_outlier_proportion
        

    def load_dataset(self, load_description=False):
        # check if dataset is present
        # add source information
        data = read_aiff(f"{PATH_TO_DATA}/{self.dataset_name}")
        df = pd.DataFrame(data[0])
        df.columns = df.columns.str.lower()
        if "class" in df.columns:
            df["class"] = df["class"].str.decode("utf-8")
        if load_description:
            return pd.DataFrame(df), data[1]
        return pd.DataFrame(df)

    def calculate_dataframe_shape(self, rows=True, cols=True):
        shape = self.pd_dataframe.shape
        if rows:
            if cols:
                return shape
            return shape[0]
        elif cols:
            return shape[1]
        return None

    def calculate_number_of_clusters(self):
        return self.pd_dataframe["class"].nunique()

    def calculate_is_imbalanced(self, alpha: float = 0.05) -> bool:
        _, p = stats.shapiro(self.pd_dataframe["class"].value_counts().to_list())
        if p < alpha:
            return True
        return False

    def calculate_number_of_outliers(self, number_of_std = 2):
        if self.is_class_present:
            return self.pd_dataframe.groupby('class').apply(lambda x: np.abs(stats.zscore(x) > 2)).any(axis=1).sum()/self.number_of_datapoint
        else:
            return self.pd_dataframe.apply(lambda x: np.abs(stats.zscore(x) > 2)).any(axis=1).sum()/self.number_of_datapoint


def main():
    for file in os.listdir(PATH_TO_DATA):
        dataset = Dataset(file)
        print("=" * 30)
        print(file)
        print("is_class_present", dataset.is_class_present)
        print("is_imbalanced", dataset.is_imbalanced)
        print("number_of_clusters", dataset.number_of_clusters)
        print("number_of_datapoint", dataset.number_of_datapoint)
        print("number_of_dimensions", dataset.number_of_dimensions)
        print("weak_outlier_proportion", dataset.weak_outlier_proportion)
        print("strong_outlier_proportion", dataset.strong_outlier_proportion)


if __name__ == "__main__":
    PATH_TO_DATA = "./datasets/data"
    main()
