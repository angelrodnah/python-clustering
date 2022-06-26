from typing import List
from utilities import read_utilities


class Clustering:
    def __init__(self) -> None:
        pass

    def load_dataset(dataset_name: str, load_description=False):
        return read_utilities.load(dataset_name, load_description)

    def download_dataset(dataset_name: str):
        pass

    def download_datasets(dataset_name_list: List):
        pass

    def list_datasets():
        pass
