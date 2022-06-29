from typing import List
from utilities import read_utilities


class Dataset:
    def __init__(self) -> None:
        pass

    def load(dataset_name: str, load_description=False):
        return read_utilities.load(dataset_name, load_description)

    def download(dataset_names: str or List):
        pass

    def list():
        pass

class Methods:
    def __init__(self) -> None:
        pass
