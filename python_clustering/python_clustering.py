#!/usr/bin/env python3

from typing import List
from .utilities import read_utilities, catalogue_utilities
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture


class Dataset:
    def __init__(self) -> None:
        pass

    def load(dataset_name: str, load_description=False):
        return read_utilities.load(dataset_name, load_description)

    def download(dataset_names: str or List, overwrite=False):
        read_utilities.download(dataset_names, overwrite=overwrite)

    def list():
        read_utilities.list_local_datasets()

    def update_catalogue():
        catalogue = catalogue_utilities.Catalogue()
        catalogue.create()

    def update_local():
        pass


class Methods:
    def __init__(self) -> None:
        pass

    def KMeans(*args):
        return KMeans(args)

    def DBSCAN(*args):
        return DBSCAN(args)

    def AgglomerativeClustering(*args):
        return AgglomerativeClustering(args)

    def GaussianMixture(*args):
        return GaussianMixture(args)

    def Birch(*args):
        return Birch(args)
