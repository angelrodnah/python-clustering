#!/usr/bin/env python3

from typing import List
from .utilities import read_utilities, catalogue_utilities
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture


class Dataset:
    def __init__(self) -> None:
        pass

    def load(self, dataset_name: str, load_description=False, download=False):
        '''
        Load a specific dataset from local
        '''
        if dataset_name not in self.list():
            if download == True:
                self.download(dataset_name)
            else:
                print("Dataset isn't present in local environment. To allow downloading, specify load(download=True)")
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


class Tasks:
    def __init__():
        pass

    def AnomalyDetection(dataset):
        '''
        Detect anomalies in the passed dataset
        '''
        pass

    def calculate_number_of_clusters(dataset):
        '''
        Calculater the number of clusters in the passed dataset
        '''
        pass

    def cluster_ensembling(dataset, methods = ['kmeans', 'gmm', 'dbscan']):
        '''Provide cluster ensempling'''
        pass

    def cluster_cimilarity(dataset, method='knn', nearest_neighbors=3):
        '''Provides similar known datasets to provided one using knn'''
        pass
