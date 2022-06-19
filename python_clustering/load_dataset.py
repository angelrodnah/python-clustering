import python_clustering.dataset_utilities as dutils


def load_dataset(dataset_name: str, load_description=False):
    return dutils.load(dataset_name, load_description)
