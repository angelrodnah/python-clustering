import pytest  # type: ignore
import python_clustering

"""
    dataset = python_clustering.Dataset()
    dataset_list = ["atom", "D31", "cpu", "banana", "gaussians1", "circle"]
    # assert(dataset.list() == dataset_list)
    data = dataset.load("D31")
    print(data.shape)
    data = dataset.load("D3aa1")
    print(data)
    data = dataset.load("blobs")
    print(data)
    data = dataset.load("blobs", download=True)
    print(data)
    dataset.download("rings")
    description = dataset.load_description("blobs")
    print(description)
    stats = dataset.load_stats("rings")
    print(stats)
    dataset.update_local_info_files()

    tasks = python_clustering.Tasks()
    data = dataset.load("D31").values
    result, detected_outliers, classifiers = tasks.detect_anomalies(data)

"""


@pytest.fixture
def get_default_datasets():
    return ["atom", "D31", "cpu", "banana", "gaussians1", "circle"]


@pytest.fixture
def get_d31_shape():
    return (3100, 3)


@pytest.fixture
def get_blobs_description():
    return "       0\n0      x\n1      y\n2  class"


@pytest.fixture
def get_rings_stats():
    return {
        "is_imbalanced": False,
        "number_of_clusters": 3,
        "number_of_datapoint": 1000,
        "number_of_dimensions": 2,
        "weak_outlier_proportion": 0.015,
        "strong_outlier_proportion": 0.004,
        "estimated_max_overlap": 0.7269,
        "estimated_mean_overlap": 0.5125,
        "missing_values_proportion": 0.0,
    }


def test_default_datasets(get_default_datasets):
    dataset = python_clustering.Dataset()
    assert dataset.list() == get_default_datasets


def test_positive_load(get_d31_shape):
    dataset = python_clustering.Dataset()
    data = dataset.load("D31")
    assert data.shape == get_d31_shape


def test_negative_load():
    dataset = python_clustering.Dataset()
    data = dataset.load("D31aaa")
    assert data == None


def test_description(get_blobs_description):
    dataset = python_clustering.Dataset()
    description = dataset.load_description("blobs")
    assert description == get_blobs_description


def test_stats(get_rings_stats):
    dataset = python_clustering.Dataset()
    stats = dataset.load_stats("rings")
    assert stats == get_rings_stats


def test_local_update():
    dataset = python_clustering.Dataset()
    assert dataset.update_local_info_files()
