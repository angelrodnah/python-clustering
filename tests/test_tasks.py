import numpy as np
import pytest  # type: ignore
import python_clustering


@pytest.fixture
def get_d31_results():
    return np.array(
        [0.0, 0.105263, 0.0, 0.052632, 0.0, 0.315789, 0.0, 0.0, 0.052632, 0.0]
    )


@pytest.fixture
def get_d31_detected_outliers():
    return np.array(
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
        ]
    )


"""
# functionality test
def test_default_datasets(get_d31_results, get_d31_detected_outliers):
    tasks = python_clustering.Tasks()
    dataset = python_clustering.Dataset()
    data = dataset.load("D31").values
    result, detected_outliers, classifiers = tasks.detect_anomalies(data)
    np.testing.assert_array_equal(result[:10, -1], get_d31_results)
    np.testing.assert_array_equal(detected_outliers[31.0], get_d31_detected_outliers)

"""
