from scipy.io import arff
import pandas as pd

def read_aiff(filename: str) -> pd.DataFrame:
    data = arff.loadarff(filename)
    return pd.DataFrame(data[0])