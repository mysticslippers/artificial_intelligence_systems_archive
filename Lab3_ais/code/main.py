from typing import Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main(path: str = "california_housing_train.csv"):
    df = load_data(path)


if __name__ == "__main__":
    main()
