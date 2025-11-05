from typing import Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def dataset_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.describe(include='all').T
    stats['missing_count'] = df.isnull().sum()
    stats['missing_pct'] = df.isnull().mean().round(4)
    stats['unique_values'] = df.nunique()
    stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                   'missing_count', 'missing_pct', 'unique_values']]
    return stats


def main(path: str = "california_housing_train.csv"):
    df = load_data(path)

    print("\n=== Информация о датасете ===")
    print(df.info())

    print("\n=== Расширенная статистика по данным ===")
    stats = dataset_statistics(df)
    pd.set_option("display.max_columns", None)
    print(stats)


if __name__ == "__main__":
    main()
