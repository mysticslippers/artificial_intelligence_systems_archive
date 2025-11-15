def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
  

def dataset_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.describe(include="all").T

    stats["missing_count"] = df.isnull().sum()
    stats["missing_pct"] = df.isnull().mean().round(4)
    stats["unique_values"] = df.nunique()

    desired_cols = [
        "count", "mean", "std", "min",
        "25%", "50%", "75%", "max",
        "missing_count", "missing_pct", "unique_values"
    ]
    existing_cols = [column for column in desired_cols if column in stats.columns]

    return stats[existing_cols]
