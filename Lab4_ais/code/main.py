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


def preprocess_data(data: pd.DataFrame, target_col: str = "Wine") -> Tuple[pd.DataFrame, pd.Series]:
    X = data.drop(columns=[target_col])
    y = data[target_col]

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=X.columns,
        index=X.index
    )

    return X_scaled, y
