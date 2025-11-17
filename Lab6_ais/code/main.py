def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    zero_as_nan_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    for column in zero_as_nan_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

            data.loc[data[column] == 0, column] = pd.NA

            median_value = data[column].median(skipna=True)
            data[column] = data[column].fillna(median_value)

    data = data.drop_duplicates().reset_index(drop=True)

    return data


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


def main(path: str = "diabetes.csv") -> None:
    try:
        data = load_data(path)
    except FileNotFoundError:
        print(f"Файл '{path}' не найден. Проверь путь к датасету.")
        return

    print("=== Исходные данные ===")
    print(f"Размер: {data.shape[0]} строк, {data.shape[1]} столбцов\n")

    processed_data = preprocess_data(data)
    print("=== После предобработки ===")
    print(f"Размер: {processed_data.shape[0]} строк, {processed_data.shape[1]} столбцов\n")

    print("=== Статистика по предобработанным данным ===")
    pd.set_option("display.max_columns", None)
    stats = dataset_statistics(processed_data)
    print(stats, "\n")


if __name__ == "__main__":
    main()
