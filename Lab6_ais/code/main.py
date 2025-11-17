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


def main(path: str = "diabetes.csv") -> None:
    try:
        data = load_data(path)
    except FileNotFoundError:
        print(f"Файл '{path}' не найден. Проверь путь к датасету.")
        return


if __name__ == "__main__":
    main()
