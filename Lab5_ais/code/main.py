def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def select_random_features(data: pd.DataFrame, y_label: str) -> List[str]:
    feature_cols = [col for col in data.columns if col != y_label]
    n_features = int(np.round(np.sqrt(len(feature_cols))))

    rng = np.random.RandomState(0)
    shuffled = feature_cols.copy()
    rng.shuffle(shuffled)

    selected_features = shuffled[:n_features]
    print(f"Выбрано признаков: {len(selected_features)}")
    print("Признаки:", selected_features)
    return selected_features


def fill_missing_with_mode(data: pd.DataFrame, feature_names: List[str], missing_value: str = "?") -> pd.DataFrame:
    data = data.copy()

    for col in feature_names:
        null_idx = data.index[data[col] == missing_value]

        if len(null_idx) == 0:
            continue

        mode_series = data.loc[data[col] != missing_value, col].mode()
        if mode_series.empty:
            continue

        mode_value = mode_series.iloc[0]
        n_nulls = len(null_idx)

        data.loc[null_idx, col] = mode_value
        print(f"{n_nulls} null values in {col} changed to {mode_value}")

    return data


def main(path: str = "AgaricusLepiota.csv") -> None:
    try:
        data = load_data(path)
    except FileNotFoundError:
        print(f"Файл '{path}' не найден. Проверь путь к датасету.")
        return


if __name__ == "__main__":
    main()
