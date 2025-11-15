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


@dataclass
class DecisionTreeNode:
    parent_attribute: Optional[str] = None
    parent_attribute_value: Optional[Any] = None

    attribute: Optional[str] = None
    entropy: float = 0.0
    samples_count: int = 0

    samples: Dict[Any, int] = field(default_factory=dict)
    probability: Dict[Any, float] = field(default_factory=dict)
    prediction: Optional[Any] = None

    children: List["DecisionTreeNode"] = field(default_factory=list)

    def predict(self, x: pd.Series) -> Any:
        if not self.children or self.attribute is None:
            return self.prediction

        value = x[self.attribute]
        for child in self.children:
            if child.parent_attribute_value == value:
                return child.predict(x)

        return self.prediction

    def predict_proba(self, x: pd.Series) -> Dict[Any, float]:
        if not self.children or self.attribute is None:
            return self.probability

        value = x[self.attribute]
        for child in self.children:
            if child.parent_attribute_value == value:
                return child.predict_proba(x)

        return self.probability

    def __str__(self) -> str:
        if not self.probability:
            return "<Empty node>"

        key_row, count_row, prob_row = "", "", ""
        for key in sorted(self.probability):
            key_str = str(key)
            count_str = f"{self.samples[key]}"
            prob_str = f"{self.probability[key]:.5f}"
            max_len = max(len(key_str), len(count_str), len(prob_str))

            key_row += f" {key_str.rjust(max_len)} |"
            count_row += f" {count_str.rjust(max_len)} |"
            prob_row += f" {prob_str.rjust(max_len)} |"

        parent_attribute = str(self.parent_attribute)
        parent_attribute_value = str(self.parent_attribute_value)
        row_len = len(key_row) - 1

        hline = "+" + "-" * row_len + "+"
        attr_row = parent_attribute + " " * (row_len - len(parent_attribute))
        attr_val_row = parent_attribute_value + " " * (row_len - len(parent_attribute_value))
        lines = [
            hline,
            "|" + attr_row + "|",
            "|" + attr_val_row + "|",
            "|" + "Count &" + " " * (row_len - len("Count &")) + "|",
            "|" + "Probability" + " " * (row_len - len("Probability")) + "|",
            hline,
            "|" + key_row,
            hline,
            "|" + count_row,
            hline,
            "|" + prob_row,
            hline
        ]

        return "\n".join(lines)


class InformationEntropy:
    def __init__(self, df: pd.DataFrame, y_label: str):
        self.y_label = y_label
        self.y_classes = set(df[y_label].unique())

        self.X_values: Dict[str, set] = {}
        for col in df.columns:
            if col != y_label:
                self.X_values[col] = set(df[col].unique())

    def info(self, df: pd.DataFrame) -> float:
        n = df.shape[0]
        if n == 0:
            return 0.0

        counts = df[self.y_label].value_counts()
        probs = counts / n
        return float(-(probs * np.log2(probs)).sum())

    def info_X(self, df: pd.DataFrame, X_label: str) -> float:
        n = df.shape[0]
        if n == 0:
            return 0.0

        result = 0.0
        for attr in self.X_values[X_label]:
            df_i = df.loc[df[X_label] == attr]
            if df_i.shape[0] == 0:
                continue
            result += df_i.shape[0] * self.info(df_i)

        return result / n

    def split_info_X(self, df: pd.DataFrame, X_label: str) -> float:
        n = df.shape[0]
        if n == 0:
            return 1e-9

        result = 0.0
        for attr in self.X_values[X_label]:
            df_i = df.loc[df[X_label] == attr]
            if df_i.shape[0] == 0:
                continue
            p = df_i.shape[0] / n
            result -= p * np.log2(p)

        return result if result > 0 else 1e-9

    def gain_ratio_X(self, df: pd.DataFrame, X_label: str, info_df: Optional[float] = None) -> float:
        if info_df is None:
            info_df = self.info(df)
        return (info_df - self.info_X(df, X_label)) / self.split_info_X(df, X_label)


def main(path: str = "AgaricusLepiota.csv") -> None:
    try:
        data = load_data(path)
    except FileNotFoundError:
        print(f"Файл '{path}' не найден. Проверь путь к датасету.")
        return


if __name__ == "__main__":
    main()
