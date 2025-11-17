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


def _axes_to_list(axes: Axes | np.ndarray) -> List[Axes]:
    if isinstance(axes, Axes):
        return [axes]
    return list(np.array(axes).ravel())


def visualize_statistics(df: pd.DataFrame,
                         corr_figsize: Tuple[int, int] = (15, 10)) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    num_cols = numeric_df.columns.tolist()

    if not num_cols:
        print("В датасете нет числовых признаков для визуализации.")
        return

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.1)

    means = numeric_df.mean()

    plt.figure(figsize=(18, 6))
    ax_mean = sns.barplot(x=means.index, y=means.values)
    ax_mean.set_title("Средние значения (mean) по числовым признакам",
                      fontsize=16)
    ax_mean.set_xlabel("Признаки")
    ax_mean.set_ylabel("Среднее значение")

    for label in ax_mean.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    for i, v in enumerate(means.values):
        ax_mean.text(i, v, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    stds = numeric_df.std()

    plt.figure(figsize=(18, 6))
    ax_std = sns.barplot(x=stds.index, y=stds.values)
    ax_std.set_title("Стандартное отклонение (std) по числовым признакам",
                     fontsize=16)
    ax_std.set_xlabel("Признаки")
    ax_std.set_ylabel("Стандартное отклонение")

    for label in ax_std.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    for i, v in enumerate(stds.values):
        ax_std.text(i, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    n = len(num_cols)
    cols = 3
    rows = int(math.ceil(n / cols))

    fig_hist, axes_hist_raw = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes_hist = _axes_to_list(axes_hist_raw)

    for i, col in enumerate(num_cols):
        ax = axes_hist[i]
        sns.histplot(numeric_df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Распределение: {col}", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    for extra_ax in axes_hist[i + 1:]:
        extra_ax.set_visible(False)

    plt.tight_layout()
    plt.suptitle("Гистограммы распределения признаков",
                 fontsize=16, y=1.02)
    plt.show()

    fig_box, axes_box_raw = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes_box = _axes_to_list(axes_box_raw)

    for i, col in enumerate(num_cols):
        ax = axes_box[i]
        sns.boxplot(x=numeric_df[col].dropna(), ax=ax)
        ax.set_title(f"Boxplot: {col}", fontsize=12)
        ax.set_xlabel(col)

    for extra_ax in axes_box[i + 1:]:
        extra_ax.set_visible(False)

    plt.tight_layout()
    plt.suptitle("Boxplot для всех числовых признаков",
                 fontsize=16, y=1.02)
    plt.show()

    corr = numeric_df.corr()

    plt.figure(figsize=corr_figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
    )
    plt.title("Корреляционная матрица признаков", fontsize=16)
    plt.tight_layout()
    plt.show()


def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    if target_column not in df.columns:
        raise ValueError(f"Столбец '{target_column}' не найден в датасете.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("=== Разделение на обучающую и тестовую выборки ===")
    print(f"Размер исходного датасета: {len(df)} объектов")
    print(f"Размер обучающей выборки: {len(X_train)} объектов")
    print(f"Размер тестовой выборки: {len(X_test)} объектов")
    print(f"Доля тестовой выборки: {test_size:.2f}\n")

    print("Распределение целевого класса в обучающей выборке:")
    print(y_train.value_counts(normalize=True).round(3))
    print("\nРаспределение целевого класса в тестовой выборке:")
    print(y_test.value_counts(normalize=True).round(3))
    print()

    return X_train, X_test, y_train, y_test


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
