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


def visualize_dataset_statistics(data: pd.DataFrame, target_col: str = "Wine", random_state: Optional[int] = 42) -> None:
    rows_count, _ = data.shape

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("Нет числовых признаков для построения гистограмм.")
        return

    means = data[numeric_cols].mean()
    stds = data[numeric_cols].std()

    bins_count = max(5, 1 + int(math.log(rows_count, 2)))
    n_features = len(numeric_cols)
    n_cols_subplot = 4
    n_rows_subplot = math.ceil(n_features / n_cols_subplot)

    fig, axes = plt.subplots(
        n_rows_subplot,
        n_cols_subplot,
        figsize=(4 * n_cols_subplot, 3 * n_rows_subplot)
    )
    axes = axes.ravel()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        hist = ax.hist(
            data[col],
            bins=bins_count,
            edgecolor="black"
        )

        max_height = np.max(hist[0])
        mu = means[col]
        sigma = stds[col]

        ax.plot([mu, mu], [0, max_height], color="red", label="mean")
        ax.plot([mu - sigma, mu - sigma], [0, max_height], color="yellow", label="-1σ")
        ax.plot([mu + sigma, mu + sigma], [0, max_height], color="yellow", label="+1σ")

        ax.set_title(col, fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Распределение признаков", fontsize=14)
    fig.tight_layout()
    plt.show()

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_norm = (X - X.mean()) / (X.max() - X.min())

    unique_classes = sorted(y.unique())
    base_colors = ["red", "green", "blue", "orange", "purple", "brown"]
    class_to_color = {
        cls: base_colors[i % len(base_colors)]
        for i, cls in enumerate(unique_classes)
    }

    if random_state is not None:
        np.random.seed(random_state)

    feature_names = list(X_norm.columns)
    if len(feature_names) < 3:
        print("Недостаточно признаков для 3D-визуализации.")
        return

    chosen_features = np.random.choice(feature_names, size=3, replace=False)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for cls in unique_classes:
        values = X_norm[y == cls]
        ax.scatter(
            values[chosen_features[0]],
            values[chosen_features[1]],
            values[chosen_features[2]],
            label=f"Класс {cls}",
            color=class_to_color[cls]
        )

    ax.set_xlabel(chosen_features[0])
    ax.set_ylabel(chosen_features[1])
    ax.set_zlabel(chosen_features[2])
    ax.set_title("3D-визуализация случайных нормализованных признаков")
    ax.legend()
    plt.show()

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_norm)

    df_pca = pd.DataFrame(X_pca, columns=["pca1", "pca2", "pca3"])
    df_pca[target_col] = y.values

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for cls in unique_classes:
        values = df_pca[df_pca[target_col] == cls]
        ax.scatter(
            values["pca1"],
            values["pca2"],
            values["pca3"],
            label=f"Класс {cls}",
            color=class_to_color[cls]
        )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.set_title("3D-визуализация первых трёх главных компонент (PCA)")
    ax.legend()
    ax.view_init(elev=-140, azim=60)

    plt.show()


def knn_predict(X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, list],
    query_point: Union[np.ndarray, pd.Series, list], k: int = 3) -> int:

    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.asarray(X)

    y_array = np.asarray(y).astype(int)

    query_point = np.asarray(query_point).astype(float).ravel()

    if X_array.shape[1] != query_point.shape[0]:
        raise ValueError(
            f"Размерности не совпадают: "
            f"{X_array.shape[1]} признаков в X и {query_point.shape[0]} в query_point."
        )

    if k <= 0:
        raise ValueError("Параметр k должен быть положительным целым числом.")

    if k > len(X_array):
        raise ValueError(
            f"k = {k} больше количества образцов в X = {len(X_array)}."
        )

    diff = X_array - query_point

    sq_distances = np.sum(diff ** 2, axis=1)

    distances = np.sqrt(sq_distances)

    k_indices = np.argsort(distances)[:k]

    k_nearest_labels = y_array[k_indices]

    most_common_label = np.bincount(k_nearest_labels).argmax()

    return int(most_common_label)


def get_confusion_matrices(k_values: List[int], X_model: pd.DataFrame, y: pd.Series,
                           test_size: float = 0.2, random_state: int = 42) -> List[np.ndarray]:

    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    classes = np.sort(y.unique())
    classes = [int(c) for c in classes]
    n_classes = len(classes)

    class_to_index: Dict[int, int] = {
        cls: idx for idx, cls in enumerate(classes)
    }

    confusion_matrices: List[np.ndarray] = []

    for k in k_values:
        y_pred = [
            knn_predict(X_train.values, y_train.values, x, k)
            for x in X_test.values
        ]

        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

        for true_label, pred_label in zip(y_test, y_pred):
            ti = class_to_index[int(true_label)]
            pi = class_to_index[int(pred_label)]
            confusion_matrix[ti, pi] += 1

        confusion_matrices.append(confusion_matrix)

    return confusion_matrices


def build_and_evaluate_two_knn_models(data: pd.DataFrame, target_col: str, fixed_features: List[str],
                                      k_values: List[int], random_state: int = 42) -> Dict[str, List[np.ndarray]]:
    np.random.seed(random_state)

    y = data[target_col]
    all_features = [col for col in data.columns if col != target_col]

    missing = [f for f in fixed_features if f not in all_features]
    if missing:
        print(f"ВНИМАНИЕ: отсутствуют фиксированные признаки: {missing}")
        fallback_features = all_features[:len(fixed_features)]
        print("Использую вместо них признаки:", fallback_features)
        fixed_features = fallback_features

    X_fixed = data[fixed_features]

    n_features_random = len(fixed_features)
    random_features = np.random.choice(all_features, size=n_features_random, replace=False)
    X_random = data[random_features]

    print("Случайно выбранные признаки (Model 1):", list(random_features))
    print("Фиксированные признаки (Model 2):", fixed_features)

    confusion_random = get_confusion_matrices(k_values, X_random, y, random_state=random_state)
    confusion_fixed = get_confusion_matrices(k_values, X_fixed, y, random_state=random_state)

    return {
        "random": confusion_random,
        "fixed": confusion_fixed
    }
