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


def plot_basic_stats(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(num_cols)
    cols = 3
    rows = int(np.ceil(n / cols))

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.1)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f"Распределение: {col}", fontsize=12)
    plt.tight_layout()
    plt.suptitle("Гистограммы распределения признаков", fontsize=16, y=1.02)
    plt.show()

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col].dropna(), ax=axes[i])
        axes[i].set_title(f"Boxplot: {col}", fontsize=12)
    plt.tight_layout()
    plt.suptitle("Boxplot для всех числовых признаков", fontsize=16, y=1.02)
    plt.show()

    corr = df[num_cols].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Корреляционная матрица признаков", fontsize=16)
    plt.show()


def preprocess_dataset(df: pd.DataFrame, numerical_features: list,
                       categorical_features: list = None, target_column: str = None) -> pd.DataFrame:
    df = df.copy()
    categorical_features = categorical_features or []

    print("Удаление строк с отсутствующими значениями...")
    df = df.dropna().reset_index(drop=True)

    print("Масштабирование числовых признаков...")
    scaler = StandardScaler()

    features_to_scale = [col for col in numerical_features if col != target_column]

    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])

    if categorical_features:
        print("Кодирование категориальных признаков...")
        df_scaled = pd.get_dummies(df_scaled, columns=categorical_features, drop_first=True)

    return df_scaled


def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    if target_column not in df.columns:
        raise ValueError(f"Целевая переменная '{target_column}' отсутствует в DataFrame!")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"Разделение данных: тестовая выборка {test_size*100:.0f}%, обучающая {100 - test_size*100:.0f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Разделение завершено успешно!")
    print(f"Размер X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Размер y_train: {y_train.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def linear_regression(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.5,
                      n_steps: int = 1500) -> tuple[np.ndarray, float]:

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for step in range(n_steps):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y

        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("Длины y_true и y_pred должны совпадать!")

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


def evaluate_model(model: tuple[np.ndarray, float], X: np.ndarray, y: np.ndarray) -> float:
    y_pred = np.dot(X, model[0]) + model[1]
    return r2_score_manual(y, y_pred)


def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\n Модель 1: все признаки")
    model1 = linear_regression(X_train.values, y_train.values)
    r2_1 = evaluate_model(model1, X_test.values, y_test.values)
    print(f"R² для модели 1 (все признаки): {r2_1:.4f}")

    geo_features = ['longitude', 'latitude']
    print("\n Модель 2: географические признаки")
    model2 = linear_regression(X_train[geo_features].values, y_train.values)
    r2_2 = evaluate_model(model2, X_test[geo_features].values, y_test.values)
    print(f"R² для модели 2 (только координаты): {r2_2:.4f}")

    socio_features = ['median_income', 'total_rooms', 'population', 'households']
    print("\n Модель 3: социально-экономические признаки")
    model3 = linear_regression(X_train[socio_features].values, y_train.values)
    r2_3 = evaluate_model(model3, X_test[socio_features].values, y_test.values)
    print(f"R² для модели 3 (соц.-эконом. признаки): {r2_3:.4f}")

    print("\n Сравнение моделей по R²:")
    print(f"Модель 1 (все признаки):        {r2_1:.4f}")
    print(f"Модель 2 (географические):      {r2_2:.4f}")
    print(f"Модель 3 (соц.-экономические):  {r2_3:.4f}")

    return {
        "model_1": r2_1,
        "model_2": r2_2,
        "model_3": r2_3
    }


def build_model_with_synthetic_feature(X_train, X_test, y_train, y_test):
    X_train_syn = X_train.copy()
    X_test_syn = X_test.copy()

    X_train_syn["income_per_household"] = X_train_syn["median_income"] * X_train_syn["households"]
    X_test_syn["income_per_household"] = X_test_syn["median_income"] * X_test_syn["households"]

    print("Добавлен признак 'income_per_household' (median_income * households)")

    model = linear_regression(X_train_syn.values, y_train.values, learning_rate=0.5, n_steps=1500)

    y_pred = np.dot(X_test_syn.values, model[0]) + model[1]

    r2 = r2_score_manual(y_test.values, y_pred)

    print(f"R² для модели с синтетическим признаком: {r2:.4f}")
    return r2


def compare_models(r2_results: dict, r2_synthetic: float) -> None:
    results = r2_results.copy()
    results["model_4_synthetic"] = r2_synthetic

    print("\nТаблица R² значений:")
    print("-" * 45)
    for name, score in results.items():
        print(f"{name:<25} | R² = {score:.4f}")
    print("-" * 45)

    best_model = max(results, key=results.get)
    print(f"\n Лучшая модель: {best_model} (R² = {results[best_model]:.4f})")


def tune_hyperparameters(X_train, X_test, y_train, y_test, feature_sets: dict):
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    step_counts = [500, 1000, 1500, 2000, 2500, 3000]

    results = {}

    for model_name, features in feature_sets.items():
        best_r2 = -np.inf
        best_lr, best_steps = None, None

        print(f"\nОптимизация для {model_name} ({len(features)} признаков):")

        for lr in learning_rates:
            for steps in step_counts:
                weights, bias = linear_regression(
                    X_train[features].values, y_train.values,
                    learning_rate=lr, n_steps=steps
                )

                y_pred = np.dot(X_test[features].values, weights) + bias

                if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                    print(f"Пропуск: lr={lr}, steps={steps}, т.к. переполнение")
                    continue

                r2 = r2_score_manual(y_test.values, y_pred)

                if r2 > best_r2:
                    best_r2 = r2
                    best_lr = lr
                    best_steps = steps

        results[model_name] = {
            "R²": best_r2,
            "learning_rate": best_lr,
            "steps": best_steps
        }

        print(f"Лучшая комбинация для {model_name}: R²={best_r2:.4f}, lr={best_lr}, steps={best_steps}")

    print("\n=== Сводная таблица лучших параметров ===")
    print(f"{'Модель':<25} | {'R²':<8} | {'Learning Rate':<13} | {'Steps'}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<25} | {res['R²']:<8.4f} | {res['learning_rate']:<13} | {res['steps']}")
    print("-" * 55)

    return results


def main(path: str = "california_housing_train.csv"):
    df = load_data(path)

    print("\n=== Информация о датасете ===")
    print(df.info())

    print("\n=== Расширенная статистика по данным ===")
    stats = dataset_statistics(df)
    pd.set_option("display.max_columns", None)
    print(stats)

    print("\n=== Визуализация признаков ===")
    plot_basic_stats(df)

    print("\n=== Предварительная обработка данных ===")

    numerical_characteristics = [
        'longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms', 'population',
        'households', 'median_income'
    ]
    categorical_characteristics = []
    target_column = 'median_house_value'

    preprocessed_df = preprocess_dataset(df, numerical_characteristics, categorical_characteristics, target_column)
    print(preprocessed_df.head())

    print("\n=== Разделение данных на обучающую и тестовую выборки ===")
    X_train, X_test, y_train, y_test = split_dataset(preprocessed_df, target_column)

    print("\n=== Реализация линейной регрессии ===")
    r2_results = build_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\n=== Модель с синтетическим признаком ===")
    r2_synthetic = build_model_with_synthetic_feature(X_train, X_test, y_train, y_test)

    print("\n=== Оценка производительности и сравнение моделей ===")
    compare_models(r2_results, r2_synthetic)

    print("\n=== Поиск оптимальных гиперпараметров ===")

    X_train["income_per_household"] = X_train["median_income"] * X_train["households"]
    X_test["income_per_household"] = X_test["median_income"] * X_test["households"]

    feature_sets = {
        "Модель 1 (все признаки)": [col for col in X_train.columns if col != "income_per_household"],
        "Модель 2 (географические признаки)": ['longitude', 'latitude'],
        "Модель 3 (соц.-экономические признаки)": ['median_income', 'total_rooms', 'population', 'households'],
        "Модель 4 (с синтетическим признаком)": list(X_train.columns)
    }

    tune_hyperparameters(X_train, X_test, y_train, y_test, feature_sets)


if __name__ == "__main__":
    main()
