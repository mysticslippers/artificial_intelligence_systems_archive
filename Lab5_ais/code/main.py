import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Hashable
from collections import namedtuple

from sklearn.model_selection import train_test_split


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


class DecisionTree:
    def __init__(self, max_leaf_entropy: float = 0.0, max_leaf_samples: int = 1):
        assert max_leaf_samples > 0, "max_leaf_samples должен быть > 0"
        self.root: Optional[DecisionTreeNode] = None
        self.max_leaf_entropy = max_leaf_entropy
        self.max_leaf_samples = max_leaf_samples
        self.info_entropy: Optional[InformationEntropy] = None

    def _fill_node_stats(self, node: DecisionTreeNode, df: pd.DataFrame):
        n = df.shape[0]
        node.samples_count = n

        samples: Dict[Any, int] = {}
        probs: Dict[Any, float] = {}
        max_count = 0
        best_class = None

        for y_class in self.info_entropy.y_classes:
            count = df.loc[df[self.info_entropy.y_label] == y_class].shape[0]
            samples[y_class] = count
            probs[y_class] = count / n if n > 0 else 0.0

            if count > max_count:
                max_count = count
                best_class = y_class

        node.samples = samples
        node.probability = probs
        node.prediction = best_class
        node.entropy = self.info_entropy.info(df)

    def _choose_best_attribute(self, df: pd.DataFrame, available_features: List[str]) -> Optional[str]:
        best_attr = None
        best_ratio = -1.0
        info_df = self.info_entropy.info(df)

        for attr in available_features:
            ratio = self.info_entropy.gain_ratio_X(df, attr, info_df=info_df)
            if ratio > best_ratio:
                best_ratio = ratio
                best_attr = attr
        return best_attr

    def _build_tree(self, df: pd.DataFrame, node: DecisionTreeNode, available_features: List[str]):
        self._fill_node_stats(node, df)

        unique_classes = df[self.info_entropy.y_label].nunique()
        if (unique_classes == 1 or
                node.entropy <= self.max_leaf_entropy or
                node.samples_count <= self.max_leaf_samples or
                len(available_features) == 0):
            return

        best_attr = self._choose_best_attribute(df, available_features)
        if best_attr is None:
            return

        node.attribute = best_attr
        remaining_features = [f for f in available_features if f != best_attr]

        for attr_value in self.info_entropy.X_values[best_attr]:
            subset = df.loc[df[best_attr] == attr_value]
            if subset.shape[0] == 0:
                continue

            child = DecisionTreeNode(
                parent_attribute=best_attr,
                parent_attribute_value=attr_value
            )
            node.children.append(child)
            self._build_tree(subset, child, remaining_features)

    def fit(self, df: pd.DataFrame, y_label: str) -> "DecisionTree":
        self.info_entropy = InformationEntropy(df, y_label)
        feature_cols = [col for col in df.columns if col != y_label]

        self.root = DecisionTreeNode(
            parent_attribute=None,
            parent_attribute_value="ROOT"
        )
        self._build_tree(df, self.root, feature_cols)
        return self

    def predict(self, X_test: pd.DataFrame) -> List[Any]:
        assert self.root is not None, "Сначала вызовите fit()"
        y_pred: List[Any] = []
        for i in range(X_test.shape[0]):
            x = X_test.iloc[i]
            y_pred.append(self.root.predict(x))
        return y_pred

    def predict_proba(self, X_test: pd.DataFrame) -> List[Dict[Any, float]]:
        assert self.root is not None, "Сначала вызовите fit()"
        y_prob: List[Dict[Any, float]] = []
        for i in range(X_test.shape[0]):
            x = X_test.iloc[i]
            y_prob.append(self.root.predict_proba(x))
        return y_prob

    def __str__(self) -> str:
        return str(self.root) if self.root is not None else "<Empty tree>"


class TreePrinter:
    def __init__(self, get_children):
        self.elbow = "└── "
        self.tee = "├── "
        self.pipe = "│   "
        self.blank = "    "
        self.get_children = get_children

    def _build_lines(self, node, prefix="", is_last=True):
        if node is None:
            return []

        block_lines = str(node).strip().split("\n")
        if not block_lines:
            return []

        connector = self.elbow if is_last else self.tee
        lines = [prefix + connector + block_lines[0]]

        child_prefix = prefix + (self.blank if is_last else self.pipe)
        for line in block_lines[1:]:
            lines.append(child_prefix + line)

        children = self.get_children(node)
        for i, child in enumerate(children):
            last_child = (i == len(children) - 1)
            lines.extend(self._build_lines(child, child_prefix, last_child))

        return lines

    def print(self, root):
        if root is None:
            print("<Empty tree>")
            return
        lines = self._build_lines(root, prefix="", is_last=True)
        print("\n".join(lines))


ConfusionMatrix = namedtuple("ConfusionMatrix", ["TP", "FP", "FN", "TN"])


def confusion_matrix(y_true: Sequence[Hashable], y_pred: Sequence[Hashable], positive_label: Optional[Hashable] = None) -> ConfusionMatrix:
    if not y_true:
        return ConfusionMatrix(TP=0, FP=0, FN=0, TN=0)

    if positive_label is None:
        classes = list(set(y_true))
        positive_label = classes[0]

    TP = FP = FN = TN = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == positive_label:
            if yt == positive_label:
                TP += 1
            else:
                FP += 1
        else:
            if yt == positive_label:
                FN += 1
            else:
                TN += 1
    return ConfusionMatrix(TP=TP, FP=FP, FN=FN, TN=TN)


def accuracy_score(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def precision_score(y_true: Sequence[Hashable], y_pred: Sequence[Hashable], positive_label: Optional[Hashable] = None) -> float:
    cm = confusion_matrix(y_true, y_pred, positive_label=positive_label)
    denom = cm.TP + cm.FP
    return cm.TP / denom if denom > 0 else 0.0


def recall_score(y_true: Sequence[Hashable], y_pred: Sequence[Hashable], positive_label: Optional[Hashable] = None) -> float:
    cm = confusion_matrix(y_true, y_pred, positive_label=positive_label)
    denom = cm.TP + cm.FN
    return cm.TP / denom if denom > 0 else 0.0


def integrate_traps(x_values, y_values):
    area = 0.0
    for i in range(len(x_values) - 1):
        x0, x1 = x_values[i], x_values[i + 1]
        y0, y1 = y_values[i], y_values[i + 1]
        area += (y0 + y1) * (x1 - x0) / 2
    return area


def TPR_by_FPR(y_true: Sequence[Hashable], y_probs: Sequence[Dict[Any, float]],
               y_positive: Hashable, y_negative: Hashable, lines_count: int = 0):
    if not y_true:
        return [0.0], [0.0]

    use_probs = lines_count <= 0
    probs_pos = [p.get(y_positive, 0.0) for p in y_probs]

    if use_probs:
        thresholds = sorted(set(probs_pos), reverse=True)
    else:
        thresholds = [1 - i / lines_count for i in range(lines_count + 1)]

    FPR_values = [0.0]
    TPR_values = [0.0]

    for threshold in thresholds:
        y_pred = [
            y_positive if p.get(y_positive, 0.0) >= threshold else y_negative
            for p in y_probs
        ]
        cm = confusion_matrix(y_true, y_pred, positive_label=y_positive)

        denom_fpr = cm.FP + cm.TN
        denom_tpr = cm.TP + cm.FN
        if denom_fpr == 0 or denom_tpr == 0:
            continue

        FPR = cm.FP / denom_fpr
        TPR = cm.TP / denom_tpr

        FPR_values.append(FPR)
        TPR_values.append(TPR)

    return FPR_values, TPR_values


def Precision_by_Recall(y_true: Sequence[Hashable], y_probs: Sequence[Dict[Any, float]],
                        y_positive: Hashable, y_negative: Hashable, lines_count: int = 0):
    if not y_true:
        return [0.0], [1.0]

    use_probs = lines_count <= 0
    probs_pos = [p.get(y_positive, 0.0) for p in y_probs]

    if use_probs:
        thresholds = sorted(set(probs_pos), reverse=True)
    else:
        thresholds = [1 - i / lines_count for i in range(lines_count + 1)]

    Recall_values = [0.0]
    Precision_values = [1.0]

    for threshold in thresholds:
        y_pred = [
            y_positive if p.get(y_positive, 0.0) >= threshold else y_negative
            for p in y_probs
        ]
        cm = confusion_matrix(y_true, y_pred, positive_label=y_positive)

        denom_recall = cm.TP + cm.FN
        denom_prec = cm.TP + cm.FP
        if denom_recall == 0 or denom_prec == 0:
            continue

        recall = cm.TP / denom_recall
        precision = cm.TP / denom_prec

        Recall_values.append(recall)
        Precision_values.append(precision)

    return Recall_values, Precision_values


def evaluate_model(model: DecisionTree, X_test: pd.DataFrame, y_test: pd.Series,
                   positive_label: Optional[Hashable] = None, negative_label: Optional[Hashable] = None,
                   lines_count: int = 0, plot: bool = True,):
    y_true = list(y_test)
    if not y_true:
        print("Пустая тестовая выборка, метрики не считаются.")
        return {}

    classes = sorted(set(y_true))
    if positive_label is None:
        positive_label = classes[0]
    if negative_label is None:
        negative_label = classes[1] if len(classes) > 1 else classes[0]

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, positive_label=positive_label)
    rec = recall_score(y_true, y_pred, positive_label=positive_label)

    y_probs = model.predict_proba(X_test)

    roc_x, roc_y = TPR_by_FPR(
        y_true=y_true,
        y_probs=y_probs,
        y_positive=positive_label,
        y_negative=negative_label,
        lines_count=lines_count,
    )

    pr_x, pr_y = Precision_by_Recall(
        y_true=y_true,
        y_probs=y_probs,
        y_positive=positive_label,
        y_negative=negative_label,
        lines_count=lines_count,
    )

    roc_pairs = sorted(zip(roc_x, roc_y), key=lambda p: p[0])
    roc_x, roc_y = zip(*roc_pairs)

    pr_pairs = sorted(zip(pr_x, pr_y), key=lambda p: p[0])
    pr_x, pr_y = zip(*pr_pairs)

    auc_roc = integrate_traps(list(roc_x), list(roc_y))
    auc_pr = integrate_traps(list(pr_x), list(pr_y))

    print("=== Оценка модели ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f} (positive = '{positive_label}')")
    print(f"Recall   : {rec:.4f} (positive = '{positive_label}')")
    print()
    print(f"AUC-ROC  : {auc_roc:.4f}")
    print(f"AUC-PR   : {auc_pr:.4f}")

    if plot:
        plt.figure()
        plt.plot(roc_x, roc_y, marker='s', linestyle='-', markersize=4,
                 label='ROC')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC-кривая')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(pr_x, pr_y, marker='o', linestyle='-', markersize=4,
                 label='Precision–Recall')
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('Recall (полнота)')
        plt.ylabel('Precision (точность)')
        plt.title('Precision–Recall кривая')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.show()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": rec,
        "roc_x": list(roc_x),
        "roc_y": list(roc_y),
        "pr_x": list(pr_x),
        "pr_y": list(pr_y),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }


def count_leaves(node: Optional[DecisionTreeNode]) -> int:
    if node is None:
        return 0
    if not node.children:
        return 1
    return sum(count_leaves(child) for child in node.children)


def tree_depth(node: Optional[DecisionTreeNode]) -> int:
    if node is None or not node.children:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)


def main(path: str = "AgaricusLepiota.csv") -> None:
    try:
        data = load_data(path)
    except FileNotFoundError:
        print(f"Файл '{path}' не найден. Проверь путь к датасету.")
        return

    y_label = "classes"

    print("Размер исходных данных:", data.shape)

    X_labels = select_random_features(data, y_label)

    data_clean = fill_missing_with_mode(data, X_labels, missing_value="?")

    data_n = data_clean[X_labels + [y_label]]
    print("Размер данных после отбора признаков:", data_n.shape)

    classes = sorted(data_n[y_label].unique())
    print("Классы целевой переменной:", classes)

    X = data_n.drop(columns=[y_label])
    y = data_n[y_label]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.9,
        random_state=0,
        stratify=y
    )

    print("Размер train:", X_train.shape[0])
    print("Размер test :", X_test.shape[0])

    df_train = X_train.join(y_train)
    dt = DecisionTree(max_leaf_entropy=0.001, max_leaf_samples=10).fit(df_train, y_label)

    print("\n=== Дерево решений (фрагмент) ===")
    printer = TreePrinter(lambda node: node.children)
    printer.print(dt.root)

    print("\nДоп. характеристики дерева:")
    print("Глубина дерева:", tree_depth(dt.root))
    print("Число листьев:", count_leaves(dt.root))

    evaluate_model(
        model=dt,
        X_test=X_test,
        y_test=y_test,
        positive_label=classes[0] if len(classes) > 0 else None,
        negative_label=classes[1] if len(classes) > 1 else None,
        lines_count=0,
        plot=True,
    )


if __name__ == "__main__":
    main()
