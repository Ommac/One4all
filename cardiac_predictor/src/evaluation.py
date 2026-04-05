"""
Leakage and duplication aware evaluation utilities for the cardiac arrhythmia models.

This module is intentionally separate from the Flask routes and inference flow.
It investigates dataset-level leakage and duplication, then evaluates the saved
PyTorch models using a reproducible split that keeps very high-similarity samples
on the same side of the train/test boundary.
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, normalize

try:
    from src.ensemble import ensemble_predict
    from src.model_definitions import ECGCNNModel, ECGLSTMModel
    from src.preprocessing import load_dataset
except ModuleNotFoundError:
    from ensemble import ensemble_predict
    from model_definitions import ECGCNNModel, ECGLSTMModel
    from preprocessing import load_dataset


DATASET_PATH = "data/ecg_dataset.csv"
CNN_MODEL_PATH = "models/cnn_model.h5"
LSTM_MODEL_PATH = "models/lstm_model.h5"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SIGNAL_LENGTH = 2500
LSTM_LENGTH = 500
DOWNsample_FACTOR = 5
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_LABELS = {0, 1, 2}
LABEL_TO_INT = {"AFib": 0, "Normal": 1, "VFib": 2}
SIMILARITY_GROUP_THRESHOLD = 0.998
STRICT_DIAGNOSTIC_THRESHOLD = 0.995
NEAR_DUPLICATE_SAMPLE_SIZE = 100


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def check_duplicates(X: np.ndarray) -> int:
    seen = set()
    duplicates = 0

    for x in X:
        key = tuple(np.round(x, 5))
        if key in seen:
            duplicates += 1
        seen.add(key)

    print("Duplicate samples:", duplicates)
    return duplicates


def check_overlap(X_train: np.ndarray, X_test: np.ndarray) -> int:
    train_set = {tuple(np.round(x, 5)) for x in X_train}
    test_set = {tuple(np.round(x, 5)) for x in X_test}

    overlap = train_set.intersection(test_set)
    print("Train-Test Overlap:", len(overlap))
    return len(overlap)


def check_similarity(X_train: np.ndarray, X_test: np.ndarray) -> int:
    sample_size = min(NEAR_DUPLICATE_SAMPLE_SIZE, len(X_train), len(X_test))
    sim = cosine_similarity(X_test[:sample_size], X_train[:sample_size])
    high_sim = int((sim > 0.98).sum())
    print("Highly similar pairs:", high_sim)
    return high_sim


def _signal_hash(signal: np.ndarray) -> int:
    signal = np.asarray(signal)
    return hash(tuple(signal[:100].tolist() + signal[-100:].tolist()))


def _validate_no_overlap(X_train: np.ndarray, X_test: np.ndarray) -> None:
    train_hashes = {_signal_hash(signal) for signal in X_train}
    test_hashes = {_signal_hash(signal) for signal in X_test}
    overlap = train_hashes & test_hashes
    if overlap:
        raise ValueError(f"Data leakage detected: {len(overlap)} overlapping samples between train/test")


def _load_trained_models() -> Tuple[ECGCNNModel, ECGLSTMModel]:
    cnn_model = ECGCNNModel(input_length=SIGNAL_LENGTH).to(DEVICE)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    cnn_model.eval()

    lstm_model = ECGLSTMModel(input_length=LSTM_LENGTH).to(DEVICE)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE, weights_only=True))
    lstm_model.eval()

    return cnn_model, lstm_model


def _compute_similarity_counts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    thresholds: Tuple[float, ...] = (0.98, 0.99, 0.995, 0.998),
) -> Dict[float, Dict[str, float]]:
    X_train_norm = normalize(X_train.astype(np.float32))
    X_test_norm = normalize(X_test.astype(np.float32))

    counts = {threshold: 0 for threshold in thresholds}
    same_label_counts = {threshold: 0 for threshold in thresholds}
    max_similarities = []

    for start in range(0, len(X_test_norm), BATCH_SIZE):
        sims = X_test_norm[start:start + BATCH_SIZE] @ X_train_norm.T
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(sims.shape[0]), best_idx]
        max_similarities.extend(best_sim.tolist())
        nearest_labels = y_train[best_idx]
        y_batch = y_test[start:start + BATCH_SIZE]

        for threshold in thresholds:
            mask = best_sim > threshold
            counts[threshold] += int(mask.sum())
            same_label_counts[threshold] += int(np.sum(mask & (nearest_labels == y_batch)))

    return {
        threshold: {
            "count": counts[threshold],
            "same_label": same_label_counts[threshold],
        }
        for threshold in thresholds
    } | {
        "summary": {
            "max_similarity_min": float(np.min(max_similarities)),
            "max_similarity_mean": float(np.mean(max_similarities)),
            "max_similarity_median": float(np.median(max_similarities)),
            "max_similarity_max": float(np.max(max_similarities)),
        }
    }


def _print_similarity_counts(title: str, stats: Dict[float, Dict[str, float]]) -> None:
    print(title)
    for threshold, values in stats.items():
        if threshold == "summary":
            continue
        print(
            f"Nearest-train cosine > {threshold}: {values['count']} "
            f"(same label: {values['same_label']})"
        )
    summary = stats["summary"]
    print("Nearest-train cosine min:", summary["max_similarity_min"])
    print("Nearest-train cosine mean:", summary["max_similarity_mean"])
    print("Nearest-train cosine median:", summary["max_similarity_median"])
    print("Nearest-train cosine max:", summary["max_similarity_max"])


def _build_similarity_groups(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = SIMILARITY_GROUP_THRESHOLD,
) -> Tuple[np.ndarray, Dict[int, int], Dict[str, object]]:
    """
    Build proxy record IDs by clustering same-label signals with extremely high
    cosine similarity on the LSTM-view signal (downsampled to 500 points).

    Using a stricter threshold than the diagnostic 0.995 keeps class balance
    viable while still preventing near-identical synthetic windows from crossing
    the split boundary.
    """
    X_downsampled = normalize(X[:, ::DOWNsample_FACTOR].astype(np.float32))
    uf = UnionFind(len(X_downsampled))
    pair_counts: Dict[int, int] = {}

    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        similarities = X_downsampled[idxs] @ X_downsampled[idxs].T
        rows, cols = np.where(np.triu(similarities > threshold, k=1))
        pair_counts[int(label)] = int(len(rows))

        for row, col in zip(rows.tolist(), cols.tolist()):
            uf.union(int(idxs[row]), int(idxs[col]))

    groups = np.array([uf.find(i) for i in range(len(X_downsampled))], dtype=np.int64)
    unique_groups, group_sizes = np.unique(groups, return_counts=True)

    diagnostics = {
        "num_groups": int(len(unique_groups)),
        "largest_groups": sorted(group_sizes.tolist(), reverse=True)[:10],
        "pair_counts": pair_counts,
        "threshold": threshold,
    }

    return groups, pair_counts, diagnostics


def _labelwise_group_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_parts = []
    test_parts = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        label_groups = groups[label_indices]
        splitter = GroupShuffleSplit(test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_local, test_local = next(
            splitter.split(label_indices, y[label_indices], groups=label_groups)
        )
        train_parts.append(label_indices[train_local])
        test_parts.append(label_indices[test_local])

    train_idx = np.sort(np.concatenate(train_parts))
    test_idx = np.sort(np.concatenate(test_parts))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _prepare_evaluation_data(
    dataset_path: str = DATASET_PATH,
    record_ids: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, int],
    Dict[str, object],
]:
    X, y, label_encoder = load_dataset(dataset_path)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D signal matrix, got shape {X.shape}")
    if X.shape[1] != SIGNAL_LENGTH:
        raise ValueError(
            f"Expected signals with length {SIGNAL_LENGTH}, got length {X.shape[1]}"
        )

    diagnostics: Dict[str, object] = {
        "duplicates": check_duplicates(X),
    }

    random_X_train, random_X_test, random_y_train, random_y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    diagnostics["random_split_overlap"] = check_overlap(random_X_train, random_X_test)
    diagnostics["random_sample_similarity"] = check_similarity(random_X_train, random_X_test)
    diagnostics["random_similarity_stats"] = _compute_similarity_counts(
        random_X_train[:, ::DOWNsample_FACTOR],
        random_X_test[:, ::DOWNsample_FACTOR],
        random_y_train,
        random_y_test,
    )
    diagnostics["random_train_distribution"] = Counter(random_y_train)
    diagnostics["random_test_distribution"] = Counter(random_y_test)

    if record_ids is not None:
        split_groups = np.asarray(record_ids)
        split_mode = "record_id_group_split"
        X_train, X_test, y_train, y_test = _labelwise_group_split(X, y, split_groups)
        diagnostics["group_build"] = {
            "source": "provided record_ids",
        }
    else:
        split_groups, pair_counts, group_build = _build_similarity_groups(X, y)
        split_mode = "similarity_group_split"
        X_train, X_test, y_train, y_test = _labelwise_group_split(X, y, split_groups)
        diagnostics["group_build"] = group_build | {
            "pair_counts": pair_counts,
        }

    diagnostics["split_mode"] = split_mode
    diagnostics["group_split_overlap"] = check_overlap(X_train, X_test)
    diagnostics["group_train_distribution"] = Counter(y_train)
    diagnostics["group_test_distribution"] = Counter(y_test)
    diagnostics["group_similarity_stats"] = _compute_similarity_counts(
        X_train[:, ::DOWNsample_FACTOR],
        X_test[:, ::DOWNsample_FACTOR],
        y_train,
        y_test,
    )

    _validate_no_overlap(X_train, X_test)

    if not set(np.unique(y_test)).issubset(VALID_LABELS):
        raise ValueError(f"Unexpected labels in y_test: {set(np.unique(y_test))}")

    label_to_int = {label: idx for idx, label in enumerate(label_encoder.classes_)}

    return X_train, X_test, y_train, y_test, list(label_encoder.classes_), label_to_int, diagnostics


def _fit_scalers(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Tuple[int, ...]]]:
    scaler_cnn = StandardScaler()
    X_train_cnn = scaler_cnn.fit_transform(X_train)
    X_test_cnn = scaler_cnn.transform(X_test)

    if scaler_cnn.n_samples_seen_ != len(X_train):
        raise ValueError(
            "Scaler leakage detected for CNN preprocessing: scaler was not fit only on X_train"
        )

    X_train_lstm_raw = X_train[:, ::DOWNsample_FACTOR]
    X_test_lstm_raw = X_test[:, ::DOWNsample_FACTOR]

    scaler_lstm = StandardScaler()
    X_train_lstm = scaler_lstm.fit_transform(X_train_lstm_raw)
    X_test_lstm = scaler_lstm.transform(X_test_lstm_raw)

    if scaler_lstm.n_samples_seen_ != len(X_train_lstm_raw):
        raise ValueError(
            "Scaler leakage detected for LSTM preprocessing: scaler was not fit only on X_train"
        )

    shape_info = {
        "cnn_single": (1, SIGNAL_LENGTH, 1),
        "lstm_single": (1, LSTM_LENGTH, 1),
    }

    return X_train_cnn, X_test_cnn, X_train_lstm, X_test_lstm, shape_info


def _batched_predict(
    model: torch.nn.Module,
    X_input: np.ndarray,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    probs_list = []
    preds_list = []

    with torch.no_grad():
        for start in range(0, len(X_input), BATCH_SIZE):
            batch = torch.FloatTensor(X_input[start:start + BATCH_SIZE]).unsqueeze(2).to(DEVICE)
            outputs = model(batch)
            probs = outputs.detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            if start == 0 and model_name == "LSTM":
                print("LSTM input shape:", tuple(batch.shape))
                print("LSTM raw output:", probs[0].tolist())

            probs_list.append(probs)
            preds_list.append(preds)

    return np.vstack(probs_list), np.concatenate(preds_list)


def _build_ensemble_predictions(
    cnn_preds: np.ndarray,
    lstm_preds: np.ndarray,
    cnn_probs: np.ndarray,
    label_to_int: Dict[str, int],
) -> np.ndarray:
    ensemble_preds = []

    for cnn_pred, lstm_pred, sample_probs in zip(cnn_preds, lstm_preds, cnn_probs):
        result = ensemble_predict(int(cnn_pred), int(lstm_pred), sample_probs.tolist())
        ensemble_preds.append(label_to_int[result["prediction"]])

    return np.asarray(ensemble_preds, dtype=np.int64)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, object]:
    return {
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def _print_model_report(name: str, metrics: Dict[str, object], preds: np.ndarray) -> None:
    print(f"=== {name} ===")
    print("F1 weighted:", metrics["f1_weighted"])
    print("F1 macro:", metrics["f1_macro"])
    print(metrics["classification_report"])
    print(metrics["confusion_matrix"])
    print(f"{name} unique predictions:", set(np.asarray(preds).tolist()))
    if len(set(np.asarray(preds).tolist())) == 1:
        print(f"WARNING: {name} predicted only one class.")


def evaluate_models(
    X_test_cnn: np.ndarray,
    X_test_lstm: np.ndarray,
    y_true: np.ndarray,
    cnn_model: torch.nn.Module,
    lstm_model: torch.nn.Module,
    class_names: List[str],
    label_to_int: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    label_to_int = label_to_int or LABEL_TO_INT

    cnn_probs, cnn_preds = _batched_predict(cnn_model, X_test_cnn, "CNN")
    _, lstm_preds = _batched_predict(lstm_model, X_test_lstm, "LSTM")
    ensemble_preds = _build_ensemble_predictions(cnn_preds, lstm_preds, cnn_probs, label_to_int)

    cnn_metrics = _compute_metrics(y_true, cnn_preds, class_names)
    lstm_metrics = _compute_metrics(y_true, lstm_preds, class_names)
    ensemble_metrics = _compute_metrics(y_true, ensemble_preds, class_names)

    return {
        "cnn_preds": cnn_preds,
        "lstm_preds": lstm_preds,
        "ensemble_preds": ensemble_preds,
        "cnn_metrics": cnn_metrics,
        "lstm_metrics": lstm_metrics,
        "ensemble_metrics": ensemble_metrics,
    }


def run_clean_evaluation(
    dataset_path: str = DATASET_PATH,
    record_ids: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Run the full evaluation pipeline with duplication/leakage diagnostics.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")
    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(f"LSTM model not found: {LSTM_MODEL_PATH}")

    X_train, X_test, y_train, y_test, class_names, label_to_int, diagnostics = _prepare_evaluation_data(
        dataset_path=dataset_path,
        record_ids=record_ids,
    )
    X_train_cnn, X_test_cnn, X_train_lstm, X_test_lstm, shape_info = _fit_scalers(X_train, X_test)

    cnn_model, lstm_model = _load_trained_models()
    results = evaluate_models(
        X_test_cnn=X_test_cnn,
        X_test_lstm=X_test_lstm,
        y_true=y_test,
        cnn_model=cnn_model,
        lstm_model=lstm_model,
        class_names=class_names,
        label_to_int=label_to_int,
    )

    print("=== DATASET DIAGNOSTICS ===")
    print("Dataset:", dataset_path)
    print("Split mode:", diagnostics["split_mode"])
    print("Device:", DEVICE)
    print("Train distribution:", diagnostics["group_train_distribution"])
    print("Test distribution:", diagnostics["group_test_distribution"])
    print("Random split distribution:", diagnostics["random_test_distribution"])
    print("CNN input shape:", shape_info["cnn_single"])
    print("LSTM input shape:", shape_info["lstm_single"])
    print("True labels:", set(np.asarray(y_test).tolist()))
    print("Group build diagnostics:", diagnostics["group_build"])

    print("=== RANDOM SPLIT DIAGNOSTICS ===")
    print("Duplicate samples:", diagnostics["duplicates"])
    print("Train-Test Overlap:", diagnostics["random_split_overlap"])
    print("Highly similar pairs:", diagnostics["random_sample_similarity"])
    _print_similarity_counts(
        "Random split nearest-neighbor similarity:",
        diagnostics["random_similarity_stats"],
    )

    print("=== FIXED SPLIT DIAGNOSTICS ===")
    print("Train-Test Overlap:", diagnostics["group_split_overlap"])
    _print_similarity_counts(
        "Group-aware split nearest-neighbor similarity:",
        diagnostics["group_similarity_stats"],
    )

    print("CNN predictions:", set(np.asarray(results["cnn_preds"]).tolist()))
    print("LSTM predictions:", set(np.asarray(results["lstm_preds"]).tolist()))
    print("True labels:", set(np.asarray(y_test).tolist()))

    _print_model_report("CNN", results["cnn_metrics"], results["cnn_preds"])
    _print_model_report("LSTM", results["lstm_metrics"], results["lstm_preds"])
    _print_model_report("ENSEMBLE", results["ensemble_metrics"], results["ensemble_preds"])

    strict_group_counts = diagnostics["group_similarity_stats"][STRICT_DIAGNOSTIC_THRESHOLD]["count"]
    if strict_group_counts > 0:
        print(
            "WARNING: High similarity remains above 0.995 even after group-aware splitting. "
            "This dataset appears to have limited source diversity, especially in VFib."
        )

    if diagnostics["group_similarity_stats"][SIMILARITY_GROUP_THRESHOLD]["count"] == 0:
        print(
            f"INFO: Similarity-group split removed all nearest-neighbor matches above "
            f"{SIMILARITY_GROUP_THRESHOLD}."
        )

    if results["lstm_metrics"]["f1_macro"] < 0.4:
        print("WARNING: LSTM predictions remain collapsed; inspect saved weights and training alignment.")

    if results["cnn_metrics"]["f1_weighted"] >= 0.99:
        print(
            "INFO: CNN F1 is still near 1.0 after leakage diagnostics. "
            "That now points more strongly to dataset simplicity / synthetic separability than split leakage."
        )

    return {
        "class_names": class_names,
        "y_test": y_test,
        "diagnostics": diagnostics,
        **results,
    }


if __name__ == "__main__":
    run_clean_evaluation()
