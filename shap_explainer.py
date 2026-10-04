"""Model-backed SHAP prototype for tabular classification."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


def _class_shap_values(values, class_index, n_features):
    """Normalize SHAP's version-dependent classifier output to (rows, features)."""
    if isinstance(values, list):
        selected = np.asarray(values[min(class_index, len(values) - 1)])
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            # SHAP newer APIs commonly return (rows, features, classes).
            if arr.shape[1] == n_features:
                selected = arr[:, :, min(class_index, arr.shape[2] - 1)]
            # Some versions return (classes, rows, features).
            elif arr.shape[2] == n_features:
                selected = arr[min(class_index, arr.shape[0] - 1), :, :]
            else:
                raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")
        elif arr.ndim == 2:
            selected = arr
        else:
            raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")
    if selected.ndim != 2 or selected.shape[1] != n_features:
        raise ValueError(f"Unexpected SHAP attribution shape: {selected.shape}")
    return selected


def run_shap_experiment(train_df, test_df, target, explain_count=20, seed=42):
    """Train a baseline RF and generate actual SHAP explanations for held-out rows.

    The caller must pass train/test splits already preprocessed without leakage.
    Outputs are experimental for the supplied upload, not universal benchmark results.
    """
    if target not in train_df or target not in test_df:
        raise ValueError("Target column must exist in both training and test data.")
    feature_names = [c for c in train_df.columns if c != target]
    if not feature_names:
        raise ValueError("No model features were found.")
    X_train = train_df[feature_names].astype(float)
    X_test = test_df[feature_names].astype(float).head(int(explain_count))
    y_train = train_df[target]
    if y_train.nunique() < 2:
        raise ValueError("Training target must contain at least two classes.")

    model = RandomForestClassifier(
        n_estimators=120, random_state=int(seed), n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)

    background = X_train.sample(min(40, len(X_train)), random_state=int(seed))
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional",
        model_output="probability"
    )
    started = time.perf_counter()
    raw_values = explainer.shap_values(X_test)
    elapsed = time.perf_counter() - started

    probabilities = model.predict_proba(X_test)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    local_rows = []
    for row_i, (idx, row) in enumerate(X_test.iterrows()):
        class_idx = int(np.argmax(probabilities[row_i]))
        matrix = _class_shap_values(raw_values, class_idx, len(feature_names))
        vals = matrix[row_i]
        for feature, value in zip(feature_names, vals):
            local_rows.append({
                "instance": int(row_i + 1),
                "feature": feature,
                "feature_value": float(row[feature]),
                "shap_value": float(value),
                "abs_shap": float(abs(value)),
                "predicted_class": str(predictions[row_i]),
                "predicted_probability": float(probabilities[row_i, class_idx]),
            })

    local = pd.DataFrame(local_rows)
    global_importance = (local.groupby("feature", as_index=False)["abs_shap"]
                         .mean().rename(columns={"abs_shap": "mean_abs_shap"})
                         .sort_values("mean_abs_shap", ascending=False)
                         .reset_index(drop=True))

    # Faithfulness estimate for first explained instance: attribution mass over
    # nested feature subsets correlated with change in predicted-class probability
    # after replacing selected features with training medians.
    x0 = X_test.iloc[[0]].copy()
    vals0 = _class_shap_values(raw_values, int(np.argmax(probabilities[0])), len(feature_names))[0]
    order = np.argsort(-np.abs(vals0))
    baseline = X_train.median(numeric_only=True).reindex(feature_names).fillna(0.0)
    rng = np.random.default_rng(int(seed))
    subsets, attribution_mass, output_change = [], [], []
    original_class_idx = int(np.argmax(probabilities[0]))
    original_prob = float(probabilities[0, original_class_idx])
    n_features = len(feature_names)
    repeats = min(20, max(5, n_features * 2))
    for _ in range(repeats):
        permutation = rng.permutation(n_features)
        k = int(rng.integers(1, n_features + 1))
        selected = permutation[:k]
        perturbed = x0.copy()
        for feature_idx in selected:
            perturbed.iloc[0, feature_idx] = baseline.iloc[feature_idx]
        changed_prob = float(model.predict_proba(perturbed)[0, original_class_idx])
        attribution_mass.append(float(np.abs(vals0[selected]).sum()))
        output_change.append(abs(original_prob - changed_prob))
        subsets.append(selected)
    if len(set(np.round(attribution_mass, 12))) > 1 and len(set(np.round(output_change, 12))) > 1:
        faithfulness = float(np.corrcoef(attribution_mass, output_change)[0, 1])
    else:
        faithfulness = float("nan")

    abs_sorted = np.sort(np.abs(vals0))
    total = float(abs_sorted.sum())
    if total == 0:
        gini = 0.0
    else:
        n = len(abs_sorted)
        gini = float((2 * np.dot(np.arange(1, n + 1), abs_sorted) /
                      (n * total)) - (n + 1) / n)

    return {
        "model": model,
        "explainer": explainer,
        "local": local,
        "global_importance": global_importance,
        "faithfulness_correlation": faithfulness,
        "sparseness_gini": gini,
        "runtime_seconds": float(elapsed),
        "runtime_per_instance_seconds": float(elapsed / max(1, len(X_test))),
        "explained_count": int(len(X_test)),
        "feature_names": feature_names,
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "baseline": baseline.to_dict(),
    }
