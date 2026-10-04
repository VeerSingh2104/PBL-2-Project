"""Core logic for XAIEvalAgent's tabular classification prototype."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

METRIC_DIRECTIONS = {
    "Faithfulness Correlation": "higher",
    "Max-Sensitivity": "lower",
    "Local Lipschitz Estimate": "lower",
    "Cross-Explainer Agreement": "higher",
    "Sparseness (Gini)": "higher",
    "Runtime (s)": "lower",
}

@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    feature_names: list[str]
    class_names: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    accuracy: float


def profile_data(df: pd.DataFrame, target: str) -> dict[str, Any]:
    X = df.drop(columns=[target])
    return {
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "numeric_features": int(X.select_dtypes(include=np.number).shape[1]),
        "categorical_features": int(X.select_dtypes(exclude=np.number).shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "target": target,
        "classes": [str(v) for v in pd.Series(df[target].dropna().unique()).tolist()],
        "task": "classification",
        "modality": "tabular",
    }


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def train_models(df: pd.DataFrame, target: str, test_size: float = .25, random_state: int = 42) -> dict[str, TrainedModel]:
    data = df.dropna(subset=[target]).copy()
    X = data.drop(columns=[target])
    y = data[target].astype(str)
    if X.empty or y.nunique() < 2:
        raise ValueError("Select a target with at least two classes and one feature.")
    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two rows for a stratified split.")
    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y), index=y.index)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    trained = {}
    for name, estimator in [
        ("Random Forest", RandomForestClassifier(n_estimators=120, random_state=random_state, class_weight="balanced")),
        ("Neural Network (MLP)", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True, random_state=random_state)),
    ]:
        pipe = Pipeline([("preprocess", _make_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        trained[name] = TrainedModel(
            name, pipe, X.columns.tolist(), encoder.classes_.tolist(),
            X_train, X_test, y_test, float(accuracy_score(y_test, pred))
        )
    return trained


def _positive_probability(model: TrainedModel, rows: pd.DataFrame) -> np.ndarray:
    probs = model.pipeline.predict_proba(rows)
    if probs.shape[1] == 2:
        return probs[:, 1]
    return np.max(probs, axis=1)


def _encoded_features(model: TrainedModel, rows: pd.DataFrame) -> np.ndarray:
    return model.pipeline.named_steps["preprocess"].transform(rows)


def explain_instance(model: TrainedModel, instance: pd.DataFrame, methods: list[str]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return original-column attribution vectors and wall-clock runtimes."""
    explanations, runtimes = {}, {}
    background = model.X_train.sample(min(40, len(model.X_train)), random_state=42)
    for method in methods:
        start = time.perf_counter()
        if method == "LIME":
            from lime.lime_tabular import LimeTabularExplainer
            prep = model.pipeline.named_steps["preprocess"]
            xbg = prep.transform(background)
            xi = prep.transform(instance)
            names = list(prep.get_feature_names_out())
            explainer = LimeTabularExplainer(
                xbg, feature_names=names, class_names=model.class_names,
                mode="classification", discretize_continuous=False, random_state=42
            )
            def predict_encoded(z):
                # LIME supplies transformed columns; invert is unavailable, so explain raw tabular
                # features using a model-agnostic perturbation fallback below.
                raise NotImplementedError
            # LIME on original columns with categorical metadata and a raw-row prediction adapter.
            numeric = instance.select_dtypes(include=np.number).columns.tolist()
            categorical = [c for c in model.feature_names if c not in numeric]
            raw = model.X_train.copy()
            cat_idx = [model.feature_names.index(c) for c in categorical]
            lime = LimeTabularExplainer(
                raw.to_numpy(dtype=object), feature_names=model.feature_names,
                class_names=model.class_names, categorical_features=cat_idx,
                mode="classification", discretize_continuous=True, random_state=42
            )
            def predict_raw(arr):
                frame = pd.DataFrame(arr, columns=model.feature_names)
                for c in categorical:
                    frame[c] = frame[c].astype(str)
                for c in numeric:
                    frame[c] = pd.to_numeric(frame[c], errors="coerce")
                return model.pipeline.predict_proba(frame)
            exp = lime.explain_instance(instance.iloc[0].to_numpy(dtype=object), predict_raw, num_features=len(model.feature_names), num_samples=500)
            vals = dict(exp.as_map().get(int(np.argmax(model.pipeline.predict_proba(instance)[0])), []))
            attr = np.zeros(len(model.feature_names))
            for idx, val in vals.items():
                if idx < len(attr): attr[idx] = val
            explanations[method] = attr
        elif method == "SHAP":
            import shap
            # KernelExplainer keeps this prototype model-agnostic for mixed tabular inputs.
            def predict_raw(arr):
                frame = pd.DataFrame(arr, columns=model.feature_names)
                for c in model.X_train.columns:
                    if c in model.X_train.select_dtypes(exclude=np.number).columns:
                        frame[c] = frame[c].astype(str)
                    else:
                        frame[c] = pd.to_numeric(frame[c], errors="coerce")
                return model.pipeline.predict_proba(frame)
            background_raw = background.to_numpy(dtype=object)
            explainer = shap.KernelExplainer(predict_raw, background_raw)
            sv = explainer.shap_values(instance.to_numpy(dtype=object), nsamples=100)
            if isinstance(sv, list):
                arr = np.asarray(sv[-1])[0]
            else:
                arr = np.asarray(sv)
                if arr.ndim == 3: arr = arr[0, :, -1]
                elif arr.ndim == 2: arr = arr[0]
            explanations[method] = np.asarray(arr, dtype=float).reshape(-1)[:len(model.feature_names)]
        else:
            continue
        runtimes[method] = time.perf_counter() - start
    return explanations, runtimes


def _gini(values: np.ndarray) -> float:
    x = np.sort(np.abs(np.asarray(values, dtype=float)))
    if len(x) == 0 or np.sum(x) == 0: return 0.0
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1) * x)) / (n * np.sum(x))) - (n + 1) / n)


def evaluate_attributions(model: TrainedModel, instance: pd.DataFrame, attributions: dict[str, np.ndarray], runtimes: dict[str, float], perturbations: int = 12) -> pd.DataFrame:
    rows, base = [], _positive_probability(model, instance)[0]
    names = model.feature_names
    for method, attr in attributions.items():
        attr = np.nan_to_num(np.asarray(attr, dtype=float))
        order = np.argsort(np.abs(attr))[::-1]
        faith_x, faith_y = [], []
        for count in range(1, len(names) + 1):
            chosen = [names[i] for i in order[:count]]
            changed = instance.copy()
            changed.loc[:, chosen] = model.X_train[chosen].median(numeric_only=True).reindex(chosen).fillna(0).to_dict() if all(pd.api.types.is_numeric_dtype(model.X_train[c]) for c in chosen) else changed[chosen]
            # Replace individual features with representative training values.
            for c in chosen:
                if not pd.api.types.is_numeric_dtype(model.X_train[c]):
                    changed.loc[:, c] = model.X_train[c].mode(dropna=True).iloc[0]
                else:
                    changed.loc[:, c] = model.X_train[c].median()
            delta = abs(base - _positive_probability(model, changed)[0])
            faith_x.append(float(np.sum(np.abs(attr[order[:count]]))))
            faith_y.append(float(delta))
        corr = float(np.corrcoef(faith_x, faith_y)[0, 1]) if len(faith_x) > 1 and np.std(faith_x) > 0 and np.std(faith_y) > 0 else 0.0

        sens, lips = [], []
        for _ in range(perturbations):
            changed = instance.copy()
            for c in names:
                if pd.api.types.is_numeric_dtype(model.X_train[c]):
                    sd = float(model.X_train[c].std() or 0)
                    changed.loc[:, c] = changed[c] + np.random.default_rng().normal(0, max(sd * .01, 1e-8))
                else:
                    if np.random.default_rng().random() < .02:
                        changed.loc[:, c] = model.X_train[c].mode(dropna=True).iloc[0]
            # Attribution perturbation proxy uses input-induced prediction difference normalized by feature distance.
            d = abs(base - _positive_probability(model, changed)[0])
            sens.append(float(np.linalg.norm(attr) * d))
            distance = max(float(np.linalg.norm(_encoded_features(model, instance) - _encoded_features(model, changed))), 1e-8)
            lips.append(float(d / distance))
        rows.append({
            "Explainer": method,
            "Faithfulness Correlation": corr,
            "Max-Sensitivity": max(sens, default=0.0),
            "Local Lipschitz Estimate": float(np.mean(lips)) if lips else 0.0,
            "Sparseness (Gini)": _gini(attr),
            "Runtime (s)": float(runtimes.get(method, 0.0)),
            "Attributions": attr,
        })
    if len(rows) > 1:
        ranks = [pd.Series(np.abs(r["Attributions"])).rank(method="average").to_numpy() for r in rows]
        for i, row in enumerate(rows):
            agreements = []
            for j, other in enumerate(ranks):
                if i == j: continue
                tau = kendalltau(ranks[i], other).statistic
                agreements.append(float(tau) if np.isfinite(tau) else 0.0)
            row["Cross-Explainer Agreement"] = float(np.mean(agreements)) if agreements else 1.0
    else:
        for row in rows: row["Cross-Explainer Agreement"] = 1.0
    return pd.DataFrame(rows)


def rank_explainers(metrics: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if metrics.empty: return metrics
    out = metrics.copy()
    for name, direction in METRIC_DIRECTIONS.items():
        values = pd.to_numeric(out[name], errors="coerce").fillna(0).astype(float)
        lo, hi = float(values.min()), float(values.max())
        normalized = (values - lo) / (hi - lo) if hi > lo else pd.Series(np.ones(len(values)), index=values.index)
        out[name + " (normalized)"] = normalized if direction == "higher" else 1 - normalized
    active = {k: max(0.0, float(v)) for k, v in weights.items() if k in METRIC_DIRECTIONS}
    total = sum(active.values())
    if total <= 0: raise ValueError("At least one metric weight must be positive.")
    out["XAIScore"] = sum(out[k + " (normalized)"] * (v / total) for k, v in active.items())
    return out.sort_values("XAIScore", ascending=False).reset_index(drop=True)
