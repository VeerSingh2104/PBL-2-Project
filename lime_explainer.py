"""LIME tabular experiment for XAIEvalAgent."""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def run_lime_experiment(train_df, test_df, target, explain_count=20, seed=42):
    from lime.lime_tabular import LimeTabularExplainer

    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError("Target must be present in train and test data.")
    features = [c for c in train_df.columns if c != target]
    if not features:
        raise ValueError("No features available.")
    X_train = train_df[features].astype(float)
    X_test = test_df[features].astype(float).head(int(explain_count))
    y_train = train_df[target]
    if y_train.nunique() < 2:
        raise ValueError("Training target must contain at least two classes.")

    model = RandomForestClassifier(n_estimators=120, random_state=int(seed),
                                   n_jobs=-1, class_weight="balanced_subsample")
    model.fit(X_train, y_train)
    explainer = LimeTabularExplainer(
        X_train.to_numpy(), feature_names=features,
        class_names=[str(c) for c in model.classes_],
        mode="classification", discretize_continuous=True,
        random_state=int(seed)
    )
    baseline = X_train.median().fillna(0.0)
    rng = np.random.default_rng(int(seed))
    rows, faithfulness = [], []
    total_start = time.perf_counter()
    for row_i, (_, row) in enumerate(X_test.iterrows()):
        exp = explainer.explain_instance(
            row.to_numpy(), model.predict_proba,
            num_features=len(features), num_samples=1000
        )
        probs = model.predict_proba(row.to_frame().T)[0]
        class_idx = int(np.argmax(probs))
        weights = dict(exp.as_map().get(class_idx, []))
        attrs = np.array([float(weights.get(i, 0.0)) for i in range(len(features))])
        for j, name in enumerate(features):
            rows.append({"instance": row_i + 1, "feature": name,
                         "feature_value": float(row.iloc[j]),
                         "lime_value": float(attrs[j]), "abs_lime": float(abs(attrs[j])),
                         "predicted_class": str(model.classes_[class_idx]),
                         "predicted_probability": float(probs[class_idx])})
        masses, changes = [], []
        for _ in range(min(20, max(5, len(features) * 2))):
            selected = rng.permutation(len(features))
            selected = selected[:int(rng.integers(1, len(features) + 1))]
            perturbed = row.to_frame().T.copy()
            for j in selected:
                perturbed.iloc[0, j] = baseline.iloc[j]
            new_prob = float(model.predict_proba(perturbed)[0, class_idx])
            masses.append(float(np.abs(attrs[selected]).sum()))
            changes.append(abs(float(probs[class_idx]) - new_prob))
        if np.std(masses) > 0 and np.std(changes) > 0:
            faithfulness.append(float(np.corrcoef(masses, changes)[0, 1]))

    local = pd.DataFrame(rows)
    importance = (local.groupby("feature", as_index=False)["abs_lime"].mean()
                  .rename(columns={"abs_lime": "mean_abs_lime"})
                  .sort_values("mean_abs_lime", ascending=False).reset_index(drop=True))
    elapsed = time.perf_counter() - total_start
    return {
        "model": model, "local": local, "global_importance": importance,
        "faithfulness_correlation": float(np.nanmean(faithfulness)) if faithfulness else float("nan"),
        "runtime_seconds": float(elapsed),
        "runtime_per_instance_seconds": float(elapsed / max(1, len(X_test))),
        "explained_count": int(len(X_test)), "feature_names": features,
    }
