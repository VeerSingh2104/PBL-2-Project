"""Tabular preprocessing utilities for XAIEvalAgent Phase 1.

Preprocessing is fitted on the training partition only to reduce leakage.
No predictive model is trained or executed here.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target


def prepare_tabular_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    scaling: str = "StandardScaler",
    missing_numeric: str = "median",
    missing_categorical: str = "most_frequent",
    drop_duplicates: bool = True,
    random_state: int = 42,
):
    """Clean, split, fit feature transformations on train, and transform both splits."""
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    if target not in df.columns:
        raise ValueError("Select a valid target column.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test size must be between 10% and 40%.")

    original_rows = len(df)
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    if len(set(work.columns)) != len(work.columns):
        raise ValueError("Column names must be unique after trimming whitespace.")
    target = str(target).strip()
    if target not in work.columns:
        raise ValueError("Target column name changed after trimming whitespace.")

    duplicate_count = int(work.duplicated().sum())
    if drop_duplicates:
        work = work.drop_duplicates().copy()

    missing_target_count = int(work[target].isna().sum())
    work = work.dropna(subset=[target]).copy()
    if len(work) < 3:
        raise ValueError("At least three rows with a non-missing target are required.")

    X = work.drop(columns=[target])
    y = work[target]

    # Phase 1 uses RandomForestClassifier + SHAP/LIME classification metrics.
    # Reject continuous targets here so the model-backed pages cannot fail later
    # with scikit-learn's less helpful "Unknown label type: continuous" error.
    target_kind = type_of_target(y)
    if target_kind not in ("binary", "multiclass"):
        if target_kind == "continuous":
            raise ValueError(
                f"Target '{target}' contains continuous numeric values. "
                "Phase 1 expects a classification target (for example 0/1). "
                "Select the actual class/label column, such as DEATH_EVENT or Credit Risk."
            )
        raise ValueError(
            f"Target '{target}' is not a supported classification target "
            f"(detected as {target_kind}). Select a binary or multiclass label column."
        )

    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    if scaling not in ("StandardScaler", "MinMaxScaler", "None"):
        raise ValueError("Unsupported scaling option.")
    scaler = StandardScaler() if scaling == "StandardScaler" else MinMaxScaler() if scaling == "MinMaxScaler" else "passthrough"

    transformers = []
    if numeric_cols:
        numeric_steps = [("imputer", SimpleImputer(strategy=missing_numeric))]
        if scaling != "None":
            numeric_steps.append(("scaler", scaler))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=missing_categorical)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    transformer = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    stratify = y if y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    X_train_t = transformer.fit_transform(X_train)
    X_test_t = transformer.transform(X_test)
    names = transformer.get_feature_names_out().tolist()
    train = pd.DataFrame(X_train_t, columns=names, index=X_train.index)
    test = pd.DataFrame(X_test_t, columns=names, index=X_test.index)
    train[target] = y_train
    test[target] = y_test

    audit = {
        "input_rows": original_rows,
        "duplicate_rows_found": duplicate_count,
        "duplicate_rows_removed": duplicate_count if drop_duplicates else 0,
        "rows_missing_target_removed": missing_target_count,
        "rows_after_cleaning": len(work),
        "train_rows": len(train),
        "test_rows": len(test),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "output_feature_count": len(names),
        "output_features": names,
        "target": target,
        "target_type": target_kind,
        "target_classes": [str(v) for v in sorted(y.dropna().unique(), key=lambda v: str(v))],
        "target_class_count": int(y.nunique(dropna=True)),
        "scaling": scaling,
        "numeric_imputation": missing_numeric,
        "categorical_imputation": missing_categorical,
        "encoding": "OneHotEncoder(handle_unknown='ignore')",
    }
    return train.reset_index(drop=True), test.reset_index(drop=True), audit
