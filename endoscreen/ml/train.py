"""CRISP-DM modelling track: clean -> prepare -> model -> evaluate -> export.

Trains and compares Logistic Regression, Random Forest and Support Vector
Machine on the hormonal health dataset, tests the research hypothesis and
exports every fitted pipeline plus a metrics report.

Run:  python -m endoscreen.ml.train --data data/hormonal_health_dataset.csv
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .. import schema as S

SEED = 42
METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc"]
ALGORITHMS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
}
DEFAULT_THRESHOLDS = {"moderate": 0.35, "high": 0.65}


class DatasetError(ValueError):
    """Raised when an uploaded dataset cannot be used for training."""


# --------------------------------------------------------------------------
# 1. Data understanding and cleaning
# --------------------------------------------------------------------------
def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicates, unusable labels and impossible values.

    Returns the cleaned frame and a report of what was done, shown on the
    model performance page so the preprocessing is auditable.
    """
    missing_cols = [c for c in S.DATASET_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise DatasetError("Dataset is missing required columns: " + ", ".join(missing_cols))
    df = raw[S.DATASET_COLUMNS].copy()
    report = {"rows_raw": int(len(df))}

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    df[S.LABEL_COLUMN] = df[S.LABEL_COLUMN].astype(str).str.strip()
    known = df[S.LABEL_COLUMN].isin(S.CLASSES)
    report["unknown_labels_removed"] = int((~known).sum())
    df = df[known]

    out_of_range = 0
    for col, (_label, _unit, lo, hi, *_rest) in S.CORE_NUMERIC.items():
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.notna() & ~values.between(lo, hi)
        out_of_range += int(bad.sum())
        df[col] = values.mask(bad)
    for col, (_label, _unit, lo, hi, *_rest) in S.LABS.items():
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.notna() & ~values.between(lo, hi)
        out_of_range += int(bad.sum())
        df[col] = values.mask(bad)
    report["out_of_range_values_blanked"] = out_of_range

    # A record with no usable age, weight or height cannot be screened.
    essential = df[["age", "weight_kg", "height_cm"]].notna().all(axis=1)
    report["rows_dropped_missing_essentials"] = int((~essential).sum())
    df = df[essential].reset_index(drop=True)

    report["rows_clean"] = int(len(df))
    report["missing_by_column"] = {
        c: int(v) for c, v in df[S.RAW_COLUMNS].isna().sum().items() if v > 0}
    report["class_distribution"] = {c: int((df[S.LABEL_COLUMN] == c).sum()) for c in S.CLASSES}
    smallest = min(report["class_distribution"].values())
    if smallest < 20:
        raise DatasetError("Every class needs at least 20 clean records to train and "
                           f"cross-validate; the smallest class has {smallest}.")
    return df, report


# --------------------------------------------------------------------------
# 2. Pipelines
# --------------------------------------------------------------------------
def _pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),         # missing values
        ("scale", StandardScaler()),                          # z-score normalisation
        ("select", SelectKBest(f_classif, k="all")),          # feature selection
        ("clf", classifier),
    ])


def candidates(fast: bool = False) -> dict:
    k_grid = [24, "all"] if not fast else ["all"]
    return {
        "logistic_regression": (
            _pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            {"select__k": k_grid, "clf__C": [0.1, 1, 10] if not fast else [1]},
        ),
        "random_forest": (
            _pipeline(RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
            {"select__k": k_grid,
             "clf__n_estimators": [300] if not fast else [120],
             "clf__max_depth": [None, 14] if not fast else [14],
             "clf__min_samples_leaf": [1, 3] if not fast else [2]},
        ),
        "svm": (
            _pipeline(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
            {"select__k": k_grid,
             "clf__C": [1, 10] if not fast else [3],
             "clf__gamma": ["scale", 0.01] if not fast else ["scale"]},
        ),
    }


# --------------------------------------------------------------------------
# 3. Evaluation helpers
# --------------------------------------------------------------------------
def _evaluate(model, X, y, classes, thresholds=None) -> dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=classes, zero_division=0)
    normal_idx = list(model.classes_).index(S.NORMAL)
    p_imbalance = 1 - proba[:, normal_idx]
    y_binary = (y != S.NORMAL).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, p_imbalance)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 60).round().astype(int))
    flagged = p_imbalance >= thresholds["moderate"]
    high = p_imbalance >= thresholds["high"]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=model.classes_)),
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "per_class": [
            {"label": c, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for c, p, r, f, s in zip(classes, precision, recall, f1, support)],
        "screening": {   # any imbalance vs none, at the 'moderate risk' flag threshold
            "auc": float(roc_auc_score(y_binary, p_imbalance)),
            "sensitivity": float(flagged[y_binary == 1].mean()),
            "specificity": float((~flagged)[y_binary == 0].mean()),
            "high_band_ppv": float(y_binary[high].mean()) if high.any() else None,
            "low_band_npv": float(1 - y_binary[~flagged].mean()) if (~flagged).any() else None,
            "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        },
    }


def derive_thresholds(y_binary, p_imbalance) -> dict:
    """Risk-band cut-offs from out-of-fold training predictions (never the test set).

    moderate: the highest cut-off that still flags at least 95% of true cases,
              because in screening a missed case costs more than a referral.
    high:     the lowest cut-off at which at least 95% of healthy people fall below.
    """
    fpr, tpr, thr = roc_curve(y_binary, p_imbalance)
    ok = np.where(tpr >= 0.95)[0]
    moderate = float(np.clip(thr[ok[0]] if len(ok) else 0.35, 0.15, 0.50))
    ok = np.where(fpr <= 0.05)[0]
    high = float(np.clip(thr[ok[-1]] if len(ok) else 0.65, moderate + 0.10, 0.90))
    return {"moderate": round(moderate, 2), "high": round(high, 2)}


def _mask_labs(X: pd.DataFrame) -> pd.DataFrame:
    """Simulate a facility with no laboratory at all (symptoms and vitals only)."""
    X = X.copy()
    X[S.LAB_FEATURES] = np.nan
    X[S.MISSING_FLAGS] = 1.0
    return X


def _hypothesis_test(cv_scores: dict) -> dict:
    """Friedman test across the three algorithms on matched CV folds (H0 vs H1)."""
    names = list(cv_scores)
    result = {"alpha": 0.05, "folds": len(next(iter(cv_scores.values()))["f1"]), "metrics": {}}
    for metric in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        samples = [cv_scores[n][metric] for n in names]
        statistic, p_value = stats.friedmanchisquare(*samples)
        pairs, raw = [], []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = np.array(samples[i]) - np.array(samples[j])
                p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(samples[i], samples[j]).pvalue)
                raw.append(p)
                pairs.append({"a": names[i], "b": names[j], "mean_diff": float(diff.mean())})
        order = np.argsort(raw)                       # Holm step-down correction
        adjusted, running = [0.0] * len(raw), 0.0
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, raw[idx] * (len(raw) - rank)))
            adjusted[idx] = running
        for pair, p, adj in zip(pairs, raw, adjusted):
            pair.update({"p_value": p, "p_holm": adj, "significant": bool(adj < 0.05)})
        result["metrics"][metric] = {
            "statistic": float(statistic), "p_value": float(p_value),
            "reject_h0": bool(p_value < 0.05), "pairwise": pairs}
    rejected = [m for m, r in result["metrics"].items() if r["reject_h0"]]
    result["reject_h0"] = bool(rejected)
    result["rejected_on"] = rejected
    return result


# --------------------------------------------------------------------------
# 4. Orchestration
# --------------------------------------------------------------------------
def train_all(csv_path, out_dir, version: str = "v1", fast: bool = False, log=print) -> dict:
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        raw = pd.read_csv(csv_path, keep_default_na=False, na_values=["", "NA", "NaN", "nan"])
    except Exception as exc:  # unreadable upload
        raise DatasetError(f"Could not read the CSV file: {exc}") from exc
    df, cleaning = clean_dataset(raw)
    log(f"Clean rows: {cleaning['rows_clean']} of {cleaning['rows_raw']}")

    X = S.build_features(df)
    y = df[S.LABEL_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    inner = StratifiedKFold(n_splits=5 if not fast else 3, shuffle=True, random_state=SEED)
    outer = StratifiedKFold(n_splits=10 if not fast else 5, shuffle=True, random_state=SEED + 1)

    models, cv_scores, fitted = {}, {}, {}
    for key, (pipe, grid) in candidates(fast).items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        # 10-fold out-of-fold predictions: reliability check and hypothesis-test samples
        oof = cross_val_predict(best, X_train, y_train, cv=outer, method="predict_proba", n_jobs=-1)
        labels = np.array(sorted(y_train.unique()))
        cv_scores[key] = {m: [] for m in METRICS}
        for _fit_idx, idx in outer.split(X_train, y_train):
            y_f, p_f = y_train.iloc[idx], oof[idx]
            pred_f = labels[p_f.argmax(axis=1)]
            cv_scores[key]["accuracy"].append(float(accuracy_score(y_f, pred_f)))
            cv_scores[key]["precision"].append(float(precision_score(y_f, pred_f, average="macro", zero_division=0)))
            cv_scores[key]["recall"].append(float(recall_score(y_f, pred_f, average="macro", zero_division=0)))
            cv_scores[key]["f1"].append(float(f1_score(y_f, pred_f, average="macro", zero_division=0)))
            cv_scores[key]["auc_roc"].append(float(roc_auc_score(y_f, p_f, multi_class="ovr", labels=labels)))
        thresholds = derive_thresholds((y_train != S.NORMAL).astype(int),
                                       1 - oof[:, list(labels).index(S.NORMAL)])
        test = _evaluate(best, X_test, y_test, S.CLASSES, thresholds)
        no_labs = _evaluate(best, _mask_labs(X_test), y_test, S.CLASSES, thresholds)
        selected = [f for f, keep in zip(S.FEATURES, best.named_steps["select"].get_support()) if keep]
        models[key] = {
            "name": ALGORITHMS[key],
            "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                            for k, v in search.best_params_.items()},
            "cv": {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in cv_scores[key].items()},
            "test": test,
            "risk_thresholds": thresholds,
            "test_without_labs": {k: no_labs[k] for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]}
                                 | {"screening": {k: no_labs["screening"][k] for k in ["auc", "sensitivity", "specificity"]}},
            "n_features_selected": len(selected),
            "selected_features": selected,
            "file": f"{key}.joblib",
            "train_seconds": round(time.time() - t0, 1),
        }
        fitted[key] = best
        joblib.dump(best, out_dir / f"{key}.joblib", compress=3)
        log(f"{ALGORITHMS[key]}: CV F1 {models[key]['cv']['f1']['mean']:.3f}, "
            f"test accuracy {test['accuracy']:.3f}, AUC {test['auc_roc']:.3f}")

    # Model selection uses cross-validation on the training split only, so the
    # held-out test set stays an unbiased estimate (statistical learning theory).
    best_key = max(models, key=lambda k: models[k]["cv"]["f1"]["mean"])

    importance = permutation_importance(
        fitted[best_key], X_test, y_test, scoring="f1_macro", n_repeats=8, random_state=SEED, n_jobs=-1)
    ranked = sorted(zip(S.FEATURES, importance.importances_mean), key=lambda t: -t[1])
    f_scores, _ = f_classif(SimpleImputer(strategy="median").fit_transform(X_train), y_train)

    normal_rows = X_train[y_train == S.NORMAL]
    reference = {f: (None if np.isnan(v) else float(v)) for f, v in normal_rows.median().items()}

    metrics = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {"file": Path(csv_path).name, "cleaning": cleaning,
                    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                    "n_features": len(S.FEATURES), "split": "80:20 stratified"},
        "classes": S.CLASSES,
        "features": S.FEATURES,
        "models": models,
        "best_model": best_key,
        "selection_rule": "Highest mean macro F1 over 10-fold cross-validation on the training split",
        "hypothesis_test": _hypothesis_test(cv_scores),
        "feature_importance": [{"feature": f, "label": S.FEATURE_LABELS.get(f, f), "importance": float(v)}
                               for f, v in ranked[:15]],
        "anova_f_top": [{"feature": f, "label": S.FEATURE_LABELS.get(f, f), "score": float(v)}
                        for f, v in sorted(zip(S.FEATURES, np.nan_to_num(f_scores)), key=lambda t: -t[1])[:15]],
        "reference": reference,
        "fast_mode": fast,
        "total_seconds": round(time.time() - started, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    log(f"Best model: {ALGORITHMS[best_key]} ({metrics['total_seconds']}s total)")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train and compare LR, RF and SVM.")
    parser.add_argument("--data", default="data/hormonal_health_dataset.csv")
    parser.add_argument("--out", default="models/v1")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--fast", action="store_true", help="Small grids, for low-memory hosts")
    args = parser.parse_args()
    train_all(args.data, args.out, args.version, args.fast)


if __name__ == "__main__":
    main()
