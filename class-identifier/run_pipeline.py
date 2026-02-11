from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geom_features import compute_features, load_master_dataset


@dataclass
class FoldMetrics:
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    mcc: float
    roc_auc_ovr_macro: float | None


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Small-sample ML pipeline for orthodontic class from 3D landmarks")
    p.add_argument(
        "--data",
        type=str,
        default=str(script_dir / "master_dataset.csv"),
        help="Path to master_dataset.csv",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(script_dir / "outputs"),
        help="Output directory",
    )
    p.add_argument(
        "--include-aligned-coords",
        action="store_true",
        help="Add aligned (local-frame) coordinates as additional features (69 dims)",
    )
    p.add_argument(
        "--no-gender",
        action="store_true",
        help="Exclude gender covariate",
    )
    p.add_argument(
        "--outer-splits",
        type=int,
        default=5,
        help="Outer CV folds (default: 5)",
    )
    p.add_argument(
        "--outer-repeats",
        type=int,
        default=20,
        help="Outer CV repeats (default: 20)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_master_dataset(args.data)
    y = df["class"].astype(int).to_numpy()

    X = compute_features(
        df,
        include_aligned_coords=bool(args.include_aligned_coords),
        include_gender=not bool(args.no_gender),
    )

    # Pipeline: standardization is essential for elastic-net logistic regression
    model = LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        max_iter=50_000,
        n_jobs=None,
        random_state=args.seed,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )

    param_grid = {
        "clf__C": np.logspace(-3, 3, 13),
        "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }

    outer_cv = RepeatedStratifiedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.outer_repeats,
        random_state=args.seed,
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)

    fold_metrics: list[FoldMetrics] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_proba_all: list[np.ndarray] = []
    best_params: list[dict[str, float]] = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=args.n_jobs,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_params.append(search.best_params_)

        y_pred = search.predict(X_test)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        roc_auc = None
        if hasattr(search, "predict_proba"):
            proba = search.predict_proba(X_test)
            y_proba_all.append(proba)
            try:
                roc_auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
            except ValueError:
                roc_auc = None

        fold_metrics.append(
            FoldMetrics(
                accuracy=float(accuracy_score(y_test, y_pred)),
                balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
                f1_macro=float(f1_score(y_test, y_pred, average="macro")),
                f1_weighted=float(f1_score(y_test, y_pred, average="weighted")),
                mcc=float(matthews_corrcoef(y_test, y_pred)),
                roc_auc_ovr_macro=roc_auc,
            )
        )

    y_true_all_arr = np.array(y_true_all, dtype=int)
    y_pred_all_arr = np.array(y_pred_all, dtype=int)

    cm = confusion_matrix(y_true_all_arr, y_pred_all_arr)

    summary = {
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "outer_cv": {
            "splits": int(args.outer_splits),
            "repeats": int(args.outer_repeats),
            "total_folds": int(args.outer_splits * args.outer_repeats),
            "seed": int(args.seed),
        },
        "metrics_mean": {
            "accuracy": float(np.mean([m.accuracy for m in fold_metrics])),
            "balanced_accuracy": float(np.mean([m.balanced_accuracy for m in fold_metrics])),
            "f1_macro": float(np.mean([m.f1_macro for m in fold_metrics])),
            "f1_weighted": float(np.mean([m.f1_weighted for m in fold_metrics])),
            "mcc": float(np.mean([m.mcc for m in fold_metrics])),
            "roc_auc_ovr_macro": (
                float(np.nanmean([m.roc_auc_ovr_macro for m in fold_metrics if m.roc_auc_ovr_macro is not None]))
                if any(m.roc_auc_ovr_macro is not None for m in fold_metrics)
                else None
            ),
        },
        "metrics_std": {
            "accuracy": float(np.std([m.accuracy for m in fold_metrics])),
            "balanced_accuracy": float(np.std([m.balanced_accuracy for m in fold_metrics])),
            "f1_macro": float(np.std([m.f1_macro for m in fold_metrics])),
            "f1_weighted": float(np.std([m.f1_weighted for m in fold_metrics])),
            "mcc": float(np.std([m.mcc for m in fold_metrics])),
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true_all_arr, y_pred_all_arr, digits=4),
        "best_params": best_params,
        "feature_columns": list(X.columns),
    }

    (outdir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Fit final model on all data using the most frequent best hyperparameters (mode)
    # This is only for interpretation/feature-importance; not used for reported performance.
    def _mode_param(key: str):
        vals = [bp[key] for bp in best_params]
        # break ties by choosing the median value among tied counts
        unique, counts = np.unique(vals, return_counts=True)
        top = unique[counts == counts.max()]
        return float(np.median(top))

    final_C = _mode_param("clf__C")
    final_l1 = _mode_param("clf__l1_ratio")

    final_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    C=final_C,
                    l1_ratio=final_l1,
                    max_iter=50_000,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    final_pipe.fit(X, y)

    # Permutation importance on full dataset (interpretation only)
    perm = permutation_importance(
        final_pipe,
        X,
        y,
        scoring="f1_macro",
        n_repeats=200,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    imp = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    imp.to_csv(outdir / "permutation_importance.csv", index=False)

    print("Wrote:")
    print(f"  {outdir / 'cv_summary.json'}")
    print(f"  {outdir / 'permutation_importance.csv'}")


if __name__ == "__main__":
    main()
