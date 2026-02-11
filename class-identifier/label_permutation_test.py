from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geom_features import compute_features, load_master_dataset


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Label permutation test (macro-F1) for small-sample multi-class classification")
    p.add_argument("--data", type=str, default=str(script_dir / "master_dataset.csv"))
    p.add_argument("--out", type=str, default=str(script_dir / "outputs" / "permutation_test.json"))
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--outer-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def mean_cv_macro_f1(X, y, *, C: float, l1_ratio: float, outer_cv) -> float:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    C=C,
                    l1_ratio=l1_ratio,
                    max_iter=50_000,
                    random_state=0,
                ),
            ),
        ]
    )

    scores: list[float] = []
    for tr, te in outer_cv.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        scores.append(float(f1_score(y[te], pred, average="macro")))
    return float(np.mean(scores))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_master_dataset(args.data)
    X = compute_features(df, include_aligned_coords=False, include_gender=True)
    y = df["class"].astype(int).to_numpy()

    # Select hyperparameters once on true labels (keeps permutation test computationally feasible)
    base_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="saga", max_iter=50_000, random_state=args.seed)),
        ]
    )
    grid = {
        "clf__C": np.logspace(-3, 3, 13),
        "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    search = GridSearchCV(base_pipe, grid, scoring="f1_macro", cv=inner_cv, n_jobs=args.n_jobs, refit=True)
    search.fit(X, y)
    C = float(search.best_params_["clf__C"])
    l1_ratio = float(search.best_params_["clf__l1_ratio"])

    outer_cv = RepeatedStratifiedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.outer_repeats,
        random_state=args.seed,
    )

    observed = mean_cv_macro_f1(X, y, C=C, l1_ratio=l1_ratio, outer_cv=outer_cv)

    rng = np.random.default_rng(args.seed)
    perm_scores = []
    for _ in range(args.n_perm):
        y_perm = rng.permutation(y)
        perm_scores.append(mean_cv_macro_f1(X, y_perm, C=C, l1_ratio=l1_ratio, outer_cv=outer_cv))

    perm_scores = np.array(perm_scores, dtype=float)
    p_value = float((1.0 + np.sum(perm_scores >= observed)) / (1.0 + args.n_perm))

    result = {
        "metric": "macro_f1",
        "observed": observed,
        "n_perm": int(args.n_perm),
        "p_value": p_value,
        "perm_mean": float(np.mean(perm_scores)),
        "perm_std": float(np.std(perm_scores)),
        "selected_hyperparams": {"C": C, "l1_ratio": l1_ratio},
        "outer_cv": {"splits": args.outer_splits, "repeats": args.outer_repeats, "seed": args.seed},
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
