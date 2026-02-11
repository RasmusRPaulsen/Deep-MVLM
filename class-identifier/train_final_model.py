from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geom_features import compute_features, load_master_dataset


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Train a final (deployable) model on all data")
    p.add_argument("--data", type=str, default=str(script_dir / "master_dataset.csv"))
    p.add_argument("--outdir", type=str, default=str(script_dir / "outputs"))
    p.add_argument("--include-aligned-coords", action="store_true")
    p.add_argument("--no-gender", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
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

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=50_000,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__C": np.logspace(-3, 3, 13),
        "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=args.n_jobs,
        refit=True,
    )
    search.fit(X, y)

    model_path = outdir / "final_model.joblib"
    meta_path = outdir / "final_model_meta.json"

    # joblib is a dependency of scikit-learn
    import joblib  # type: ignore

    joblib.dump(search.best_estimator_, model_path)

    meta = {
        "data": str(Path(args.data).resolve()),
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "include_aligned_coords": bool(args.include_aligned_coords),
        "include_gender": not bool(args.no_gender),
        "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in search.best_params_.items()},
        "cv": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "seed": int(args.seed)},
        "scoring": "f1_macro",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"  {model_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
