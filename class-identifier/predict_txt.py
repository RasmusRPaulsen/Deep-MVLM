from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geom_features import compute_features, dataframe_from_txt


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Predict orthodontic class from a single landmark .txt")
    p.add_argument("txt", type=str, help="Path to landmark .txt file")
    p.add_argument(
        "--model",
        type=str,
        default=str(script_dir / "outputs" / "final_model.joblib"),
        help="Path to trained joblib model (Pipeline)",
    )
    p.add_argument("--gender", type=int, choices=[0, 1], required=True, help="0=F, 1=M")
    p.add_argument("--include-aligned-coords", action="store_true", help="Must match how the model was trained")
    p.add_argument("--no-gender", action="store_true", help="Must match how the model was trained")
    p.add_argument("--out", type=str, default="", help="Optional: write JSON prediction to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # joblib is a dependency of scikit-learn
    import joblib  # type: ignore

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            "Model not found. Train it first with: "
            "/Users/eck/Documents/yüksek-lisans/MorphoMark/.venv/bin/python class-identifier/train_final_model.py"
        )

    pipe = joblib.load(model_path)

    df_one = dataframe_from_txt(args.txt, gender=args.gender)
    X_one = compute_features(
        df_one,
        include_aligned_coords=bool(args.include_aligned_coords),
        include_gender=not bool(args.no_gender),
    )

    pred = int(pipe.predict(X_one)[0])
    proba = None
    if hasattr(pipe, "predict_proba"):
        proba_arr = pipe.predict_proba(X_one)[0]
        proba = [float(x) for x in proba_arr]

    label_map = {0: "Class I", 1: "Class II", 2: "Class III"}

    result = {
        "predicted_class_index": pred,
        "predicted_class_label": label_map.get(pred, str(pred)),
        "probabilities_order": ["Class I", "Class II", "Class III"],
        "probabilities": proba,
        "model": str(model_path.resolve()),
    }

    print(json.dumps(result, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
