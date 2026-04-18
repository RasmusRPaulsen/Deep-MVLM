# 3D Facial Landmark Classification Pipeline

This project trains multi-class models to classify patients into **Class 1 / 2 / 3** using 23 facial 3D landmarks and gender.

## What the pipeline does

- Loads all `Class<k>_<M|F><id>.txt` files from `data/`
- Parses each sample as `(23, 3)` coordinates
- Encodes gender as binary (`F=0`, `M=1`)
- Normalizes by:
  - Centering at **Nasion** (point index `2`)
  - Scaling by interocular distance `End_L-End_R` (indices `15-14`)
- Engineers geometric features, including:
  - Pairwise 3D distances (`23 choose 2 = 253`)
  - Face/nose/mouth/eye/chin ratios
  - Angles (nose-chin, jaw, etc.)
  - Symmetry metrics (left-right differences/ratios)
  - Midpoint-based bilateral geometry
  - Optional flattened normalized coordinates
- Trains and compares:
  - Random Forest
  - XGBoost
  - SVM (RBF)
  - Logistic Regression
  - PyTorch MLP (with early stopping)
- Uses stratified 5-fold CV and randomized hyperparameter search
- Saves evaluation reports, confusion matrices, feature importances, and plots

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python train_pipeline.py --data-dir data --landmarks-file landmarks.txt --output-dir outputs
```

## Production run (recommended)

```bash
python train_pipeline.py \
  --data-dir data \
  --landmarks-file landmarks.txt \
  --output-dir outputs_no_xgb_allclasses_topk_ovr \
  --no-xgboost \
  --n-iter-search 20 \
  --skip-tsne \
  --top-k 300
```

### Useful options

- `--n-iter-search 20` : increase hyperparameter search iterations
- `--no-flat-coords` : use only engineered geometry (no flattened coordinates)
- `--skip-tsne` : skip t-SNE plot for faster runtime
- `--top-k 300` : apply fold-internal SelectKBest with top-k features
- `--topk-method f_classif|chi2` : SelectKBest score function (default: `f_classif`)

## Outputs

Generated in the directory passed via `--output-dir`:

- `model_metrics_summary.csv`
- `report_<MODEL>.csv`
- `ovr_report_<MODEL>.csv` (one-vs-rest precision/recall/specificity/accuracy)
- `confusion_matrix_<MODEL>.csv` and `.png`
- `top20_features_<TREE_MODEL>.csv` and `.png`
- `top20_features_<TREE_MODEL>_interpretation.txt`
- `top20_features_LogisticRegression.csv` and `.png`
- `top20_features_LogisticRegression_interpretation.txt`
- `model_comparison.png`
- `landmarks_3d_examples.png`
- `embedding_pca.png`, `embedding_tsne.png` (unless disabled)
- `best_model_summary.txt`
- `dataset_overview.txt`
