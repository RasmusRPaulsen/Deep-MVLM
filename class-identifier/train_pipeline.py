#!/usr/bin/env python3
"""
3D Facial Landmark Classification Pipeline

- Loads patient landmark files from the format: Class<1|2|3>_<M|F><id>.txt
- Normalizes landmarks with Nasion-centered coordinates and interocular scaling
- Engineers geometric features (distances, ratios, angles, symmetry, midpoint features)
- Trains and compares multiple models with stratified 5-fold CV
- Tunes hyperparameters with randomized search
- Evaluates with accuracy, macro-F1, confusion matrices, class-wise metrics
- Exports feature importance for tree-based models
- Includes optional 3D and embedding visualizations (PCA/t-SNE)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from xgboost import XGBClassifier


# Landmark indices used in geometry definitions
IDX_TR = 0
IDX_N = 2
IDX_PRN = 3
IDX_SN = 5
IDX_LS = 7
IDX_LI = 9
IDX_PG = 10
IDX_GN = 11
IDX_ME = 12
IDX_END_R = 14
IDX_END_L = 15
IDX_EX_R = 13
IDX_EX_L = 16
IDX_AL_R = 17
IDX_AL_L = 18
IDX_CH_R = 19
IDX_CH_L = 20
IDX_GO_R = 21
IDX_GO_L = 22


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetBundle:
    file_names: List[str]
    labels: np.ndarray
    genders: np.ndarray
    coords_raw: np.ndarray
    coords_norm: np.ndarray
    landmark_names: List[str]


def parse_landmark_names(landmarks_file: Path) -> List[str]:
    names: Dict[int, str] = {}
    with landmarks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)\s+([^\s(]+)", line)
            if match:
                idx = int(match.group(1))
                name = match.group(2)
                names[idx] = name

    ordered = [names.get(i, f"P{i}") for i in range(23)]
    return ordered


def parse_single_coordinate_file(file_path: Path) -> np.ndarray:
    coords = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                raise ValueError(f"Invalid coordinate row in {file_path.name}: {line}")
            x, y, z = map(float, parts[-3:])
            coords.append([x, y, z])

    arr = np.asarray(coords, dtype=np.float32)
    if arr.shape != (23, 3):
        raise ValueError(f"Expected (23,3) in {file_path.name}, got {arr.shape}")
    return arr


def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords[IDX_N]
    interocular = np.linalg.norm(coords[IDX_END_L] - coords[IDX_END_R])
    interocular = max(float(interocular), 1e-8)
    normalized = centered / interocular
    return normalized


def load_dataset(data_dir: Path, landmarks_file: Path) -> DatasetBundle:
    pattern = re.compile(r"^Class([123])_([MF])(\d+)\.txt$", re.IGNORECASE)

    file_names: List[str] = []
    labels: List[int] = []
    genders: List[int] = []
    coords_raw: List[np.ndarray] = []
    coords_norm: List[np.ndarray] = []

    for file_path in sorted(data_dir.glob("*.txt")):
        m = pattern.match(file_path.name)
        if not m:
            continue

        class_id = int(m.group(1))
        gender_code = m.group(2).upper()

        coords = parse_single_coordinate_file(file_path)
        norm_coords = normalize_landmarks(coords)

        file_names.append(file_path.name)
        labels.append(class_id - 1)
        genders.append(1 if gender_code == "M" else 0)
        coords_raw.append(coords)
        coords_norm.append(norm_coords)

    if not file_names:
        raise RuntimeError(f"No valid data files found in {data_dir}")

    return DatasetBundle(
        file_names=file_names,
        labels=np.asarray(labels, dtype=np.int64),
        genders=np.asarray(genders, dtype=np.int64),
        coords_raw=np.asarray(coords_raw, dtype=np.float32),
        coords_norm=np.asarray(coords_norm, dtype=np.float32),
        landmark_names=parse_landmark_names(landmarks_file),
    )


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-12:
        return 0.0
    cos_val = float(np.dot(ba, bc) / denom)
    cos_val = max(-1.0, min(1.0, cos_val))
    return float(np.degrees(np.arccos(cos_val)))


def safe_ratio(num: float, den: float) -> float:
    den = den if abs(den) > 1e-8 else 1e-8
    return float(num / den)


def point_to_line_distance(point: np.ndarray, line_a: np.ndarray, line_b: np.ndarray) -> float:
    line_vec = line_b - line_a
    denom = np.linalg.norm(line_vec)
    if denom <= 1e-12:
        return euclidean(point, line_a)
    return float(np.linalg.norm(np.cross(line_vec, line_a - point)) / denom)


def signed_point_plane_distance(point: np.ndarray, plane_a: np.ndarray, plane_b: np.ndarray, plane_c: np.ndarray) -> float:
    normal = np.cross(plane_b - plane_a, plane_c - plane_a)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        return 0.0
    normal = normal / norm
    return float(np.dot(point - plane_a, normal))


def bilateral_pairs() -> List[Tuple[int, int, str]]:
    return [
        (IDX_EX_R, IDX_EX_L, "exocanthion"),
        (IDX_END_R, IDX_END_L, "endocanthion"),
        (IDX_AL_R, IDX_AL_L, "alare"),
        (IDX_CH_R, IDX_CH_L, "cheilion"),
        (IDX_GO_R, IDX_GO_L, "gonion"),
    ]


def engineer_features(
    coords_norm: np.ndarray,
    genders: np.ndarray,
    landmark_names: Sequence[str],
    include_flatten: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    n_samples = coords_norm.shape[0]
    n_landmarks = coords_norm.shape[1]

    feature_rows: List[np.ndarray] = []
    feature_names: List[str] = []

    pair_indices = [(i, j) for i in range(n_landmarks) for j in range(i + 1, n_landmarks)]

    for sample_idx in range(n_samples):
        p = coords_norm[sample_idx]
        feats: List[float] = []
        names: List[str] = []

        if include_flatten:
            flat = p.flatten()
            feats.extend(flat.tolist())
            for i in range(n_landmarks):
                names.extend([
                    f"coord_{landmark_names[i]}_x",
                    f"coord_{landmark_names[i]}_y",
                    f"coord_{landmark_names[i]}_z",
                ])

        # Comprehensive pairwise distances (23 choose 2 = 253 features)
        for i, j in pair_indices:
            feats.append(euclidean(p[i], p[j]))
            names.append(f"dist_{landmark_names[i]}_{landmark_names[j]}")

        # Core clinically interpretable distances
        face_height = euclidean(p[IDX_TR], p[IDX_ME])
        face_width = euclidean(p[IDX_GO_R], p[IDX_GO_L])
        nose_width = euclidean(p[IDX_AL_L], p[IDX_AL_R])
        mouth_width = euclidean(p[IDX_CH_L], p[IDX_CH_R])
        eye_inner_width = euclidean(p[IDX_END_L], p[IDX_END_R])
        eye_outer_width = euclidean(p[IDX_EX_L], p[IDX_EX_R])
        chin_projection = euclidean(p[IDX_GN], p[IDX_ME])

        feats.extend(
            [
                face_height,
                face_width,
                nose_width,
                mouth_width,
                eye_inner_width,
                eye_outer_width,
                chin_projection,
            ]
        )
        names.extend(
            [
                "face_height_Tr_Me",
                "face_width_GoR_GoL",
                "nose_width_AlL_AlR",
                "mouth_width_ChL_ChR",
                "eye_inner_width_EndL_EndR",
                "eye_outer_width_ExL_ExR",
                "chin_projection_Gn_Me",
            ]
        )

        # Requested ratios
        feats.extend(
            [
                safe_ratio(face_height, face_width),
                safe_ratio(nose_width, face_width),
                safe_ratio(mouth_width, face_width),
                safe_ratio(eye_inner_width, eye_outer_width),
                safe_ratio(chin_projection, face_height),
            ]
        )
        names.extend(
            [
                "ratio_face_height_over_width",
                "ratio_nose_width_over_face_width",
                "ratio_mouth_width_over_face_width",
                "ratio_inner_eye_over_outer_eye",
                "ratio_chin_projection_over_face_height",
            ]
        )

        # Requested angles + extra profile/jaw angles
        nose_chin_angle = angle_3d(p[IDX_PRN], p[IDX_N], p[IDX_GN])
        jaw_angle = angle_3d(p[IDX_GO_R], p[IDX_GN], p[IDX_GO_L])
        nasion_subnasale_chin = angle_3d(p[IDX_N], p[IDX_SN], p[IDX_GN])
        lip_chin_angle = angle_3d(p[IDX_LS], p[8], p[IDX_GN])
        nasolabial_angle = angle_3d(p[IDX_PRN], p[IDX_SN], p[IDX_LS])

        dist_sn_pg_z = abs(float(p[IDX_SN][2] - p[IDX_PG][2]))
        dist_ls_li_z = abs(float(p[IDX_LS][2] - p[IDX_LI][2]))

        # Mentolabial sulcus depth proxy:
        # perpendicular distance of Li to Sn-Pg profile line (3D) + sagittal residual from line midpoint
        mentolabial_sulcus_depth = point_to_line_distance(p[IDX_LI], p[IDX_SN], p[IDX_PG])
        mentolabial_sagittal_residual = abs(float(p[IDX_LI][2] - 0.5 * (p[IDX_SN][2] + p[IDX_PG][2])))

        feats.extend(
            [
                nose_chin_angle,
                jaw_angle,
                nasion_subnasale_chin,
                lip_chin_angle,
                nasolabial_angle,
                dist_sn_pg_z,
                dist_ls_li_z,
                mentolabial_sulcus_depth,
                mentolabial_sagittal_residual,
            ]
        )
        names.extend(
            [
                "angle_nose_chin_Prn_N_Gn",
                "angle_jaw_GoR_Gn_GoL",
                "angle_N_Sn_Gn",
                "angle_Ls_Stm_Gn",
                "angle_Prn_Sn_Ls",
                "dist_Sn_Pg_z",
                "dist_Ls_Li_z",
                "mentolabial_sulcus_depth",
                "mentolabial_sulcus_depth_z_residual",
            ]
        )

        # Advanced sagittal/profile features for hard-negative separation (Class1 vs Class3)
        profile_chain = [IDX_TR, 1, IDX_N, IDX_PRN, 4, IDX_SN, IDX_LS, 8, IDX_LI, IDX_PG, IDX_GN, IDX_ME]
        profile_chain_names = [landmark_names[idx] for idx in profile_chain]

        # Signed z-differences across profile chain
        for i in range(len(profile_chain)):
            for j in range(i + 1, len(profile_chain)):
                i_idx = profile_chain[i]
                j_idx = profile_chain[j]
                i_name = profile_chain_names[i]
                j_name = profile_chain_names[j]
                dz = float(p[i_idx][2] - p[j_idx][2])
                feats.extend([dz, abs(dz)])
                names.extend([f"sag_dz_{i_name}_{j_name}", f"sag_abs_dz_{i_name}_{j_name}"])

        # Sagittal slopes in (y,z) for adjacent profile points
        for i in range(len(profile_chain) - 1):
            a_idx = profile_chain[i]
            b_idx = profile_chain[i + 1]
            a_name = profile_chain_names[i]
            b_name = profile_chain_names[i + 1]
            dy = float(p[b_idx][1] - p[a_idx][1])
            dz = float(p[b_idx][2] - p[a_idx][2])
            slope_zy = safe_ratio(dz, dy)
            feats.extend([slope_zy, abs(slope_zy)])
            names.extend([f"profile_slope_zy_{a_name}_{b_name}", f"profile_abs_slope_zy_{a_name}_{b_name}"])

        # Local profile curvature angles (2D yz-curve projected to 3D points)
        for i in range(1, len(profile_chain) - 1):
            a_idx = profile_chain[i - 1]
            b_idx = profile_chain[i]
            c_idx = profile_chain[i + 1]
            b_name = profile_chain_names[i]
            curv = angle_3d(p[a_idx], p[b_idx], p[c_idx])
            feats.append(curv)
            names.append(f"profile_curvature_{b_name}")

        # Depth wrt orbital plane (N, Ex_R, Ex_L)
        orbital_a = p[IDX_N]
        orbital_b = p[IDX_EX_R]
        orbital_c = p[IDX_EX_L]
        depth_targets = [IDX_PRN, 4, IDX_SN, IDX_LS, 8, IDX_LI, IDX_PG, IDX_GN, IDX_ME]
        for idx in depth_targets:
            d_signed = signed_point_plane_distance(p[idx], orbital_a, orbital_b, orbital_c)
            feats.extend([d_signed, abs(d_signed)])
            names.extend([f"depth_orbital_plane_signed_{landmark_names[idx]}", f"depth_orbital_plane_abs_{landmark_names[idx]}"])

        # Depth-ratio stability features among key profile points
        d_sn = abs(signed_point_plane_distance(p[IDX_SN], orbital_a, orbital_b, orbital_c))
        d_ls = abs(signed_point_plane_distance(p[IDX_LS], orbital_a, orbital_b, orbital_c))
        d_li = abs(signed_point_plane_distance(p[IDX_LI], orbital_a, orbital_b, orbital_c))
        d_pg = abs(signed_point_plane_distance(p[IDX_PG], orbital_a, orbital_b, orbital_c))
        d_gn = abs(signed_point_plane_distance(p[IDX_GN], orbital_a, orbital_b, orbital_c))
        feats.extend(
            [
                safe_ratio(d_pg, d_sn),
                safe_ratio(d_gn, d_sn),
                safe_ratio(d_ls, d_li),
                safe_ratio(d_pg, d_li),
            ]
        )
        names.extend(
            [
                "depth_ratio_Pg_over_Sn",
                "depth_ratio_Gn_over_Sn",
                "depth_ratio_Ls_over_Li",
                "depth_ratio_Pg_over_Li",
            ]
        )

        # Symmetry features from bilateral landmarks
        midline_ref_points = [(IDX_N, "N"), (IDX_GN, "Gn"), (IDX_ME, "Me")]
        for r_idx, r_name in midline_ref_points:
            for left_idx, right_idx, pair_name in bilateral_pairs():
                d_left = euclidean(p[left_idx], p[r_idx])
                d_right = euclidean(p[right_idx], p[r_idx])
                feats.extend([d_left, d_right, abs(d_left - d_right), safe_ratio(d_left, d_right)])
                names.extend(
                    [
                        f"sym_{pair_name}_{r_name}_left_dist",
                        f"sym_{pair_name}_{r_name}_right_dist",
                        f"sym_{pair_name}_{r_name}_abs_diff",
                        f"sym_{pair_name}_{r_name}_left_over_right",
                    ]
                )

        # Midpoint-based features
        central_points = [IDX_TR, IDX_N, IDX_PRN, 5, 6, 8, 10, IDX_GN, IDX_ME]
        central_names = [landmark_names[idx] for idx in central_points]

        for left_idx, right_idx, pair_name in bilateral_pairs():
            midpoint = 0.5 * (p[left_idx] + p[right_idx])
            for cp_idx, cp_name in zip(central_points, central_names):
                d_mid = euclidean(midpoint, p[cp_idx])
                feats.append(d_mid)
                names.append(f"mid_{pair_name}_to_{cp_name}")

            # relation of midpoint to N-Gn line (approx using x-distance after normalization)
            feats.append(abs(midpoint[0]))
            names.append(f"mid_{pair_name}_abs_x")

        # Gender as numeric feature
        feats.append(float(genders[sample_idx]))
        names.append("gender_binary")

        feature_rows.append(np.asarray(feats, dtype=np.float32))
        if not feature_names:
            feature_names = names

    X_tabular = np.vstack(feature_rows)
    X_structured = coords_norm.copy()
    return X_tabular, feature_names, X_structured


def build_hard_negative_feature_subset(
    X: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    selected_indices: List[int] = []
    selected_names: List[str] = []

    sagittal_landmarks = ["Prn", "Sn", "Ls", "Li", "Pg", "Gn", "Me", "Stm", "Col"]
    explicit_features = {
        "dist_Sn_Pg_z",
        "dist_Ls_Li_z",
        "mentolabial_sulcus_depth",
        "mentolabial_sulcus_depth_z_residual",
        "angle_Prn_Sn_Ls",
        "angle_N_Sn_Gn",
        "angle_Ls_Stm_Gn",
        "angle_nose_chin_Prn_N_Gn",
    }

    for idx, name in enumerate(feature_names):
        is_explicit = name in explicit_features
        is_profile_angle_ratio = name.startswith("angle_") or name.startswith("ratio_")
        is_symmetry = name.startswith("sym_")
        is_sagittal_coord = name.startswith("coord_") and name.endswith("_z")
        is_profile_distance = name.startswith("dist_") and any(k in name for k in sagittal_landmarks)
        is_profile_engineered = (
            name.startswith("sag_")
            or name.startswith("profile_")
            or name.startswith("depth_")
            or name.startswith("mentolabial_")
        )

        if (
            is_explicit
            or is_profile_angle_ratio
            or is_symmetry
            or is_sagittal_coord
            or is_profile_distance
            or is_profile_engineered
        ):
            selected_indices.append(idx)
            selected_names.append(name)

    if not selected_indices:
        raise ValueError("Hard-negative feature selection produced zero features.")

    selected_X = X[:, selected_indices]
    return selected_X, selected_names, np.asarray(selected_indices, dtype=np.int64)


class TorchMLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], dropout: float, n_classes: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 250,
        weight_decay: float = 1e-4,
        patience: int = 25,
        val_size: float = 0.2,
        random_state: int = 42,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_size = val_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.random_state)
        train_idx, val_idx = next(sss.split(X, y))

        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)

        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        dev = torch.device(self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model_ = TorchMLPNet(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            n_classes=len(self.classes_),
        ).to(dev)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_loss = math.inf
        best_state = None
        no_improve = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            for xb, yb in train_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_val.to(dev))
                val_loss = criterion(val_logits, y_val.to(dev)).item()

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if self.verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:03d}: val_loss={val_loss:.6f}")

            if no_improve >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.to("cpu")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


@dataclass
class ModelResult:
    name: str
    best_estimator: object
    best_params: Dict
    accuracy: float
    f1_macro: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    confusion_matrix: np.ndarray


def maybe_wrap_with_topk_selector(
    estimator: object,
    param_space: Dict[str, Sequence],
    top_k: int | None,
    topk_method: str,
) -> Tuple[object, Dict[str, Sequence]]:
    if top_k is None:
        return estimator, param_space

    score_func = f_classif if topk_method == "f_classif" else chi2
    selector = SelectKBest(score_func=score_func, k=top_k)

    steps: List[Tuple[str, object]] = []
    if topk_method == "chi2":
        steps.append(("topk_minmax", MinMaxScaler()))
    steps.append(("topk_select", selector))
    steps.append(("model", estimator))

    wrapped_estimator = Pipeline(steps=steps)
    wrapped_space = {f"model__{k}": v for k, v in param_space.items()}
    return wrapped_estimator, wrapped_space


def get_model_search_space(
    random_state: int,
    top_k: int | None,
    topk_method: str,
) -> Dict[str, Tuple[object, Dict[str, Sequence]]]:
    models: Dict[str, Tuple[object, Dict[str, Sequence]]] = {}

    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_space = {
        "n_estimators": [200, 400, 600, 900],
        "max_depth": [None, 6, 10, 16, 24],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8],
        "class_weight": [None, "balanced"],
    }
    models["RandomForest"] = maybe_wrap_with_topk_selector(rf, rf_space, top_k=top_k, topk_method=topk_method)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_space = {
        "n_estimators": [150, 300, 500, 700],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    models["XGBoost"] = maybe_wrap_with_topk_selector(xgb, xgb_space, top_k=top_k, topk_method=topk_method)

    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ]
    )
    svm_space = {
        "clf__C": [0.1, 0.5, 1.0, 3.0, 10.0, 30.0],
        "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1],
        "clf__class_weight": [None, "balanced"],
    }
    models["SVM_RBF"] = maybe_wrap_with_topk_selector(svm, svm_space, top_k=top_k, topk_method=topk_method)

    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=6000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    lr_space = {
        "clf__C": [0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 10.0],
        "clf__solver": ["lbfgs", "newton-cg", "saga"],
        "clf__class_weight": [None, "balanced"],
    }
    models["LogisticRegression"] = maybe_wrap_with_topk_selector(lr, lr_space, top_k=top_k, topk_method=topk_method)

    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                TorchMLPClassifier(
                    random_state=random_state,
                    device="cpu",
                    verbose=False,
                ),
            ),
        ]
    )
    mlp_space = {
        "clf__hidden_dims": [(64, 32), (128, 64), (256, 128), (128, 64, 32)],
        "clf__dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
        "clf__lr": [5e-4, 1e-3, 2e-3, 5e-3],
        "clf__batch_size": [16, 24, 32, 48],
        "clf__weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
        "clf__max_epochs": [150, 220, 300],
        "clf__patience": [15, 25, 35],
    }
    models["MLP_PyTorch"] = maybe_wrap_with_topk_selector(mlp, mlp_space, top_k=top_k, topk_method=topk_method)

    return models


def build_one_vs_rest_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []

    for cls, cls_name in zip(class_labels, class_names):
        y_true_bin = (y_true == cls).astype(np.int64)
        y_pred_bin = (y_pred == cls).astype(np.int64)
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm_bin.ravel()

        recall = safe_ratio(tp, tp + fn)
        specificity = safe_ratio(tn, tn + fp)
        precision = safe_ratio(tp, tp + fp)
        f1 = safe_ratio(2.0 * precision * recall, precision + recall)
        accuracy = safe_ratio(tp + tn, tp + tn + fp + fn)

        rows.append(
            {
                "class": cls_name,
                "precision": precision,
                "recall_sensitivity": recall,
                "specificity": specificity,
                "f1": f1,
                "ovr_accuracy": accuracy,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "support_positive": int(tp + fn),
                "support_negative": int(tn + fp),
            }
        )

    return pd.DataFrame(rows)


def resolve_tree_estimator_and_feature_names(
    fitted_estimator: object,
    feature_names: Sequence[str],
) -> Tuple[object, List[str]]:
    estimator = fitted_estimator
    resolved_feature_names = list(feature_names)

    if isinstance(estimator, Pipeline):
        selector = estimator.named_steps.get("topk_select")
        if selector is not None and hasattr(selector, "get_support"):
            support_idx = selector.get_support(indices=True)
            resolved_feature_names = np.asarray(feature_names)[support_idx].tolist()

        if "model" in estimator.named_steps:
            estimator = estimator.named_steps["model"]

    return estimator, resolved_feature_names


def resolve_logistic_estimator_and_feature_names(
    fitted_estimator: object,
    feature_names: Sequence[str],
) -> Tuple[object | None, List[str]]:
    estimator = fitted_estimator
    resolved_feature_names = list(feature_names)

    if isinstance(estimator, Pipeline):
        selector = estimator.named_steps.get("topk_select")
        if selector is not None and hasattr(selector, "get_support"):
            support_idx = selector.get_support(indices=True)
            resolved_feature_names = np.asarray(feature_names)[support_idx].tolist()

        model_step = estimator.named_steps.get("model")
        if model_step is not None and isinstance(model_step, Pipeline):
            estimator = model_step.named_steps.get("clf")
        else:
            estimator = estimator.named_steps.get("clf")

    if estimator is None or not hasattr(estimator, "coef_"):
        return None, resolved_feature_names

    return estimator, resolved_feature_names


def save_logistic_feature_importance(
    fitted_estimator: object,
    feature_names: Sequence[str],
    output_dir: Path,
) -> None:
    estimator, resolved_feature_names = resolve_logistic_estimator_and_feature_names(
        fitted_estimator=fitted_estimator,
        feature_names=feature_names,
    )
    if estimator is None:
        return

    coef = np.asarray(estimator.coef_, dtype=np.float64)
    if coef.ndim != 2:
        return

    if coef.shape[1] != len(resolved_feature_names):
        return

    # Multi-class: aggregate one-vs-rest coefficients using mean absolute value.
    mean_abs_coef = np.mean(np.abs(coef), axis=0)
    idx_sorted = np.argsort(mean_abs_coef)[::-1]
    top_idx = idx_sorted[:20]

    top_df = pd.DataFrame(
        {
            "feature": np.asarray(resolved_feature_names)[top_idx],
            "importance": mean_abs_coef[top_idx],
        }
    )
    top_df.to_csv(output_dir / "top20_features_LogisticRegression.csv", index=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_df, x="importance", y="feature", palette="magma")
    plt.title("Top 20 Feature Importances - LogisticRegression")
    plt.tight_layout()
    plt.savefig(output_dir / "top20_features_LogisticRegression.png", dpi=180)
    plt.close()

    interpretation = interpret_top_features(top_df["feature"].tolist())
    with (output_dir / "top20_features_LogisticRegression_interpretation.txt").open("w", encoding="utf-8") as f:
        for line in interpretation:
            f.write(line + "\n")


def fit_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    output_dir: Path,
    random_state: int,
    n_iter_search: int,
    top_k: int | None,
    topk_method: str,
    selected_models: Sequence[str] | None = None,
    class_name_map: Dict[int, str] | None = None,
) -> List[ModelResult]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    model_spaces = get_model_search_space(
        random_state=random_state,
        top_k=top_k,
        topk_method=topk_method,
    )

    if selected_models is not None:
        selected_set = set(selected_models)
        model_spaces = {k: v for k, v in model_spaces.items() if k in selected_set}
        if not model_spaces:
                available_models = list(
                    get_model_search_space(
                        random_state=random_state,
                        top_k=top_k,
                        topk_method=topk_method,
                    ).keys()
                )
                raise ValueError(f"No valid models selected. Available: {available_models}")

    results: List[ModelResult] = []
    class_labels = np.unique(y)
    class_names = [
        class_name_map.get(int(lbl), f"Class{int(lbl) + 1}") if class_name_map else f"Class{int(lbl) + 1}"
        for lbl in class_labels
    ]

    for model_name, (base_estimator, param_space) in model_spaces.items():
        print(f"\n=== Tuning {model_name} ===")
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_space,
            n_iter=n_iter_search if model_name != "MLP_PyTorch" else max(6, n_iter_search // 2),
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            random_state=random_state,
            refit=True,
            verbose=1,
        )
        search.fit(X, y)

        best_estimator = search.best_estimator_
        y_pred = cross_val_predict(best_estimator, X, y, cv=cv, n_jobs=1)

        acc = accuracy_score(y, y_pred)
        f1m = f1_score(y, y_pred, average="macro")
        cm = confusion_matrix(y, y_pred, labels=class_labels)
        p, r, _, _ = precision_recall_fscore_support(
            y,
            y_pred,
            average=None,
            labels=class_labels,
            zero_division=0,
        )

        result = ModelResult(
            name=model_name,
            best_estimator=best_estimator,
            best_params=search.best_params_,
            accuracy=float(acc),
            f1_macro=float(f1m),
            per_class_precision=p.tolist(),
            per_class_recall=r.tolist(),
            confusion_matrix=cm,
        )
        results.append(result)

        report = classification_report(
            y,
            y_pred,
            labels=class_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df.to_csv(output_dir / f"report_{model_name}.csv", index=True)

        ovr_df = build_one_vs_rest_report(
            y_true=y,
            y_pred=y_pred,
            class_labels=class_labels,
            class_names=class_names,
        )
        ovr_df.to_csv(output_dir / f"ovr_report_{model_name}.csv", index=False)

        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(output_dir / f"confusion_matrix_{model_name}.csv")

        plot_confusion_matrix(cm, model_name, output_dir / f"confusion_matrix_{model_name}.png")

        # Feature importance for tree-based models
        if model_name in {"RandomForest", "XGBoost"}:
            tree_estimator, tree_feature_names = resolve_tree_estimator_and_feature_names(
                fitted_estimator=best_estimator,
                feature_names=feature_names,
            )
            save_tree_feature_importance(
                model_name=model_name,
                fitted_estimator=tree_estimator,
                feature_names=tree_feature_names,
                output_dir=output_dir,
            )

        if model_name == "LogisticRegression":
            save_logistic_feature_importance(
                fitted_estimator=best_estimator,
                feature_names=feature_names,
                output_dir=output_dir,
            )

    summary_rows: List[Dict[str, object]] = []
    for r in results:
        row: Dict[str, object] = {
            "model": r.name,
            "accuracy": r.accuracy,
            "f1_macro": r.f1_macro,
            "best_params": json.dumps(r.best_params),
        }
        for i, cls_name in enumerate(class_names):
            cls_token = cls_name.replace(" ", "")
            row[f"precision_{cls_token}"] = r.per_class_precision[i]
            row[f"recall_{cls_token}"] = r.per_class_recall[i]
        summary_rows.append(row)

    metrics_df = pd.DataFrame(summary_rows)
    metrics_df.to_csv(output_dir / "model_metrics_summary.csv", index=False)
    plot_model_comparison(metrics_df, output_dir / "model_comparison.png")

    return results


def save_tree_feature_importance(
    model_name: str,
    fitted_estimator: object,
    feature_names: Sequence[str],
    output_dir: Path,
) -> None:
    if not hasattr(fitted_estimator, "feature_importances_"):
        return

    importances = np.asarray(fitted_estimator.feature_importances_, dtype=np.float64)
    if importances.shape[0] != len(feature_names):
        return

    idx_sorted = np.argsort(importances)[::-1]
    top_idx = idx_sorted[:20]

    top_df = pd.DataFrame(
        {
            "feature": np.asarray(feature_names)[top_idx],
            "importance": importances[top_idx],
        }
    )
    top_df.to_csv(output_dir / f"top20_features_{model_name}.csv", index=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top 20 Feature Importances - {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"top20_features_{model_name}.png", dpi=180)
    plt.close()

    interpretation = interpret_top_features(top_df["feature"].tolist())
    with (output_dir / f"top20_features_{model_name}_interpretation.txt").open("w", encoding="utf-8") as f:
        for line in interpretation:
            f.write(line + "\n")


def interpret_top_features(top_features: Sequence[str]) -> List[str]:
    lines = []
    category_counts = {
        "distance": 0,
        "ratio": 0,
        "angle": 0,
        "symmetry": 0,
        "midpoint": 0,
        "coordinate": 0,
        "gender": 0,
        "other": 0,
    }

    for feat in top_features:
        if feat.startswith("dist_"):
            category_counts["distance"] += 1
        elif feat.startswith("ratio_"):
            category_counts["ratio"] += 1
        elif feat.startswith("angle_"):
            category_counts["angle"] += 1
        elif feat.startswith("sym_"):
            category_counts["symmetry"] += 1
        elif feat.startswith("mid_"):
            category_counts["midpoint"] += 1
        elif feat.startswith("coord_"):
            category_counts["coordinate"] += 1
        elif feat == "gender_binary":
            category_counts["gender"] += 1
        else:
            category_counts["other"] += 1

    lines.append("Top-feature category counts:")
    for k, v in category_counts.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Clinical interpretation guide:")
    lines.append("- Distance-heavy importance suggests global craniofacial size/proportion differences.")
    lines.append("- Symmetry-heavy importance suggests left-right asymmetry contributes to class separation.")
    lines.append("- Ratio/angle importance suggests shape configuration is more discriminative than absolute size.")
    lines.append("- Midpoint features indicate bilateral relational geometry around facial midline.")
    lines.append("- Gender importance, if present, may reflect sex-related dimorphism and should be interpreted cautiously.")
    return lines


def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_path: Path) -> None:
    plt.figure(figsize=(5.8, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics_df.melt(id_vars=["model"], value_vars=["accuracy", "f1_macro"], var_name="metric", value_name="value")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="model", y="value", hue="metric")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_landmarks_3d_examples(coords_norm: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    if n_classes == 0:
        return

    fig = plt.figure(figsize=(4 * n_classes, 4))

    for pos, cls in enumerate(unique_classes):
        idx = int(np.where(y == cls)[0][0])
        pts = coords_norm[idx]

        ax = fig.add_subplot(1, n_classes, pos + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=26)
        ax.set_title(f"Class{int(cls) + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_embeddings(X: np.ndarray, y: np.ndarray, output_dir: Path, random_state: int) -> None:
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(Xs)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set1", s=45)
    plt.title("PCA Projection")
    plt.tight_layout()
    plt.savefig(output_dir / "embedding_pca.png", dpi=180)
    plt.close()

    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=random_state)
    X_tsne = tsne.fit_transform(Xs)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="Set1", s=45)
    plt.title("t-SNE Projection")
    plt.tight_layout()
    plt.savefig(output_dir / "embedding_tsne.png", dpi=180)
    plt.close()


def save_dataset_overview(
    bundle: DatasetBundle,
    feature_count: int,
    output_dir: Path,
    top_k: int | None,
    topk_method: str,
) -> None:
    class_counts = pd.Series(bundle.labels).value_counts().sort_index().to_dict()
    gender_counts = pd.Series(bundle.genders).value_counts().sort_index().to_dict()

    lines = [
        "Dataset Overview",
        "----------------",
        f"Samples: {len(bundle.file_names)}",
        f"Landmarks per sample: {bundle.coords_raw.shape[1]}",
        f"Coordinate dims: {bundle.coords_raw.shape[2]}",
        f"Engineered feature count: {feature_count}",
        f"Class distribution (0,1,2): {class_counts}",
        f"Gender distribution (0=F,1=M): {gender_counts}",
        "Normalization: center at Nasion (idx=2), scale by interocular distance End_L-End_R (idx=15-14)",
        f"Top-k feature selection in CV folds: {top_k if top_k is not None else 'disabled'}",
        f"Top-k scoring method: {topk_method if top_k is not None else 'n/a'}",
    ]

    with (output_dir / "dataset_overview.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D facial landmark multi-class classification pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing Class*_*.txt files")
    parser.add_argument("--landmarks-file", type=str, default="landmarks.txt", help="Landmark definition text file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to write reports and plots")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter-search", type=int, default=16, help="RandomizedSearch iterations for classical models")
    parser.add_argument(
        "--no-flat-coords",
        action="store_true",
        help="Disable raw coordinate flatten features (uses pure engineered geometry features)",
    )
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE plot (faster)")
    parser.add_argument("--rf-only", action="store_true", help="Train only RandomForest model")
    parser.add_argument("--no-xgboost", action="store_true", help="Exclude XGBoost from training")
    parser.add_argument(
        "--hard-negative-13",
        action="store_true",
        help="Use only Class1 vs Class3 samples with a focused hard-negative feature subset",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Apply SelectKBest inside each CV fold with this many features (0 disables)",
    )
    parser.add_argument(
        "--topk-method",
        type=str,
        default="f_classif",
        choices=["f_classif", "chi2"],
        help="Scoring function for SelectKBest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.random_state)

    data_dir = Path(args.data_dir)
    landmarks_file = Path(args.landmarks_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(data_dir=data_dir, landmarks_file=landmarks_file)

    X_tab, feature_names, X_struct = engineer_features(
        coords_norm=bundle.coords_norm,
        genders=bundle.genders,
        landmark_names=bundle.landmark_names,
        include_flatten=(not args.no_flat_coords),
    )
    y = bundle.labels
    class_name_map: Dict[int, str] | None = None
    top_k = args.top_k if args.top_k and args.top_k > 0 else None

    if args.hard_negative_13:
        mask = np.isin(y, [0, 2])
        X_tab = X_tab[mask]
        X_struct = X_struct[mask]
        y = y[mask]

        # Remap labels to contiguous binary labels required by some estimators (e.g., XGBoost)
        # Original meaning is preserved via class_name_map in reporting.
        y = (y == 2).astype(np.int64)
        class_name_map = {0: "Class1", 1: "Class3"}

        X_tab, feature_names, selected_idx = build_hard_negative_feature_subset(X_tab, feature_names)
        selected_feature_df = pd.DataFrame(
            {
                "selected_index": selected_idx,
                "feature_name": feature_names,
            }
        )
        selected_feature_df.to_csv(output_dir / "hard_negative_feature_set_class1_vs_class3.csv", index=False)

    print("Loaded samples:", len(y))
    print("Feature count:", X_tab.shape[1])

    save_dataset_overview(
        bundle=bundle,
        feature_count=X_tab.shape[1],
        output_dir=output_dir,
        top_k=top_k,
        topk_method=args.topk_method,
    )

    selected_models: Sequence[str] | None = None
    if args.rf_only:
        selected_models = ["RandomForest"]
    elif args.no_xgboost:
        selected_models = ["RandomForest", "SVM_RBF", "LogisticRegression", "MLP_PyTorch"]

    results = fit_and_evaluate_models(
        X=X_tab,
        y=y,
        feature_names=feature_names,
        output_dir=output_dir,
        random_state=args.random_state,
        n_iter_search=args.n_iter_search,
        top_k=top_k,
        topk_method=args.topk_method,
        selected_models=selected_models,
        class_name_map=class_name_map,
    )

    best = max(results, key=lambda r: r.f1_macro)
    with (output_dir / "best_model_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Best model by macro-F1: {best.name}\n")
        f.write(f"Accuracy: {best.accuracy:.4f}\n")
        f.write(f"F1-macro: {best.f1_macro:.4f}\n")
        f.write("Best params:\n")
        f.write(json.dumps(best.best_params, indent=2))

    plot_landmarks_3d_examples(X_struct, y, output_dir / "landmarks_3d_examples.png")
    if not args.skip_tsne:
        plot_embeddings(X_tab, y, output_dir, random_state=args.random_state)

    print("\nPipeline complete. Outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
