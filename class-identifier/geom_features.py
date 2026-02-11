from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LANDMARKS: list[str] = [
    "Tr",
    "G",
    "N",
    "Prn",
    "Col",
    "Sn",
    "Phi",
    "Ls",
    "Stm",
    "Li",
    "Pg",
    "Gn",
    "Me",
    "Ex_R",
    "End_R",
    "End_L",
    "Ex_L",
    "Al_R",
    "Al_L",
    "Ch_R",
    "Ch_L",
    "Go_R",
    "Go_L",
]


def _p(row: pd.Series, name: str) -> np.ndarray:
    return np.array([row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"]], dtype=float)


def dist3(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in R^3."""
    return float(np.linalg.norm(a - b))


def angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray, *, degrees: bool = True) -> float:
    """Angle ABC in 3D formed by vectors BA and BC."""
    ba = a - b
    bc = c - b
    ba_n = np.linalg.norm(ba)
    bc_n = np.linalg.norm(bc)
    if ba_n == 0.0 or bc_n == 0.0:
        return float("nan")
    cos_theta = float(np.dot(ba, bc) / (ba_n * bc_n))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = float(np.arccos(cos_theta))
    return float(np.degrees(theta) if degrees else theta)


def safe_ratio(num: float, den: float) -> float:
    if den == 0.0 or np.isnan(den):
        return float("nan")
    return float(num / den)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0:
        return np.full(3, np.nan)
    return v / n


@dataclass(frozen=True)
class LocalFrame:
    origin: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    scale: float


def build_local_frame(row: pd.Series) -> LocalFrame:
    """Construct a subject-specific anatomical frame.

    - Origin: N (nasion)
    - x-axis: left-right via Ex_R - Ex_L
    - y-axis: superior-inferior via Tr - Me
    - z-axis: antero-posterior as x × y, oriented to point toward Prn

    This yields signed, orientation-stabilized features (useful for sagittal classification)
    while remaining robust to global coordinate system differences.
    """

    origin = _p(row, "N")
    x_axis = _unit(_p(row, "Ex_R") - _p(row, "Ex_L"))
    y0 = _unit(_p(row, "Tr") - _p(row, "Me"))
    z_axis = _unit(np.cross(x_axis, y0))

    # Ensure z points anteriorly (towards the nose tip Prn)
    prn_vec = _p(row, "Prn") - origin
    if np.dot(z_axis, prn_vec) < 0:
        z_axis = -z_axis

    # Re-orthonormalize y to enforce right-handed basis
    y_axis = _unit(np.cross(z_axis, x_axis))

    # Scale: biocular width as a stable soft tissue reference
    scale = dist3(_p(row, "Ex_R"), _p(row, "Ex_L"))
    if scale == 0.0:
        scale = 1.0

    return LocalFrame(origin=origin, x=x_axis, y=y_axis, z=z_axis, scale=scale)


def to_local(frame: LocalFrame, point: np.ndarray) -> np.ndarray:
    v = point - frame.origin
    coords = np.array([np.dot(v, frame.x), np.dot(v, frame.y), np.dot(v, frame.z)], dtype=float)
    return coords / frame.scale


def compute_features(
    df: pd.DataFrame,
    *,
    include_aligned_coords: bool = False,
    include_gender: bool = True,
) -> pd.DataFrame:
    """Compute a publication-oriented feature table.

    Returns a DataFrame aligned with df index.
    """

    feature_rows: list[dict[str, float]] = []

    for _, row in df.iterrows():
        frame = build_local_frame(row)

        # Core points
        pts = {name: _p(row, name) for name in LANDMARKS}

        # Local (signed) projections for sagittal profile
        local = {name: to_local(frame, pts[name]) for name in pts}

        def z(name: str) -> float:
            return float(local[name][2])

        def y(name: str) -> float:
            return float(local[name][1])

        # Reference distances (scale-free via division by biocular width)
        d_ex = 1.0  # already scaled in local coords

        # Distances (normalized)
        d_n_sn = dist3(local["N"], local["Sn"])
        d_n_me = dist3(local["N"], local["Me"])
        d_sn_me = dist3(local["Sn"], local["Me"])
        d_n_prn = dist3(local["N"], local["Prn"])
        d_sn_prn = dist3(local["Sn"], local["Prn"])
        d_sn_pg = dist3(local["Sn"], local["Pg"])
        d_n_pg = dist3(local["N"], local["Pg"])
        d_g_sn = dist3(local["G"], local["Sn"])
        d_ch = dist3(local["Ch_R"], local["Ch_L"])
        d_al = dist3(local["Al_R"], local["Al_L"])
        d_end = dist3(local["End_R"], local["End_L"])
        d_go = dist3(local["Go_R"], local["Go_L"])

        # Angles (degrees)
        ang_convexity_sn = angle3(local["G"], local["Sn"], local["Pg"])  # soft-tissue convexity
        ang_convexity_n = angle3(local["G"], local["N"], local["Pg"])  # facial convexity at N
        ang_nasolabial = angle3(local["Col"], local["Sn"], local["Ls"])  # nasolabial
        ang_mentolabial = angle3(local["Li"], local["Phi"], local["Pg"])  # approx mentolabial
        ang_profile_n_sn_pg = angle3(local["N"], local["Sn"], local["Pg"])  # profile angle at Sn

        # Signed sagittal relationships (z-axis is anterior-positive)
        z_pg_minus_sn = z("Pg") - z("Sn")
        z_prn_minus_sn = z("Prn") - z("Sn")
        z_ls_minus_sn = z("Ls") - z("Sn")
        z_li_minus_sn = z("Li") - z("Sn")
        z_g_minus_n = z("G") - z("N")

        # Vertical proportions (y-axis is superior-positive)
        y_sn_minus_n = y("Sn") - y("N")
        y_me_minus_sn = y("Me") - y("Sn")

        feature_row: dict[str, float] = {
            # widths (normalized)
            "width_biocular_ex": d_ex,
            "width_intercanthal_end": d_end,
            "width_nasal_al": d_al,
            "width_mouth_ch": d_ch,
            "width_bigonal_go": d_go,
            # key normalized distances
            "d_N_Sn": d_n_sn,
            "d_N_Me": d_n_me,
            "d_Sn_Me": d_sn_me,
            "d_N_Prn": d_n_prn,
            "d_Sn_Prn": d_sn_prn,
            "d_Sn_Pg": d_sn_pg,
            "d_N_Pg": d_n_pg,
            "d_G_Sn": d_g_sn,
            # clinically-inspired ratios (dimensionless)
            "ratio_lower_face": safe_ratio(d_sn_me, d_n_me),
            "ratio_nose_to_face": safe_ratio(d_n_prn, d_n_me),
            "ratio_chin_to_face": safe_ratio(d_sn_pg, d_n_me),
            "ratio_mouth_to_nose_width": safe_ratio(d_ch, d_al),
            "ratio_nasal_to_biocular": safe_ratio(d_al, d_ex),
            # angles
            "ang_convexity_G_Sn_Pg": ang_convexity_sn,
            "ang_convexity_G_N_Pg": ang_convexity_n,
            "ang_profile_N_Sn_Pg": ang_profile_n_sn_pg,
            "ang_nasolabial_Col_Sn_Ls": ang_nasolabial,
            "ang_mentolabial_Li_Phi_Pg": ang_mentolabial,
            # signed sagittal (key for Class I/II/III)
            "z_Pg_minus_Sn": z_pg_minus_sn,
            "z_Prn_minus_Sn": z_prn_minus_sn,
            "z_Ls_minus_Sn": z_ls_minus_sn,
            "z_Li_minus_Sn": z_li_minus_sn,
            "z_G_minus_N": z_g_minus_n,
            # vertical
            "y_Sn_minus_N": y_sn_minus_n,
            "y_Me_minus_Sn": y_me_minus_sn,
        }

        if include_aligned_coords:
            for name in LANDMARKS:
                feature_row[f"{name}_lx"] = float(local[name][0])
                feature_row[f"{name}_ly"] = float(local[name][1])
                feature_row[f"{name}_lz"] = float(local[name][2])

        if include_gender:
            feature_row["gender"] = float(row.get("gender", np.nan))

        feature_rows.append(feature_row)

    return pd.DataFrame(feature_rows, index=df.index)


def load_master_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if "class" not in df.columns:
        raise ValueError("Expected a 'class' column in master_dataset.csv")
    return df


def parse_landmark_txt(txt_path: str | Path) -> np.ndarray:
    """Parse a landmark .txt file into an array of shape (23, 3).

    Expected per-line format is comma-separated with x,y,z in columns 2..4
    (consistent with build_dataset.py). Landmark names are assumed to follow
    the fixed LANDMARKS order.
    """

    txt_path = Path(txt_path)
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    coords: list[list[float]] = []

    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        coords.append([x, y, z])
        if len(coords) == len(LANDMARKS):
            break

    arr = np.asarray(coords, dtype=float)
    if arr.shape != (len(LANDMARKS), 3):
        raise ValueError(
            f"Expected {len(LANDMARKS)} landmarks with 3D coords; got shape {arr.shape} from {txt_path}"
        )
    return arr


def dataframe_from_txt(
    txt_path: str | Path,
    *,
    subject_id: str | None = None,
    gender: int | None = None,
) -> pd.DataFrame:
    """Create a one-row DataFrame with the same coordinate columns as master_dataset.csv."""

    coords = parse_landmark_txt(txt_path)
    row: dict[str, object] = {
        "id": subject_id if subject_id is not None else Path(txt_path).stem,
        "gender": np.nan if gender is None else int(gender),
        "class": np.nan,
    }

    for name, (x, y, z) in zip(LANDMARKS, coords):
        row[f"{name}_x"] = float(x)
        row[f"{name}_y"] = float(y)
        row[f"{name}_z"] = float(z)

    return pd.DataFrame([row])
