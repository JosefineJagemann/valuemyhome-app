from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib

# Project root = .../house-price-predictor (works anywhere)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def _pick_first_existing(patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        matches = sorted(MODELS_DIR.glob(pat))
        if matches:
            return matches[0]
    return None


def _load_single_model(path: Path):
    artifact = joblib.load(path)

    pipe = artifact.get("pipeline") or artifact.get("pipe") or artifact.get("model")
    if pipe is None:
        raise KeyError(
            f"Could not find a 'pipeline' (or 'pipe'/'model') key in {path.name}"
        )

    feature_cols = (
        artifact.get("feature_columns")
        or artifact.get("features")
        or artifact.get("feature_names")
        or artifact.get("X_columns")
        or artifact.get("X_cols")
        or artifact.get("columns")
        or []
    )
    target_name = artifact.get("target") or artifact.get("target_name") or "price"
    base_defaults: Dict = artifact.get("feature_defaults", {})

    return pipe, list(feature_cols), target_name, base_defaults


def load_models_and_columns():
    MODELS_DIR.mkdir(exist_ok=True)

    xgb_path = _pick_first_existing(
        ["*xgb*.joblib", "*xgboost*.joblib", "model-xgb-*.joblib"]
    )
    lr_path = _pick_first_existing(
        ["*linreg*.joblib", "*linear*.joblib", "model-lr-*.joblib"]
    )

    pipe_lr = pipe_xgb = None
    feature_cols: List[str] = []
    target_name = "price"
    base_defaults: Dict = {}

    if xgb_path is not None:
        pipe_xgb, cols, tgt, defaults = _load_single_model(xgb_path)
        if cols:
            feature_cols = cols
        target_name = tgt or target_name
        base_defaults = defaults or base_defaults

    if lr_path is not None:
        pipe_lr_tmp, cols, tgt, defaults = _load_single_model(lr_path)
        pipe_lr = pipe_lr_tmp
        if not feature_cols and cols:
            feature_cols = cols
        if not target_name and tgt:
            target_name = tgt
        if not base_defaults and defaults:
            base_defaults = defaults

    if not feature_cols:
        feature_cols = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_above",
            "sqft_basement",
            "floors",
            "house_age",
            "zipcode",
            "lat",
            "long",
        ]

    if not base_defaults:
        base_defaults = {c: 0 for c in feature_cols}

    return (
        pipe_lr,
        pipe_xgb,
        feature_cols,
        target_name,
        lr_path,
        xgb_path,
        base_defaults,
    )
