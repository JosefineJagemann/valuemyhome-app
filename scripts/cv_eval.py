from __future__ import annotations

from pathlib import Path
import click
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# XGBoost is optional; import only if used
try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False


DATA_PATH_DEFAULT = Path("data/processed/dataset.csv")
OUT_DIR = Path("reports/metrics")


def detect_feature_types(df: pd.DataFrame, target: str):
    features = [c for c in df.columns if c != target]
    # treat columns with dtype 'string' as categoricals
    categorical = [c for c in features if str(df[c].dtype) == "string"]
    numeric = [c for c in features if c not in categorical]
    return numeric, categorical


@click.command()
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, path_type=Path),
    default=DATA_PATH_DEFAULT,
)
@click.option("--target", type=str, required=True)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(["linreg", "xgb"], case_sensitive=False),
    required=True,
)
@click.option("--folds", type=int, default=5, show_default=True)
@click.option("--random-state", type=int, default=42, show_default=True)
def main(data_path: Path, target: str, model_name: str, folds: int, random_state: int):
    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not found in {data_path}")

    numeric, categorical = detect_feature_types(df, target)
    X = df.drop(columns=[target])
    y = df[target]

    # Preprocessing: match your training choices
    if model_name == "linreg":
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ],
            remainder="drop",
        )
        model = LinearRegression()
    else:
        if not HAS_XGB:
            raise SystemExit("xgboost is not installed in this environment.")
        pre = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ],
            remainder="drop",
        )
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
        )

    pipe = Pipeline([("pre", pre), ("model", model)])

    # Use metrics compatible with older sklearn:
    scoring = {
        "neg_mse": "neg_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    res = cross_validate(pipe, X, y, scoring=scoring, cv=cv, return_train_score=False)

    # Convert to RMSE/MAE (positive) and summarize
    rmse = np.sqrt(-res["test_neg_mse"])
    mae = -res["test_neg_mae"]
    r2 = res["test_r2"]

    summary = pd.DataFrame(
        {
            "metric": ["RMSE", "MAE", "R2"],
            "mean": [rmse.mean(), mae.mean(), r2.mean()],
            "std": [rmse.std(ddof=1), mae.std(ddof=1), r2.std(ddof=1)],
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"cv_{model_name}_{folds}fold.csv"
    summary.to_csv(out_csv, index=False)

    print("\n=== Cross-Validation Results ===")
    print(f"Model: {model_name} | Folds: {folds}")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean": lambda v: f"{v:,.4f}",
                "std": lambda v: f"{v:,.4f}",
            },
        )
    )
    print(f"\nSaved -> {out_csv}\n")


if __name__ == "__main__":
    main()
