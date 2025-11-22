from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


def detect_feature_types(df: pd.DataFrame, target: str):
    feature_cols = [c for c in df.columns if c != target]
    categorical = [c for c in feature_cols if str(df[c].dtype) == "string"]
    numeric = [c for c in feature_cols if c not in categorical]
    return numeric, categorical


@click.command()
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed/dataset.csv"),
)
@click.option("--target", type=str, required=True)
def main(data_path: Path, target: str):
    # Load data
    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not found in {data_path}")

    # Feature types & split
    numeric, categorical = detect_feature_types(df, target)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess: passthrough numeric, one-hot categorical
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )

    # XGBoost baseline
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("model", xgb)])

    # Fit
    pipe.fit(X_train, y_train)

    # Metrics
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(mse**0.5)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"R²:   {r2:,.4f}")

    # Save artifacts
    Path("models").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = Path("models") / f"model-xgb-{stamp}.joblib"

    # Save pipeline + target + feature columns
    model_package = {
        "pipeline": pipe,
        "target": target,
        "columns": X_train.columns.tolist(),
    }
    joblib.dump(model_package, model_path)

    print(f"\nSaved model -> {model_path}")

    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    with open(Path("reports/metrics") / f"xgb-{stamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved metrics -> reports/metrics/xgb-{stamp}.json")


if __name__ == "__main__":
    main()
