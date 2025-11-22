from __future__ import annotations

from pathlib import Path
import pandas as pd

import click
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_PATH_DEFAULT = Path("data/processed/dataset.csv")
MODELS_DIR = Path("models")
FIG_DIR = Path("reports/figures")
PRED_CSV = Path("reports/metrics/predictions_xgb.csv")


def latest_xgb_model() -> Path | None:
    """Return the newest saved XGBoost model artifact, or None if missing."""
    candidates = sorted(MODELS_DIR.glob("model-xgb-*.joblib"))
    return candidates[-1] if candidates else None


@click.command()
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, path_type=Path),
    default=DATA_PATH_DEFAULT,
    help="Processed dataset CSV (same used in training).",
)
@click.option("--target", type=str, required=True, help="Target column, e.g. price")
@click.option(
    "--model-path",
    type=click.Path(path_type=Path),
    help="Optional: path to a specific saved model .joblib. If omitted, loads latest XGB model.",
)
def main(data_path: Path, target: str, model_path: Path | None):
    # Load data
    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not found in {data_path}")

    # Pick model
    if model_path is None:
        model_path = latest_xgb_model()
        if model_path is None:
            raise SystemExit(
                "No XGB model found in models/. Train with scripts/train_xgb.py first."
            )
    print(f"Using model: {model_path}")

    artifact = joblib.load(model_path)
    pipe = artifact["pipeline"]

    # Recreate the same split used in training
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Predict on test set
    y_pred = pipe.predict(X_test)

    # Create output folders
    from pathlib import Path

    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_path = Path("reports/metrics/predictions_xgb.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(pred_path, index=False)
    print(f"Saved predictions -> {pred_path}")

    # === Plot 1: Predicted vs Actual ===
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    lo, hi = (
        float(min(y_test.min(), y_pred.min())),
        float(max(y_test.max(), y_pred.max())),
    )
    plt.plot([lo, hi], [lo, hi], color="red", lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual — XGBoost")
    out1 = Path("reports/figures/pred_vs_actual_xgb.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved figure -> {out1}")

    # === Plot 2: Residuals histogram ===
    residuals = y_pred - y_test.values
    plt.figure()
    plt.hist(residuals, bins=50, color="steelblue", edgecolor="black")
    plt.xlabel("Residual (Predicted - Actual)")
    plt.ylabel("Count")
    plt.title("Residuals Histogram — XGBoost")
    out2 = Path("reports/figures/residuals_hist_xgb.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved figure -> {out2}")

    print("Done!")


if __name__ == "__main__":
    main()
