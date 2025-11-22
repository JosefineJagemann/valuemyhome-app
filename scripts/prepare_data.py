from __future__ import annotations
from pathlib import Path
import click
import pandas as pd

from src.your_project.data.clean import basic_clean, CAT_COLS_DEFAULT
from src.your_project.features.build_features import (
    add_age_features,
    reorder_columns_price_first,
)


@click.command()
@click.option(
    "--raw-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to raw CSV",
)
@click.option(
    "--target", type=str, required=True, help="Target column name (e.g., price)"
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=Path("data/processed/dataset.csv"),
)
def main(raw_file: Path, target: str, out_path: Path) -> None:
    df = pd.read_csv(raw_file)
    click.echo(f"Loaded raw: {df.shape[0]:,} rows x {df.shape[1]} cols")

    df = basic_clean(df, categorical_cols=CAT_COLS_DEFAULT)
    df = add_age_features(df)

    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found after cleaning.")
    df = reorder_columns_price_first(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    click.echo(
        f"Saved processed dataset -> {out_path} ({df.shape[0]:,} rows x {df.shape[1]} cols)"
    )


if __name__ == "__main__":
    main()
