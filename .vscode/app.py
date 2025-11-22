import numpy as np
import pandas as pd


def prepare_features(form_dict, model_columns):
    """
    Takes the form input and returns a DataFrame with ALL columns
    expected by the model. Missing columns are filled automatically.
    """

    # Convert form input to a DataFrame
    df = pd.DataFrame([form_dict])

    # Create any engineered features your training used
    if "yr_built" in model_columns:
        df["house_age"] = 2025 - df["yr_built"]

    if "yr_renovated" in model_columns:
        df["renovation_age"] = df["yr_renovated"].replace(0, np.nan)
        df["renovation_age"] = 2025 - df["renovation_age"]
        df["renovation_age"] = df["renovation_age"].fillna(0)

    if "is_renovated" in model_columns:
        df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)

    # Add **missing columns** with default values
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0  # safe default

    # Reorder columns
    df = df[model_columns]

    return df
