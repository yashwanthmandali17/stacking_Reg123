"""Data preprocessing pipeline for House Price Stacking Regressor."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    # --- Feature Engineering ---
    df["House_Age"]    = 2023 - df["Year_Built"]
    df["Remodel_Age"]  = 2023 - df["Remodel_Year"]
    df["Remodeled"]    = (df["Remodel_Year"] != df["Year_Built"]).astype(int)
    df["Price_Per_SqFt_approx"] = df["Lot_Area"] / (df["Total_SqFt"] + 1)
    df["Total_Rooms"]  = df["Bedrooms"] + df["Bathrooms"]

    df.drop(columns=["Year_Built", "Remodel_Year"], inplace=True)

    # --- Encode categoricals ---
    cat_cols = ["Neighborhood", "House_Condition", "Garage_Type", "Heating_Type"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # --- Split features / target ---
    target = "Sale_Price"
    X = df.drop(columns=[target])
    y = np.log1p(df[target])          # log-transform target for better regression

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- Scale ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, list(X.columns)
