"""Stacking Regressor training module for House Price prediction."""

import os
import joblib
import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import preprocess_data


def build_stacking_model() -> StackingRegressor:
    """Build and return the configured StackingRegressor."""
    base_estimators = [
        ("rf",  RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
        ("gb",  GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42)),
        ("svr", Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10, epsilon=0.1, kernel="rbf"))])),
        ("knn", KNeighborsRegressor(n_neighbors=8, weights="distance")),
    ]

    meta_learner = Ridge(alpha=5.0)

    model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )
    return model


def train_and_save(data_path: str, model_path: str) -> None:
    """Load data, train stacking model, and persist to disk."""
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)

    print("Building stacking regressor …")
    model = build_stacking_model()

    print("Training (this may take a minute) …")
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅  Model saved → {model_path}")


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_and_save(
        data_path=os.path.join(BASE, "data", "houses.csv"),
        model_path=os.path.join(BASE, "models", "stacking_model.pkl"),
    )
