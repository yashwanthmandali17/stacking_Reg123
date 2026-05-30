"""Regression evaluation metrics for the House Price Stacking Regressor."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    """
    Compute regression metrics on the test set.

    Returns
    -------
    metrics : dict  (r2, mae, rmse, mape)
    y_test_orig : np.ndarray  (original scale $)
    y_pred_orig : np.ndarray  (original scale $)
    """
    y_pred_log = model.predict(X_test)

    # Invert log1p transform
    y_test_orig = np.expm1(np.array(y_test))
    y_pred_orig = np.expm1(y_pred_log)

    metrics = {
        "r2":   r2_score(y_test_orig, y_pred_orig),
        "mae":  mean_absolute_error(y_test_orig, y_pred_orig),
        "rmse": np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
        "mape": np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1e-9))) * 100,
    }
    return metrics, y_test_orig, y_pred_orig
