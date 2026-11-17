import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def _make_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.values})
    df["dow"] = np.arange(len(df)) % 7
    df["lag_1"] = df["y"].shift(1).bfill()
    df["lag_7"] = df["y"].shift(7).bfill()
    df["roll_7"] = df["y"].rolling(7, min_periods=1).mean()
    return df

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0

def benchmark_and_forecast(demand_df: pd.DataFrame, facility_id: str, medicine: str, horizon: int = 30):
    sub = demand_df[(demand_df["facility_id"] == facility_id) & (demand_df["medicine"] == medicine)].sort_values("date")
    y = sub["consumption"].values.astype(float)
    feat = _make_features(pd.Series(y))
    X = feat[["dow", "lag_1", "lag_7", "roll_7"]].values

    # Strict chronological 28-day hold-out split (no data leakage)
    split = len(y) - 28
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    models = {
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=80),
        "Random Forest": RandomForestRegressor(random_state=42, max_depth=5, n_estimators=80),
    }

    metrics = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        pred = np.maximum(0, mdl.predict(X_test))
        metrics[name] = round(wape(y_test, pred), 2)

    # Simple baselines on exact same chronological hold-out
    ma_pred = np.full_like(y_test, np.mean(y_train[-7:]))
    sn_pred = np.array([y_train[-7 + (i % 7)] for i in range(len(y_test))])
    metrics["Moving Average (7d)"] = round(wape(y_test, ma_pred), 2)
    metrics["Seasonal Naive"] = round(wape(y_test, sn_pred), 2)

    # Fit Gradient Boosting on full history for 30-day forward forecast
    best_model = GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=80)
    best_model.fit(X, y)

    hist = list(y)
    future_preds = []
    for step in range(horizon):
        dow = (len(hist)) % 7
        lag_1 = hist[-1]
        lag_7 = hist[-7]
        roll_7 = np.mean(hist[-7:])
        p = float(max(1.0, best_model.predict([[dow, lag_1, lag_7, roll_7]])[0]))
        future_preds.append(round(p, 1))
        hist.append(p)

    return {
        "avg_daily_forecast": round(float(np.mean(future_preds)), 1),
        "total_30d_forecast": int(round(sum(future_preds))),
        "daily_series": future_preds,
        "holdout_wape": metrics,
    }
