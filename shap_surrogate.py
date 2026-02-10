# shap_surrogate.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np


def _train_test_split(X: np.ndarray, y: np.ndarray, test_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_split)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import r2_score, mean_squared_error
    pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def _fit_xgb(X_tr, y_tr, X_te, y_te, seed: int):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return model


def _fit_lgbm(X_tr, y_tr, X_te, y_te, seed: int):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.02,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="l2", verbose=-1)
    return model


def _fit_rf(X_tr, y_tr, seed: int):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=800,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def train_surrogate_and_shap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_explain: np.ndarray,
    surrogate: str,
    test_split: float,
    seed: int,
    feature_names: Optional[List[str]],
    outdir: Path,
    save_beeswarm: bool = True,
) -> Dict[str, Any]:
    """
    Train a tree surrogate f_hat(s) ≈ Q(s,pi(s)) and compute TreeSHAP.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    X_tr, X_te, y_tr, y_te = _train_test_split(X_train, y_train, test_split=test_split, seed=seed)

    if surrogate == "xgb":
        model = _fit_xgb(X_tr, y_tr, X_te, y_te, seed=seed)
    elif surrogate == "lgbm":
        model = _fit_lgbm(X_tr, y_tr, X_te, y_te, seed=seed)
    elif surrogate == "rf":
        model = _fit_rf(X_tr, y_tr, seed=seed)
    else:
        raise ValueError(f"Unknown surrogate: {surrogate}")

    metrics = _evaluate(model, X_te, y_te)

    # TreeSHAP
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)  # (n_explain, n_features)
    base_value = explainer.expected_value

    # Save SHAP beeswarm only when requested.
    if save_beeswarm:
        try:
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(outdir / "shap_beeswarm.png", dpi=220)
            plt.close()
        except Exception as e:
            (outdir / "shap_plot_warning.txt").write_text(f"{e}\n", encoding="utf-8")

    return {
        "model": model,
        "metrics": metrics,
        "shap_values": np.asarray(shap_values),
        "base_value": base_value,
    }
