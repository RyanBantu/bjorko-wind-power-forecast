#!/usr/bin/env python3
"""Linear wind-speed bias correction: observed WS30 ≈ a * forecast_ws + b.

Fitted on the 311 aligned backtest points (forecast 15-min wind vs mast WS30).
Applied to NWP wind *before* the empirical power curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ALIGNED_CSV = OUT / "backtest_aligned_points.csv"
COEFF_JSON = OUT / "wind_bias_correction.json"


def signed_bias(forecast: np.ndarray, observed: np.ndarray) -> float:
    """Mean signed error of the forecast: mean(forecast − observed)."""
    return float(np.mean(np.asarray(forecast, dtype=float) - np.asarray(observed, dtype=float)))


def fit_linear_correction(
    forecast_ws: np.ndarray,
    observed_ws: np.ndarray,
) -> dict:
    """OLS: observed ≈ a * forecast + b."""
    x = np.asarray(forecast_ws, dtype=float)
    y = np.asarray(observed_ws, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two points to fit a linear correction")
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "a": float(a),
        "b": float(b),
        "r2": float(r2),
        "n_points": int(x.size),
        "bias_fcst_minus_obs_ms": signed_bias(x, y),
        "model": "observed_ws30 = a * forecast_ws + b",
        "source": str(ALIGNED_CSV),
    }


def save_coefficients(coeffs: dict, path: Path | str = COEFF_JSON) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coeffs, indent=2))
    return path


def load_coefficients(path: Path | str = COEFF_JSON) -> dict:
    return json.loads(Path(path).read_text())


def apply_wind_correction(wind, coeffs: dict | None = None):
    """corrected = a * wind + b, clipped at 0. Accepts ndarray or Series."""
    if coeffs is None:
        coeffs = load_coefficients()
    a, b = float(coeffs["a"]), float(coeffs["b"])
    if isinstance(wind, pd.Series):
        out = a * wind.astype(float) + b
        return out.clip(lower=0.0)
    arr = np.asarray(wind, dtype=float)
    return np.clip(a * arr + b, 0.0, None)


def leave_one_day_out(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a, b on all days except one; apply to the held-out day.

    Returns
    -------
    oos_points
        Copy of ``aligned`` with ``fcst_ws_ms`` / implied power targets replaced
        by the fold-specific correction (caller applies the power curve).
    folds
        One row per held-out day: a, b, r2, n_train, n_test.
    """
    if "date" not in aligned.columns:
        raise ValueError("aligned frame must include a date column")
    days = sorted(aligned["date"].astype(str).unique())
    oos_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for day in days:
        train = aligned.loc[aligned["date"].astype(str) != day]
        test = aligned.loc[aligned["date"].astype(str) == day].copy()
        if len(train) < 2 or test.empty:
            continue
        coeffs = fit_linear_correction(train["fcst_ws_ms"], train["obs_ws_ms"])
        test["fcst_ws_raw_ms"] = test["fcst_ws_ms"]
        test["fcst_ws_ms"] = apply_wind_correction(test["fcst_ws_raw_ms"].to_numpy(), coeffs)
        test["fold_a"] = coeffs["a"]
        test["fold_b"] = coeffs["b"]
        oos_parts.append(test)
        fold_rows.append(
            {
                "held_out_date": day,
                "a": coeffs["a"],
                "b": coeffs["b"],
                "r2_train": coeffs["r2"],
                "n_train": coeffs["n_points"],
                "n_test": int(len(test)),
            }
        )
    if not oos_parts:
        raise RuntimeError("leave-one-day-out produced no folds")
    return pd.concat(oos_parts, ignore_index=True), pd.DataFrame(fold_rows)


def fold_stability(folds: pd.DataFrame) -> dict:
    return {
        "n_folds": int(len(folds)),
        "a_mean": float(folds["a"].mean()),
        "a_std": float(folds["a"].std(ddof=1)),
        "a_min": float(folds["a"].min()),
        "a_max": float(folds["a"].max()),
        "b_mean": float(folds["b"].mean()),
        "b_std": float(folds["b"].std(ddof=1)),
        "b_min": float(folds["b"].min()),
        "b_max": float(folds["b"].max()),
    }


def fit_from_aligned(aligned_csv: Path | str = ALIGNED_CSV) -> dict:
    aligned = pd.read_csv(aligned_csv)
    coeffs = fit_linear_correction(aligned["fcst_ws_ms"], aligned["obs_ws_ms"])
    save_coefficients(coeffs)
    return coeffs


def main() -> None:
    coeffs = fit_from_aligned()
    print(
        f"bias (fcst − obs) = {coeffs['bias_fcst_minus_obs_ms']:+.3f} m/s   "
        f"obs ≈ {coeffs['a']:.4f} * fcst + {coeffs['b']:.4f}   "
        f"R²={coeffs['r2']:.3f}   n={coeffs['n_points']}"
    )
    print(f"  wrote {COEFF_JSON}")


if __name__ == "__main__":
    main()
