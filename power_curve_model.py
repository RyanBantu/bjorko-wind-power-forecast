#!/usr/bin/env python3
"""Monotonic empirical power curve for the Björkö turbine.

Fits scipy.interpolate.PchipInterpolator to the well-populated 0.5 m/s
bins in outputs/bjorko_power_curve_20hz_hf.csv, then exposes

    power_kw = power_curve_fn(wind_speed_ms)

Output is clipped to 0 below cut-in and to the plateau above the highest
populated bin. The low-wind means (1–3 m/s) are excluded: they are
noisy residual rectifier power, not a physical cubic rise. Sparse tail
bins (n < n_min) are also dropped so the 15 m/s collapse does not pull
the plateau down.

The fitted interpolator is saved as pickle + JSON so it can be reloaded
without refitting.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
DEFAULT_CURVE_CSV = ROOT / "outputs" / "bjorko_power_curve_20hz_hf.csv"
DEFAULT_PKL = ROOT / "outputs" / "power_curve_pchip.pkl"
DEFAULT_JSON = ROOT / "outputs" / "power_curve_pchip.json"
DEFAULT_PLOT = ROOT / "outputs" / "power_curve_pchip_fit.png"

# Bins with fewer samples than this are treated as unpopulated.
DEFAULT_N_MIN = 2000


@dataclass
class PowerCurveParams:
    cut_in_ms: float
    ws_max_ms: float
    plateau_kw: float
    n_min: int
    source: str
    x_ms: list[float]
    y_kw: list[float]


class PowerCurve:
    """Callable P(v) with cut-in / plateau clipping."""

    def __init__(self, interpolator: PchipInterpolator, params: PowerCurveParams):
        self.interpolator = interpolator
        self.params = params

    def __call__(self, wind_speed):
        return power_from_interpolator(self.interpolator, self.params, wind_speed)

    def save(self, pkl_path: Path | str = DEFAULT_PKL, json_path: Path | str = DEFAULT_JSON) -> None:
        pkl_path = Path(pkl_path)
        json_path = Path(json_path)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with pkl_path.open("wb") as f:
            pickle.dump({"interpolator": self.interpolator, "params": asdict(self.params)}, f, protocol=4)
        json_path.write_text(json.dumps(asdict(self.params), indent=2))
        print(f"  wrote {pkl_path}")
        print(f"  wrote {json_path}")


def power_from_interpolator(interpolator: PchipInterpolator, params: PowerCurveParams, wind_speed):
    """Evaluate P(v) [kW], clipped below cut-in and above the last populated bin."""
    scalar = np.isscalar(wind_speed) or (isinstance(wind_speed, np.ndarray) and wind_speed.ndim == 0)
    ws = np.atleast_1d(np.asarray(wind_speed, dtype=float))
    out = np.full(ws.shape, np.nan, dtype=float)
    valid = np.isfinite(ws)
    if valid.any():
        x = np.clip(ws[valid], interpolator.x[0], interpolator.x[-1])
        p = np.asarray(interpolator(x), dtype=float)
        p = np.where(ws[valid] < params.cut_in_ms, 0.0, p)
        p = np.where(ws[valid] > params.ws_max_ms, params.plateau_kw, p)
        p = np.clip(p, 0.0, params.plateau_kw)
        out[valid] = p
    if scalar:
        return float(out[0])
    return out


def _well_populated(curve: pd.DataFrame, n_min: int) -> pd.DataFrame:
    if "n_samples" not in curve.columns:
        raise ValueError("power-curve CSV must include n_samples")
    kept = curve.loc[curve["n_samples"] >= n_min].sort_values("ws_bin_center_ms").copy()
    if kept.empty:
        raise ValueError(f"no bins with n_samples >= {n_min}")
    return kept


def _detect_cut_in(populated: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Walk left from the peak to the trough that starts the cubic rise."""
    y = populated["mean_power_kw"].to_numpy()
    x = populated["ws_bin_center_ms"].to_numpy()
    i_peak = int(np.argmax(y))
    i_trough = i_peak
    for i in range(i_peak, 0, -1):
        # Still on the rising flank while y[i-1] <= y[i] + small tolerance.
        if y[i - 1] <= y[i] + 0.25:
            i_trough = i - 1
        else:
            break
    # Cut-in at the left edge of the trough bin (or 0.5 m/s below its center).
    cut_in = float(populated.iloc[i_trough]["ws_bin_left_ms"])
    rising = populated.iloc[i_trough:].copy()
    print(
        f"  peak {x[i_peak]:.2f} m/s → {y[i_peak]:.2f} kW; "
        f"trough {x[i_trough]:.2f} m/s → {y[i_trough]:.2f} kW; "
        f"cut-in {cut_in:.2f} m/s"
    )
    return cut_in, rising


def fit_power_curve(
    curve_csv: Path | str = DEFAULT_CURVE_CSV,
    n_min: int = DEFAULT_N_MIN,
) -> PowerCurve:
    """Fit a globally monotonic PCHIP on well-populated bin centers vs mean power."""
    curve_csv = Path(curve_csv)
    raw = pd.read_csv(curve_csv)
    populated = _well_populated(raw, n_min)
    print(
        f"  well-populated bins (n≥{n_min}): {len(populated)} / {len(raw)}  "
        f"WS {populated['ws_bin_center_ms'].min():.2f}–"
        f"{populated['ws_bin_center_ms'].max():.2f} m/s"
    )

    cut_in, rising = _detect_cut_in(populated)
    # Drop the trough knot itself: it sits at ~3.5 kW right next to cut-in and
    # makes PCHIP jump like a step. Rise from (cut_in, 0) through the next bins.
    if len(rising) > 1:
        rising = rising.iloc[1:]
    x = rising["ws_bin_center_ms"].to_numpy(dtype=float)
    y = rising["mean_power_kw"].to_numpy(dtype=float)
    # Force a physical curve: non-decreasing from the trough, starting at 0 kW.
    y = np.maximum.accumulate(y)
    if x[0] > cut_in:
        x = np.insert(x, 0, cut_in)
        y = np.insert(y, 0, 0.0)
    else:
        y[0] = 0.0

    interpolator = PchipInterpolator(x, y, extrapolate=False)
    params = PowerCurveParams(
        cut_in_ms=float(cut_in),
        ws_max_ms=float(x[-1]),
        plateau_kw=float(y[-1]),
        n_min=int(n_min),
        source=str(curve_csv),
        x_ms=[float(v) for v in x],
        y_kw=[float(v) for v in y],
    )
    print(f"  plateau {params.plateau_kw:.2f} kW above {params.ws_max_ms:.2f} m/s")
    print("  fit knots (m/s → kW):")
    for xv, yv in zip(x, y):
        print(f"    {xv:6.2f}  {yv:7.3f}")
    return PowerCurve(interpolator, params)


def load_power_curve(pkl_path: Path | str = DEFAULT_PKL) -> PowerCurve:
    with Path(pkl_path).open("rb") as f:
        payload = pickle.load(f)
    params = PowerCurveParams(**payload["params"])
    interpolator = payload.get("interpolator")
    if interpolator is None:
        interpolator = PchipInterpolator(params.x_ms, params.y_kw, extrapolate=False)
    return PowerCurve(interpolator, params)


def load_power_curve_from_json(json_path: Path | str = DEFAULT_JSON) -> PowerCurve:
    """Rebuild the interpolator from the JSON knots (pickle-independent)."""
    params = PowerCurveParams(**json.loads(Path(json_path).read_text()))
    interpolator = PchipInterpolator(params.x_ms, params.y_kw, extrapolate=False)
    return PowerCurve(interpolator, params)


# Public alias used by the forecasting pipeline.
def power_curve_fn(wind_speed, model: PowerCurve | None = None):
    if model is None:
        model = load_power_curve()
    return model(wind_speed)


def plot_fit(
    model: PowerCurve,
    curve_csv: Path | str = DEFAULT_CURVE_CSV,
    png_path: Path | str = DEFAULT_PLOT,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = pd.read_csv(curve_csv)
    png_path = Path(png_path)
    ws_grid = np.linspace(0.0, 16.0, 401)
    p_fit = model(ws_grid)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sizes = 20 + 40 * np.sqrt(raw["n_samples"].to_numpy() / raw["n_samples"].max())
    used = raw["n_samples"] >= model.params.n_min
    ax.scatter(
        raw.loc[~used, "ws_bin_center_ms"],
        raw.loc[~used, "mean_power_kw"],
        s=sizes[~used.to_numpy()],
        c="#b0b8c0",
        label="Excluded bin mean (sparse / below cut-in)",
        zorder=2,
    )
    ax.scatter(
        raw.loc[used, "ws_bin_center_ms"],
        raw.loc[used, "mean_power_kw"],
        s=sizes[used.to_numpy()],
        c="#1f4e79",
        label="Well-populated bin mean",
        zorder=3,
    )
    ax.plot(ws_grid, p_fit, color="#c45c26", linewidth=2.2, label="PCHIP P(v)", zorder=4)
    ax.axvline(model.params.cut_in_ms, color="#7a8b99", linestyle="--", linewidth=1, label=f"Cut-in {model.params.cut_in_ms:.2f} m/s")
    ax.axhline(model.params.plateau_kw, color="#7a8b99", linestyle=":", linewidth=1, label=f"Plateau {model.params.plateau_kw:.2f} kW")
    ax.set_xlabel("Wind speed, 30 m mast (m/s)")
    ax.set_ylabel("Electrical power (kW)")
    ax.set_title("Björkö fitted empirical power curve (PCHIP, Running-mode bins)")
    ax.set_xlim(0, 16)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {png_path}")
    return png_path


def main() -> PowerCurve:
    print("=" * 72)
    print("FIT MONOTONIC PCHIP POWER CURVE")
    print("=" * 72)
    model = fit_power_curve()
    model.save()
    plot_fit(model)

    # Quick evaluation checks
    checks = [0.0, 2.0, model.params.cut_in_ms, 6.0, 8.0, 10.0, 14.0, 20.0]
    print("  power_curve_fn checks:")
    for v in checks:
        print(f"    {v:5.2f} m/s → {model(v):7.3f} kW")
    return model


if __name__ == "__main__":
    main()
