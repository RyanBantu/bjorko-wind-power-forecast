#!/usr/bin/env python3
"""Backtest the historical NWP → power pipeline on Björkö campaign days.

For each unique UTC date in outputs/bjorko_15min_running_20hz.csv, pull the
Open-Meteo Historical Forecast (stitched short-lead-time proxy), convert it
to 15-minute wind and power, and score it against observed WS30 and power.

This is not a true archived day-ahead backtest: the 44 days are
non-continuous SHM campaigns, and the NWP series is a stitched short-lead
proxy, not a single-run forecast issued the day before.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_pipeline import forecast_power
from power_curve_model import load_power_curve
from wind_bias_correction import (
    apply_wind_correction,
    fit_from_aligned,
    fold_stability,
    leave_one_day_out,
    save_coefficients,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OBS_CSV = OUT / "bjorko_15min_running_20hz.csv"
RESULTS_CSV = OUT / "backtest_results.csv"
RESULTS_CORRECTED_CSV = OUT / "backtest_results_corrected.csv"
RESULTS_OOS_CSV = OUT / "backtest_results_oos.csv"
ALIGNED_CSV = OUT / "backtest_aligned_points.csv"
ALIGNED_CORRECTED_CSV = OUT / "backtest_aligned_points_corrected.csv"
ALIGNED_OOS_CSV = OUT / "backtest_aligned_points_oos.csv"
FOLDS_CSV = OUT / "wind_bias_correction_lodo_folds.csv"
SCATTER_PNG = OUT / "backtest_power_scatter.png"
TIMESERIES_PNG = OUT / "backtest_power_timeseries.png"

TOLERANCE = pd.Timedelta(minutes=2)
MIN_POINTS_FOR_DAY = 1
N_TIMESERIES_DAYS = 4


def _load_observations(path: Path = OBS_CSV) -> pd.DataFrame:
    obs = pd.read_csv(path, parse_dates=["timestamp"])
    if obs["timestamp"].dt.tz is None:
        obs["timestamp"] = obs["timestamp"].dt.tz_localize("UTC")
    else:
        obs["timestamp"] = obs["timestamp"].dt.tz_convert("UTC")
    obs = obs.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    obs["date"] = obs["timestamp"].dt.date
    return obs


def _align_day(obs_day: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    left = obs_day[["timestamp", "wind_speed_ws30_ms", "power_kw"]].rename(
        columns={"wind_speed_ws30_ms": "obs_ws_ms", "power_kw": "obs_power_kw"}
    )
    right = forecast.reset_index()
    # forecast index may be unnamed or "timestamp"
    time_col = "timestamp" if "timestamp" in right.columns else right.columns[0]
    right = right.rename(
        columns={
            time_col: "timestamp",
            "wind_speed_15min_ms": "fcst_ws_ms",
            "power_kw": "fcst_power_kw",
        }
    )
    right = right[["timestamp", "fcst_ws_ms", "fcst_power_kw"]].sort_values("timestamp")
    if right["timestamp"].dt.tz is None:
        right["timestamp"] = right["timestamp"].dt.tz_localize("UTC")
    else:
        right["timestamp"] = right["timestamp"].dt.tz_convert("UTC")

    aligned = pd.merge_asof(
        left.sort_values("timestamp"),
        right,
        on="timestamp",
        direction="nearest",
        tolerance=TOLERANCE,
    )
    aligned = aligned.dropna(subset=["obs_ws_ms", "obs_power_kw", "fcst_ws_ms", "fcst_power_kw"])
    return aligned


def _metrics(obs: np.ndarray, fcst: np.ndarray) -> dict[str, float]:
    err = np.asarray(fcst, dtype=float) - np.asarray(obs, dtype=float)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def _fetch_historical(day, retries: int = 3, apply_bias_correction: bool = False) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return forecast_power(
                date=day,
                use_live=False,
                add_variability=False,
                apply_bias_correction=apply_bias_correction,
            )
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"historical forecast failed for {day}: {last_err}") from last_err


def _day_rows_from_aligned(aligned_all: pd.DataFrame, plateau: float) -> pd.DataFrame:
    rows = []
    for day, block in aligned_all.groupby("date", sort=True):
        wind = _metrics(block["obs_ws_ms"], block["fcst_ws_ms"])
        power = _metrics(block["obs_power_kw"], block["fcst_power_kw"])
        rows.append(
            {
                "date": day,
                "n_points": int(len(block)),
                "wind_speed_MAE": wind["mae"],
                "wind_speed_RMSE": wind["rmse"],
                "wind_speed_bias": wind["bias"],
                "power_MAE": power["mae"],
                "power_RMSE": power["rmse"],
                "power_nRMSE": power["rmse"] / plateau,
                "power_bias": power["bias"],
            }
        )
    wind_all = _metrics(aligned_all["obs_ws_ms"], aligned_all["fcst_ws_ms"])
    power_all = _metrics(aligned_all["obs_power_kw"], aligned_all["fcst_power_kw"])
    rows.append(
        {
            "date": "ALL",
            "n_points": int(len(aligned_all)),
            "wind_speed_MAE": wind_all["mae"],
            "wind_speed_RMSE": wind_all["rmse"],
            "wind_speed_bias": wind_all["bias"],
            "power_MAE": power_all["mae"],
            "power_RMSE": power_all["rmse"],
            "power_nRMSE": power_all["rmse"] / plateau,
            "power_bias": power_all["bias"],
        }
    )
    return pd.DataFrame(rows)


def _summary_from_aligned(aligned_all: pd.DataFrame, n_dates: int, plateau: float, failed: list[str]) -> dict:
    wind_all = _metrics(aligned_all["obs_ws_ms"], aligned_all["fcst_ws_ms"])
    power_all = _metrics(aligned_all["obs_power_kw"], aligned_all["fcst_power_kw"])
    mean_obs_ws = float(aligned_all["obs_ws_ms"].mean())
    n_days = aligned_all["date"].nunique()
    return {
        "n_campaign_dates": n_dates,
        "n_days_processed": n_dates,
        "n_days_failed": len(failed),
        "failed_dates": failed,
        "n_days_zero_overlap": n_dates - n_days,
        "zero_overlap_dates": [],
        "n_days_with_overlap": n_days,
        "n_aligned_points": int(len(aligned_all)),
        "plateau_kw": plateau,
        "wind_speed_MAE": wind_all["mae"],
        "wind_speed_RMSE": wind_all["rmse"],
        "wind_speed_bias": wind_all["bias"],
        "wind_speed_nRMSE_vs_mean_obs": wind_all["rmse"] / mean_obs_ws if mean_obs_ws else np.nan,
        "mean_obs_ws_ms": mean_obs_ws,
        "power_MAE": power_all["mae"],
        "power_RMSE": power_all["rmse"],
        "power_nRMSE": power_all["rmse"] / plateau,
        "power_bias": power_all["bias"],
    }


def apply_correction_to_aligned(aligned: pd.DataFrame, coeffs: dict) -> pd.DataFrame:
    """Re-run the power-curve step on bias-corrected 15-min forecast wind."""
    model = load_power_curve()
    out = aligned.copy()
    out["fcst_ws_raw_ms"] = out["fcst_ws_ms"]
    out["fcst_power_raw_kw"] = out["fcst_power_kw"]
    out["fcst_ws_ms"] = apply_wind_correction(out["fcst_ws_raw_ms"].to_numpy(), coeffs)
    out["fcst_power_kw"] = model(out["fcst_ws_ms"].to_numpy())
    return out


def run_backtest(apply_bias_correction: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    obs = _load_observations()
    dates = sorted(obs["date"].unique())
    plateau = float(load_power_curve().params.plateau_kw)

    aligned_parts: list[pd.DataFrame] = []
    failed: list[str] = []

    print(
        f"Campaign dates: {len(dates)}  plateau={plateau:.4f} kW  "
        f"tolerance=±{TOLERANCE}  bias_correction={apply_bias_correction}"
    )
    for i, day in enumerate(dates, start=1):
        day_s = day.isoformat()
        obs_day = obs.loc[obs["date"] == day]
        try:
            forecast = _fetch_historical(day, apply_bias_correction=apply_bias_correction)
            aligned = _align_day(obs_day, forecast)
        except Exception as exc:
            print(f"  [{i:02d}/{len(dates)}] {day_s}  FAILED  {exc}")
            failed.append(day_s)
            aligned = pd.DataFrame()

        n = len(aligned)
        if n == 0:
            print(f"  [{i:02d}/{len(dates)}] {day_s}  n=0  (no overlap)")
            continue

        aligned = aligned.copy()
        aligned["date"] = day_s
        aligned_parts.append(aligned)
        wind = _metrics(aligned["obs_ws_ms"].to_numpy(), aligned["fcst_ws_ms"].to_numpy())
        power = _metrics(aligned["obs_power_kw"].to_numpy(), aligned["fcst_power_kw"].to_numpy())
        print(
            f"  [{i:02d}/{len(dates)}] {day_s}  n={n:3d}  "
            f"WS MAE={wind['mae']:.2f} m/s  P MAE={power['mae']:.2f} kW"
        )

    aligned_all = pd.concat(aligned_parts, ignore_index=True) if aligned_parts else pd.DataFrame()
    if aligned_all.empty:
        raise RuntimeError("no aligned forecast/observation points")

    results = _day_rows_from_aligned(aligned_all, plateau)
    summary = _summary_from_aligned(aligned_all, len(dates), plateau, failed)
    return results, aligned_all, summary


def compare_bias_correction() -> None:
    """Uncorrected vs in-sample correction vs leave-one-day-out correction."""
    if not ALIGNED_CSV.exists():
        raise FileNotFoundError(f"{ALIGNED_CSV} missing — run an uncorrected backtest first")

    raw = pd.read_csv(ALIGNED_CSV, parse_dates=["timestamp"])
    raw["date"] = raw["date"].astype(str)
    plateau = float(load_power_curve().params.plateau_kw)
    model = load_power_curve()

    coeffs = fit_from_aligned(ALIGNED_CSV)
    in_sample = apply_correction_to_aligned(raw, coeffs)

    oos_wind, folds = leave_one_day_out(raw)
    oos = oos_wind.copy()
    oos["fcst_power_raw_kw"] = oos.get("fcst_power_kw", np.nan)
    oos["fcst_power_kw"] = model(oos["fcst_ws_ms"].to_numpy())

    stability = fold_stability(folds)
    coeffs_ship = dict(coeffs)
    coeffs_ship["cross_validation"] = {
        "scheme": "leave-one-day-out",
        **stability,
        "note": (
            "a, b shipped here are fit on all 44 days. "
            "LODO metrics are the honest estimate for a new unseen day."
        ),
    }
    save_coefficients(coeffs_ship)

    before = _summary_from_aligned(raw, raw["date"].nunique(), plateau, [])
    after_is = _summary_from_aligned(in_sample, in_sample["date"].nunique(), plateau, [])
    after_oos = _summary_from_aligned(oos, oos["date"].nunique(), plateau, [])

    _day_rows_from_aligned(in_sample, plateau).to_csv(RESULTS_CORRECTED_CSV, index=False)
    _day_rows_from_aligned(oos, plateau).to_csv(RESULTS_OOS_CSV, index=False)
    in_sample.to_csv(ALIGNED_CORRECTED_CSV, index=False)
    oos.to_csv(ALIGNED_OOS_CSV, index=False)
    folds.to_csv(FOLDS_CSV, index=False)
    print(f"  wrote {OUT / 'wind_bias_correction.json'}  (all-44-day coefficients for live use)")
    print(f"  wrote {FOLDS_CSV}")
    print(f"  wrote {RESULTS_OOS_CSV}")

    print()
    print("=" * 78)
    print("BIAS CORRECTION  —  uncorrected | in-sample | leave-one-day-out")
    print(f"n={len(raw)} aligned 15-min points, {raw['date'].nunique()} days, plateau={plateau:.2f} kW")
    print("=" * 78)
    print(
        f"{'metric':<22} {'uncorrected':>12} {'in-sample':>12} {'LODO OOS':>12}"
    )
    rows = [
        ("Wind MAE (m/s)", "wind_speed_MAE"),
        ("Wind RMSE (m/s)", "wind_speed_RMSE"),
        ("Wind bias (m/s)", "wind_speed_bias"),
        ("Power MAE (kW)", "power_MAE"),
        ("Power RMSE (kW)", "power_RMSE"),
        ("Power nRMSE", "power_nRMSE"),
        ("Power bias (kW)", "power_bias"),
    ]
    for label, key in rows:
        print(f"{label:<22} {before[key]:12.3f} {after_is[key]:12.3f} {after_oos[key]:12.3f}")

    print()
    print("Shipped coefficients (fit on all 44 days)")
    print(f"  observed_WS30 = {coeffs['a']:.4f} * forecast_ws + {coeffs['b']:.4f}   R²={coeffs['r2']:.3f}")
    print(f"  uncorrected bias (fcst − obs) = {coeffs['bias_fcst_minus_obs_ms']:+.3f} m/s")
    print()
    print(f"Leave-one-day-out fold stability  ({stability['n_folds']} folds)")
    print(
        f"  a  mean={stability['a_mean']:.4f}  std={stability['a_std']:.4f}  "
        f"min={stability['a_min']:.4f}  max={stability['a_max']:.4f}"
    )
    print(
        f"  b  mean={stability['b_mean']:.4f}  std={stability['b_std']:.4f}  "
        f"min={stability['b_min']:.4f}  max={stability['b_max']:.4f}"
    )
    print()
    print(
        "  LODO is the honest estimate for a new campaign day. "
        "The live pipeline should keep the all-44-day (a, b)."
    )
    print("=" * 78)


def plot_scatter(aligned: pd.DataFrame, plateau: float, png_path: Path = SCATTER_PNG) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.scatter(aligned["obs_power_kw"], aligned["fcst_power_kw"], s=14, alpha=0.45, c="#1f4e79")
    lim = max(float(aligned["obs_power_kw"].max()), float(aligned["fcst_power_kw"].max()), plateau)
    ax.plot([0, lim], [0, lim], color="#c45c26", linewidth=1.4, label="1:1")
    ax.set_xlabel("Observed power (kW)")
    ax.set_ylabel("Forecasted power (kW)")
    ax.set_title("Backtest: forecasted vs observed power (all aligned 15-min points)")
    ax.set_xlim(0, lim * 1.02)
    ax.set_ylim(0, lim * 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {png_path}")
    return png_path


def plot_timeseries(
    aligned: pd.DataFrame,
    results: pd.DataFrame,
    png_path: Path = TIMESERIES_PNG,
    n_days: int = N_TIMESERIES_DAYS,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = (
        results.loc[results["date"] != "ALL"]
        .sort_values(["n_points", "date"], ascending=[False, True])
        .head(n_days)["date"]
        .tolist()
    )
    n = len(days)
    fig, axes = plt.subplots(n, 1, figsize=(9.2, 2.4 * n), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, day in zip(axes, days):
        block = aligned.loc[aligned["date"] == day].sort_values("timestamp")
        ax.plot(block["timestamp"], block["obs_power_kw"], color="#1f4e79", linewidth=1.6, label="Observed")
        ax.plot(block["timestamp"], block["fcst_power_kw"], color="#c45c26", linewidth=1.6, label="Forecast")
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"{day}  ({len(block)} aligned 15-min points)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("UTC")
    fig.suptitle("Backtest: observed vs forecasted power on the busiest campaign days", y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")
    return png_path


def print_summary(results: pd.DataFrame, summary: dict) -> None:
    print()
    print("=" * 72)
    print("BACKTEST SUMMARY")
    print("=" * 72)
    print(f"  Campaign days processed:     {summary['n_days_processed']}")
    print(f"  Days with aligned points:    {summary['n_days_with_overlap']}")
    print(f"  Days with zero overlap:      {summary['n_days_zero_overlap']}")
    if summary["zero_overlap_dates"]:
        print(f"    {', '.join(summary['zero_overlap_dates'])}")
    print(f"  Days that failed to fetch:   {summary['n_days_failed']}")
    if summary["failed_dates"]:
        print(f"    {', '.join(summary['failed_dates'])}")
    print(f"  Aligned 15-min points:       {summary['n_aligned_points']}")
    print()
    print("  Wind speed (forecast 15-min vs observed WS30)")
    print(f"    MAE    {summary['wind_speed_MAE']:.3f} m/s")
    print(f"    RMSE   {summary['wind_speed_RMSE']:.3f} m/s")
    print(f"    bias   {summary['wind_speed_bias']:+.3f} m/s  (mean forecast − observed)")
    print(
        f"    nRMSE  {summary['wind_speed_nRMSE_vs_mean_obs']:.3f}  "
        f"(RMSE / mean observed WS30 = {summary['mean_obs_ws_ms']:.2f} m/s)"
    )
    print()
    print("  Power (forecast vs observed, Running-mode 15-min)")
    print(f"    MAE    {summary['power_MAE']:.3f} kW")
    print(f"    RMSE   {summary['power_RMSE']:.3f} kW")
    print(f"    bias   {summary.get('power_bias', float('nan')):+.3f} kW")
    print(
        f"    nRMSE  {summary['power_nRMSE']:.3f}  "
        f"(RMSE / fitted plateau {summary['plateau_kw']:.2f} kW, not 45 kW nameplate)"
    )
    print()
    print("  CAVEAT")
    print(
        "  This evaluates the historical forecast pipeline against only "
        f"{summary['n_campaign_dates']} non-continuous campaign days, using a "
        "stitched short-lead-time Open-Meteo Historical Forecast proxy rather "
        "than true archived day-ahead (single-run) forecasts. Those single-run "
        "archives start in 2024 and do not cover the 2022–2023 Björkö record."
    )
    print("=" * 72)
    print()
    show = results.copy()
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Re-download all 44 historical forecasts instead of using saved alignments.",
    )
    parser.add_argument(
        "--bias-correction",
        action="store_true",
        help="When refetching, apply the saved linear wind correction before the power curve.",
    )
    args = parser.parse_args()

    if args.refetch:
        results, aligned, summary = run_backtest(apply_bias_correction=args.bias_correction)
        OUT.mkdir(exist_ok=True)
        dest_results = RESULTS_CORRECTED_CSV if args.bias_correction else RESULTS_CSV
        dest_aligned = ALIGNED_CORRECTED_CSV if args.bias_correction else ALIGNED_CSV
        results.to_csv(dest_results, index=False)
        aligned.to_csv(dest_aligned, index=False)
        print(f"  wrote {dest_results}")
        print(f"  wrote {dest_aligned}")
        plot_scatter(aligned, summary["plateau_kw"])
        plot_timeseries(aligned, results)
        print_summary(results, summary)
        return

    compare_bias_correction()


if __name__ == "__main__":
    main()
