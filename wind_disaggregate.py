#!/usr/bin/env python3
"""Hourly NWP wind → 15-minute estimates for the Björkö forecast pipeline.

Default: linear interpolation between hourly knots (hour-on-the-hour values
are preserved).

Optional enhancement: add zero-mean noise on the interior 15-minute stamps
so they wiggle like the campaign data, without changing the hourly knots.
Amplitude comes from a variability profile extracted from the 20 Hz
Running-mode 15-minute windows.

Two profiles are stored:

- intra_window_std: std of WS30 inside completed 15-minute windows
  (direct 20 Hz statistic the user asked for). Median ≈ 0.83 m/s — too
  large to add raw onto 15-minute *means*.
- intra_hour_resid: std of 15-minute means around their hourly mean, for
  hours with ≥3 quarters. Median-scale ≈ 0.41 m/s — this is the noise
  that belongs on hourly→15-minute disaggregation.

`add_variability=True` uses intra_hour_resid by default, looked up as a
function of the interpolated wind speed. Pass noise_scale to turn it up
or down; 0 disables the enhancement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIFTEEN_CSV = ROOT / "outputs" / "bjorko_15min_running_20hz.csv"
PROFILE_CSV = ROOT / "outputs" / "subhourly_variability_profile.csv"
PROFILE_JSON = ROOT / "outputs" / "subhourly_variability_profile.json"
DEMO_CSV = ROOT / "outputs" / "demo_hourly_to_15min.csv"
DEMO_PLOT = ROOT / "outputs" / "demo_hourly_to_15min.png"

COMPLETED_SECONDS = 800.0
BIN_WIDTH = 0.5
DEFAULT_NOISE_SCALE = 1.0


def extract_variability_profile(
    fifteen_csv: Path | str = FIFTEEN_CSV,
    completed_seconds: float = COMPLETED_SECONDS,
) -> tuple[pd.DataFrame, dict]:
    """Build wind-speed-binned variability tables from 20 Hz-derived 15-min stats."""
    df = pd.read_csv(fifteen_csv, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    completed = df.loc[df["n_seconds"] >= completed_seconds].copy()
    if completed.empty:
        raise ValueError(f"no 15-min windows with n_seconds ≥ {completed_seconds}")

    # --- std of 20 Hz WS30 inside each completed 15-min window ---
    ws_bin = BIN_WIDTH * np.floor(completed["wind_speed_ws30_ms"].to_numpy() / BIN_WIDTH) + BIN_WIDTH / 2.0
    completed = completed.assign(ws_bin=ws_bin)
    intra_window = (
        completed.groupby("ws_bin", as_index=False)
        .agg(
            n_windows=("ws30_std", "size"),
            intra_window_std_median=("ws30_std", "median"),
            intra_window_std_mean=("ws30_std", "mean"),
        )
        .sort_values("ws_bin")
    )

    # --- 15-min means vs hourly mean (the right sigma for disaggregation) ---
    df = df.sort_values("timestamp")
    df["hour"] = df["timestamp"].dt.floor("h")
    resid_rows = []
    for hour, block in df.groupby("hour"):
        if len(block) < 3:
            continue
        mu = float(block["wind_speed_ws30_ms"].mean())
        for _, row in block.iterrows():
            resid_rows.append(
                {
                    "hour": hour,
                    "quarter_min": int(row["timestamp"].minute),
                    "ws_15": float(row["wind_speed_ws30_ms"]),
                    "resid": float(row["wind_speed_ws30_ms"] - mu),
                    "ws_hour": mu,
                }
            )
    resid = pd.DataFrame(resid_rows)
    if resid.empty:
        raise ValueError("need hours with ≥3 fifteen-minute stamps to estimate intra-hour residuals")

    resid["ws_bin"] = BIN_WIDTH * np.floor(resid["ws_hour"].to_numpy() / BIN_WIDTH) + BIN_WIDTH / 2.0
    intra_hour = (
        resid.groupby("ws_bin", as_index=False)
        .agg(
            n_15min=("resid", "size"),
            intra_hour_resid_std=("resid", "std"),
            intra_hour_resid_mae=("resid", lambda s: float(np.mean(np.abs(s)))),
        )
        .sort_values("ws_bin")
    )

    profile = pd.merge(intra_window, intra_hour, on="ws_bin", how="outer").sort_values("ws_bin")

    summary = {
        "n_completed_windows": int(len(completed)),
        "completed_seconds": completed_seconds,
        "intra_window_std_median_ms": float(completed["ws30_std"].median()),
        "intra_window_std_mean_ms": float(completed["ws30_std"].mean()),
        "n_hours_with_3plus_quarters": int(resid["hour"].nunique()),
        "intra_hour_resid_std_ms": float(resid["resid"].std(ddof=1)),
        "source": str(Path(fifteen_csv)),
        "note": (
            "intra_window_std is the 20 Hz WS30 std inside a 15-min window. "
            "intra_hour_resid_std is the scatter of 15-min means around the hourly mean "
            "and is the default noise amplitude for hourly→15-min disaggregation."
        ),
    }
    print(
        f"  completed 15-min windows: {summary['n_completed_windows']}  "
        f"intra-window std median {summary['intra_window_std_median_ms']:.3f} m/s"
    )
    print(
        f"  hours with ≥3 quarters: {summary['n_hours_with_3plus_quarters']}  "
        f"intra-hour residual std {summary['intra_hour_resid_std_ms']:.3f} m/s"
    )
    return profile, summary


def save_variability_profile(
    profile: pd.DataFrame,
    summary: dict,
    csv_path: Path | str = PROFILE_CSV,
    json_path: Path | str = PROFILE_JSON,
) -> None:
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")


def _sigma_lookup(profile: pd.DataFrame, wind_ms: np.ndarray, column: str, fallback: float) -> np.ndarray:
    if profile is None or profile.empty or column not in profile.columns:
        return np.full(wind_ms.shape, fallback, dtype=float)
    table = profile.dropna(subset=[column]).sort_values("ws_bin")
    if table.empty:
        return np.full(wind_ms.shape, fallback, dtype=float)
    return np.interp(wind_ms, table["ws_bin"].to_numpy(), table[column].to_numpy(), left=table[column].iloc[0], right=table[column].iloc[-1])


def _zero_end_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-std, mean-zero interior noise that is exactly 0 at both endpoints.

    Independent Gaussians on the interior 15-min stamps, recentered so they
    do not bias the hourly mean. A free Brownian bridge was too wander-prone
    on 4-point hours.
    """
    noise = np.zeros(n)
    if n < 3:
        return noise
    interior = rng.normal(0.0, 1.0, size=n - 2)
    interior = interior - interior.mean()
    std = interior.std(ddof=0)
    if std > 0:
        interior = interior / std
    noise[1:-1] = interior
    return noise


def hourly_to_15min(
    hourly: pd.Series,
    add_variability: bool = False,
    profile: pd.DataFrame | None = None,
    summary: dict | None = None,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    noise_column: str = "intra_hour_resid_std",
    rng: np.random.Generator | int | None = None,
) -> pd.Series:
    """Linear interpolation of hourly wind onto a 15-minute grid.

    Parameters
    ----------
    hourly
        Wind speed [m/s] indexed by timezone-aware (or naive) timestamps
        at hourly spacing. Include both endpoints of the horizon.
    add_variability
        If True, add mean-zero noise on :15/:30/:45. Hourly knots stay exact.
    profile, summary
        Output of extract_variability_profile. summary supplies the fallback
        residual std when a wind-speed bin is missing.
    noise_scale
        Multiplier on the looked-up sigma. 0 is equivalent to add_variability=False.
    noise_column
        Profile column used as sigma. Default is the intra-hour 15-min residual.
        Pass "intra_window_std_median" only if you intentionally want 20 Hz
        intra-window scatter (usually too large for 15-min means).
    """
    if not isinstance(hourly, pd.Series):
        raise TypeError("hourly must be a pandas Series of wind speed [m/s]")
    series = hourly.sort_index().astype(float)
    grid = pd.date_range(series.index.min(), series.index.max(), freq="15min", tz=series.index.tz)
    interpolated = series.reindex(series.index.union(grid)).interpolate(method="time").reindex(grid)
    interpolated.name = interpolated.name or "wind_speed_ms"

    if not add_variability or noise_scale == 0:
        return interpolated.clip(lower=0.0)

    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(rng)

    fallback = 0.40
    if summary is not None:
        fallback = float(summary.get("intra_hour_resid_std_ms", fallback))

    values = interpolated.to_numpy(dtype=float, copy=True)
    index = interpolated.index
    # Apply an independent bridge on each hour-long segment so 00:00, 01:00, … stay exact.
    hour_starts = [i for i, ts in enumerate(index) if ts.minute == 0]
    if index[-1] not in series.index and hour_starts and hour_starts[-1] != len(index) - 1:
        hour_starts.append(len(index) - 1)

    for a, b in zip(hour_starts, hour_starts[1:]):
        sl = slice(a, b + 1)
        n = b - a + 1
        if n < 3:
            continue
        sigma = _sigma_lookup(profile, values[sl], noise_column, fallback)
        # Sparse wind-speed bins have noisy residual std; cap at the global figure.
        sigma = np.minimum(sigma, fallback)
        noise = _zero_end_noise(n, rng)
        values[sl] = values[sl] + noise_scale * sigma * noise

    out = pd.Series(values, index=index, name=interpolated.name).clip(lower=0.0)
    return out


def demo_on_campaign_day(
    fifteen_csv: Path | str = FIFTEEN_CSV,
    profile: pd.DataFrame | None = None,
    summary: dict | None = None,
    min_quarters: int = 8,
) -> pd.DataFrame:
    """Hold out a campaign day: hourly means → 15-min, compare to observed."""
    df = pd.read_csv(fifteen_csv, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["date"] = df["timestamp"].dt.date
    counts = df.groupby("date").size()
    eligible = counts[counts >= min_quarters]
    if eligible.empty:
        raise ValueError(f"no campaign day with ≥{min_quarters} fifteen-minute stamps")
    day = eligible.idxmax()
    obs = df.loc[df["date"] == day].set_index("timestamp")["wind_speed_ws30_ms"].sort_index()

    hourly = obs.resample("1h").mean().dropna()
    if len(hourly) < 2:
        raise ValueError("campaign day does not cover two hourly stamps")

    linear = hourly_to_15min(hourly, add_variability=False)
    noisy = hourly_to_15min(
        hourly,
        add_variability=True,
        profile=profile,
        summary=summary,
        noise_scale=1.0,
        rng=0,
    )

    aligned = pd.DataFrame(
        {
            "observed_15min": obs,
            "linear_15min": linear,
            "linear_plus_variability": noisy,
        }
    )
    # Compare only where we have an observed 15-min stamp inside the hourly span.
    cmp = aligned.dropna(subset=["observed_15min", "linear_15min"])
    if cmp.empty:
        raise ValueError("no overlapping 15-min stamps on the demo day")
    mae = float(np.mean(np.abs(cmp["linear_15min"] - cmp["observed_15min"])))
    print(f"  demo day {day}: {len(obs)} observed 15-min, {len(hourly)} hourly stamps, linear MAE {mae:.3f} m/s")
    return aligned


def plot_demo(demo: pd.DataFrame, png_path: Path | str = DEMO_PLOT) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path = Path(png_path)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    if demo["observed_15min"].notna().any():
        ax.plot(demo.index, demo["observed_15min"], "o", color="#1f4e79", markersize=4, label="Observed 15-min WS30")
    ax.plot(demo.index, demo["linear_15min"], color="#c45c26", linewidth=1.8, label="Linear hourly → 15-min")
    ax.plot(
        demo.index,
        demo["linear_plus_variability"],
        color="#5b8a72",
        linewidth=1.2,
        alpha=0.9,
        label="Linear + intra-hour variability",
    )
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title("Hourly-to-15-minute disaggregation vs campaign observations")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {png_path}")
    return png_path


def main() -> None:
    print("=" * 72)
    print("SUB-HOURLY VARIABILITY + HOURLY→15-MIN DISAGGREGATION")
    print("=" * 72)
    profile, summary = extract_variability_profile()
    save_variability_profile(profile, summary)
    print("\n  variability profile:")
    print(profile.to_string(index=False))

    demo = demo_on_campaign_day(profile=profile, summary=summary)
    demo_out = demo.dropna(how="all")
    demo_out.to_csv(DEMO_CSV)
    print(f"  wrote {DEMO_CSV}")
    plot_demo(demo)

    # End-to-end: disaggregated wind through the fitted power curve.
    try:
        from power_curve_model import load_power_curve

        model = load_power_curve()
        wind = demo["linear_15min"].dropna()
        power = pd.Series(model(wind.to_numpy()), index=wind.index, name="power_kw")
        print("  sample P(v) on disaggregated wind:")
        print(pd.DataFrame({"wind_ms": wind, "power_kw": power}).head(8).to_string())
    except FileNotFoundError:
        print("  fitted power curve not found yet; skip P(v) demo")


if __name__ == "__main__":
    main()
