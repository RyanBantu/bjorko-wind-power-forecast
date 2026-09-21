#!/usr/bin/env python3
"""Empirical power curve and 15-minute SCADA dataset for the Björkö turbine.

There is no dedicated electrical-power channel. Instantaneous power is
computed as DC current × DC voltage from the generator rectifier:

    power_w = DCC [A] * DCV [V]

Wind speed is the 30 m meteorological-mast measurement (WS30). The nacelle
anemometer (WSN) is kept as a secondary column but is not used for the
curve: it drops to zero for long stretches even while the turbine is
running.

Normal production is SysMode == 12 ("Running"). Other modes (idle, start,
park, service, emergency) have near-zero mean power.

B1_CL4_100.csv is high-frequency SHM snapshots (nine bursts, four days).
B1_CL4_20.csv covers 2022-07-05 to 2023-08-02 and is the source used for
the forecasting dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METADATA_CSV = ROOT / "Bjorko_Sensors_Specs_Metadata.csv"
MODES_JSON = ROOT / "Bjorko_modes_mapping.json"
MODES_CSV = ROOT / "Bjorko_modes_mapping.csv"

LABVIEW_ORIGIN = "1904-01-01"
RUNNING_MODE = 12
BIN_WIDTH_MS = 0.5
# A 15-minute stamp is kept if it contains at least this much raw data.
MIN_SECONDS_PER_BIN = 30.0
# IEC-style wind-speed range for the binned curve.
WS_BIN_LO = 0.5
WS_BIN_HI = 16.0

# Only these columns are read from the 69-channel files.
LOAD_COLS = ["Time", "SysMode", "WS30", "WSN", "DCC", "DCV"]


def print_sensor_channels() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_CSV, encoding="latin-1")
    name_col = "Internal Signal Name"
    print("=" * 72)
    print("SENSOR CHANNELS (Bjorko_Sensors_Specs_Metadata.csv)")
    print("=" * 72)
    for _, row in meta.iterrows():
        idx = row.iloc[0]
        name = str(row[name_col]).strip()
        unit = str(row.get("Unit", "")).strip()
        desc = str(row.get("Description of Signal Name", "")).strip()
        print(f"  [{idx:>2}] {name:<14} {unit:<10} {desc}")

    print("\nIdentified columns for this analysis")
    print("  Time     timestamps (LabVIEW seconds since 1904-01-01 UTC)")
    print("  WS30     wind speed, 30 m met mast [m/s]  ← used for the curve")
    print("  WSN      wind speed, nacelle [m/s]        ← secondary, many zeros")
    print("  WindEst  FFR estimate [m/s]               ← unused (often stuck at 25)")
    print("  DCC, DCV rectifier current [A] and voltage [V]")
    print("  power    DCC * DCV [W]                    ← no dedicated power channel")
    print("  MaxPwrEst FFR max-power estimate [W]      ← unused (available, not measured)")
    print("  SysMode  controller mode [enum]")
    return meta


def print_mode_mapping() -> dict[int, str]:
    if MODES_CSV.exists():
        modes_df = pd.read_csv(MODES_CSV)
        mapping = {int(k): str(v) for k, v in zip(modes_df.iloc[:, 0], modes_df.iloc[:, 1])}
        source = MODES_CSV.name
    else:
        raw = json.loads(MODES_JSON.read_text())
        mapping = {int(k): str(v) for k, v in raw.items()}
        source = MODES_JSON.name
        print(f"\nNote: {MODES_CSV.name} is not in this directory; using {source}.")

    print("\n" + "=" * 72)
    print(f"OPERATING MODES ({source})")
    print("=" * 72)
    for code in sorted(mapping):
        marker = "  ← normal power production" if code == RUNNING_MODE else ""
        print(f"  {code:>2}: {mapping[code]}{marker}")
    print(
        "\nNormal production filter: SysMode == 12 (Running). "
        "Start / operation-ready / park / service / emergency modes "
        "are excluded."
    )
    return mapping


def load_needed_columns(csv_path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Load timestamp, wind, DC electrical, and mode columns only."""
    print(f"\nLoading {csv_path.name}  usecols={LOAD_COLS}")
    pieces: list[pd.DataFrame] = []
    n_raw = 0
    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=LOAD_COLS, chunksize=chunksize), start=1):
        n_raw += len(chunk)
        pieces.append(chunk)
        if i == 1 or i % 5 == 0:
            print(f"  chunk {i}: {n_raw:,} rows")
    df = pd.concat(pieces, ignore_index=True)
    print(f"  loaded {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["Time"], unit="s", origin=LABVIEW_ORIGIN, utc=True)
    out["power_w"] = out["DCC"] * out["DCV"]
    out["power_kw"] = out["power_w"] / 1000.0
    return out


def filter_running(df: pd.DataFrame) -> pd.DataFrame:
    running = df.loc[df["SysMode"] == RUNNING_MODE].copy()
    print(
        f"  Running (SysMode={RUNNING_MODE}): {len(running):,} / {len(df):,} "
        f"({100 * len(running) / max(len(df), 1):.1f}%)"
    )
    return running


def resample_15min(running: pd.DataFrame, sample_dt_s: float) -> pd.DataFrame:
    """Mean wind and power on a 15-minute calendar grid, Running samples only."""
    ts = running.set_index("timestamp").sort_index()
    agg = ts.resample("15min").agg(
        wind_speed_ws30_ms=("WS30", "mean"),
        wind_speed_wsn_ms=("WSN", "mean"),
        power_w=("power_w", "mean"),
        power_kw=("power_kw", "mean"),
        n_samples=("power_w", "size"),
        ws30_std=("WS30", "std"),
        power_w_std=("power_w", "std"),
    )
    agg = agg.dropna(subset=["wind_speed_ws30_ms", "power_w"])
    agg["n_seconds"] = agg["n_samples"] * sample_dt_s
    agg["enough_samples"] = agg["n_seconds"] >= MIN_SECONDS_PER_BIN
    # Measurement noise around zero when barely producing.
    agg["power_w"] = agg["power_w"].clip(lower=0.0)
    agg["power_kw"] = agg["power_w"] / 1000.0
    kept = agg.loc[agg["enough_samples"]].copy()
    print(
        f"  15-min bins with any Running data: {len(agg):,}; "
        f"kept (≥{MIN_SECONDS_PER_BIN:.0f}s): {len(kept):,}"
    )
    return kept.reset_index()


def _bin_power_curve(ws: np.ndarray, p_kw: np.ndarray, count_name: str) -> pd.DataFrame:
    edges = np.arange(WS_BIN_LO, WS_BIN_HI + BIN_WIDTH_MS, BIN_WIDTH_MS)
    centers = edges[:-1] + BIN_WIDTH_MS / 2.0
    idx = np.digitize(ws, edges) - 1
    rows = []
    for i, center in enumerate(centers):
        mask = (idx == i) & (ws >= edges[0]) & (ws < edges[-1])
        if not np.any(mask):
            continue
        pw = p_kw[mask]
        ww = ws[mask]
        rows.append(
            {
                "ws_bin_center_ms": round(float(center), 3),
                "ws_bin_left_ms": float(edges[i]),
                "ws_bin_right_ms": float(edges[i + 1]),
                count_name: int(mask.sum()),
                "mean_wind_speed_ms": float(ww.mean()),
                "mean_power_kw": float(pw.mean()),
                "std_power_kw": float(pw.std(ddof=1)) if mask.sum() > 1 else 0.0,
                "p10_power_kw": float(np.quantile(pw, 0.10)),
                "p50_power_kw": float(np.quantile(pw, 0.50)),
                "p90_power_kw": float(np.quantile(pw, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def empirical_power_curve(series_15min: pd.DataFrame) -> pd.DataFrame:
    """IEC 61400-12-1 method of bins on 15-minute means (0.5 m/s bins)."""
    curve = _bin_power_curve(
        series_15min["wind_speed_ws30_ms"].to_numpy(),
        series_15min["power_kw"].to_numpy(),
        count_name="n_15min",
    )
    print(f"  15-min power-curve bins with data: {len(curve)}")
    return curve


def empirical_power_curve_hf(running: pd.DataFrame) -> pd.DataFrame:
    """Same 0.5 m/s bins, but on every Running sample. More stable than 15-min bins."""
    power_kw = np.clip(running["power_w"].to_numpy() / 1000.0, a_min=0.0, a_max=None)
    curve = _bin_power_curve(running["WS30"].to_numpy(), power_kw, count_name="n_samples")
    print(f"  high-frequency power-curve bins with data: {len(curve)}")
    return curve


def plot_curve(
    series_15min: pd.DataFrame,
    curve: pd.DataFrame,
    png_path: Path,
    title: str,
    hf_curve: pd.DataFrame | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(
        series_15min["wind_speed_ws30_ms"],
        series_15min["power_kw"],
        s=12,
        alpha=0.25,
        c="#7a8b99",
        label="15-min means (Running)",
        zorder=1,
    )
    if not curve.empty:
        ax.plot(
            curve["mean_wind_speed_ms"],
            curve["mean_power_kw"],
            color="#1f4e79",
            marker="o",
            linewidth=2,
            markersize=5,
            label="15-min binned mean (0.5 m/s)",
            zorder=3,
        )
        ax.fill_between(
            curve["mean_wind_speed_ms"],
            curve["p10_power_kw"],
            curve["p90_power_kw"],
            color="#1f4e79",
            alpha=0.15,
            label="15-min 10th–90th percentile",
            zorder=2,
        )
    if hf_curve is not None and not hf_curve.empty:
        ax.plot(
            hf_curve["mean_wind_speed_ms"],
            hf_curve["mean_power_kw"],
            color="#c45c26",
            marker="s",
            linewidth=2,
            markersize=4,
            label="High-frequency binned mean",
            zorder=4,
        )
    ax.set_xlabel("Wind speed, 30 m mast (m/s)")
    ax.set_ylabel("Electrical power (kW)")
    ax.set_title(title)
    ax.set_xlim(0, 16)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {png_path}")


def process_file(csv_path: Path, out_dir: Path, sample_hz: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag = f"{int(sample_hz)}hz"
    print("\n" + "=" * 72)
    print(f"PROCESS {csv_path.name}  (nominal {sample_hz:g} Hz)")
    print("=" * 72)

    raw = load_needed_columns(csv_path)
    df = prepare(raw)
    print(
        f"  time span: {df['timestamp'].min()} → {df['timestamp'].max()}  "
        f"({df['timestamp'].dt.normalize().nunique()} distinct UTC dates)"
    )

    running = filter_running(df)
    series_15 = resample_15min(running, sample_dt_s=1.0 / sample_hz)
    curve = empirical_power_curve(series_15)
    hf_curve = empirical_power_curve_hf(running)

    out_dir.mkdir(exist_ok=True)
    series_path = out_dir / f"bjorko_15min_running_{tag}.csv"
    curve_path = out_dir / f"bjorko_power_curve_{tag}.csv"
    hf_path = out_dir / f"bjorko_power_curve_{tag}_hf.csv"
    png_path = out_dir / f"bjorko_power_curve_{tag}.png"

    series_out = series_15[
        [
            "timestamp",
            "wind_speed_ws30_ms",
            "wind_speed_wsn_ms",
            "power_kw",
            "power_w",
            "n_samples",
            "n_seconds",
            "ws30_std",
            "power_w_std",
        ]
    ]
    series_out.to_csv(series_path, index=False)
    curve.to_csv(curve_path, index=False)
    hf_curve.to_csv(hf_path, index=False)
    print(f"  wrote {series_path}")
    print(f"  wrote {curve_path}")
    print(f"  wrote {hf_path}")

    title = (
        f"Björkö empirical power curve · {tag} source · SysMode=Running · "
        f"{series_15['timestamp'].min().date()} to {series_15['timestamp'].max().date()}"
        if len(series_15)
        else f"Björkö empirical power curve · {tag}"
    )
    plot_curve(series_15, curve, png_path, title, hf_curve=hf_curve)

    if len(hf_curve):
        print("\n  High-frequency binned curve (kW):")
        print(
            hf_curve[["ws_bin_center_ms", "n_samples", "mean_wind_speed_ms", "mean_power_kw"]].to_string(
                index=False
            )
        )
    return series_15, curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        choices=("20", "100", "both"),
        default="both",
        help="Which time-series file to process (default: both).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print_sensor_channels()
    print_mode_mapping()

    out_dir = ROOT / "outputs"
    if args.source in ("100", "both"):
        process_file(ROOT / "B1_CL4_100.csv", out_dir, sample_hz=100.0)
    if args.source in ("20", "both"):
        process_file(ROOT / "B1_CL4_20.csv", out_dir, sample_hz=20.0)


if __name__ == "__main__":
    main()
