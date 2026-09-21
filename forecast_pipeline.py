#!/usr/bin/env python3
"""NWP → 15-minute wind → empirical power for the Björkö turbine.

Live path
    SMHI SNOW1g (PMP3g fallback) → hourly_to_15min → power_curve_fn

Historical path
    Open-Meteo Historical Forecast (stitched short-lead) → same conversion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from openmeteo_historical import fetch_historical_forecast
from power_curve_model import load_power_curve
from smhi_forecast import fetch_smhi_forecast
from wind_bias_correction import apply_wind_correction
from wind_disaggregate import extract_variability_profile, hourly_to_15min

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def _wind_to_power(
    hourly_ws: pd.Series,
    add_variability: bool,
    rng: int | None,
    apply_bias_correction: bool = False,
) -> pd.DataFrame:
    profile = summary = None
    if add_variability:
        profile, summary = extract_variability_profile()
    wind_15 = hourly_to_15min(
        hourly_ws,
        add_variability=add_variability,
        profile=profile,
        summary=summary,
        rng=rng,
    )
    hourly_out = hourly_ws
    if apply_bias_correction:
        wind_15 = apply_wind_correction(wind_15)
        hourly_out = apply_wind_correction(hourly_ws)
        wind_15.attrs["bias_corrected"] = True
    model = load_power_curve()
    power = pd.Series(model(wind_15.to_numpy()), index=wind_15.index, name="power_kw")
    frame = pd.DataFrame(
        {
            "wind_speed_hourly_ms": hourly_out.reindex(wind_15.index),
            "wind_speed_15min_ms": wind_15,
            "power_kw": power,
        }
    )
    if "wd" in hourly_ws.attrs:
        wd = hourly_ws.attrs["wd"].astype(float)
        frame["wind_direction_deg"] = (
            wd.reindex(wd.index.union(wind_15.index)).interpolate(method="time").reindex(wind_15.index)
        )
    return frame


def run_live_forecast(
    add_variability: bool = False,
    rng: int | None = 0,
    apply_bias_correction: bool = False,
) -> pd.DataFrame:
    hourly = fetch_smhi_forecast()
    frame = _wind_to_power(
        hourly,
        add_variability=add_variability,
        rng=rng,
        apply_bias_correction=apply_bias_correction,
    )
    frame.attrs["source"] = hourly.attrs.get("api")
    frame.attrs["reference_time"] = hourly.attrs.get("reference_time")
    frame.attrs["source_url"] = hourly.attrs.get("source_url")
    frame.attrs["bias_corrected"] = apply_bias_correction
    return frame


def forecast_power(
    date=None,
    use_live: bool = True,
    add_variability: bool = False,
    rng: int | None = 0,
    apply_bias_correction: bool = False,
) -> pd.DataFrame:
    """Day-ahead-style 15-minute wind + power forecast.

    use_live=True  → SMHI (date ignored)
    use_live=False → Open-Meteo Historical Forecast for ``date``
    apply_bias_correction → linear WS30 mapping before the power curve
    """
    if use_live:
        return run_live_forecast(
            add_variability=add_variability,
            rng=rng,
            apply_bias_correction=apply_bias_correction,
        )
    if date is None:
        raise ValueError("date is required when use_live=False")
    return run_historical_forecast(
        date,
        add_variability=add_variability,
        rng=rng,
        apply_bias_correction=apply_bias_correction,
    )


def run_historical_forecast(
    date,
    add_variability: bool = False,
    rng: int | None = 0,
    apply_bias_correction: bool = False,
) -> pd.DataFrame:
    hourly = fetch_historical_forecast(date)
    frame = _wind_to_power(
        hourly,
        add_variability=add_variability,
        rng=rng,
        apply_bias_correction=apply_bias_correction,
    )
    frame.attrs["source"] = hourly.attrs.get("source")
    frame.attrs["date"] = hourly.attrs.get("date")
    frame.attrs["note"] = hourly.attrs.get("note")
    frame.attrs["bias_corrected"] = apply_bias_correction
    return frame


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    print(f"  wrote {path}  ({len(frame)} rows, {frame['power_kw'].sum()/4:.1f} kWh if 15-min energy)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "historical", "both"), default="both")
    parser.add_argument("--date", default="2022-08-11", help="UTC date for historical mode")
    parser.add_argument("--add-variability", action="store_true")
    parser.add_argument("--bias-correction", action="store_true")
    args = parser.parse_args()

    if args.mode in ("live", "both"):
        print("=" * 72)
        print("LIVE SMHI → 15-min → power")
        print("=" * 72)
        live = run_live_forecast(
            add_variability=args.add_variability,
            apply_bias_correction=args.bias_correction,
        )
        print(
            f"  source={live.attrs.get('source')}  "
            f"ref={live.attrs.get('reference_time')}  "
            f"{live.index.min()} → {live.index.max()}"
        )
        print(live.head(8).to_string())
        _write(live, OUT / "live_forecast_15min.csv")

    if args.mode in ("historical", "both"):
        print("=" * 72)
        print(f"HISTORICAL Open-Meteo {args.date} → 15-min → power")
        print("=" * 72)
        hist = run_historical_forecast(
            args.date,
            add_variability=args.add_variability,
            apply_bias_correction=args.bias_correction,
        )
        print(f"  {hist.index.min()} → {hist.index.max()}  note={hist.attrs.get('note')}")
        print(hist.head(8).to_string())
        _write(hist, OUT / f"historical_forecast_15min_{args.date}.csv")


if __name__ == "__main__":
    main()
