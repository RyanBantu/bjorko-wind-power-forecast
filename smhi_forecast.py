#!/usr/bin/env python3
"""Live SMHI wind forecast for the Björkö turbine.

The URL the pipeline was specified against,

    /api/category/pmp3g/version/2/geotype/point/lon/{lon}/lat/{lat}/data.json

was retired by SMHI on 2026-03-31 (PMP3gv2). It now returns HTTP 404.
This module still *tries* that endpoint first, then falls back to the
replacement SNOW1gv1 API:

    /api/category/snow1g/version/1/geotype/point/lon/{lon}/lat/{lat}/data.json

Both payloads are normalised to the old names:

    ws  wind speed [m/s]
    wd  wind direction [deg]

and a UTC-indexed pandas Series of ``ws`` covering the next 24 hours
from ``referenceTime``. Direction is attached as ``series.attrs["wd"]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nwp_http import get_json

BJORKO_LAT = 57.71818820625921
BJORKO_LON = 11.683382148764485

PMP3G_URL = (
    "https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2"
    "/geotype/point/lon/{lon}/lat/{lat}/data.json"
)
SNOW1G_URL = (
    "https://opendata-download-metfcst.smhi.se/api/category/snow1g/version/1"
    "/geotype/point/lon/{lon}/lat/{lat}/data.json"
)

# SNOW1g uses 9999 as a missing-value sentinel.
MISSING = 9999
HORIZON = pd.Timedelta(hours=24)


def _format_coord(value: float) -> str:
    """SMHI accepts at most 6 decimal places."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return float(value) == MISSING
    except (TypeError, ValueError):
        return True


def _param_value(parameters: list[dict], name: str) -> float:
    for item in parameters:
        if item.get("name") == name and item.get("values"):
            value = item["values"][0]
            if _is_missing(value):
                return np.nan
            return float(value)
    return np.nan


def _parse_pmp3g(payload: dict) -> pd.DataFrame:
    rows = []
    for entry in payload.get("timeSeries", []):
        rows.append(
            {
                "timestamp": entry.get("validTime"),
                "ws": _param_value(entry.get("parameters", []), "ws"),
                "wd": _param_value(entry.get("parameters", []), "wd"),
            }
        )
    return pd.DataFrame(rows)


def _parse_snow1g(payload: dict) -> pd.DataFrame:
    rows = []
    for entry in payload.get("timeSeries", []):
        data = entry.get("data") or {}
        ws = data.get("wind_speed")
        wd = data.get("wind_from_direction")
        rows.append(
            {
                "timestamp": entry.get("time") or entry.get("validTime"),
                "ws": np.nan if _is_missing(ws) else float(ws),
                "wd": np.nan if _is_missing(wd) else float(wd),
            }
        )
    return pd.DataFrame(rows)


def _to_frame(payload: dict) -> pd.DataFrame:
    sample = (payload.get("timeSeries") or [{}])[0]
    if "parameters" in sample:
        frame = _parse_pmp3g(payload)
        api = "pmp3g"
    else:
        frame = _parse_snow1g(payload)
        api = "snow1g"
    if frame.empty:
        raise ValueError("SMHI timeSeries is empty")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame.attrs["api"] = api
    return frame


def _clip_horizon(frame: pd.DataFrame, reference_time: pd.Timestamp) -> pd.DataFrame:
    """Keep the next 24 h from referenceTime, plus the next knot for interpolation."""
    start = reference_time
    end = reference_time + HORIZON
    in_window = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
    after = frame.loc[frame["timestamp"] > end]
    if in_window.empty:
        raise ValueError(f"no SMHI steps between {start} and {end}")
    # Hourly valid times sit on the hour, so the 24 h mark (e.g. 21:15+24h)
    # usually falls between hours. Keep the next knot so 15-min interpolation
    # covers a full 24-hour horizon.
    if not after.empty:
        extra = after.iloc[[0]]
        if extra["timestamp"].iloc[0] not in set(in_window["timestamp"]):
            in_window = pd.concat([in_window, extra], ignore_index=True)
    return in_window.reset_index(drop=True)


def fetch_smhi_forecast(
    lat: float = BJORKO_LAT,
    lon: float = BJORKO_LON,
) -> pd.Series:
    """Download the SMHI point forecast and return 24 h of wind speed [m/s].

    Parameters
    ----------
    lat, lon
        WGS84 coordinates. Defaults are the Chalmers Björkö turbine
        (57.71818820625921, 11.683382148764485).

    Returns
    -------
    pandas.Series
        Hourly wind speed [m/s], UTC DatetimeIndex, name ``ws``.
        ``series.attrs["wd"]`` is the matching wind-direction Series [deg].
        ``series.attrs["reference_time"]`` is the model reference time.
    """
    lon_s, lat_s = _format_coord(lon), _format_coord(lat)
    pmp_url = PMP3G_URL.format(lon=lon_s, lat=lat_s)
    snow_url = SNOW1G_URL.format(lon=lon_s, lat=lat_s)

    payload = None
    source_url = pmp_url
    try:
        payload = get_json(pmp_url)
    except Exception:
        payload = get_json(snow_url)
        source_url = snow_url

    frame = _to_frame(payload)
    reference_time = pd.to_datetime(payload.get("referenceTime"), utc=True)
    if pd.isna(reference_time):
        reference_time = frame["timestamp"].iloc[0]
    clipped = _clip_horizon(frame, reference_time)

    ws = pd.Series(
        clipped["ws"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(clipped["timestamp"], name="timestamp"),
        name="ws",
    )
    wd = pd.Series(
        clipped["wd"].to_numpy(dtype=float),
        index=ws.index,
        name="wd",
    )
    ws.attrs["wd"] = wd
    ws.attrs["reference_time"] = reference_time
    ws.attrs["created_time"] = payload.get("createdTime") or payload.get("approvedTime")
    ws.attrs["source_url"] = source_url
    ws.attrs["api"] = frame.attrs.get("api")
    ws.attrs["geometry"] = payload.get("geometry")
    return ws


def main() -> None:
    ws = fetch_smhi_forecast()
    print(f"SMHI {ws.attrs.get('api')}  ref={ws.attrs['reference_time']}  n={len(ws)}")
    print(f"  {ws.index.min()} → {ws.index.max()}  source={ws.attrs.get('source_url')}")
    print(pd.DataFrame({"ws": ws, "wd": ws.attrs["wd"]}).to_string())


if __name__ == "__main__":
    main()
