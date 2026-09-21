#!/usr/bin/env python3
"""Open-Meteo Historical Forecast wind for a single UTC day at Björkö.

This is a *stitched short-lead-time* series, not a true archived single-run
day-ahead forecast.

Open-Meteo's Historical Forecast API concatenates the first few hours of
each successive model run into one continuous hourly timeline. That is
close to analysis-quality weather, but it is **not** "the forecast that
was issued yesterday for tomorrow." Those archived full-horizon runs
exist only on the Single Runs API, and only from 2024 onward (ECMWF IFS
HRES from March 2024; other models later). They do not cover the
Björkö campaign dates (2022-07-05 – 2023-08-02).

Use this fetcher to pair NWP wind with the campaign 15-minute SCADA
days, with that caveat attached to any skill score you compute.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Union

import pandas as pd

from nwp_http import get_json

BJORKO_LAT = 57.71818820625921
BJORKO_LON = 11.683382148764485

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

DateLike = Union[str, date, datetime, pd.Timestamp]


def _as_iso_date(value: DateLike) -> str:
    if isinstance(value, str):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"date must be str/date/datetime, got {type(value)!r}")


def fetch_historical_forecast(
    date: DateLike,
    lat: float = BJORKO_LAT,
    lon: float = BJORKO_LON,
) -> pd.Series:
    """Hourly 10 m wind speed [m/s] for one UTC day from Open-Meteo.

    Calls the Historical Forecast API with ``start_date`` and ``end_date``
    both set to ``date`` and ``hourly=wind_speed_10m``. Wind is requested
    in m/s (the API default is km/h).

    This is a stitched short-lead-time series, not a true archived
    single-run day-ahead forecast (those only exist from 2024 onward via
    the Single Runs API, which does not cover our 2022–2023 dates).
    """
    day = _as_iso_date(date)
    payload = get_json(
        HISTORICAL_FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": day,
            "end_date": day,
            "hourly": "wind_speed_10m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
    )
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    values = hourly.get("wind_speed_10m") or []
    if not times or len(times) != len(values):
        raise ValueError(f"Open-Meteo hourly payload missing or misaligned for {day}")

    index = pd.to_datetime(times, utc=True)
    series = pd.Series(values, index=index, name="ws", dtype=float)
    series.index.name = "timestamp"
    series.attrs["source"] = "open-meteo-historical-forecast"
    series.attrs["note"] = (
        "Stitched short-lead-time Historical Forecast API series, "
        "not an archived single-run day-ahead forecast."
    )
    series.attrs["latitude"] = payload.get("latitude", lat)
    series.attrs["longitude"] = payload.get("longitude", lon)
    series.attrs["elevation_m"] = payload.get("elevation")
    series.attrs["hourly_units"] = payload.get("hourly_units")
    series.attrs["date"] = day
    return series


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", nargs="?", default="2022-08-11", help="UTC date YYYY-MM-DD")
    args = parser.parse_args()
    ws = fetch_historical_forecast(args.date)
    print(f"Open-Meteo historical  {ws.attrs['date']}  n={len(ws)}")
    print(f"  {ws.index.min()} → {ws.index.max()}  unit={ws.attrs.get('hourly_units')}")
    print(ws.to_string())


if __name__ == "__main__":
    main()
