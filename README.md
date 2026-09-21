# Björkö Wind Power Forecast

Day-ahead, 15-minute wind power forecasting pipeline for the Chalmers wind turbine (Björkö, Sweden) — a two-stage approach combining NWP wind speed forecasts with an empirically fitted power curve, bias-corrected and validated with leave-one-day-out (LODO) cross-validation.

## Overview

This project builds a day-ahead power forecast for a 45 kW research turbine using:

1. **An empirical power curve**, fitted from real SCADA data rather than a theoretical/manufacturer curve
2. **NWP wind speed forecasts** (SMHI for live operational use, Open-Meteo's Historical Forecast API for backtesting)
3. **Hourly ? 15-minute disaggregation**, with an optional realistic variability layer
4. **A linear bias correction** (model output statistics) between NWP wind speed and the turbine's own mast measurements
5. **Leave-one-day-out cross-validation** across 44 measurement-campaign days, to report honest out-of-sample forecast skill

Full methodology and results are written up in [`wind_forecast_paper.docx`](./wind_forecast_paper.docx).

## Why this approach

The turbine's own historical SCADA record is **not a continuous time series** — it consists of 44 non-contiguous measurement-campaign days across a 13-month span, with gaps of up to several weeks. That rules out training a sequence model (LSTM, etc.) directly on turbine history. Instead, this pipeline separates the problem into a meteorological forecast (which *is* available continuously, from any NWP provider) and a turbine response model (which only needs to characterize the turbine's static wind-to-power behavior, and can be fit reliably even from non-continuous data given a large enough sample).

## Data source

**Raw SCADA data is not included in this repository** (file sizes exceed GitHub's limits, and the data is already properly published elsewhere).

Download it from Zenodo before running any pipeline scripts:

> Fogelström, S., Johansson, H., Carlson, O., Hofsäß, M., Bischoff, O., Marykovskiy, Y., & Abdallah, I. (2023). *Björkö Wind Turbine Version 1 (45kW) high frequency Structural Health Monitoring (SHM) data* (Version 3) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8230330

Place these files in the project root:
- `B1_CL4_20.csv` — 20 Hz SCADA record (the usable operational data, ~2.8 GB)
- `B1_CL4_100.csv` — 100 Hz SCADA record (only 9 short SHM bursts on 4 days — not used for forecasting)
- `Bjorko_Sensors_Specs_Metadata.csv` — channel name/unit reference
- `Bjorko_modes_mapping.json` — controller `SysMode` code reference
- `Bjorko_digital_io_states_mappings.csv` — digital I/O state reference

## Setup

```bash
git clone https://github.com/RyanBantu/bjorko-wind-power-forecast.git
cd bjorko-wind-power-forecast
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then download the Zenodo files above into the project root before running any script.

## Pipeline scripts

Run in this order for a full rebuild from raw data:

| Script | Purpose | Key outputs |
|---|---|---|
| `build_power_curve.py` | Loads `Time`, `SysMode`, `WS30`, `WSN`, `DCC`, `DCV` from the 20 Hz file (chunked), filters to `SysMode == 12` (Running), bins wind speed into 0.5 m/s buckets | `outputs/bjorko_power_curve_20hz_hf.csv`, `outputs/bjorko_power_curve_20hz.csv`, `outputs/bjorko_15min_running_20hz.csv` |
| `power_curve_model.py` | Fits a monotonic PCHIP interpolator to the well-populated bins (n ? 2000), with programmatic cut-in detection | `outputs/power_curve_pchip.pkl`, `outputs/power_curve_pchip.json`, `outputs/power_curve_pchip_fit.png` |
| `wind_disaggregate.py` | Extracts a sub-hourly variability profile from real 20 Hz data; disaggregates hourly wind speed to 15-minute steps (linear, with optional injected variability) | `outputs/subhourly_variability_profile.csv/json`, `outputs/demo_hourly_to_15min.png` |
| `smhi_forecast.py` | Live NWP fetcher — SMHI open forecast API (SNOW1gv1, since PMP3g was retired 31 Mar 2026) | — |
| `openmeteo_historical.py` | Historical NWP fetcher — Open-Meteo Historical Forecast API, used for backtesting against 2022–2023 campaign dates | — |
| `forecast_pipeline.py` | End-to-end: fetch wind forecast ? disaggregate to 15-min ? apply bias correction (optional) ? convert through power curve | `outputs/live_forecast_15min.csv`, `outputs/historical_forecast_15min_<date>.csv` |
| `backtest.py` | Runs the historical pipeline across all 44 campaign days, aligns against observed power, computes MAE/RMSE/nRMSE | `outputs/backtest_results.csv`, `outputs/backtest_power_scatter.png`, `outputs/backtest_power_timeseries.png` |
| `wind_bias_correction.py` | Fits the linear wind-speed bias correction and runs leave-one-day-out cross-validation | `outputs/wind_bias_correction.json`, `outputs/wind_bias_correction_lodo_folds.csv` |

## Usage

**Live day-ahead forecast (next 24h, from now):**
```bash
python3 forecast_pipeline.py --mode live
```

**Historical forecast for a specific past campaign day:**
```bash
python3 forecast_pipeline.py --mode historical --date 2022-08-11
```

**Reproduce the full backtest:**
```bash
python3 backtest.py
```

## Results summary

Validated out-of-sample (leave-one-day-out) across 311 aligned 15-minute points spanning 44 campaign days:

| Metric | Uncorrected | Bias-corrected (LODO out-of-sample) |
|---|---:|---:|
| Wind speed MAE | 1.289 m/s | 1.003 m/s |
| Wind speed bias | ?0.957 m/s | +0.009 m/s |
| Power RMSE | 5.776 kW | 5.479 kW |
| Power nRMSE (vs. 19.99 kW plateau) | 0.289 | 0.274 |

Full results, error decomposition, and discussion of limitations are in the paper.

## Known limitations

- Ground truth is 44 non-continuous days, not a full year of continuous SCADA — reported skill should be read as indicative rather than a robust year-round validation.
- Historical backtesting uses Open-Meteo's Historical Forecast API, a stitched short-lead-time proxy — true archived single-model-run forecasts (exact day-ahead reproduction) are only available from 2024 onward and don't cover this turbine's 2022–2023 campaign dates.
- "Power" in this dataset is DC rectifier power (current × voltage), not AC power exported to the grid.
- The turbine's measured plateau (~20 kW) is well below its 45 kW nameplate rating, likely due to a combination of yaw error, mast-height vs. hub-height mismatch, and FFR/curtailment behavior tied to the turbine's grid-services research program — see the paper's discussion section.

## Citation

If you use this pipeline, please cite the underlying SCADA dataset:

```
Fogelström, S., Johansson, H., Carlson, O., Hofsäß, M., Bischoff, O., Marykovskiy, Y., & Abdallah, I. (2023).
Björkö Wind Turbine Version 1 (45kW) high frequency Structural Health Monitoring (SHM) data (Version 3) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.8230330
```

## Author

Ryan Bantu, Goperch Innovations Pvt Ltd
Project supervised by K. Victor Sam Moses Babu
