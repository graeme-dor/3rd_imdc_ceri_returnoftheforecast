# Weekly Step Climate Forecasts with Normals Fallback (Scenario B)

This directory contains the code and generated dataset for weekly step forecasts: **`forecasting_climate_step_fallback_weekly.csv`**.

This dataset provides municipal-level weekly forecasts for target periods of all 5 validation/forecast rounds, starting from the week immediately following each round's training cutoff date. It maps monthly forecasts to weekly step-like values inside the 6-month forecast window, falling back to climatological weekly normals for target weeks falling inside the reference month (no forecast available) or extending beyond the 6-month horizon.

---

## 1. File Structure

- **[generate_forecasts_step_fallback.py](generate_forecasts_step_fallback.py)**: Python generator script.
- **[forecasting_climate_step_fallback_weekly.csv](forecasting_climate_step_fallback_weekly.csv)**: Concatenated step-forecast dataset (shape: `(1762932, 9)`).

---

## 2. Variables (9 Columns)

| Column | Type | Description |
|:---|:---|:---|
| `round` | Categorical | The validation or forecast round ID (e.g. `round_1`, `round_5_forecast_2026_2027`). |
| `geocode` | Integer | IBGE municipality administrative code (7 digits). Excludes state `ES`. |
| `date` | Date (YYYY-MM-DD) | Weekly Sunday representing the epidemiological week end. |
| `week` | Integer | Calendar week of the year (1 to 53). |
| `temp_min` | Float | Weekly minimum temperature (°C). |
| `temp_med` | Float | Weekly median temperature (°C). |
| `umid_med` | Float | Weekly median relative humidity (%). |
| `precip_tot` | Float | Weekly total precipitation (mm). |
| `rainy_days` | Float | Weekly number of rainy days (precipitation > 0mm, range: 0.0 to 7.0). |

---

For each validation round:
1. Target weeks falling inside the **6-month forecast window** (up to 6 months after the cutoff, excluding reference month) are filled with the weekly disaggregated Copernicus forecasts (lead times 1-6) for `temp_med`, `umid_med`, and `precip_tot`.
   * `temp_min` is computed by shifting its weekly climatological normal by the weekly step median temperature anomaly:
     $$\text{temp\_min} = \text{temp\_min\_normal} + (\text{temp\_med} - \text{temp\_med\_normal})$$
   * `rainy_days` retains its weekly climatological normal:
     $$\text{rainy\_days} = \text{rainy\_days\_normal}$$
2. Target weeks falling **outside the 6-month window** (more than 6 months after the cutoff) OR falling **inside the reference month** (where Copernicus does not have a 1-6 month forecast) are filled with the historical weekly climatological normals computed using only data $\le$ cutoff.
3. If a municipality contains missing values in the raw Copernicus forecasting file, they are left empty (NaN) inside the 6-month window prior to the final spatial-fill.
4. If a municipality is missing from the observed weather data (e.g. island municipalities `2605459`, `2916104`, and `2919926`), its missing values are resolved using a neighboring/spatial forward-fill and backward-fill across geocodes, guaranteeing zero null values in the final dataset.

---

## 4. How to Reproduce

Run the generator script from the root of the workspace:
```bash
python climate_projections/generate_forecasts_step_fallback.py
```
