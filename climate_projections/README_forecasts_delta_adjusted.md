# Weekly Delta-Adjusted Climate Forecasts with Normals Fallback (Scenario C)

This directory contains the code and generated dataset for weekly delta-adjusted forecasts: **`forecasting_climate_delta_adjusted_weekly.csv`**.

This dataset provides municipal-level weekly forecasts for the target periods of all 5 validation/forecast rounds, starting from the week immediately following each round's training cutoff date. Inside the 6-month forecast window, weekly climatological normals are shifted by the Copernicus forecast anomalies (deltas) relative to monthly observed normals. Beyond the 6-month window (or for weeks falling in the reference month where Copernicus has no forecast), it falls back to standard weekly normals.

---

## 1. File Structure

- **[generate_forecasts_delta_adjusted.py](generate_forecasts_delta_adjusted.py)**: Python generator script.
- **[forecasting_climate_delta_adjusted_weekly.csv](forecasting_climate_delta_adjusted_weekly.csv)**: Concatenated delta-adjusted forecasts dataset (shape: `(1762932, 15)`).

---

## 2. Variables (15 Columns)

| Column | Type | Description |
|:---|:---|:---|
| `round` | Categorical | The validation or forecast round ID (e.g. `round_1`, `round_5_forecast_2026_2027`). |
| `geocode` | Integer | IBGE municipality administrative code (7 digits). Excludes state `ES`. |
| `date` | Date (YYYY-MM-DD) | Weekly Sunday representing the epidemiological week end. |
| `week` | Integer | Calendar week of the year (1 to 53). |
| `temp_min` | Float | Weekly minimum temperature (°C), adjusted by median temp anomaly. |
| `temp_med` | Float | Weekly median temperature (°C), adjusted by median temp anomaly. |
| `temp_max` | Float | Weekly maximum temperature (°C), adjusted by median temp anomaly. |
| `precip_min` | Float | Weekly minimum precipitation (mm), adjusted by precipitation anomaly. |
| `precip_med` | Float | Weekly median precipitation (mm), adjusted by precipitation anomaly. |
| `precip_max` | Float | Weekly maximum precipitation (mm), adjusted by precipitation anomaly. |
| `rel_humid_min` | Float | Weekly minimum relative humidity (%), adjusted by humidity anomaly. |
| `rel_humid_med` | Float | Weekly median relative humidity (%), adjusted by humidity anomaly. |
| `rel_humid_max` | Float | Weekly maximum relative humidity (%), adjusted by humidity anomaly. |
| `thermal_range` | Float | Diurnal temperature range (°C). Retains weekly climatological normal. |
| `rainy_days` | Float | Weekly number of rainy days (precipitation > 0mm). Retains weekly climatological normal. |

---

## 3. Delta Adjustment & Fallback Methodology

For each validation round:

### A. Inside the 6-Month Forecast Window (Target dates $\le$ Cutoff + 6 months, excluding reference month)
1. We compute the historical monthly observed normal for each geocode and calendar month $M \in \{1..12\}$ using only data $\le$ cutoff.
2. For each lead forecast month $M$, we compute the monthly forecast anomaly (delta or ratio) on the primary monthly forecast variables:
   $$\Delta_{temp} = \text{Forecast}_{temp\_med} - \text{Normal}_{temp\_med}$$
   $$\Delta_{humid} = \text{Forecast}_{umid\_med} - \text{Normal}_{rel\_humid\_med}$$
   $$\text{Ratio}_{precip} = \min\left(3.0, \frac{\text{Forecast}_{precip\_tot\_weekly}}{\max(1.0, \text{Normal}_{precip\_med})}\right)$$
3. We adjust the weekly climatological normals ($w$) for target weeks falling inside month $M$:
   * **Temperature**:
     $$\hat{w}_{temp\_min} = w_{temp\_min} + \Delta_{temp}$$
     $$\hat{w}_{temp\_med} = w_{temp\_med} + \Delta_{temp}$$
     $$\hat{w}_{temp\_max} = w_{temp\_max} + \Delta_{temp}$$
   * **Relative Humidity** (clipped to $[0.0, 100.0]$):
     $$\hat{w}_{rel\_humid\_min} = \max(0.0, \min(100.0, w_{rel\_humid\_min} + \Delta_{humid}))$$
     $$\hat{w}_{rel\_humid\_med} = \max(0.0, \min(100.0, w_{rel\_humid\_med} + \Delta_{humid}))$$
     $$\hat{w}_{rel\_humid\_max} = \max(0.0, \min(100.0, w_{rel\_humid\_max} + \Delta_{humid}))$$
   * **Precipitation** (multiplicative scaling):
     $$\hat{w}_{precip\_min} = w_{precip\_min} \times \text{Ratio}_{precip}$$
     $$\hat{w}_{precip\_med} = w_{precip\_med} \times \text{Ratio}_{precip}$$
     $$\hat{w}_{precip\_max} = w_{precip\_max} \times \text{Ratio}_{precip}$$
   * **Thermal Range & Rainy Days**:
     $$\hat{w}_{thermal\_range} = w_{thermal\_range}$$
     $$\hat{w}_{rainy\_days} = w_{rainy\_days}$$

### B. Outside the 6-Month Forecast Window OR Reference Month (Target dates $>$ Cutoff + 6 months, or within reference month)
1. The forecast falls back to the standard weekly climatological normals:
   $$\hat{w} = w$$
2. If a municipality contains missing values in the raw Copernicus forecasting file, they are left empty (NaN) inside the 6-month window prior to the final spatial-fill.
3. If a municipality is completely missing from the observed weather data (e.g. island municipalities `2605459`, `2916104` and `2919926`), its missing values are resolved using a neighboring/spatial forward-fill and backward-fill across geocodes, guaranteeing zero null values in the final dataset.

---

## 4. How to Reproduce

Run the generator script from the root of the workspace:
```bash
python climate_projections/generate_forecasts_delta_adjusted.py
```
