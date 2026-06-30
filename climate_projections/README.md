# Weekly Delta-Adjusted Climate Forecasts

This directory contains the code and documentation for the delta-adjusted climate forecasts: **`forecasting_climate_delta_adjusted_weekly.csv`**.

This dataset provides municipal-level weekly forecasts for the target periods of all 4 validation rounds, starting from the week immediately following each round's training cutoff date. Inside the 6-month forecast window, weekly climatological normals are shifted by the Copernicus forecast anomalies (deltas) relative to monthly observed normals. Beyond the 6-month window (or for weeks falling in the reference month where Copernicus has no forecast), it falls back to standard weekly normals.

---

## 1. File Structure

- **[generate_forecasts_delta_adjusted.py](generate_forecasts_delta_adjusted.py)**: Python generator script.
- **[forecasting_climate_delta_adjusted_weekly.csv](forecasting_climate_delta_adjusted_weekly.csv)**: Concatenated delta-adjusted forecasts dataset.

---

## 2. Variables (15 Columns)

| Column | Type | Description |
|:---|:---|:---|
| `round` | Categorical | The validation round ID (e.g. `round_1`, `round_4`). |
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
   $$
   \begin{aligned}
   \Delta_{\text{temp}} &= \text{Forecast}_{\text{temp, med}} - \text{Normal}_{\text{temp, med}} \\
   \Delta_{\text{humid}} &= \text{Forecast}_{\text{humidity, med}} - \text{Normal}_{\text{humidity, med}} \\
   \text{Ratio}_{\text{precip}} &= \min\left(3.0, \frac{\text{Forecast}_{\text{precipitation, total}}}{\max(1.0, \text{Normal}_{\text{precipitation, med}})}\right)
   \end{aligned}
   $$
3. We adjust the weekly climatological normals ($w$) for target weeks falling inside month $M$:
   * **Temperature**:
     $$
     \begin{aligned}
     \hat{w}_{\text{temp, min}} &= w_{\text{temp, min}} + \Delta_{\text{temp}} \\
     \hat{w}_{\text{temp, med}} &= w_{\text{temp, med}} + \Delta_{\text{temp}} \\
     \hat{w}_{\text{temp, max}} &= w_{\text{temp, max}} + \Delta_{\text{temp}}
     \end{aligned}
     $$
   * **Relative Humidity** (clipped to $[0.0, 100.0]$):
     $$
     \begin{aligned}
     \hat{w}_{\text{humid, min}} &= \max\left(0.0, \min\left(100.0, w_{\text{humid, min}} + \Delta_{\text{humid}}\right)\right) \\
     \hat{w}_{\text{humid, med}} &= \max\left(0.0, \min\left(100.0, w_{\text{humid, med}} + \Delta_{\text{humid}}\right)\right) \\
     \hat{w}_{\text{humid, max}} &= \max\left(0.0, \min\left(100.0, w_{\text{humid, max}} + \Delta_{\text{humid}}\right)\right)
     \end{aligned}
     $$
   * **Precipitation** (multiplicative scaling):
     $$
     \begin{aligned}
     \hat{w}_{\text{precip, min}} &= w_{\text{precip, min}} \times \text{Ratio}_{\text{precip}} \\
     \hat{w}_{\text{precip, med}} &= w_{\text{precip, med}} \times \text{Ratio}_{\text{precip}} \\
     \hat{w}_{\text{precip, max}} &= w_{\text{precip, max}} \times \text{Ratio}_{\text{precip}}
     \end{aligned}
     $$
   * **Thermal Range & Rainy Days**:
     $$
     \begin{aligned}
     \hat{w}_{\text{thermal range}} &= w_{\text{thermal range}} \\
     \hat{w}_{\text{rainy days}} &= w_{\text{rainy days}}
     \end{aligned}
     $$

### B. Outside the 6-Month Forecast Window OR Reference Month (Target dates $>$ Cutoff + 6 months, or within reference month)
1. The forecast falls back to the standard weekly climatological normals: $\hat{w} = w$
2. If a municipality contains missing values in the raw Copernicus forecasting file, they are left empty (NaN) inside the 6-month window prior to the final spatial-fill.
3. If a municipality is completely missing from the observed weather data (e.g. island municipalities `2605459`, `2916104` and `2919926`), its missing values are resolved using a neighboring/spatial forward-fill and backward-fill across geocodes, guaranteeing zero null values in the final dataset.

---

## 4. How to Reproduce

Run the generator script from the root of the workspace:
```bash
python climate_projections/generate_forecasts_delta_adjusted.py
```
