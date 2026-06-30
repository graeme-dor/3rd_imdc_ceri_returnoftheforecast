# Municipal-Level Climate Forecasts (Weekly Normals)

This directory contains the code and the generated baseline forecast dataset at the municipal level: **`forecasting_climate_normals_weekly.csv`**.

This dataset contains the weekly climatological normals computed causally for each validation and forecast round, restricted to the weeks extending beyond each round's training cutoff date.

---

## 1. File Structure

- **[generate_municipal_forecasts.py](generate_municipal_forecasts.py)**: Python generator script.
- **[forecasting_climate_normals_weekly.csv](forecasting_climate_normals_weekly.csv)**: Concatenated forecast dataset across all rounds (shape: `(1762932, 15)`).

---

## 2. Variables (15 Columns)

| Column | Type | Description |
|:---|:---|:---|
| `round` | Categorical | The validation or forecast round ID (e.g. `round_1`, `round_5_forecast_2026_2027`). |
| `geocode` | Integer | IBGE municipality administrative code (7 digits). Excludes state `ES`. |
| `date` | Date (YYYY-MM-DD) | Weekly Sunday representing the epidemiological week end. |
| `week` | Integer | Calendar week of the year (1 to 53). |
| `temp_min` | Float | Climatological weekly normal of minimum temperature (°C). |
| `temp_med` | Float | Climatological weekly normal of median temperature (°C). |
| `temp_max` | Float | Climatological weekly normal of maximum temperature (°C). |
| `precip_min` | Float | Climatological weekly normal of minimum precipitation (mm). |
| `precip_med` | Float | Climatological weekly normal of median precipitation (mm). |
| `precip_max` | Float | Climatological weekly normal of maximum precipitation (mm). |
| `rel_humid_min` | Float | Climatological weekly normal of minimum relative humidity (%). |
| `rel_humid_med` | Float | Climatological weekly normal of median relative humidity (%). |
| `rel_humid_max` | Float | Climatological weekly normal of maximum relative humidity (%). |
| `thermal_range` | Float | Climatological weekly normal of diurnal temperature range (°C). |
| `rainy_days` | Float | Climatological weekly normal of the number of rainy days in the week. |

---

## 3. Date Cutoffs & Forecast Approach

The forecast values are calculated causally using the **historical arithmetic mean** (climatological normals) of the observed variables for each calendar week (grouped by municipality and week) over all available training years on or before the cutoff date:

$$\text{Normal}_{g, w} = \frac{1}{|Y_{\le \text{cutoff}}|} \sum_{y \in Y_{\le \text{cutoff}}} x_{g, y, w}$$

Where $x_{g, y, w}$ is the observed climate value for municipality $g$, year $y$, and calendar week $w$.

| Round | Training Cutoff | Forecast Start | Forecast End | Weeks |
|:---|:---:|:---:|:---:|:---:|
| `round_1` | `2022-06-19` | `2022-06-26` | `2023-10-01` | 67 |
| `round_2` | `2023-06-18` | `2023-06-25` | `2024-09-29` | 67 |
| `round_3` | `2024-06-16` | `2024-06-23` | `2025-09-28` | 67 |
| `round_4` | `2025-06-15` | `2025-06-22` | `2026-03-08` | 38 |
| `round_5_forecast_2026_2027` | `2026-03-08` | `2026-03-15` | `2027-10-03` | 82 |

---

## 4. How to Reproduce

Run the generator script from the root of the workspace:
```bash
python climate_projections/generate_municipal_forecasts.py
```
This script will:
1. Load valid municipality geocodes from `data/data_imdc_2026/map_regional_health.csv`.
2. Load and filter `data/data_imdc_2026/climate.csv` in chunks.
3. Compute municipal-level weekly normals.
4. Build the continuous target grids and merge lookup values.
5. Save the unified forecasts.
