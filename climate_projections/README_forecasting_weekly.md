# Weekly Disaggregated Copernicus Climate Forecasts

This directory contains the code and generated dataset for weekly disaggregated climate forecasts: **`forecasting_climate_weekly.csv`**.

This dataset disaggregates the provided monthly forecasts (`data/data_imdc_2026/forecasting_climate.csv`) to a weekly Sunday resolution. It starts from the training cutoff of the first validation round (June 2022 reference month) onward.

---

## 1. File Structure

- **[disaggregate_forecasts.py](disaggregate_forecasts.py)**: Python script performing the disaggregation.
- **[forecasting_climate_weekly.csv](forecasting_climate_weekly.csv)**: Concatenated weekly forecasts dataset (shape: `(719452, 9)`).

---

## 2. Variables (9 Columns)

| Column | Type | Description |
|:---|:---|:---|
| `round` | Categorical | The validation or forecast round ID (e.g. `round_1`, `round_5_forecast_2026_2027`). |
| `reference_month` | Date (YYYY-MM-DD) | The month the forecast was issued (reference start date). |
| `forecast_months_ahead` | Integer | The monthly lead time (1 to 6 months ahead). |
| `geocode` | Integer | IBGE municipality administrative code (7 digits). Excludes state `ES`. |
| `date` | Date (YYYY-MM-DD) | Weekly Sunday representing the epidemiological week end. |
| `week` | Integer | Calendar week of the year (1 to 53). |
| `temp_med` | Float | Predicted median temperature for the week (°C). Left empty if missing. |
| `umid_med` | Float | Predicted median relative humidity for the week (%). Left empty if missing. |
| `precip_tot` | Float | Predicted total precipitation for the week (mm). Left empty if missing. |

---

## 3. Disaggregation Methodology & Causal Setup

The disaggregation is performed by mapping monthly forecasts to weekly Sunday dates, restricted to the 6-month forecast available **at the training cutoff** for each respective round:

1. For each round, we identify the forecast issued at the training cutoff (represented by the `reference_month` on or immediately before the cutoff).
2. For this `reference_month`, we retrieve the monthly predictions for lead times `forecast_months_ahead` $k \in \{1, 2, 3, 4, 5, 6\}$ (July to December for June cutoffs).
3. The target calendar month for each lead time is calculated as:
   $$\text{Target Month} = \text{reference\_month} + k \text{ months}$$
4. All weekly Sundays (epidemiological week end dates) falling inside that target calendar month are identified and assigned the corresponding monthly forecast values (`temp_med`, `umid_med`, `precip_tot`) for each municipality.
5. If a municipality contains missing values in the raw `forecasting_climate.csv` (e.g. geocode `2605459`), they are left empty (NaN) in the output.

### Date Reference Mapping
*   **`round_1`**: reference month `2022-06-01`, targets: July 2022 to December 2022.
*   **`round_2`**: reference month `2023-06-01`, targets: July 2023 to December 2023.
*   **`round_3`**: reference month `2024-06-01`, targets: July 2024 to December 2024.
*   **`round_4`**: reference month `2025-06-01`, targets: July 2025 to December 2025.
*   **`round_5_forecast_2026_2027`**: reference month `2026-03-01`, targets: April 2026 to September 2026.

---

## 4. How to Reproduce

Run the disaggregation script from the root of the workspace:
```bash
python climate_projections/disaggregate_forecasts.py
```
This script will:
1. Filter the raw monthly forecasts to keep only the specific reference months at each round's cutoff.
2. Filter out municipalities in state `ES`.
3. Compute the target calendar months and expand them to weekly Sunday dates.
4. Output the concatenated weekly forecast file.
