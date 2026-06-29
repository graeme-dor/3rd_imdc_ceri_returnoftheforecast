# 2026 Sprint: 3rd Infodengue-Mosqlimate Dengue Challenge (IMDC)

## Team and Contributors

**CERI - Return of the Forecast - Stellenbosch University / Google DeepMind**

Carlin Foka<sup>1</sup>, Jenicca Poongavanan<sup>1</sup>, Monika Moir<sup>1</sup>, Graeme Dor<sup>1</sup>, Houriiyah Tegally<sup>1</sup>, Isabela Albuquerque<sup>2</sup>, Petar Veličković<sup>2,3</sup>

<sup>1</sup> Centre for Epidemic Response and Innovation (CERI), School of Data Science and Computational Thinking, Stellenbosch University, Stellenbosch, South Africa  
<sup>2</sup> Google DeepMind  
<sup>3</sup> University of Cambridge, Cambridge, United Kingdom

---

## Summary

This repository contains the dengue predictions submitted to the **3rd Infodengue-Mosqlimate Dengue Challenge (IMDC) 2026** by Team **CERI - Return of the Forecast**. Our final submission relies on a Dynamic Expert Bridge Ensemble (Top-9 Bridge) that dynamically weights multiple statistical and machine-learning models (including a Negative Binomial Generalized Linear Mixed Model and an ensembled spatial GNN-LSTM) based on their predicted forecast errors.

---

## Repository Structure:

*   `data/`
    *   `processed/` (clean state-week features and forecast climate variables)
    *   `submissions/` (formatted predictions to be submitted to the competition)
*   `src/`
    *   `preprocess_data.py` (script to aggregate municipal case and environmental features to state-week level)
    *   `models.py` (wrapper class registering the model architectures)
    *   `evaluate.py` (script to evaluate forecasting performance)
    *   `generate_submissions.py` (script to format predictions into challenge-compliant submission files)
    *   `bayesian/` (contains core parameter estimation and inference code for the Bayesian Thermal model)
        *   `bayesian_nb_glmm.py` (PyTorch Negative Binomial GLMM implementation)
    *   `dengue_dynamic_expert_ensemble.R` (R script to train internal experts and produce out-of-fold predictions)
    *   `add_external_experts_top3_bridge.R` (integrates GNN-LSTM and Bayesian NB-GLMM predictions as external experts)
    *   `create_final_top9_temp04_outputs.R` (R script to run bridge error modeling and weights combination)
    *   `plot_top3_final_validation_outputs.R` (script to plot validation trajectories)
    *   `plot_final_top9_vs_external_models.R` (script to compare models)
*   `requirements.txt` (Python project package dependencies)

---

## Libraries and dependencies

This project relies on a hybrid Python/R stack:
*   **Python Dependencies:** Main dependencies are defined in `requirements.txt`. The Bayesian Thermal model utilizes `PyTorch` for Maximum A Posteriori (MAP) parameter optimization and `SciPy` for predictive distributions. The GNN-LSTM model was trained using `PyTorch` and `PyTorch Geometric`. `Pandas` and `NumPy` were used for data preparation.
*   **R Dependencies:** The dynamic bridge ensemble and internal experts were built in R. Main libraries include `data.table` for data manipulation, `mgcv` for Generalized Additive Models (GAMs), `ranger` for Random Forest implementation, and `MASS` for Negative Binomial distributions.

---

## Data and Variables

The project uses the following datasets:
*   `dengue.csv.gz`: Weekly dengue case counts at the municipal level, aggregated to state-week level.
*   `climate.csv.gz`: Historical observed climate variables (used only during model training).
*   `forecasting_climate_delta_adjusted_weekly.csv`: Weekly forecasted climate variables (used strictly during target forecast periods).
*   `datasus_population_2001_2025.csv.gz`: Population data used for state aggregation and model offset.
*   `ocean_climate_oscillations.csv.gz`: Ocean index data containing ENSO, IOD, and PDO anomalies.
*   `environ_vars.csv`: Static environmental variables including Köppen climate classes and biomes.
*   `map_regional_health.csv`: State geocode and region mapping tables.

### Processed Variables:

Municipal dengue cases were aggregated to the state-week level, excluding Espírito Santo (ES) as per competition guidelines. Climate reanalysis data was aggregated to state-week level using population-weighted averages to accurately represent exposure in populated areas. For target validation and test periods, forecast climate variables from the delta-adjusted climate forecasts were aggregated to the state-week level to prevent data leakage.

Fixed effects covariates include Brière-transformed temperature suitabilities. These represent vector physiological thresholds based on literature-derived bounds: Lower $T_{min} = 17.8^\circ\text{C}$ and Upper $T_{max} = 34.6^\circ\text{C}$ (Mordecai et al., 2017). Covariates were lagged by their optimal correlation offsets: Brière-transformed minimum temperature suitability (11-week lag), Brière-transformed median temperature suitability (14-week lag), relative humidity (4-week lag), and rainy days (9-week lag). Ocean indices were lagged by 1 (ENSO), 7 (PDO), and 12 weeks (IOD). Seasonality was encoded using sine and cosine harmonics of epidemiological weeks. State populations were included as log-offsets.

---

## Model Training

The final ensemble model is a **Dynamic Expert Bridge (Top-9 Bridge)** with a softmax temperature of `0.4`. It combines 9 individual experts:
*   **GNN-LSTM Ensemble (Model 11-FC):** Post-hoc weighted quantile ensemble (0.53 border adjacency GNN + 0.47 human mobility flow GNN) with layer normalization, residual GCN blocks, and single-layer LSTM.
*   **Bayesian NB-GLMM (Bayesian Thermal):** Biological Negative Binomial GLMM fitted via MAP estimation in PyTorch, using Brière temperature suitabilities and state-level anomaly masking for historical Zika/COVID-19 periods.
*   **7 Internal Experts:** Seasonal GAM (cyclic weekly GAM), Historical Risk (negative-binomial sampling from state-week historical summaries), Climate RF (random forest on log cases), Outbreak RF (two-stage outbreak classifier and magnitude model), Low-Incidence RF (hurdle-style magnitude model), Spatial GAM (macroregion/synchrony effects), and Peak-Only RF (peak classifier/regression).

To perform the ensemble:
1. A Random Forest meta-model (400 trees) was trained for each expert to predict that expert's expected absolute error for each state-week.
2. Predicted errors were converted into weights using inverse-error weighting with a softmax temperature parameter of 0.4.

The GNN-LSTM training code is outline-configured in `src/models.py`, the Bayesian Thermal model in `src/bayesian/bayesian_nb_glmm.py`, and the R dynamic bridge ensemble execution in `src/dengue_dynamic_expert_ensemble.R`, `src/add_external_experts_top3_bridge.R`, and `src/create_final_top9_temp04_outputs.R`.

---

## Post-processing

For GNN-LSTM predictions, quantile monotonicity was enforced post-prediction using a cumulative maximum over quantile columns. For the final bridge output, quantile predictions were sorted to ensure monotonicity, and any negative values were set to zero.

---

## Data Usage Restrictions

Validation predictions were generated using strictly causal constraints:
*   Observed dengue cases and observed climate variables from target forecast periods were not used.
*   Weather variables for target forecast periods were strictly sourced from the aggregated delta-adjusted climate forecasts.
*   Historical dengue summaries and climate climatologies were recomputed inside each cross-validation split using training-period data only.

---

## Predictive Uncertainty

All experts generated quantile predictions or full predictive distributions. The Dynamic Bridge combined uncertainty by drawing samples from each expert's predictive distribution weighted by the bridge weights, and taking empirical quantiles to produce predictions across the required quantiles: `[0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]`.

---

## References

Mordecai, E.A. et al. (2017) 'Detecting the impact of temperature on transmission of Zika, dengue, and chikungunya using mechanistic models', *PLOS Neglected Tropical Diseases*, 11(4), p. e0005568. doi: 10.1371/journal.pntd.0005568.
