# Dengue Forecasting Challenge 2026 - Team CERI - Return of the Forecast

This repository contains the source code, data preprocessing, and predictions submitted by Team **CERI - Return of the Forecast** for the **3rd Infodengue–Mosqlimate Dengue Challenge (IMDC) 2026** (Dengue forecasting at the state level in Brazil).

---

## 1. Team and Contributors
*   **Team Name:** CERI - Return of the Forecast
*   **Team Members:** Carlin Foka<sup>1</sup>, Jenicca Poongavanan<sup>1</sup>, Monika Moir<sup>1</sup>, Graeme Dor<sup>1</sup>, Houriiyah Tegally<sup>1</sup>, Isabela Albuquerque<sup>2</sup>, Petar Veličković<sup>2,3</sup>

<sup>1</sup> Centre for Epidemic Response and Innovation (CERI), School of Data Science and Computational Thinking, Stellenbosch University, Stellenbosch, South Africa  
<sup>2</sup> Google DeepMind  
<sup>3</sup> University of Cambridge, Cambridge, United Kingdom

---

## 2. Repository Structure

*   `src/`: Python source files for preprocessing, modeling, validation, and prediction formatting.
    *   `src/preprocess_data.py`: Preprocessing script that aggregates probable cases to the state level, computes population-weighted climate features, and merges demographic/ocean features.
    *   `src/models.py`: Python wrapper registering model architectures.
    *   `src/evaluate.py`: Validation script to evaluate forecasting metrics.
    *   `src/generate_submissions.py`: Formatter generating standardized challenge-compliant submission files.
    *   `src/bayesian/`: Package containing core model components and parameter estimation routines.
*   `data/`: Directory structure for inputs, intermediates, and outputs.
    *   `data/processed/`: Folder holding clean merged modeling features.
    *   `data/submissions/`: Folder holding formatted prediction files.
*   `requirements.txt`: Python package requirements and environment dependencies.

---

## 3. Execution Workflow

To run the data preprocessing, model fitting, validation, and submission generation:

1.  **Environment Setup**: Install dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Data Preprocessing**: Aggregate and align cases and environmental features:
    ```bash
    PYTHONPATH=src python src/preprocess_data.py
    ```
3.  **Model Validation**: Evaluate forecasting accuracy across historical target rounds:
    ```bash
    PYTHONPATH=src python src/evaluate.py
    ```
4.  **Submission Generation**: Fit the modeling configuration on training intervals and export predictions:
    ```bash
    PYTHONPATH=src python src/generate_submissions.py
    ```
