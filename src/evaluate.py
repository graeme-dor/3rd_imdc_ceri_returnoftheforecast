import os
import pandas as pd
import numpy as np

def compute_wis(y_true, q_dict):
    """
    Computes the Weighted Interval Score (WIS) for predictions.
    """
    ae = (y_true - q_dict[0.5]).abs()
    wis = ae.copy()
    
    intervals = [
        (0.25, 0.75, 0.50),  # 50% interval
        (0.10, 0.90, 0.20),  # 80% interval
        (0.05, 0.95, 0.10),  # 90% interval
        (0.025, 0.975, 0.05) # 95% interval
    ]
    
    for l_q, u_q, alpha in intervals:
        l = q_dict[l_q]
        u = q_dict[u_q]
        under_penalty = (l - y_true).clip(lower=0) * (2.0 / alpha)
        over_penalty = (y_true - u).clip(lower=0) * (2.0 / alpha)
        is_score = (u - l) + under_penalty + over_penalty
        wis += (alpha / 2.0) * is_score
        
    return wis / (len(intervals) + 1)

def run_validation(model_class, model_name):
    """
    Fits and evaluates the model across defined retrospective validation rounds.
    """
    print(f"Evaluating {model_name}...")
    
    # 1. Load preprocessed weekly features from data/processed/state_weekly_features.csv
    # 2. Iterate through retrospective validation splits
    # 3. Fit model on training interval
    # 4. Generate forecasts for target interval
    # 5. Compute performance metrics (WIS, MAE, RMSE, and 95% Coverage)
    
    pass

if __name__ == '__main__':
    # run_validation(...)
    pass
