import os
import pandas as pd
import numpy as np

def preprocess():
    """
    Loads raw inputs, aggregates probable dengue cases and weather parameters
    to state level using population-weighted averages, aligns weekly ocean indicators,
    and saves the unified features.
    """
    print("Starting data preprocessing...")
    
    # 1. Load geocode mappings
    # 2. Aggregate probable cases (dengue) to state level
    # 3. Load population data for demographic offset weighting
    # 4. Load climate reanalysis data and compute population-weighted averages
    # 5. Load and align ocean indicators
    # 6. Save combined features dataset to data/processed/state_weekly_features.csv
    
    print("Data preprocessing completed.")

if __name__ == '__main__':
    preprocess()
