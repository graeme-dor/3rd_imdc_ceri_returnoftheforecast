import os
import pandas as pd
import numpy as np

def aggregate_state_climate_projections():
    print("Starting weekly state-level climate projections aggregation...")
    
    # 1. Load Geocode mapping (excluding ES)
    print("Loading geocode mappings...")
    df_map = pd.read_csv('data/data_imdc_2026/map_regional_health.csv')
    geocode_to_state = df_map.set_index('geocode')['uf'].to_dict()
    
    # 2. Load Population data (use year 2025 for mapping)
    print("Loading population data for weighting...")
    df_pop = pd.read_csv('data/data_imdc_2026/datasus_population_2001_2025.csv.gz')
    df_pop_2025 = df_pop[df_pop['year'] == 2025].copy()
    pop_lookup = df_pop_2025.set_index('geocode')['population'].to_dict()
    
    # 3. Read and aggregate municipal weekly forecasts in chunks
    fc_path = 'climate_projections/forecasting_climate_delta_adjusted_weekly.csv'
    if not os.path.exists(fc_path):
        raise FileNotFoundError(f"Municipal weekly forecast not found at {fc_path}")
        
    print(f"Reading municipal forecast file: {fc_path}")
    chunk_size = 300000
    climate_vars = [
        'temp_min', 'temp_med', 'temp_max', 
        'precip_min', 'precip_med', 'precip_max', 
        'rel_humid_min', 'rel_humid_med', 'rel_humid_max', 
        'thermal_range', 'rainy_days'
    ]
    
    chunks = []
    
    for chunk in pd.read_csv(fc_path, chunksize=chunk_size):
        # Map geocode to state
        chunk['uf'] = chunk['geocode'].map(geocode_to_state.get)
        # Filter out ES and missing states
        chunk = chunk[(chunk['uf'].notna()) & (chunk['uf'] != 'ES')].copy()
        if len(chunk) == 0:
            continue
            
        # Map population for each geocode
        chunk['pop'] = chunk['geocode'].map(pop_lookup.get).fillna(1.0)
        
        # Compute population-weighted variables
        for var in climate_vars:
            chunk[f'{var}_weighted'] = chunk[var] * chunk['pop']
            
        keep_cols = ['uf', 'round', 'date', 'pop'] + [f'{var}_weighted' for var in climate_vars]
        chunks.append(chunk[keep_cols])
        
    print("Concatenating and grouping chunks...")
    df_all = pd.concat(chunks, ignore_index=True)
    
    # Group by state, round, and date
    df_state = df_all.groupby(['uf', 'round', 'date']).sum().reset_index()
    
    # Divide by total population to get weighted average
    for var in climate_vars:
        df_state[var] = df_state[f'{var}_weighted'] / df_state['pop']
        df_state = df_state.drop(columns=[f'{var}_weighted'])
        
    df_state = df_state.drop(columns=['pop'])
    
    # Sort
    df_state = df_state.sort_values(['round', 'uf', 'date']).reset_index(drop=True)
    
    # Save to processed directory
    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'state_forecasting_climate_delta_adjusted.csv')
    df_state.to_csv(out_path, index=False)
    print(f"Saved aggregated state forecasts to {out_path} (shape: {df_state.shape})")

if __name__ == '__main__':
    aggregate_state_climate_projections()
