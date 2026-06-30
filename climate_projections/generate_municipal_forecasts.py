import os
import pandas as pd
import numpy as np
import datetime

def generate_municipal_forecasts():
    print("Starting municipal-level climate forecast generation...")
    
    # 1. Load Geocode mapping to identify valid municipalities (excluding ES)
    print("Loading geocode mappings...")
    df_map = pd.read_csv('data/data_imdc_2026/map_regional_health.csv')
    valid_geocodes = set(df_map[df_map['uf'] != 'ES']['geocode'].tolist())
    print(f"Loaded {len(valid_geocodes)} valid geocodes (excluding ES).")
    
    # Define columns to process
    climate_vars = [
        'temp_min', 'temp_med', 'temp_max', 
        'precip_min', 'precip_med', 'precip_max', 
        'rel_humid_min', 'rel_humid_med', 'rel_humid_max', 
        'thermal_range', 'rainy_days'
    ]
    
    output_dir = 'climate_projections'
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Read raw climate.csv in chunks and load observed records
    raw_climate_path = 'data/data_imdc_2026/climate.csv.gz'
    if not os.path.exists(raw_climate_path):
        raise FileNotFoundError(f"Raw climate file not found at {raw_climate_path}")
        
    print("Loading raw municipal climate records from climate.csv...")
    chunks = []
    chunk_size = 500000
    
    for chunk in pd.read_csv(raw_climate_path, chunksize=chunk_size):
        # Filter valid geocodes
        chunk = chunk[chunk['geocode'].isin(valid_geocodes)].copy()
        if len(chunk) == 0:
            continue
            
        # Keep only required columns
        keep_cols = ['geocode', 'date'] + climate_vars
        chunks.append(chunk[keep_cols])
        
    print("Concatenating observed municipal records...")
    df_muni_observed = pd.concat(chunks, ignore_index=True)
    df_muni_observed['date_dt'] = pd.to_datetime(df_muni_observed['date'])
    print(f"Loaded {len(df_muni_observed)} observed rows.")
    
    # Define periods: cutoffs and target date ranges
    periods = [
        {
            'name': 'round_1',
            'cutoff': '2022-06-19',
            'target_start': '2022-06-26',
            'target_end': '2023-10-01'
        },
        {
            'name': 'round_2',
            'cutoff': '2023-06-18',
            'target_start': '2023-06-25',
            'target_end': '2024-09-29'
        },
        {
            'name': 'round_3',
            'cutoff': '2024-06-16',
            'target_start': '2024-06-23',
            'target_end': '2025-09-28'
        },
        {
            'name': 'round_4',
            'cutoff': '2025-06-15',
            'target_start': '2025-06-22',
            'target_end': '2026-03-08'
        },
        {
            'name': 'round_5_forecast_2026_2027',
            'cutoff': '2026-03-08',
            'target_start': '2026-03-15',
            'target_end': '2027-10-03'
        }
    ]
    
    all_geocodes = sorted(list(valid_geocodes))
    forecast_dfs = []
    
    # 3. Generate forecasts causally for each period using vectorized pandas
    for p in periods:
        name = p['name']
        cutoff = p['cutoff']
        target_start_str = p['target_start']
        target_end_str = p['target_end']
        
        print(f"\nProcessing {name} (cutoff: {cutoff}, forecast: {target_start_str} to {target_end_str})...")
        
        # Filter training data to cutoff
        df_train = df_muni_observed[df_muni_observed['date'] <= cutoff].copy()
        df_train['week'] = df_train['date_dt'].dt.isocalendar().week.astype(int)
        
        # Calculate weekly climatological normals per geocode
        print("  Computing climatological normals...")
        df_normals = df_train.groupby(['geocode', 'week'])[climate_vars].mean().reset_index()
        
        # Generate target forecast dates range
        d_start = datetime.datetime.strptime(target_start_str, '%Y-%m-%d').date()
        d_end = datetime.datetime.strptime(target_end_str, '%Y-%m-%d').date()
        
        dates = []
        curr = d_start
        while curr <= d_end:
            dates.append(curr.strftime('%Y-%m-%d'))
            curr += datetime.timedelta(days=7)
            
        print(f"  Building target grid for {len(all_geocodes)} geocodes and {len(dates)} weeks...")
        # Create MultiIndex grid
        df_grid = pd.MultiIndex.from_product([all_geocodes, dates], names=['geocode', 'date']).to_frame().reset_index(drop=True)
        df_grid['dt'] = pd.to_datetime(df_grid['date'])
        df_grid['week'] = df_grid['dt'].dt.isocalendar().week.astype(int)
        
        # Set up lookup week (fallback week 53 to week 52)
        df_grid['lookup_week'] = df_grid['week'].apply(lambda w: 52 if w == 53 else w)
        
        # Merge grid with normals
        df_normals_lookup = df_normals.rename(columns={'week': 'lookup_week'})
        df_grid = pd.merge(df_grid, df_normals_lookup, on=['geocode', 'lookup_week'], how='left')
        
        # Fill any missing values using forward/backward fill within each geocode group
        df_grid[climate_vars] = df_grid.groupby('geocode')[climate_vars].ffill().bfill()
        
        # Set round name and reorder columns
        df_grid['round'] = name
        final_cols = ['round', 'geocode', 'date', 'week'] + climate_vars
        df_forecast = df_grid[final_cols].copy()
        
        print(f"  Generated forecast shape: {df_forecast.shape}")
        forecast_dfs.append(df_forecast)
        
    # 4. Concatenate and save to CSV
    print("\nConcatenating all validation rounds forecasts...")
    df_all_forecasts = pd.concat(forecast_dfs, ignore_index=True)
    
    out_path = os.path.join(output_dir, 'forecasting_climate_normals_weekly.csv')
    print(f"Saving final concatenated forecasts to {out_path}...")
    df_all_forecasts.to_csv(out_path, index=False)
    print(f"Done! Saved file shape: {df_all_forecasts.shape}")

if __name__ == '__main__':
    generate_municipal_forecasts()
