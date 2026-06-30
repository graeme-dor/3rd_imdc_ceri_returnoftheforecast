import os
import pandas as pd
import numpy as np
import datetime

def generate_step_fallback():
    print("Starting weekly step forecast with normals fallback generation...")
    
    # 1. Load Geocode mapping (excluding ES)
    df_map = pd.read_csv('data/data_imdc_2026/map_regional_health.csv')
    valid_geocodes = set(df_map[df_map['uf'] != 'ES']['geocode'].tolist())
    all_geocodes = sorted(list(valid_geocodes))
    
    # Define columns in forecast and columns to calculate
    fc_vars = ['temp_med', 'umid_med', 'precip_tot']
    climate_vars = ['temp_med', 'umid_med', 'precip_tot', 'temp_min', 'rainy_days']
    
    # Load raw observed climate.csv for normals calculation
    print("Loading raw observed climate.csv...")
    raw_climate_path = 'data/data_imdc_2026/climate.csv.gz'
    chunks = []
    for chunk in pd.read_csv(raw_climate_path, chunksize=500000):
        chunk = chunk[chunk['geocode'].isin(valid_geocodes)].copy()
        if len(chunk) == 0:
            continue
        chunks.append(chunk[['geocode', 'date', 'temp_med', 'rel_humid_med', 'precip_med', 'temp_min', 'rainy_days']])
        
    df_obs = pd.concat(chunks, ignore_index=True)
    df_obs['date_dt'] = pd.to_datetime(df_obs['date'])
    # Map observed column names to forecast names for consistency
    df_obs = df_obs.rename(columns={'rel_humid_med': 'umid_med', 'precip_med': 'precip_tot'})
    
    # Load weekly disaggregated Copernicus forecasts
    print("Loading weekly Copernicus forecasts...")
    df_weekly_fc = pd.read_csv('climate_projections/forecasting_climate_weekly.csv')
    
    # Define validation and prediction periods
    periods = [
        {
            'name': 'round_1',
            'cutoff': '2022-06-19',
            'ref_month': '2022-06-01',
            'target_start': '2022-06-26',
            'target_end': '2023-10-01',
            'fc_end_date': '2022-12-31'  # 6 months after June 2022 reference month
        },
        {
            'name': 'round_2',
            'cutoff': '2023-06-18',
            'ref_month': '2023-06-01',
            'target_start': '2023-06-25',
            'target_end': '2024-09-29',
            'fc_end_date': '2023-12-31'  # 6 months after June 2023 reference month
        },
        {
            'name': 'round_3',
            'cutoff': '2024-06-16',
            'ref_month': '2024-06-01',
            'target_start': '2024-06-23',
            'target_end': '2025-09-28',
            'fc_end_date': '2024-12-31'  # 6 months after June 2024 reference month
        },
        {
            'name': 'round_4',
            'cutoff': '2025-06-15',
            'ref_month': '2025-06-01',
            'target_start': '2025-06-22',
            'target_end': '2026-03-08',
            'fc_end_date': '2025-12-31'  # 6 months after June 2025 reference month
        },
        {
            'name': 'round_5_forecast_2026_2027',
            'cutoff': '2026-03-08',
            'ref_month': '2026-03-01',
            'target_start': '2026-03-15',
            'target_end': '2027-10-03',
            'fc_end_date': '2026-09-30'  # 6 months after March 2026 reference month
        }
    ]
    
    step_dfs = []
    
    for p in periods:
        name = p['name']
        cutoff = p['cutoff']
        ref_month_str = p['ref_month']
        target_start = p['target_start']
        target_end = p['target_end']
        fc_end = p['fc_end_date']
        
        print(f"\nProcessing {name} (cutoff: {cutoff}, target end: {target_end})...")
        
        # Calculate weekly climatological normals up to cutoff
        df_train = df_obs[df_obs['date'] <= cutoff].copy()
        df_train['week'] = df_train['date_dt'].dt.isocalendar().week.astype(int)
        df_normals = df_train.groupby(['geocode', 'week'])[climate_vars].mean().reset_index()
        
        # Generate target dates
        d_start = datetime.datetime.strptime(target_start, '%Y-%m-%d').date()
        d_end = datetime.datetime.strptime(target_end, '%Y-%m-%d').date()
        
        dates = []
        curr = d_start
        while curr <= d_end:
            dates.append(curr.strftime('%Y-%m-%d'))
            curr += datetime.timedelta(days=7)
            
        # Build MultiIndex grid
        df_grid = pd.MultiIndex.from_product([all_geocodes, dates], names=['geocode', 'date']).to_frame().reset_index(drop=True)
        df_grid['dt'] = pd.to_datetime(df_grid['date'])
        df_grid['week'] = df_grid['dt'].dt.isocalendar().week.astype(int)
        
        # Load forecast for this round
        df_fc_round = df_weekly_fc[df_weekly_fc['round'] == name].copy()
        
        # Merge grid with forecast (only contains fc_vars)
        df_grid = pd.merge(df_grid, df_fc_round[['geocode', 'date'] + fc_vars], on=['geocode', 'date'], how='left')
        
        # Determine fallback rows (dates after the 6-month forecast window or during the reference month)
        ref_dt = pd.to_datetime(ref_month_str)
        fallback_mask = (df_grid['date'] > fc_end) | ((df_grid['dt'].dt.year == ref_dt.year) & (df_grid['dt'].dt.month == ref_dt.month))
        
        # Fetch weekly normals for fallback
        df_grid['lookup_week'] = df_grid['week'].apply(lambda w: 52 if w == 53 else w)
        df_normals_lookup = df_normals.rename(columns={'week': 'lookup_week'})
        # Explicitly suffix all normal variables to avoid merge suffix gotchas
        rename_dict = {var: f"{var}_normal" for var in climate_vars}
        df_normals_lookup = df_normals_lookup.rename(columns=rename_dict)
        df_grid = pd.merge(df_grid, df_normals_lookup, on=['geocode', 'lookup_week'], how='left')
        
        # Apply normals fallback to forecast variables
        for var in fc_vars:
            df_grid.loc[fallback_mask, var] = df_grid.loc[fallback_mask, f'{var}_normal']
            
        # Calculate and fallback temp_min and rainy_days
        # temp_min: Shift by step temp_med anomaly inside the forecast window, fallback to normal outside
        delta_temp = df_grid['temp_med'] - df_grid['temp_med_normal']
        df_grid['temp_min'] = df_grid['temp_min_normal'] + delta_temp
        df_grid.loc[fallback_mask, 'temp_min'] = df_grid.loc[fallback_mask, 'temp_min_normal']
        
        # rainy_days: Keep weekly climatological normal across all periods (no forecast available)
        df_grid['rainy_days'] = df_grid['rainy_days_normal']
        
        # Fill any remaining NaNs (for geocodes absent from climate.csv) using neighboring fill
        for var in climate_vars:
            df_grid[var] = df_grid.groupby('geocode')[var].ffill().bfill()
            
        # Reorder columns
        df_grid['round'] = name
        final_cols = ['round', 'geocode', 'date', 'week'] + climate_vars
        df_step_round = df_grid[final_cols].copy()
        
        print(f"  Generated step forecast shape: {df_step_round.shape}")
        step_dfs.append(df_step_round)
        
    df_all_step = pd.concat(step_dfs, ignore_index=True)
    out_path = 'climate_projections/forecasting_climate_step_fallback_weekly.csv'
    df_all_step.to_csv(out_path, index=False)
    print(f"\nSuccessfully completed Scenario B! Saved to {out_path} (shape: {df_all_step.shape})")

if __name__ == '__main__':
    generate_step_fallback()
