import os
import pandas as pd
import numpy as np
import datetime

def generate_delta_adjusted():
    print("Starting weekly delta-adjusted forecast with normals fallback generation...")
    
    # 1. Load Geocode mapping (excluding ES)
    df_map = pd.read_csv('data/data_imdc_2026/map_regional_health.csv')
    valid_geocodes = set(df_map[df_map['uf'] != 'ES']['geocode'].tolist())
    all_geocodes = sorted(list(valid_geocodes))
    
    # Define columns to process (all 11 standard variables)
    climate_vars = [
        'temp_min', 'temp_med', 'temp_max', 
        'precip_min', 'precip_med', 'precip_max', 
        'rel_humid_min', 'rel_humid_med', 'rel_humid_max', 
        'thermal_range', 'rainy_days'
    ]
    
    # Load raw observed climate.csv for normals calculation
    print("Loading raw observed climate.csv...")
    raw_climate_path = 'data/data_imdc_2026/climate.csv.gz'
    chunks = []
    for chunk in pd.read_csv(raw_climate_path, chunksize=500000):
        chunk = chunk[chunk['geocode'].isin(valid_geocodes)].copy()
        if len(chunk) == 0:
            continue
        chunks.append(chunk[['geocode', 'date'] + climate_vars])
        
    df_obs = pd.concat(chunks, ignore_index=True)
    df_obs['date_dt'] = pd.to_datetime(df_obs['date'])
    
    # Load raw forecasting_climate.csv
    print("Loading Copernicus monthly forecasts...")
    df_fc = pd.read_csv('data/data_imdc_2026/forecasting_climate.csv.gz')
    df_fc['precip_tot'] = df_fc['precip_tot'] * 280180.0  # Convert kg/m2/s (mm/s) to mm/week median scale
    df_fc['reference_month_dt'] = pd.to_datetime(df_fc['reference_month'], format='mixed')
    df_fc['reference_month'] = df_fc['reference_month_dt'].dt.strftime('%Y-%m-%d')
    df_fc = df_fc[df_fc['geocode'].isin(valid_geocodes)].copy()
    
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
    
    # Helpers for date math
    def add_months(sourcedate, months):
        month = sourcedate.month - 1 + months
        year = sourcedate.year + month // 12
        month = month % 12 + 1
        return year, month
        
    delta_dfs = []
    
    for p in periods:
        name = p['name']
        cutoff = p['cutoff']
        ref_month_str = p['ref_month']
        target_start = p['target_start']
        target_end = p['target_end']
        fc_end = p['fc_end_date']
        
        print(f"\nProcessing {name} (cutoff: {cutoff}, target end: {target_end})...")
        
        # Split observed training history to cutoff
        df_train = df_obs[df_obs['date'] <= cutoff].copy()
        df_train['week'] = df_train['date_dt'].dt.isocalendar().week.astype(int)
        df_train['year'] = df_train['date_dt'].dt.year
        df_train['month'] = df_train['date_dt'].dt.month
        
        # A. Calculate climatological weekly normals per geocode
        print("  Computing weekly climatological normals...")
        df_weekly_normals = df_train.groupby(['geocode', 'week'])[climate_vars].mean().reset_index()
        
        # B. Calculate historical monthly normals per geocode and month
        print("  Computing historical monthly observed normals...")
        df_monthly_obs = df_train.groupby(['geocode', 'year', 'month'])[climate_vars].mean().reset_index()
        df_monthly_normals = df_monthly_obs.groupby(['geocode', 'month'])[climate_vars].mean().reset_index()
        
        # C. Retrieve forecasts for this reference month
        df_fc_round = df_fc[df_fc['reference_month'] == ref_month_str].copy()
        
        # Calculate target calendar month for each forecast lead time
        ref_dt = pd.to_datetime(ref_month_str)
        lead_months = sorted(df_fc_round['forecast_months_ahead'].unique().tolist())
        
        lead_to_month_map = {}
        for lead in lead_months:
            ty, tm = add_months(ref_dt, lead)
            lead_to_month_map[lead] = tm
            
        df_fc_round['month'] = df_fc_round['forecast_months_ahead'].map(lead_to_month_map)
        
        # D. Merge forecast with monthly observed normals to compute anomaly deltas
        print("  Computing forecast deltas...")
        # Explicitly suffix all monthly normal variables to avoid merge suffix gotchas
        rename_dict_monthly = {var: f"{var}_normal" for var in climate_vars}
        df_monthly_normals_lookup = df_monthly_normals.rename(columns=rename_dict_monthly)
        df_fc_round = pd.merge(df_fc_round, df_monthly_normals_lookup, on=['geocode', 'month'], how='left')
        
        df_fc_round['delta_temp'] = df_fc_round['temp_med'] - df_fc_round['temp_med_normal']
        df_fc_round['delta_humid'] = df_fc_round['umid_med'] - df_fc_round['rel_humid_med_normal']
        # Compute multiplicative ratio for precipitation with a 1.0 mm normal floor and a [0.0, 3.0] clipping range
        df_fc_round['ratio_precip'] = df_fc_round['precip_tot'] / df_fc_round['precip_med_normal'].clip(lower=1.0)
        df_fc_round['ratio_precip'] = df_fc_round['ratio_precip'].clip(lower=0.0, upper=3.0)
        
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
        df_grid['month'] = df_grid['dt'].dt.month
        
        # E. Merge grid with weekly normals
        df_grid['lookup_week'] = df_grid['week'].apply(lambda w: 52 if w == 53 else w)
        df_weekly_normals_lookup = df_weekly_normals.rename(columns={'week': 'lookup_week'})
        df_grid = pd.merge(df_grid, df_weekly_normals_lookup, on=['geocode', 'lookup_week'], how='left')
        
        # F. Merge grid with forecast deltas and ratios
        df_grid = pd.merge(df_grid, df_fc_round[['geocode', 'month', 'delta_temp', 'delta_humid', 'ratio_precip']], on=['geocode', 'month'], how='left')
        
        # G. Apply delta adjustment inside forecast window, keeping standard normals outside
        # Mask for dates inside the 6-month forecast window (dates <= fc_end) but excluding the reference month itself
        fc_window_mask = (df_grid['date'] <= fc_end) & ~((df_grid['dt'].dt.year == ref_dt.year) & (df_grid['dt'].dt.month == ref_dt.month))
        
        # Temperature adjustments: Shift min, med, max by delta_temp
        for var in ['temp_min', 'temp_med', 'temp_max']:
            df_grid[f'{var}_adj'] = df_grid[var]
            df_grid.loc[fc_window_mask, f'{var}_adj'] = df_grid.loc[fc_window_mask, var] + df_grid.loc[fc_window_mask, 'delta_temp']
            
        # Humidity adjustments: Shift min, med, max by delta_humid (clip to [0.0, 100.0])
        for var in ['rel_humid_min', 'rel_humid_med', 'rel_humid_max']:
            df_grid[f'{var}_adj'] = df_grid[var]
            df_grid.loc[fc_window_mask, f'{var}_adj'] = np.clip(
                df_grid.loc[fc_window_mask, var] + df_grid.loc[fc_window_mask, 'delta_humid'],
                0.0, 100.0
            )
            
        # Precipitation adjustments: Scale min, med, max multiplicatively by ratio_precip
        for var in ['precip_min', 'precip_med', 'precip_max']:
            df_grid[f'{var}_adj'] = df_grid[var]
            df_grid.loc[fc_window_mask, f'{var}_adj'] = df_grid.loc[fc_window_mask, var] * df_grid.loc[fc_window_mask, 'ratio_precip']
            
        # Thermal range and rainy days adjustments: Keep weekly climatological normals
        for var in ['thermal_range', 'rainy_days']:
            df_grid[f'{var}_adj'] = df_grid[var]
            
        # Fill any remaining NaNs (for geocodes absent from climate.csv) using neighboring fill
        adj_cols = [f'{var}_adj' for var in climate_vars]
        for col in adj_cols:
            df_grid[col] = df_grid.groupby('geocode')[col].ffill().bfill()
            
        # Select and rename final adjusted columns
        df_grid['round'] = name
        df_grid = df_grid.drop(columns=climate_vars)
        rename_dict = {f'{var}_adj': var for var in climate_vars}
        df_grid = df_grid.rename(columns=rename_dict)
        
        final_cols = ['round', 'geocode', 'date', 'week'] + climate_vars
        df_delta_round = df_grid[final_cols].copy()
        
        print(f"  Generated delta-adjusted forecast shape: {df_delta_round.shape}")
        delta_dfs.append(df_delta_round)
        
    df_all_delta = pd.concat(delta_dfs, ignore_index=True)
    out_path = 'climate_projections/forecasting_climate_delta_adjusted_weekly.csv'
    df_all_delta.to_csv(out_path, index=False)
    print(f"\nSuccessfully completed Scenario C! Saved to {out_path} (shape: {df_all_delta.shape})")

if __name__ == '__main__':
    generate_delta_adjusted()
