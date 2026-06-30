import os
import pandas as pd
import numpy as np
import datetime

def disaggregate_forecasts_causal():
    print("Starting causal monthly-to-weekly climate forecasts disaggregation...")
    
    # 1. Load Geocode mapping to identify valid municipalities (excluding ES)
    print("Loading geocode mappings...")
    df_map = pd.read_csv('data/data_imdc_2026/map_regional_health.csv')
    valid_geocodes = set(df_map[df_map['uf'] != 'ES']['geocode'].tolist())
    print(f"Loaded {len(valid_geocodes)} valid geocodes (excluding ES).")
    
    # 2. Load raw forecasting_climate.csv
    raw_fc_path = 'data/data_imdc_2026/forecasting_climate.csv.gz'
    if not os.path.exists(raw_fc_path):
        raise FileNotFoundError(f"Raw forecasting climate file not found at {raw_fc_path}")
        
    print("Loading monthly forecasts from forecasting_climate.csv...")
    df_fc = pd.read_csv(raw_fc_path)
    
    # Parse dates to clean format YYYY-MM-DD
    df_fc['reference_month_dt'] = pd.to_datetime(df_fc['reference_month'], format='mixed')
    df_fc['reference_month'] = df_fc['reference_month_dt'].dt.strftime('%Y-%m-%d')
    
    # Filter valid geocodes
    df_fc = df_fc[df_fc['geocode'].isin(valid_geocodes)].copy()
    
    # Define round cutoffs and reference month mapping
    periods = [
        {'name': 'round_1', 'ref_month': '2022-06-01'},
        {'name': 'round_2', 'ref_month': '2023-06-01'},
        {'name': 'round_3', 'ref_month': '2024-06-01'},
        {'name': 'round_4', 'ref_month': '2025-06-01'}
    ]
    
    # Helpers
    def add_months(sourcedate, months):
        month = sourcedate.month - 1 + months
        year = sourcedate.year + month // 12
        month = month % 12 + 1
        return year, month
        
    def get_sundays_in_month(year, month):
        d = datetime.date(year, month, 1)
        while d.weekday() != 6: # 6 is Sunday
            d += datetime.timedelta(days=1)
        sundays = []
        while d.month == month:
            sundays.append(d.strftime('%Y-%m-%d'))
            d += datetime.timedelta(days=7)
        return sundays
        
    # Process each round
    forecast_dfs = []
    
    for p in periods:
        name = p['name']
        ref_month_str = p['ref_month']
        
        print(f"\nProcessing {name} (reference month: {ref_month_str})...")
        
        # Filter raw forecasts for this specific reference month
        df_sub = df_fc[df_fc['reference_month'] == ref_month_str].copy()
        if len(df_sub) == 0:
            print(f"  Warning: No forecast data found for reference month {ref_month_str} in raw file.")
            continue
            
        # Get unique lead times (should be 1 to 6)
        leads = sorted(df_sub['forecast_months_ahead'].unique().tolist())
        
        # Build date mapping for this reference month
        mapping_rows = []
        ref_dt = pd.to_datetime(ref_month_str)
        for lead in leads:
            target_y, target_m = add_months(ref_dt, lead)
            sundays = get_sundays_in_month(target_y, target_m)
            
            for sun in sundays:
                sun_dt = pd.to_datetime(sun)
                w = int(sun_dt.isocalendar().week)
                mapping_rows.append({
                    'reference_month': ref_month_str,
                    'forecast_months_ahead': lead,
                    'date': sun,
                    'week': w
                })
                
        df_date_map = pd.DataFrame(mapping_rows)
        
        # Expand monthly forecasts to weekly Sundays
        df_weekly_sub = pd.merge(df_sub, df_date_map, on=['reference_month', 'forecast_months_ahead'], how='inner')
        
        # Add round metadata
        df_weekly_sub['round'] = name
        forecast_dfs.append(df_weekly_sub)
        print(f"  Generated forecasts shape: {df_weekly_sub.shape}")
        
    # Concatenate all periods
    print("\nConcatenating all validation rounds forecasts...")
    df_all_forecasts = pd.concat(forecast_dfs, ignore_index=True)
    
    # Reorder columns
    final_cols = ['round', 'reference_month', 'forecast_months_ahead', 'geocode', 'date', 'week', 'temp_med', 'umid_med', 'precip_tot']
    df_all_forecasts = df_all_forecasts[final_cols].copy()
    
    # Sort for consistency
    df_all_forecasts = df_all_forecasts.sort_values(['round', 'reference_month', 'forecast_months_ahead', 'geocode', 'date']).reset_index(drop=True)
    
    # Output to CSV
    output_path = 'climate_projections/forecasting_climate_weekly.csv'
    print(f"Saving causal weekly forecasts to {output_path}...")
    df_all_forecasts.to_csv(output_path, index=False)
    print(f"Successfully completed! Final dataset shape: {df_all_forecasts.shape}")

if __name__ == '__main__':
    disaggregate_forecasts_causal()
