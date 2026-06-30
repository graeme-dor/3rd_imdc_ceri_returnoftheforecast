import os
import pandas as pd
import numpy as np

# List of 26 states (excluding ES)
STATES = sorted([
    'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 
    'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO'
])

def preprocess_gravity():
    print("Preprocessing gravity model network for Brazil...")
    
    # 1. State Capital Coordinates (latitude, longitude)
    coords = {
        'AC': (-9.974, -67.808), 'AL': (-9.666, -35.735), 'AM': (-3.100, -60.016),
        'AP': (0.034, -51.069), 'BA': (-12.971, -38.510), 'CE': (-3.731, -38.526),
        'DF': (-15.780, -47.930), 'GO': (-16.678, -49.253), 'MA': (-2.530, -44.302),
        'MG': (-19.920, -43.940), 'MS': (-20.442, -54.646), 'MT': (-15.601, -56.096),
        'PA': (-1.455, -48.503), 'PB': (-7.115, -34.863), 'PE': (-8.053, -34.881),
        'PI': (-5.089, -42.801), 'PR': (-25.427, -49.273), 'RJ': (-22.903, -43.209),
        'RN': (-5.795, -35.209), 'RO': (-8.761, -63.903), 'RR': (2.819, -60.673),
        'RS': (-30.034, -51.217), 'SC': (-27.596, -48.549), 'SE': (-10.911, -37.073),
        'SP': (-23.548, -46.636), 'TO': (-10.167, -48.331)
    }
    
    # 2. Load state population data from features
    features_path = 'data/processed/state_weekly_features.csv'
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"State features not found at {features_path}. Please run src/preprocess_data.py first.")
        
    df_features = pd.read_csv(features_path)
    pop_dict = df_features.groupby('uf')['population'].mean().to_dict()
    
    # 3. Calculate pairwise Haversine distances (in km)
    num_states = len(STATES)
    dist = np.zeros((num_states, num_states))
    for i in range(num_states):
        uf_i = STATES[i]
        lat1, lon1 = coords[uf_i]
        for j in range(num_states):
            if i == j:
                dist[i, j] = 1e9  # Prevent self-loops
                continue
            uf_j = STATES[j]
            lat2, lon2 = coords[uf_j]
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            dist[i, j] = 2 * np.arcsin(np.sqrt(a)) * 6371.0
            
    # 4. Compute Gravity Forces
    G = np.zeros((num_states, num_states))
    for i in range(num_states):
        uf_i = STATES[i]
        p_i = pop_dict.get(uf_i, 1.0e6)
        for j in range(num_states):
            if i == j:
                G[i, j] = 0.0
                continue
            uf_j = STATES[j]
            p_j = pop_dict.get(uf_j, 1.0e6)
            G[i, j] = (p_i * p_j) / (dist[i, j]**2)
            
    # Convert to DataFrame
    df_gravity = pd.DataFrame(G, index=STATES, columns=STATES)
    
    # 5. Row-normalize: inflow proportions sum to 1.0
    row_sums = df_gravity.sum(axis=1)
    df_gravity_norm = df_gravity.div(row_sums, axis=0).fillna(0.0)
    
    # Save the output
    output_path = 'data/processed/state_gravity_matrix.csv'
    df_gravity_norm.to_csv(output_path)
    print(f"Saved normalized state-to-state gravity matrix to {output_path} (shape: {df_gravity_norm.shape})")

if __name__ == '__main__':
    preprocess_gravity()
