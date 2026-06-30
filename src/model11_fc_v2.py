#!/usr/bin/env python3
"""
Independent Model 11-FC pipeline.

Inputs:
  1) challenge data folder containing raw challenge files such as dengue.csv.gz,
     climate.csv.gz, datasus_population_2001_2025.csv.gz, environ_vars.csv.gz,
     enso.csv.gz, iod.csv.gz, pdo.csv.gz
  2) mobility adjacency matrix CSV
  3) forecast climate CSV, either state-level or municipality-level

Border adjacency is generated inside this script from a hard-coded Brazilian
state-neighbor map, then row-normalized. A border matrix CSV can still be
passed optionally to override the generated version.

Outputs:
  - state-week panel used by the model
  - forecast-climate adapted split tensors diagnostics
  - Model 10A-FC border predictions
  - Model 10B-FC mobility predictions
  - Model 11-FC weighted ensemble predictions and summaries

"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# =============================================================================
# Configuration
# =============================================================================

UF_CODE_TO_UF = {
    11: "RO",
    12: "AC",
    13: "AM",
    14: "RR",
    15: "PA",
    16: "AP",
    17: "TO",
    21: "MA",
    22: "PI",
    23: "CE",
    24: "RN",
    25: "PB",
    26: "PE",
    27: "AL",
    28: "SE",
    29: "BA",
    31: "MG",
    32: "ES",
    33: "RJ",
    35: "SP",
    41: "PR",
    42: "SC",
    43: "RS",
    50: "MS",
    51: "MT",
    52: "GO",
    53: "DF",
}
UF_TO_UF_CODE = {v: k for k, v in UF_CODE_TO_UF.items()}

SPLIT_YEARS = {1: 2022, 2: 2023, 3: 2024, 4: 2025}
SPLIT_CUTOFF_DATES = {
    1: pd.Timestamp("2022-06-19"),
    2: pd.Timestamp("2023-06-18"),
    3: pd.Timestamp("2024-06-16"),
    4: pd.Timestamp("2025-06-15"),
}
SPLIT_FORECAST_ROUNDS = {
    1: "round_1",
    2: "round_2",
    3: "round_3",
    4: "round_4",
}

QUANTILES = np.array(
    [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975], dtype=np.float32
)
Q_COLS = [f"q_{q:g}" for q in QUANTILES]

RAW_CLIMATE_COLS = [
    "temp_min",
    "temp_med",
    "temp_max",
    "precip_min",
    "precip_med",
    "precip_max",
    "rel_humid_min",
    "rel_humid_med",
    "rel_humid_max",
    "thermal_range",
    "rainy_days",
]

CLIMATE_LAGS = {
    "temp_min": 11,
    "temp_med": 12,
    "rainy_days": 9,
    "rel_humid_med": 4,
    "rel_humid_min": 4,
    "thermal_range": 4,
    "rel_humid_max": 5,
    "precip_max": 6,
    "precip_med": 6,
    "temp_max": 12,
    "precip_min": 6,
}

OCEAN_LAGS = {
    "iod": 12,
    "enso": 1,
    "pdo": 7,
}

# Undirected state-border graph used to build Model 10A-FC internally.
# The final matrix is filtered to the states present in the mobility matrix
# and row-normalized. Diagonal entries are zero.
BORDER_NEIGHBORS = {
    "AC": ["AM", "RO"],
    "AL": ["BA", "PE", "SE"],
    "AM": ["AC", "RO", "RR", "PA", "MT"],
    "AP": ["PA"],
    "BA": ["AL", "SE", "PE", "PI", "TO", "GO", "MG", "ES"],
    "CE": ["RN", "PB", "PE", "PI"],
    "DF": ["GO"],
    "ES": ["BA", "MG", "RJ"],
    "GO": ["MT", "MS", "MG", "BA", "TO", "DF"],
    "MA": ["PA", "TO", "PI"],
    "MG": ["BA", "ES", "RJ", "SP", "MS", "GO"],
    "MS": ["MT", "GO", "MG", "SP", "PR"],
    "MT": ["AM", "PA", "TO", "GO", "MS", "RO"],
    "PA": ["AP", "RR", "AM", "MT", "TO", "MA"],
    "PB": ["RN", "CE", "PE"],
    "PE": ["CE", "PB", "AL", "BA", "PI"],
    "PI": ["MA", "CE", "PE", "BA", "TO"],
    "PR": ["SP", "MS", "SC"],
    "RJ": ["ES", "MG", "SP"],
    "RN": ["CE", "PB"],
    "RO": ["AC", "AM", "MT"],
    "RR": ["AM", "PA"],
    "RS": ["SC"],
    "SC": ["PR", "RS"],
    "SE": ["AL", "BA"],
    "SP": ["RJ", "MG", "MS", "PR"],
    "TO": ["PA", "MA", "PI", "BA", "GO", "MT"],
}

FEATURE_COLS = [
    "log_cases",
    "log_incidence_100k",
    "temp_min_lag11",
    "temp_med_lag12",
    "rainy_days_lag9",
    "rel_humid_med_lag4",
    "rel_humid_min_lag4",
    "thermal_range_lag4",
    "rel_humid_max_lag5",
    "precip_max_lag6",
    "precip_med_lag6",
    "temp_max_lag12",
    "precip_min_lag6",
    "iod_lag12",
    "enso_lag1",
    "pdo_lag7",
    "population",
    "week_sin",
    "week_cos",
    "koppen_Af",
    "koppen_Am",
    "koppen_As",
    "koppen_Aw",
    "koppen_BSh",
    "koppen_Cfa",
    "koppen_Cfb",
    "koppen_Cwa",
    "koppen_Cwb",
    "biome_Amazônia",
    "biome_Caatinga",
    "biome_Cerrado",
    "biome_Mata Atlântica",
    "biome_Pampa",
    "biome_Pantanal",
]

Y_COL = "y_log_cases"
HISTORY_LEN = 104
BEST_MODEL11_FC_WEIGHT_BORDER = 0.53
BEST_MODEL11_FC_WEIGHT_MOBILITY = 0.47

TRAIN_CFG = {
    "gcn_hidden": 32,
    "lstm_hidden": 96,
    "lstm_layers": 1,
    "dropout": 0.15,
    "head_hidden": 64,
    "batch_size": 4,
    "epochs": 600,
    "lr": 7e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "patience": 100,
    "crossing_penalty_weight": 0.02,
}


# =============================================================================
# Utilities
# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_file(
    data_dir: Path, names: Iterable[str], required: bool = True
) -> Optional[Path]:
    for name in names:
        p = data_dir / name
        if p.exists():
            return p
    for name in names:
        hits = list(data_dir.rglob(name))
        if hits:
            return sorted(hits, key=lambda x: len(str(x)))[0]
    if required:
        raise FileNotFoundError(
            f"Could not find any of: {list(names)} inside {data_dir}"
        )
    return None


def read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    log(f"Reading {path}")
    return pd.read_csv(path, **kwargs)


def to_uf_code_from_geocode(geocode: pd.Series) -> pd.Series:
    return pd.to_numeric(geocode, errors="coerce").astype("Int64") // 100000


def add_uf_from_code(df: pd.DataFrame, code_col: str = "uf_code") -> pd.DataFrame:
    out = df.copy()
    out[code_col] = pd.to_numeric(out[code_col], errors="coerce").astype("Int64")
    out["uf"] = out[code_col].map(UF_CODE_TO_UF)
    return out


def normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        return out
    if "data" in out.columns:
        out = out.rename(columns={"data": "date"})
        out["date"] = pd.to_datetime(out["date"])
        return out
    raise ValueError("Dataframe needs a date column")


def ew25(year: int) -> int:
    return year * 100 + 25


def season_mask(epiweek: pd.Series, year: int) -> pd.Series:
    return (epiweek >= year * 100 + 41) & (epiweek <= (year + 1) * 100 + 40)


def ensure_week_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "week" not in out.columns:
        if "epiweek" not in out.columns:
            raise ValueError("Need epiweek to create week")
        out["week"] = pd.to_numeric(out["epiweek"], errors="coerce").astype(int) % 100
    return out


# =============================================================================
# Raw data panel builder
# =============================================================================


def load_state_order_from_matrix(matrix_csv: Path) -> List[str]:
    df = pd.read_csv(matrix_csv)
    # Matrix is usually 26x27 with first label column.
    first_col = df.columns[0]
    labels = df[first_col].astype(str).tolist()
    # If first column is numeric UF code, map to UF abbreviations.
    mapped = []
    for x in labels:
        xs = x.strip()
        if xs.isdigit() and int(xs) in UF_CODE_TO_UF:
            mapped.append(UF_CODE_TO_UF[int(xs)])
        else:
            mapped.append(xs)
    return mapped


def load_dengue_state(data_dir: Path, state_order: List[str]) -> pd.DataFrame:
    path = find_file(data_dir, ["dengue.csv.gz", "dengue.csv"])
    df = read_csv_any(path, parse_dates=["date"])

    required = ["date", "epiweek", "casos"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"dengue file missing required column: {c}")

    if "uf" not in df.columns:
        if "uf_code" not in df.columns:
            if "geocode" not in df.columns:
                raise ValueError("dengue file needs uf, uf_code, or geocode")
            df["uf_code"] = to_uf_code_from_geocode(df["geocode"])
        df = add_uf_from_code(df, "uf_code")

    if "uf_code" not in df.columns:
        df["uf_code"] = df["uf"].map(UF_TO_UF_CODE)

    df["uf"] = df["uf"].astype(str)
    df = df[df["uf"].isin(state_order)].copy()

    out = df.groupby(["date", "epiweek", "uf", "uf_code"], as_index=False)[
        "casos"
    ].sum()

    # Complete weekly x state grid from available dengue dates.
    dates = out[["date", "epiweek"]].drop_duplicates().sort_values("epiweek")
    states = pd.DataFrame({"uf": state_order})
    states["uf_code"] = states["uf"].map(UF_TO_UF_CODE).astype(int)
    grid = dates.assign(_k=1).merge(states.assign(_k=1), on="_k").drop(columns="_k")
    out = grid.merge(out, on=["date", "epiweek", "uf", "uf_code"], how="left")
    out["casos"] = out["casos"].fillna(0).astype(float)
    out["year"] = out["epiweek"].astype(int) // 100
    out["week"] = out["epiweek"].astype(int) % 100
    return out.sort_values(["epiweek", "uf"]).reset_index(drop=True)


def load_climate_state(data_dir: Path, state_order: List[str]) -> pd.DataFrame:
    path = find_file(data_dir, ["climate.csv.gz", "climate.csv"])
    df = read_csv_any(path, parse_dates=["date"])

    if "uf" not in df.columns:
        if "uf_code" not in df.columns:
            if "geocode" not in df.columns:
                raise ValueError("climate file needs uf, uf_code, or geocode")
            df["uf_code"] = to_uf_code_from_geocode(df["geocode"])
        df = add_uf_from_code(df, "uf_code")

    if "epiweek" not in df.columns:
        raise ValueError("climate file needs epiweek")

    df["uf"] = df["uf"].astype(str)
    df = df[df["uf"].isin(state_order)].copy()

    # Handle common aliases.
    aliases = {
        "umid_min": "rel_humid_min",
        "umid_med": "rel_humid_med",
        "umid_max": "rel_humid_max",
        "humidity_min": "rel_humid_min",
        "humidity_med": "rel_humid_med",
        "humidity_max": "rel_humid_max",
        "precip_tot": "precip_med",
        "rain_days": "rainy_days",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "thermal_range" not in df.columns and {"temp_max", "temp_min"}.issubset(
        df.columns
    ):
        df["thermal_range"] = pd.to_numeric(
            df["temp_max"], errors="coerce"
        ) - pd.to_numeric(df["temp_min"], errors="coerce")

    missing = [c for c in RAW_CLIMATE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"climate file missing required climate columns: {missing}")

    for c in RAW_CLIMATE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df.groupby(["date", "epiweek", "uf"], as_index=False)[RAW_CLIMATE_COLS].mean()
    return out


def load_population_state(data_dir: Path, state_order: List[str]) -> pd.DataFrame:
    path = find_file(
        data_dir,
        [
            "datasus_population_2001_2025.csv.gz",
            "datasus_population_2001_2025.csv",
            "population.csv.gz",
            "population.csv",
        ],
    )
    df = read_csv_any(path)

    if "year" not in df.columns:
        for c in ["ano", "Year"]:
            if c in df.columns:
                df = df.rename(columns={c: "year"})
                break
    if "year" not in df.columns:
        raise ValueError("population file needs year column")

    pop_col = None
    for c in ["population", "pop", "populacao", "população", "Pop", "POP"]:
        if c in df.columns:
            pop_col = c
            break
    if pop_col is None:
        # Last resort: choose first numeric non-key column.
        candidates = [
            c for c in df.columns if c not in ["year", "geocode", "uf", "uf_code"]
        ]
        for c in candidates:
            if pd.api.types.is_numeric_dtype(df[c]):
                pop_col = c
                break
    if pop_col is None:
        raise ValueError("Could not identify population column")

    if "uf" not in df.columns:
        if "uf_code" not in df.columns:
            if "geocode" not in df.columns:
                raise ValueError("population file needs uf, uf_code, or geocode")
            df["uf_code"] = to_uf_code_from_geocode(df["geocode"])
        df = add_uf_from_code(df, "uf_code")

    df["uf"] = df["uf"].astype(str)
    df = df[df["uf"].isin(state_order)].copy()
    df[pop_col] = pd.to_numeric(df[pop_col], errors="coerce")

    out = df.groupby(["year", "uf"], as_index=False)[pop_col].sum()
    out = out.rename(columns={pop_col: "population"})

    # Extend 2026 with latest available population.
    max_year = int(out["year"].max())
    if max_year < 2026:
        latest = out[out["year"] == max_year].copy()
        latest["year"] = 2026
        out = pd.concat([out, latest], ignore_index=True)

    return out


def load_environment_static(data_dir: Path, state_order: List[str]) -> pd.DataFrame:
    path = find_file(
        data_dir,
        [
            "environ_vars.csv.gz",
            "environ_vars.csv",
            "environment.csv.gz",
            "environment.csv",
        ],
    )
    df = read_csv_any(path)

    if "uf" not in df.columns:
        if "uf_code" not in df.columns:
            if "geocode" not in df.columns:
                raise ValueError("environment file needs uf, uf_code, or geocode")
            df["uf_code"] = to_uf_code_from_geocode(df["geocode"])
        df = add_uf_from_code(df, "uf_code")

    df["uf"] = df["uf"].astype(str)
    df = df[df["uf"].isin(state_order)].copy()

    # If already one-hot columns are present, average them.
    one_hot_cols = [
        c for c in df.columns if c.startswith("koppen_") or c.startswith("biome_")
    ]

    pieces = []
    if one_hot_cols:
        tmp = df[["uf"] + one_hot_cols].copy()
        for c in one_hot_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        pieces.append(tmp.groupby("uf", as_index=False)[one_hot_cols].mean())

    # Otherwise create one-hot from categorical columns.
    cat_candidates = {"koppen": None, "biome": None}
    for c in df.columns:
        cl = c.lower()
        if (
            "koppen" in cl
            and cat_candidates["koppen"] is None
            and not c.startswith("koppen_")
        ):
            cat_candidates["koppen"] = c
        if (
            "biome" in cl
            and cat_candidates["biome"] is None
            and not c.startswith("biome_")
        ):
            cat_candidates["biome"] = c

    for prefix, col in cat_candidates.items():
        if col is not None:
            dummies = pd.get_dummies(df[col].astype(str), prefix=prefix)
            tmp = pd.concat([df[["uf"]], dummies], axis=1)
            pieces.append(
                tmp.groupby("uf", as_index=False)[dummies.columns.tolist()].mean()
            )

    if not pieces:
        raise ValueError(
            "Could not create environment static columns from environ_vars"
        )

    out = pd.DataFrame({"uf": state_order})
    for piece in pieces:
        out = out.merge(piece, on="uf", how="left")

    # Ensure all required static columns exist.
    static_required = [
        c for c in FEATURE_COLS if c.startswith("koppen_") or c.startswith("biome_")
    ]
    for c in static_required:
        if c not in out.columns:
            out[c] = 0.0
    out[static_required] = out[static_required].fillna(0.0)

    return out[["uf"] + static_required]


def load_ocean_index(data_dir: Path, name: str) -> pd.DataFrame:
    path = find_file(data_dir, [f"{name}.csv.gz", f"{name}.csv"])
    df = read_csv_any(path)
    df = normalize_date_col(df)
    # Pick numeric value column.
    possible = [name, "value", "index", "indice", "anom", "anomaly"]
    val_col = None
    for c in possible:
        if c in df.columns and c != "date":
            val_col = c
            break
    if val_col is None:
        for c in df.columns:
            if c != "date" and pd.api.types.is_numeric_dtype(df[c]):
                val_col = c
                break
    if val_col is None:
        raise ValueError(f"Could not identify value column for {name}")
    out = df[["date", val_col]].copy().rename(columns={val_col: name})
    out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def merge_ocean_indices(panel: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    out = panel.copy().sort_values("date")
    unique_dates = out[["date"]].drop_duplicates().sort_values("date")
    for name in ["enso", "iod", "pdo"]:
        idx = load_ocean_index(data_dir, name).sort_values("date")
        # Merge-asof handles monthly indices into weekly panel.
        merged_dates = pd.merge_asof(unique_dates, idx, on="date", direction="backward")
        out = out.merge(merged_dates, on="date", how="left")
        out[name] = out[name].ffill().bfill()
    return out


def build_base_panel(
    data_dir: Path, state_order: List[str], output_dir: Path
) -> pd.DataFrame:
    log("\nBuilding state-week panel from raw challenge folder")
    dengue = load_dengue_state(data_dir, state_order)
    climate = load_climate_state(data_dir, state_order)
    population = load_population_state(data_dir, state_order)
    env_static = load_environment_static(data_dir, state_order)

    panel = dengue.merge(climate, on=["date", "epiweek", "uf"], how="left")
    panel = panel.merge(population, on=["year", "uf"], how="left")
    panel = panel.merge(env_static, on="uf", how="left")
    panel = merge_ocean_indices(panel, data_dir)

    # Sort in matrix state order.
    state_rank = {uf: i for i, uf in enumerate(state_order)}
    panel["node_idx"] = panel["uf"].map(state_rank)
    panel = panel.sort_values(["epiweek", "node_idx"]).reset_index(drop=True)

    # Feature engineering.
    panel["population"] = pd.to_numeric(panel["population"], errors="coerce")
    panel["casos"] = pd.to_numeric(panel["casos"], errors="coerce").fillna(0.0)
    panel["incidence_100k"] = (
        panel["casos"] / panel["population"].replace(0, np.nan) * 100000.0
    )
    panel["incidence_100k"] = (
        panel["incidence_100k"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    panel["log_cases"] = np.log1p(panel["casos"])
    panel["log_incidence_100k"] = np.log1p(panel["incidence_100k"])
    panel[Y_COL] = panel["log_cases"]

    panel = ensure_week_col(panel)
    panel["week_sin"] = np.sin(2.0 * np.pi * panel["week"] / 53.0)
    panel["week_cos"] = np.cos(2.0 * np.pi * panel["week"] / 53.0)

    # Climate lags by state.
    panel = panel.sort_values(["uf", "date"]).reset_index(drop=True)
    for raw_col, lag in CLIMATE_LAGS.items():
        panel[f"{raw_col}_lag{lag}"] = panel.groupby("uf", sort=False)[raw_col].shift(
            lag
        )

    for raw_col, lag in OCEAN_LAGS.items():
        panel[f"{raw_col}_lag{lag}"] = panel.groupby("uf", sort=False)[raw_col].shift(
            lag
        )

    # Target columns.
    for split_id, year in SPLIT_YEARS.items():
        panel[f"target_{split_id}"] = season_mask(panel["epiweek"].astype(int), year)
        panel[f"available_until_ew25_{split_id}"] = panel["epiweek"].astype(
            int
        ) <= ew25(year)

    # Ensure required feature columns exist.
    missing = [c for c in FEATURE_COLS if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Panel missing required feature columns after construction: {missing}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "state_panel_model11_fc_built_from_raw.csv"
    panel.to_csv(panel_path, index=False)
    log(f"Saved built panel: {panel_path} shape={panel.shape}")
    return panel


# =============================================================================
# Forecast climate handling
# =============================================================================


def read_or_build_state_forecast_climate(
    forecast_csv: Path, state_order: List[str], output_dir: Path
) -> pd.DataFrame:
    fc = read_csv_any(forecast_csv, parse_dates=["date"])

    aliases = {
        "umid_min": "rel_humid_min",
        "umid_med": "rel_humid_med",
        "umid_max": "rel_humid_max",
        "humidity_min": "rel_humid_min",
        "humidity_med": "rel_humid_med",
        "humidity_max": "rel_humid_max",
        "precip_tot": "precip_med",
        "rain_days": "rainy_days",
    }
    for old, new in aliases.items():
        if old in fc.columns and new not in fc.columns:
            fc = fc.rename(columns={old: new})

    if "thermal_range" not in fc.columns and {"temp_max", "temp_min"}.issubset(
        fc.columns
    ):
        fc["thermal_range"] = pd.to_numeric(
            fc["temp_max"], errors="coerce"
        ) - pd.to_numeric(fc["temp_min"], errors="coerce")

    if "round" not in fc.columns:
        raise ValueError("Forecast climate file needs a 'round' column")

    if "uf" not in fc.columns:
        if "uf_code" not in fc.columns:
            if "geocode" not in fc.columns:
                raise ValueError("Forecast climate file needs uf, uf_code, or geocode")
            fc["uf_code"] = to_uf_code_from_geocode(fc["geocode"])
        fc = add_uf_from_code(fc, "uf_code")

    missing = [c for c in RAW_CLIMATE_COLS if c not in fc.columns]
    if missing:
        raise ValueError(f"Forecast climate missing raw climate columns: {missing}")

    fc["uf"] = fc["uf"].astype(str)
    fc = fc[fc["uf"].isin(state_order)].copy()
    for c in RAW_CLIMATE_COLS:
        fc[c] = pd.to_numeric(fc[c], errors="coerce")

    state_fc = fc.groupby(["round", "uf", "date"], as_index=False)[
        RAW_CLIMATE_COLS
    ].mean()
    out_path = output_dir / "state_forecasting_climate_delta_adjusted_from_input.csv"
    state_fc.to_csv(out_path, index=False)
    log(f"Saved state-level forecast climate: {out_path} shape={state_fc.shape}")
    return state_fc


def recompute_lagged_climate(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().sort_values(["uf", "date"]).reset_index(drop=True)
    for raw_col, lag in CLIMATE_LAGS.items():
        df[f"{raw_col}_lag{lag}"] = df.groupby("uf", sort=False)[raw_col].shift(lag)
    return df


def apply_forecast_climate_for_split(
    panel: pd.DataFrame, state_fc: pd.DataFrame, split_id: int
) -> pd.DataFrame:
    cutoff_date = SPLIT_CUTOFF_DATES[split_id]
    round_name = SPLIT_FORECAST_ROUNDS[split_id]
    target_col = f"target_{split_id}"

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    fc_round = state_fc[state_fc["round"] == round_name].copy()
    if fc_round.empty:
        raise ValueError(f"No forecast climate rows found for {round_name}")

    fc_round = fc_round[["uf", "date"] + RAW_CLIMATE_COLS].copy()
    fc_round = fc_round.rename(columns={c: f"forecast_{c}" for c in RAW_CLIMATE_COLS})
    df = df.merge(fc_round, on=["uf", "date"], how="left")

    target_mask = (df["date"] > cutoff_date) & (df[target_col].astype(bool))
    n_target = int(target_mask.sum())
    n_match = int(df.loc[target_mask, "forecast_temp_min"].notna().sum())
    log(
        f"split {split_id} | {round_name} | target rows={n_target} | forecast matches={n_match}"
    )
    if n_match != n_target:
        raise ValueError(
            f"Forecast climate did not match all target rows for split {split_id}: {n_match}/{n_target}"
        )

    for col in RAW_CLIMATE_COLS:
        df.loc[target_mask, col] = df.loc[target_mask, f"forecast_{col}"]
        df = df.drop(columns=[f"forecast_{col}"])

    df = recompute_lagged_climate(df)
    df["forecast_climate_used"] = False
    df.loc[target_mask, "forecast_climate_used"] = True
    df["forecast_round"] = np.nan
    df.loc[target_mask, "forecast_round"] = round_name
    return df


# =============================================================================
# Split building
# =============================================================================


def matrix_for(
    df: pd.DataFrame, epiweeks: np.ndarray, cols: List[str], state_order: List[str]
) -> np.ndarray:
    state_rank = {uf: i for i, uf in enumerate(state_order)}
    tmp = df[df["epiweek"].isin(epiweeks)].copy()
    tmp["node_idx"] = tmp["uf"].map(state_rank)
    tmp = tmp.sort_values(["epiweek", "node_idx"])
    expected = len(epiweeks) * len(state_order)
    if len(tmp) != expected:
        raise ValueError(f"Expected {expected} rows for matrix, got {len(tmp)}")
    return (
        tmp[cols]
        .to_numpy(dtype=np.float32)
        .reshape(len(epiweeks), len(state_order), len(cols))
    )


def target_for(
    df: pd.DataFrame, epiweeks: np.ndarray, state_order: List[str]
) -> np.ndarray:
    return matrix_for(df, epiweeks, [Y_COL], state_order)[:, :, 0]


def pad_y(y: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros((max_len, y.shape[1]), dtype=np.float32)
    mask = np.zeros((max_len, y.shape[1]), dtype=bool)
    out[: y.shape[0]] = y
    mask[: y.shape[0]] = True
    return out, mask


def pad_X_target(X_target: np.ndarray, max_len: int) -> np.ndarray:
    L, N, F = X_target.shape
    out = np.zeros((max_len, N, F), dtype=np.float32)
    if L > 0:
        out[:L] = X_target
        if L < max_len:
            out[L:] = X_target[-1][None, :, :]
    return out


def freeze_after_cutoff(
    df: pd.DataFrame, cutoff: int, cols_to_freeze: List[str]
) -> pd.DataFrame:
    out = df.copy().sort_values(["uf", "epiweek"]).reset_index(drop=True)
    for uf, g in out.groupby("uf", sort=False):
        available = g[g["epiweek"] <= cutoff]
        if available.empty:
            continue
        last_row = available.sort_values("epiweek").iloc[-1]
        future_idx = g[g["epiweek"] > cutoff].index
        for c in cols_to_freeze:
            out.loc[future_idx, c] = last_row[c]
    return out


def fit_scaler(
    df: pd.DataFrame, cutoff: int, cols: List[str]
) -> Tuple[Dict[str, float], StandardScaler]:
    tmp = df.copy()
    available = tmp["epiweek"] <= cutoff
    medians = {}
    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        med = tmp.loc[available, c].median()
        if not np.isfinite(med):
            med = tmp[c].median()
        if not np.isfinite(med):
            med = 0.0
        medians[c] = float(med)
        tmp[c] = tmp[c].fillna(med)
    scaler = StandardScaler()
    scaler.fit(tmp.loc[available, cols])
    return medians, scaler


def apply_scaler(
    df: pd.DataFrame, cols: List[str], medians: Dict[str, float], scaler: StandardScaler
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(medians[c])
        )
    out[cols] = scaler.transform(out[cols])
    return out


def sample_meta(raw_panel: pd.DataFrame, season_year: int) -> Optional[Dict]:
    cutoff = ew25(season_year)
    hist_epiweeks = np.array(
        sorted(raw_panel.loc[raw_panel["epiweek"] <= cutoff, "epiweek"].unique()),
        dtype=np.int64,
    )
    if len(hist_epiweeks) < HISTORY_LEN:
        return None
    hist_epiweeks = hist_epiweeks[-HISTORY_LEN:]
    target_epiweeks = np.array(
        sorted(
            raw_panel.loc[
                season_mask(raw_panel["epiweek"], season_year), "epiweek"
            ].unique()
        ),
        dtype=np.int64,
    )
    if len(target_epiweeks) == 0:
        return None
    target_dates = (
        raw_panel.loc[raw_panel["epiweek"].isin(target_epiweeks), ["epiweek", "date"]]
        .drop_duplicates()
        .sort_values("epiweek")
    )
    return {
        "season_year": season_year,
        "cutoff": cutoff,
        "history_epiweeks": hist_epiweeks,
        "target_epiweeks": target_epiweeks,
        "target_dates": target_dates,
    }


def build_splits(
    panel: pd.DataFrame, panels_fc: Dict[int, pd.DataFrame], state_order: List[str]
) -> Dict[int, Dict]:
    climate_lag_cols = [f"{c}_lag{lag}" for c, lag in CLIMATE_LAGS.items()]
    static_known = ["population", "week_sin", "week_cos"] + [
        c for c in FEATURE_COLS if c.startswith("koppen_") or c.startswith("biome_")
    ]
    safe_target_cols = [c for c in climate_lag_cols + static_known if c in FEATURE_COLS]
    freeze_cols = [c for c in FEATURE_COLS if c not in safe_target_cols]

    splits = {}
    for split_id, split_year in SPLIT_YEARS.items():
        cutoff = ew25(split_year)
        train_panel = panel.copy()
        val_panel = panels_fc[split_id].copy()
        medians, scaler = fit_scaler(train_panel, cutoff, FEATURE_COLS)

        min_year = int(train_panel["epiweek"].min() // 100)
        train_metas = []
        for season_year in range(min_year, split_year):
            meta = sample_meta(train_panel, season_year)
            if meta is not None and meta["target_epiweeks"].max() <= cutoff:
                train_metas.append(meta)
        if not train_metas:
            raise ValueError(f"No training samples for split {split_id}")
        val_meta = sample_meta(val_panel, split_year)
        if val_meta is None:
            raise ValueError(f"No validation sample for split {split_id}")

        max_target_len = max(
            [len(m["target_epiweeks"]) for m in train_metas]
            + [len(val_meta["target_epiweeks"])]
        )

        def build_arrays(raw: pd.DataFrame, meta: Dict) -> Dict:
            tmp = freeze_after_cutoff(raw, meta["cutoff"], freeze_cols)
            tmp = apply_scaler(tmp, FEATURE_COLS, medians, scaler)
            X_hist = matrix_for(
                tmp, meta["history_epiweeks"], FEATURE_COLS, state_order
            )
            X_target = matrix_for(
                tmp, meta["target_epiweeks"], FEATURE_COLS, state_order
            )
            X = np.concatenate(
                [X_hist, pad_X_target(X_target, max_target_len)], axis=0
            ).astype(np.float32)
            y = target_for(raw, meta["target_epiweeks"], state_order)
            y_pad, y_mask = pad_y(y, max_target_len)
            out = dict(meta)
            out["X"] = X
            out["y"] = y_pad
            out["y_mask"] = y_mask
            return out

        train_samples = [build_arrays(train_panel, m) for m in train_metas]
        val_sample = build_arrays(val_panel, val_meta)

        splits[split_id] = {
            "split_id": split_id,
            "split_year": split_year,
            "cutoff_epiweek": cutoff,
            "train_season_years": np.array(
                [s["season_year"] for s in train_samples], dtype=np.int64
            ),
            "X_train": np.stack([s["X"] for s in train_samples]).astype(np.float32),
            "y_train": np.stack([s["y"] for s in train_samples]).astype(np.float32),
            "y_train_mask": np.stack([s["y_mask"] for s in train_samples]),
            "X_val": val_sample["X"][None, ...].astype(np.float32),
            "y_val": val_sample["y"][None, ...].astype(np.float32),
            "y_val_mask": val_sample["y_mask"][None, ...],
            "val_target_epiweeks": val_sample["target_epiweeks"],
            "val_target_dates": val_sample["target_dates"],
            "states": np.array(state_order, dtype=object),
            "feature_cols": np.array(FEATURE_COLS, dtype=object),
            "scaler": scaler,
            "imputation_medians": medians,
            "safe_target_feature_cols": np.array(safe_target_cols, dtype=object),
            "frozen_after_cutoff_cols": np.array(freeze_cols, dtype=object),
            "forecast_climate_used": True,
        }
        log(
            f"split {split_id} | X_train={splits[split_id]['X_train'].shape} X_val={splits[split_id]['X_val'].shape} y_val={splits[split_id]['y_val'].shape}"
        )
    return splits


# =============================================================================
# Model
# =============================================================================


class SeasonalTensorDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx], "mask": self.mask[idx]}


class DenseGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, A):
        return self.linear(torch.matmul(A, x))


class ResidualGCNBlock(nn.Module):
    def __init__(self, features, dropout=0.15):
        super().__init__()
        self.gcn = DenseGraphConv(features, features)
        self.norm = nn.LayerNorm(features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        residual = x
        h = self.gcn(x, A)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        return residual + h


class ResidualGCNLSTMQuantileModel(nn.Module):
    def __init__(
        self,
        in_features,
        num_nodes,
        target_len,
        num_quantiles,
        A,
        gcn_hidden=32,
        lstm_hidden=96,
        lstm_layers=1,
        dropout=0.15,
        head_hidden=64,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.target_len = target_len
        self.num_quantiles = num_quantiles
        self.gcn_hidden = gcn_hidden
        self.register_buffer("A", A.float())
        self.input_gcn = DenseGraphConv(in_features, gcn_hidden)
        self.input_norm = nn.LayerNorm(gcn_hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gcn_block1 = ResidualGCNBlock(gcn_hidden, dropout)
        self.gcn_block2 = ResidualGCNBlock(gcn_hidden, dropout)
        self.lstm = nn.LSTM(
            gcn_hidden,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, target_len * num_quantiles),
        )

    def forward(self, X):
        B, T, N, F = X.shape
        h = X.reshape(B * T, N, F)
        h = self.input_gcn(h, self.A)
        h = self.input_norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.gcn_block1(h, self.A)
        h = self.gcn_block2(h, self.A)
        h = h.reshape(B, T, N, self.gcn_hidden)
        h = h.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, self.gcn_hidden)
        lstm_out, _ = self.lstm(h)
        last = lstm_out[:, -1, :]
        out = self.head(last).reshape(B, N, self.target_len, self.num_quantiles)
        return out.permute(0, 2, 1, 3).contiguous()


def masked_quantile_loss(pred, target, mask, quantiles):
    q = quantiles.to(pred.device).view(1, 1, 1, -1)
    error = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * error, (q - 1.0) * error)
    maskf = mask.unsqueeze(-1).float()
    loss = loss * maskf
    denom = torch.clamp(maskf.sum() * pred.shape[-1], min=1.0)
    return loss.sum() / denom


def crossing_penalty(pred):
    return torch.relu(pred[..., :-1] - pred[..., 1:]).mean()


def enforce_monotone_np(pred):
    return np.maximum.accumulate(pred, axis=-1)


def read_adjacency(path: Path, state_order: List[str]) -> torch.Tensor:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    row_labels = df[first_col].astype(str).tolist()
    mapped = []
    for x in row_labels:
        xs = x.strip()
        mapped.append(
            UF_CODE_TO_UF[int(xs)] if xs.isdigit() and int(xs) in UF_CODE_TO_UF else xs
        )
    df.index = mapped
    # Keep only columns that correspond to state labels.
    col_map = {}
    for c in df.columns[1:]:
        cs = str(c).strip()
        col_map[c] = (
            UF_CODE_TO_UF[int(cs)] if cs.isdigit() and int(cs) in UF_CODE_TO_UF else cs
        )
    values = df.iloc[:, 1:].rename(columns=col_map)
    missing_rows = [s for s in state_order if s not in values.index]
    missing_cols = [s for s in state_order if s not in values.columns]
    if missing_rows or missing_cols:
        raise ValueError(
            f"Adjacency {path} missing rows {missing_rows} or cols {missing_cols}"
        )
    A = values.loc[state_order, state_order].to_numpy(dtype=np.float32)
    return torch.tensor(A, dtype=torch.float32)


def build_border_adjacency(state_order: List[str]) -> torch.Tensor:
    """
    Build the row-normalized Brazilian state-border adjacency matrix from code.

    Rows and columns follow state_order. Only neighbors also present in
    state_order are kept, so the function works for both 26-state and
    27-unit state orders.
    """
    state_order = [str(s) for s in state_order]
    state_set = set(state_order)

    missing = [s for s in state_order if s not in BORDER_NEIGHBORS]
    if missing:
        raise ValueError(
            f"No hard-coded border neighbors available for states: {missing}"
        )

    n = len(state_order)
    idx = {uf: i for i, uf in enumerate(state_order)}
    A = np.zeros((n, n), dtype=np.float32)

    for uf in state_order:
        neighbors = [v for v in BORDER_NEIGHBORS[uf] if v in state_set and v != uf]
        if len(neighbors) == 0:
            continue
        w = 1.0 / float(len(neighbors))
        for nb in neighbors:
            A[idx[uf], idx[nb]] = w

    # Defensive cleanup.
    np.fill_diagonal(A, 0.0)
    row_sums = A.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    A[nonzero] = A[nonzero] / row_sums[nonzero]

    return torch.tensor(A, dtype=torch.float32)


def save_adjacency_tensor(A: torch.Tensor, state_order: List[str], path: Path) -> None:
    """Save an adjacency tensor as a labelled CSV for reproducibility."""
    arr = A.detach().cpu().numpy() if hasattr(A, "detach") else np.asarray(A)
    pd.DataFrame(arr, index=state_order, columns=state_order).to_csv(path)


def train_one_split(
    split_id: int,
    split_obj: Dict,
    A: torch.Tensor,
    model_name: str,
    cfg: Dict,
    device: torch.device,
    seed: int,
) -> Tuple[nn.Module, np.ndarray, pd.DataFrame]:
    set_seed(seed)
    train_ds = SeasonalTensorDataset(
        split_obj["X_train"], split_obj["y_train"], split_obj["y_train_mask"]
    )
    val_ds = SeasonalTensorDataset(
        split_obj["X_val"], split_obj["y_val"], split_obj["y_val_mask"]
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = ResidualGCNLSTMQuantileModel(
        in_features=split_obj["X_train"].shape[-1],
        num_nodes=split_obj["X_train"].shape[2],
        target_len=split_obj["y_train"].shape[1],
        num_quantiles=len(QUANTILES),
        A=A.to(device),
        gcn_hidden=cfg["gcn_hidden"],
        lstm_hidden=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
        head_hidden=cfg["head_hidden"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=1e-5
    )
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32)
    best_val = np.inf
    best_state = None
    best_epoch = -1
    patience_counter = 0
    rows = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []
        cross_losses = []
        for batch in train_loader:
            Xb = batch["X"].to(device)
            yb = batch["y"].to(device)
            mb = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(Xb)
            q_loss = masked_quantile_loss(pred, yb, mb, q_tensor)
            c_loss = crossing_penalty(pred)
            loss = q_loss + cfg["crossing_penalty_weight"] * c_loss
            loss.backward()
            if cfg["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            train_losses.append(float(q_loss.detach().cpu()))
            cross_losses.append(float(c_loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["X"].to(device))
                val_loss = masked_quantile_loss(
                    pred, batch["y"].to(device), batch["mask"].to(device), q_tensor
                )
                val_losses.append(float(val_loss.detach().cpu()))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        rows.append(
            {
                "epoch": epoch,
                "train_q_loss": float(np.mean(train_losses)),
                "train_cross_loss": float(np.mean(cross_losses)),
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 25 == 0:
            log(
                f"{model_name} split={split_id} epoch={epoch} val={val_loss:.5f} best={best_val:.5f}@{best_epoch}"
            )
        if patience_counter >= cfg["patience"]:
            log(f"Early stopping {model_name} split {split_id} at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_log_q = (
            model(torch.tensor(split_obj["X_val"], dtype=torch.float32).to(device))
            .detach()
            .cpu()
            .numpy()
        )
    pred_log_q = enforce_monotone_np(pred_log_q)
    return model, pred_log_q, pd.DataFrame(rows)


# =============================================================================
# Evaluation and ensemble
# =============================================================================


def predictions_to_eval_df(
    split_id: int, split_obj: Dict, pred_log_quantiles: np.ndarray
) -> pd.DataFrame:
    pred = np.expm1(pred_log_quantiles[0])  # [L,N,Q]
    pred = np.clip(pred, 0.0, None)
    y_cases = np.expm1(split_obj["y_val"][0])
    y_cases = np.clip(y_cases, 0.0, None)
    mask = split_obj["y_val_mask"][0]
    epiweeks = split_obj["val_target_epiweeks"]
    target_dates_df = split_obj["val_target_dates"].copy()
    date_map = dict(
        zip(
            target_dates_df["epiweek"].astype(int),
            pd.to_datetime(target_dates_df["date"]),
        )
    )
    rows = []
    states = split_obj["states"].tolist()
    for t, ew in enumerate(epiweeks):
        if t >= pred.shape[0]:
            continue
        date = date_map.get(int(ew), pd.NaT)
        for n, uf in enumerate(states):
            if not mask[t, n]:
                continue
            row = {
                "split_id": split_id,
                "uf": uf,
                "epiweek": int(ew),
                "date": date,
                "casos": float(y_cases[t, n]),
            }
            for qi, q in enumerate(QUANTILES):
                row[f"q_{q:g}"] = float(pred[t, n, qi])
            row["pred_median"] = row["q_0.5"]
            row["error"] = row["pred_median"] - row["casos"]
            rows.append(row)
    df = pd.DataFrame(rows)
    return compute_scores(df)


def find_q_col(q: float) -> str:
    return f"q_{q:g}"


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q025, q05, q10, q25, q50, q75, q90, q95, q975 = [find_q_col(q) for q in QUANTILES]
    y = out["casos"].to_numpy(float)
    median = out[q50].to_numpy(float)
    out["ae"] = np.abs(y - median)
    out["se"] = (y - median) ** 2
    weighted = 0.5 * out["ae"].to_numpy(float)
    for lower_col, upper_col, alpha in [
        (q025, q975, 0.05),
        (q05, q95, 0.10),
        (q10, q90, 0.20),
        (q25, q75, 0.50),
    ]:
        lower = out[lower_col].to_numpy(float)
        upper = out[upper_col].to_numpy(float)
        interval = (
            (upper - lower)
            + (2.0 / alpha) * (lower - y) * (y < lower)
            + (2.0 / alpha) * (y - upper) * (y > upper)
        )
        weighted += (alpha / 2.0) * interval
    out["wis"] = weighted / (4 + 0.5)
    out["cov_95"] = ((out["casos"] >= out[q025]) & (out["casos"] <= out[q975])).astype(
        float
    )
    return out


def summarize_eval(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n": len(df),
            "mean_wis": df["wis"].mean(),
            "median_wis": df["wis"].median(),
            "mae": df["ae"].mean(),
            "rmse": float(np.sqrt(df["se"].mean())),
            "coverage_95": df["cov_95"].mean(),
            "mean_cases": df["casos"].mean(),
            "mean_pred_median": df["q_0.5"].mean(),
        }
    )


def ensemble_split(
    border_df: pd.DataFrame,
    mobility_df: pd.DataFrame,
    split_id: int,
    weight_border: float,
) -> pd.DataFrame:
    merge_keys = ["split_id", "uf", "epiweek"]
    b = border_df[border_df["split_id"] == split_id].copy()
    m = mobility_df[mobility_df["split_id"] == split_id].copy()
    b_keep = merge_keys + ["date", "casos"] + Q_COLS
    m_keep = merge_keys + ["casos"] + Q_COLS
    b = b[b_keep].rename(columns={c: f"{c}_border" for c in Q_COLS})
    m = (
        m[m_keep]
        .rename(columns={c: f"{c}_mobility" for c in Q_COLS})
        .drop(columns=["casos"])
    )
    merged = b.merge(m, on=merge_keys, how="inner", validate="one_to_one")
    out = merged[["split_id", "uf", "epiweek", "date", "casos"]].copy()
    for q in Q_COLS:
        out[q] = (
            weight_border * merged[f"{q}_border"]
            + (1.0 - weight_border) * merged[f"{q}_mobility"]
        )
    q_vals = np.maximum.accumulate(out[Q_COLS].to_numpy(float), axis=1)
    out[Q_COLS] = q_vals
    out = compute_scores(out)
    out["model"] = "model11_forecast_climate_ensemble"
    out["weight_border"] = weight_border
    out["weight_mobility"] = 1.0 - weight_border
    out["forecast_climate_used"] = True
    return out


# =============================================================================
# Plotting
# =============================================================================


def make_ribbon_plots(eval_df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        log("matplotlib unavailable; skipping plots")
        return
    plot_dir = output_dir / "plots_forecast_ribbons_all_states"
    plot_dir.mkdir(exist_ok=True)
    for split_id in sorted(eval_df["split_id"].unique()):
        split_df = eval_df[eval_df["split_id"] == split_id].copy()
        ufs = sorted(split_df["uf"].unique())
        n_cols = 4
        n_rows = math.ceil(len(ufs) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), sharex=True
        )
        axes = np.array(axes).flatten()
        for ax_idx, uf in enumerate(ufs):
            ax = axes[ax_idx]
            state_df = split_df[split_df["uf"] == uf].sort_values("date")
            ax.fill_between(
                state_df["date"],
                state_df["q_0.025"],
                state_df["q_0.975"],
                alpha=0.18,
                label="95% interval",
            )
            ax.fill_between(
                state_df["date"],
                state_df["q_0.1"],
                state_df["q_0.9"],
                alpha=0.28,
                label="80% interval",
            )
            ax.plot(
                state_df["date"],
                state_df["q_0.5"],
                linewidth=1.8,
                label="Median prediction",
            )
            ax.plot(
                state_df["date"],
                state_df["casos"],
                marker="o",
                markersize=2,
                linewidth=1,
                label="Observed",
            )
            ax.set_title(
                f"{uf} | WIS={state_df['wis'].mean():.1f} | MAE={state_df['ae'].mean():.1f}",
                fontsize=9,
            )
            ax.grid(alpha=0.25)
        for j in range(len(ufs), len(axes)):
            axes[j].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0)
        )
        fig.suptitle(
            f"Model 11-FC forecast intervals by state — split {split_id}",
            fontsize=16,
            y=1.02,
        )
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = plot_dir / f"model11_fc_forecast_ribbons_all_states_split_{split_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved plot: {path}")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent final Model 11-FC pipeline from raw challenge folder."
    )
    parser.add_argument(
        "--challenge-data-dir",
        type=Path,
        required=True,
        help="Folder containing raw challenge files: dengue, climate, population, environment, ocean indices.",
    )
    parser.add_argument(
        "--mobility-matrix-csv",
        type=Path,
        required=True,
        help="Mobility adjacency matrix CSV. Its row order defines the state order.",
    )
    parser.add_argument(
        "--border-matrix-csv",
        type=Path,
        default=None,
        help="Optional border adjacency CSV. If omitted, border adjacency is generated from the hard-coded state-neighbor map.",
    )
    parser.add_argument(
        "--forecast-climate-csv",
        type=Path,
        required=True,
        help="Forecast climate CSV, state-level or municipality-level.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder.")
    parser.add_argument(
        "--epochs", type=int, default=600, help="Training epochs. Use 1 for smoke test."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument(
        "--weight-border", type=float, default=BEST_MODEL11_FC_WEIGHT_BORDER
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(TRAIN_CFG)
    cfg["epochs"] = args.epochs

    set_seed(args.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log(f"Device: {device}")

    state_order = load_state_order_from_matrix(args.mobility_matrix_csv)
    log(f"State order from mobility matrix ({len(state_order)}): {state_order}")
    if len(state_order) == 0:
        raise ValueError("Could not infer state order from mobility matrix")

    panel = build_base_panel(args.challenge_data_dir, state_order, args.output_dir)
    state_fc = read_or_build_state_forecast_climate(
        args.forecast_climate_csv, state_order, args.output_dir
    )
    panels_fc = {}
    diag_rows = []
    for split_id in sorted(SPLIT_YEARS):
        panel_fc = apply_forecast_climate_for_split(panel, state_fc, split_id)
        panels_fc[split_id] = panel_fc
        target_col = f"target_{split_id}"
        target_rows = panel_fc[panel_fc[target_col].astype(bool)].copy()
        diag_rows.append(
            {
                "split_id": split_id,
                "round": SPLIT_FORECAST_ROUNDS[split_id],
                "n_target_rows": len(target_rows),
                "n_forecast_climate_used": int(
                    target_rows["forecast_climate_used"].sum()
                ),
                "all_target_rows_have_forecast": bool(
                    target_rows["forecast_climate_used"].sum() == len(target_rows)
                ),
            }
        )
        panel_fc.to_csv(
            args.output_dir / f"panel_model11_fc_split_{split_id}.csv", index=False
        )
    pd.DataFrame(diag_rows).to_csv(
        args.output_dir / "forecast_climate_diagnostics.csv", index=False
    )
    log(str(pd.DataFrame(diag_rows)))

    splits = build_splits(panel, panels_fc, state_order)

    if args.border_matrix_csv is not None:
        A_border = read_adjacency(args.border_matrix_csv, state_order)
        log(f"Loaded border adjacency from: {args.border_matrix_csv}")
    else:
        A_border = build_border_adjacency(state_order)
        generated_border_path = args.output_dir / "generated_state_border_adjacency.csv"
        save_adjacency_tensor(A_border, state_order, generated_border_path)
        log(
            f"Generated border adjacency from hard-coded state-neighbor map: {generated_border_path}"
        )

    A_mobility = read_adjacency(args.mobility_matrix_csv, state_order)

    all_eval = {}
    for model_name, A in [
        ("model10a_border_forecast_climate", A_border),
        ("model10b_mobility_forecast_climate", A_mobility),
    ]:
        model_dir = args.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        eval_dfs = []
        summary_rows = []
        for split_id, split_obj in splits.items():
            log(f"\nTraining {model_name} split {split_id}")
            model, pred_log_q, hist = train_one_split(
                split_id, split_obj, A, model_name, cfg, device, args.seed
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "train_cfg": cfg,
                    "feature_cols": FEATURE_COLS,
                    "quantiles": QUANTILES,
                    "state_order": state_order,
                    "split_id": split_id,
                },
                model_dir / f"{model_name}_split_{split_id}.pt",
            )
            hist.to_csv(
                model_dir / f"{model_name}_training_history_split_{split_id}.csv",
                index=False,
            )
            eval_df = predictions_to_eval_df(split_id, split_obj, pred_log_q)
            eval_df["model"] = model_name
            eval_df["forecast_climate_used"] = True
            eval_df.to_csv(
                model_dir / f"{model_name}_evaluation_split_{split_id}.csv", index=False
            )
            summary = summarize_eval(eval_df)
            summary["model"] = model_name
            summary["split_id"] = split_id
            summary["forecast_climate_used"] = True
            summary_df = summary.to_frame().T
            summary_df.to_csv(
                model_dir / f"{model_name}_summary_split_{split_id}.csv", index=False
            )
            eval_dfs.append(eval_df)
            summary_rows.append(summary_df)
            log(str(summary_df))
        eval_all = pd.concat(eval_dfs, ignore_index=True)
        summary_all = pd.concat(summary_rows, ignore_index=True)
        eval_all.to_csv(
            model_dir / f"{model_name}_evaluation_all_splits.csv", index=False
        )
        summary_all.to_csv(
            model_dir / f"{model_name}_summary_by_split.csv", index=False
        )
        all_eval[model_name] = eval_all

    weight_border = args.weight_border
    best_eval_dfs = []
    summary_rows = []
    for split_id in sorted(splits):
        ens = ensemble_split(
            all_eval["model10a_border_forecast_climate"],
            all_eval["model10b_mobility_forecast_climate"],
            split_id,
            weight_border,
        )
        ens.to_csv(
            args.output_dir / f"model11_fc_best_evaluation_split_{split_id}.csv",
            index=False,
        )
        summary = summarize_eval(ens)
        summary["model"] = "model11_forecast_climate_ensemble"
        summary["split_id"] = split_id
        summary["weight_border"] = weight_border
        summary["weight_mobility"] = 1.0 - weight_border
        summary["forecast_climate_used"] = True
        summary_df = summary.to_frame().T
        summary_df.to_csv(
            args.output_dir / f"model11_fc_best_summary_split_{split_id}.csv",
            index=False,
        )
        best_eval_dfs.append(ens)
        summary_rows.append(summary_df)

    best_eval_all = pd.concat(best_eval_dfs, ignore_index=True)
    best_summary_by_split = pd.concat(summary_rows, ignore_index=True)
    best_overall = summarize_eval(best_eval_all).to_frame().T
    best_overall["model"] = "model11_forecast_climate_ensemble"
    best_overall["weight_border"] = weight_border
    best_overall["weight_mobility"] = 1.0 - weight_border
    best_overall["forecast_climate_used"] = True
    best_eval_all.to_csv(
        args.output_dir / "model11_fc_best_evaluation_all_splits.csv", index=False
    )
    best_summary_by_split.to_csv(
        args.output_dir / "model11_fc_best_summary_by_split.csv", index=False
    )
    best_overall.to_csv(
        args.output_dir / "model11_fc_best_summary_overall.csv", index=False
    )

    # Minimal weight summary with the chosen final weight.
    best_overall.to_csv(
        args.output_dir / "model11_fc_weight_sweep_summary_overall.csv", index=False
    )

    log("\nFinal Model 11-FC overall summary:")
    log(str(best_overall))
    log(f"Saved outputs to {args.output_dir}")

    if args.make_plots:
        make_ribbon_plots(best_eval_all, args.output_dir)


if __name__ == "__main__":
    main()
