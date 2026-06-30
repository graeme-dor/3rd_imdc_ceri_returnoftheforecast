# Dynamic expert ensemble for year-ahead dengue forecasts in Brazil
# ------------------------------------------------------------------
# Goal
#   Forecast dengue cases without using previous/current-year dengue cases as
#   covariates inside the forecast horizon. Experts may use seasonality,
#   static location/environment variables, population, historical risk summaries
#   computed from the training period, climate covariates available at forecast
#   time, and ocean-climate oscillation covariates when available.
#
# Data expected in the working directory:
#   dengue.csv
#   climate.csv.gz
#   forecasting_climate.csv.gz
#   environ_vars.csv
#   datasus_population_2001_2025.csv
#   map_regional_health.csv
#   ocean_climate_oscillations.csv
#
# Main outputs:
#   output/expert_oof_predictions.csv
#   output/bridge_oof_predictions.csv
#   output/final_forecast_template.csv
#
# Notes
#   1. This is a full modeling scaffold, not a one-click final competition
#      submission. Start with RUN_PILOT = TRUE, check outputs, then scale up.
#   2. The probabilistic intervals are generated from simulation around each
#      expert's point forecasts and then mixed by the bridge weights. You should
#      calibrate these intervals after inspecting validation coverage.
#   3. The script is deliberately explicit so different team members can own
#      different expert blocks.

options(stringsAsFactors = FALSE)

# ----------------------------- Configuration ----------------------------- #

DATA_DIR <- "."
OUT_DIR <- Sys.getenv("OUT_DIR", file.path(DATA_DIR, "output"))
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

RUN_PILOT <- as.logical(Sys.getenv("RUN_PILOT", "FALSE"))
PILOT_UF <- c("SP", "RJ", "GO")  # change to your three pilot states
BRIDGE_TOP_K <- 3
BRIDGE_TEMPERATURE <- 0.75
INCLUDE_PEAK_EXPERT <- as.logical(Sys.getenv("INCLUDE_PEAK_EXPERT", "FALSE"))
MASK_TRAIN_SEASONS <- trimws(unlist(strsplit(Sys.getenv("MASK_TRAIN_SEASONS", ""), ",")))
MASK_TRAIN_SEASONS <- MASK_TRAIN_SEASONS[nzchar(MASK_TRAIN_SEASONS)]

# Aggregation level for this script. The user idea is state-level forecasting.
TARGET_LEVEL <- "state"

# Split columns in dengue.csv. The data contain train_1/target_1, ... train_4/target_4.
# We use these as official-style rolling splits if present.
SPLITS <- 1:4

# Forecast quantiles required by Mosqlimate-style probabilistic predictions.
QUANTILE_PROBS <- c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
QUANTILE_NAMES <- c(
  "lower_95", "lower_90", "lower_80", "lower_50", "pred",
  "upper_50", "upper_80", "upper_90", "upper_95"
)

# Number of posterior/predictive samples used to combine experts.
N_SAMPLES <- 400
set.seed(20260616)

# ------------------------------- Packages -------------------------------- #

required_pkgs <- c(
  "data.table", "lubridate", "mgcv", "MASS", "ranger", "nnet"
)

missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    "Install missing packages before running:\n  install.packages(c(",
    paste(sprintf('"%s"', missing_pkgs), collapse = ", "),
    "))"
  )
}

library(data.table)
library(lubridate)
library(mgcv)
library(MASS)
library(ranger)
library(nnet)

# ------------------------------ Small utils ------------------------------ #

message_step <- function(...) {
  message(format(Sys.time(), "%H:%M:%S"), " | ", ...)
}

first_existing_file <- function(data_dir, candidates, required = TRUE) {
  paths <- file.path(data_dir, candidates)
  hit <- paths[file.exists(paths) & file.access(paths, 4) == 0]
  if (length(hit) > 0) return(hit[[1]])
  if (required) {
    stop(
      "None of these required files were found/readable in ",
      normalizePath(data_dir, mustWork = FALSE), ": ",
      paste(candidates, collapse = ", ")
    )
  }
  NA_character_
}

safe_log1p <- function(x) log1p(pmax(x, 0))

inv_log1p <- function(x) pmax(expm1(x), 0)

epi_year <- function(epiweek) as.integer(substr(sprintf("%06d", epiweek), 1, 4))

epi_week <- function(epiweek) as.integer(substr(sprintf("%06d", epiweek), 5, 6))

challenge_season_from_epiweek <- function(epiweek) {
  yy <- epi_year(epiweek)
  ww <- epi_week(epiweek)
  start_year <- ifelse(ww >= 41, yy, yy - 1L)
  paste0(start_year, "-", start_year + 1L)
}

month_floor <- function(x) as.Date(lubridate::floor_date(as.Date(x), "month"))

make_harmonics <- function(dt, week_col = "week") {
  w <- dt[[week_col]]
  dt[, `:=`(
    sin52_1 = sin(2 * pi * w / 52.1775),
    cos52_1 = cos(2 * pi * w / 52.1775),
    sin52_2 = sin(4 * pi * w / 52.1775),
    cos52_2 = cos(4 * pi * w / 52.1775)
  )]
  dt
}

weighted_mean <- function(x, w) {
  ok <- is.finite(x) & is.finite(w) & w > 0
  if (!any(ok)) return(NA_real_)
  sum(x[ok] * w[ok]) / sum(w[ok])
}

clip_cases <- function(x) pmax(round(x), 0)

quantile_table <- function(samples) {
  qs <- stats::quantile(samples, probs = QUANTILE_PROBS, na.rm = TRUE, names = FALSE, type = 8)
  setNames(as.list(qs), QUANTILE_NAMES)
}

negative_binomial_samples <- function(mu, size, n = N_SAMPLES) {
  mu <- pmax(mu, 1e-6)
  size <- ifelse(is.finite(size) && size > 0, size, 10)
  rnbinom(n, mu = mu, size = size)
}

lognormal_count_samples <- function(mu, sigma, n = N_SAMPLES) {
  mu <- pmax(mu, 0)
  sigma <- ifelse(is.finite(sigma) && sigma > 0, sigma, 0.75)
  clip_cases(rlnorm(n, meanlog = log1p(mu), sdlog = sigma) - 1)
}

score_mae <- function(obs, pred) mean(abs(obs - pred), na.rm = TRUE)

score_rmse <- function(obs, pred) sqrt(mean((obs - pred)^2, na.rm = TRUE))

mask_training_seasons <- function(train_dt, context = "training") {
  if (length(MASK_TRAIN_SEASONS) == 0) return(train_dt)
  train_dt <- copy(train_dt)
  train_dt[, challenge_season := challenge_season_from_epiweek(epiweek)]
  before <- nrow(train_dt)
  train_dt <- train_dt[!challenge_season %in% MASK_TRAIN_SEASONS]
  removed <- before - nrow(train_dt)
  message_step(
    "Masked ", removed, " ", context, " rows from seasons: ",
    paste(MASK_TRAIN_SEASONS, collapse = ", ")
  )
  train_dt[, challenge_season := NULL]
  train_dt[]
}

validate_forecast_rules <- function(dt, group_cols = "uf", target_year = NULL) {
  # Mosqlimate-style validation checks for weekly probabilistic forecasts.
  # target_year is optional because the scaffold can be used for several splits.
  dt <- copy(dt)
  interval_cols <- QUANTILE_NAMES
  missing_cols <- setdiff(c("date", "epiweek", interval_cols), names(dt))
  if (length(missing_cols) > 0) {
    stop("Forecast table is missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  dt[, date := as.Date(date)]
  if (any(lubridate::wday(dt$date) != 1, na.rm = TRUE)) {
    stop("Validation failed: all prediction dates must be Sundays.")
  }

  if (any(as.matrix(dt[, ..interval_cols]) < 0, na.rm = TRUE)) {
    stop("Validation failed: all prediction values and intervals must be nonnegative.")
  }

  nested_ok <- dt[
    ,
    lower_95 <= lower_90 &
      lower_90 <= lower_80 &
      lower_80 <= lower_50 &
      lower_50 <= pred &
      pred <= upper_50 &
      upper_50 <= upper_80 &
      upper_80 <= upper_90 &
      upper_90 <= upper_95
  ]
  if (any(!nested_ok, na.rm = TRUE)) {
    stop("Validation failed: prediction intervals are not nested.")
  }

  group_cols <- intersect(group_cols, names(dt))
  if (length(group_cols) > 0) {
    gaps <- dt[
      order(date),
      .(has_gap = any(diff(date) != 7)),
      by = group_cols
    ][has_gap == TRUE]
    if (nrow(gaps) > 0) {
      stop("Validation failed: prediction dates are not continuous weekly Sundays for at least one group.")
    }
  }

  if (!is.null(target_year)) {
    target_year <- as.integer(target_year)
    required_epiweeks <- c(
      (target_year - 1) * 100L + 41:53,
      target_year * 100L + 1:40
    )
    required_epiweeks <- required_epiweeks[required_epiweeks %in% unique(dt$epiweek)]
    missing_ew <- setdiff(required_epiweeks, unique(dt$epiweek))
    if (length(missing_ew) > 0) {
      stop("Validation failed: missing challenge epiweeks: ", paste(missing_ew, collapse = ", "))
    }
  }

  invisible(TRUE)
}

wis_single <- function(y, q_row) {
  # Weighted interval score approximation using 50/80/90/95 intervals plus median.
  intervals <- list(
    c(alpha = 0.50, lo = q_row[["lower_50"]], hi = q_row[["upper_50"]]),
    c(alpha = 0.20, lo = q_row[["lower_80"]], hi = q_row[["upper_80"]]),
    c(alpha = 0.10, lo = q_row[["lower_90"]], hi = q_row[["upper_90"]]),
    c(alpha = 0.05, lo = q_row[["lower_95"]], hi = q_row[["upper_95"]])
  )
  median_score <- abs(y - q_row[["pred"]])
  interval_scores <- vapply(intervals, function(z) {
    alpha <- z[["alpha"]]
    lo <- z[["lo"]]
    hi <- z[["hi"]]
    (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
  }, numeric(1))
  (0.5 * median_score + sum((c(0.50, 0.20, 0.10, 0.05) / 2) * interval_scores)) /
    (0.5 + sum(c(0.50, 0.20, 0.10, 0.05) / 2))
}

# ------------------------------- Data prep ------------------------------- #

read_inputs <- function(data_dir = DATA_DIR) {
  message_step("Reading dengue, climate, population, environmental, map, and oscillation data")

  dengue_path <- first_existing_file(data_dir, "dengue.csv")
  climate_path <- first_existing_file(data_dir, c("climate.csv.gz", "climate.csv"))
  forecast_climate_path <- first_existing_file(
    data_dir,
    c(
      "forecasting_climate_delta_adjusted_weekly.csv",
      "forecasting_climate_delta_adjusted weekly.csv",
      "forecasting_climate.csv.gz",
      "forecasting_climate.csv"
    ),
    required = FALSE
  )
  environ_path <- first_existing_file(data_dir, "environ_vars.csv")
  pop_path <- first_existing_file(data_dir, "datasus_population_2001_2025.csv")
  region_map_path <- first_existing_file(data_dir, "map_regional_health.csv")
  osc_path <- first_existing_file(data_dir, "ocean_climate_oscillations.csv")

  dengue <- fread(dengue_path)
  climate <- fread(climate_path)
  if (is.na(forecast_climate_path)) {
    message_step("No forecasting_climate file found; using training-period climate climatology only")
    forecast_climate <- data.table(
      geocode = integer(),
      reference_month = as.Date(character()),
      forecast_months_ahead = integer(),
      temp_med = numeric(),
      umid_med = numeric(),
      precip_tot = numeric()
    )
  } else {
    message_step("Using forecast climate file: ", basename(forecast_climate_path))
    forecast_climate <- fread(forecast_climate_path)
  }
  environ <- fread(environ_path)
  pop <- fread(pop_path)
  region_map <- fread(region_map_path)
  osc <- fread(osc_path)

  list(
    dengue = dengue,
    climate = climate,
    forecast_climate = forecast_climate,
    environ = environ,
    pop = pop,
    region_map = region_map,
    osc = osc
  )
}

prepare_state_week_panel <- function(x) {
  dengue <- copy(x$dengue)
  climate <- copy(x$climate)
  forecast_climate <- copy(x$forecast_climate)
  environ <- copy(x$environ)
  pop <- copy(x$pop)
  region_map <- copy(x$region_map)
  osc <- copy(x$osc)

  dengue[, date := as.Date(date)]
  dengue[, `:=`(year = epi_year(epiweek), week = epi_week(epiweek))]
  pop[, year := as.integer(year)]

  # Population by state-year.
  pop_state <- merge(
    unique(dengue[, .(geocode, uf, uf_code)]),
    pop,
    by = "geocode",
    all.x = TRUE
  )[, .(population = sum(population, na.rm = TRUE)), by = .(uf, uf_code, year)]

  # Static environment composition by state. Categorical variables are converted
  # to dominant category and proportions for common categories.
  env_state <- merge(
    unique(dengue[, .(geocode, uf, uf_code)]),
    environ[, .(geocode, koppen, biome)],
    by = "geocode",
    all.x = TRUE
  )
  env_mode <- function(v) {
    v <- v[!is.na(v)]
    if (!length(v)) return(NA_character_)
    names(sort(table(v), decreasing = TRUE))[1]
  }
  env_state <- env_state[, .(
    koppen_mode = env_mode(koppen),
    biome_mode = env_mode(biome),
    prop_amazon = mean(biome == "Amazônia", na.rm = TRUE),
    prop_cerrado = mean(biome == "Cerrado", na.rm = TRUE),
    prop_mata_atlantica = mean(biome == "Mata Atlântica", na.rm = TRUE),
    n_municipalities = uniqueN(geocode)
  ), by = .(uf, uf_code)]

  # Macroregion by state.
  state_regions <- unique(region_map[, .(uf, uf_code, uf_name, macroregion_code, macroregion_name)])

  # Aggregate observed climate from municipality-week to state-week using
  # population weights where possible.
  climate[, date := as.Date(date)]
  climate[, `:=`(year = epi_year(epiweek), week = epi_week(epiweek))]
  climate <- merge(
    climate,
    unique(dengue[, .(geocode, uf, uf_code)]),
    by = "geocode",
    all.x = TRUE
  )
  climate <- merge(climate, pop, by = c("geocode", "year"), all.x = TRUE)
  climate[is.na(population), population := 1]

  climate_cols <- c(
    "temp_min", "temp_med", "temp_max",
    "precip_min", "precip_med", "precip_max",
    "pressure_min", "pressure_med", "pressure_max",
    "rel_humid_min", "rel_humid_med", "rel_humid_max",
    "thermal_range", "rainy_days"
  )
  state_climate <- climate[, lapply(.SD, weighted_mean, w = population),
    by = .(uf, uf_code, date, epiweek, year, week),
    .SDcols = climate_cols
  ]
  setnames(
    state_climate,
    climate_cols,
    paste0("obs_", climate_cols)
  )

  # Forecast climate. The newer validation file is weekly and round-specific:
  # round_1 maps to target_1, round_2 to target_2, etc. The older file is
  # monthly by reference month and is retained as a fallback for compatibility.
  if (nrow(forecast_climate) > 0 && all(c("round", "date", "temp_med", "precip_med", "rel_humid_med") %in% names(forecast_climate))) {
    message_step("Preparing weekly round-specific forecast climate")
    forecast_climate[, date := as.Date(date)]
    if (!"year" %in% names(forecast_climate)) forecast_climate[, year := epi_year(as.integer(week))]
    if (!"epiweek" %in% names(forecast_climate)) forecast_climate[, epiweek := as.integer(week)]
    forecast_climate[, `:=`(year = epi_year(epiweek), week = epi_week(epiweek))]
    forecast_climate <- merge(
      forecast_climate,
      unique(dengue[, .(geocode, uf, uf_code)]),
      by = "geocode",
      all.x = TRUE
    )
    forecast_climate <- merge(
      forecast_climate,
      pop[, .(geocode, year, population)],
      by = c("geocode", "year"),
      all.x = TRUE
    )
    forecast_climate[is.na(population), population := 1]

    fc_state <- forecast_climate[, .(
      fc_temp_med = weighted_mean(temp_med, population),
      fc_humid_med = weighted_mean(rel_humid_med, population),
      fc_precip_med = weighted_mean(precip_med, population),
      fc_rainy_days = weighted_mean(rainy_days, population),
      fc_thermal_range = weighted_mean(thermal_range, population)
    ), by = .(round, uf, uf_code, date, epiweek, year, week)]

    setorder(fc_state, round, uf, date)
    fc_base_cols <- c(
      fc_temp_med = "clim_temp_med",
      fc_humid_med = "clim_humid_med",
      fc_precip_med = "clim_precip_med",
      fc_rainy_days = "clim_rainy_days",
      fc_thermal_range = "clim_thermal_range"
    )
    for (fc_col in names(fc_base_cols)) {
      clim_col <- unname(fc_base_cols[[fc_col]])
      for (ll in c(4L, 8L, 12L)) {
        fc_state[, (paste0(clim_col, "_lag", ll, "w")) := shift(get(fc_col), ll), by = .(round, uf)]
      }
    }
    fc_state[
      ,
      `:=`(
        clim_precip_roll4w = rowMeans(cbind(fc_precip_med, clim_precip_med_lag4w), na.rm = TRUE),
        clim_precip_roll8w = rowMeans(cbind(fc_precip_med, clim_precip_med_lag4w, clim_precip_med_lag8w), na.rm = TRUE),
        clim_humid_roll4w = rowMeans(cbind(fc_humid_med, clim_humid_med_lag4w), na.rm = TRUE),
        clim_humid_roll8w = rowMeans(cbind(fc_humid_med, clim_humid_med_lag4w, clim_humid_med_lag8w), na.rm = TRUE),
        clim_temp_roll4w = rowMeans(cbind(fc_temp_med, clim_temp_med_lag4w), na.rm = TRUE),
        clim_temp_roll8w = rowMeans(cbind(fc_temp_med, clim_temp_med_lag4w, clim_temp_med_lag8w), na.rm = TRUE)
      )
    ]
  } else if (nrow(forecast_climate) > 0 && all(c("reference_month", "forecast_months_ahead") %in% names(forecast_climate))) {
    # Forecast climate is monthly by reference month and lead month.
    # We convert it to state-month. During validation, use the reference month
    # that corresponds to the forecast origin month.
    forecast_climate[, reference_month := as.Date(reference_month)]
    forecast_climate[, forecast_month := reference_month %m+% months(forecast_months_ahead)]
    forecast_climate <- merge(
      forecast_climate,
      unique(dengue[, .(geocode, uf, uf_code)]),
      by = "geocode",
      all.x = TRUE
    )
    forecast_climate[, forecast_year := year(forecast_month)]
    forecast_climate <- merge(
      forecast_climate,
      pop[, .(geocode, year, population)],
      by.x = c("geocode", "forecast_year"),
      by.y = c("geocode", "year"),
      all.x = TRUE
    )
    forecast_climate[is.na(population), population := 1]
    fc_state <- forecast_climate[, .(
      fc_temp_med = weighted_mean(temp_med, population),
      fc_humid_med = weighted_mean(umid_med, population),
      fc_precip_tot = weighted_mean(precip_tot, population)
    ), by = .(uf, uf_code, reference_month, forecast_month, forecast_months_ahead)]
  } else {
    fc_state <- data.table()
  }

  # Ocean oscillations are already weekly-ish. Attach by nearest month/week date.
  osc[, date := as.Date(date)]
  osc[, osc_month := month_floor(date)]
  osc_month <- osc[, .(
    enso = mean(enso, na.rm = TRUE),
    iod = mean(iod, na.rm = TRUE),
    pdo = mean(pdo, na.rm = TRUE)
  ), by = osc_month]

  # State-week outcome.
  train_target_cols <- grep("^(train|target)_\\d+$", names(dengue), value = TRUE)
  state_cases <- dengue[, c(
    .(cases = sum(casos, na.rm = TRUE)),
    lapply(.SD, function(z) all(as.logical(z)))
  ), by = .(uf, uf_code, date, epiweek, year, week), .SDcols = train_target_cols]

  panel <- merge(state_cases, pop_state, by = c("uf", "uf_code", "year"), all.x = TRUE)
  panel <- merge(panel, state_climate, by = c("uf", "uf_code", "date", "epiweek", "year", "week"), all.x = TRUE)
  panel <- merge(panel, env_state, by = c("uf", "uf_code"), all.x = TRUE)
  panel <- merge(panel, state_regions, by = c("uf", "uf_code"), all.x = TRUE)
  panel[, month := month(date)]
  panel[, date_month := month_floor(date)]
  panel <- merge(panel, osc_month, by.x = "date_month", by.y = "osc_month", all.x = TRUE)

  panel <- make_harmonics(panel, "week")
  panel[, incidence_per_100k := 1e5 * cases / pmax(population, 1)]
  panel[, log_population := log(pmax(population, 1))]
  panel[, state_id := factor(uf)]
  panel[, macroregion_id := factor(macroregion_name)]
  panel[, koppen_mode := factor(koppen_mode)]
  panel[, biome_mode := factor(biome_mode)]
  setorder(panel, uf, date)

  if (RUN_PILOT) {
    panel <- panel[uf %in% PILOT_UF]
    fc_state <- fc_state[uf %in% PILOT_UF]
  }

  list(panel = panel, forecast_climate_state = fc_state)
}

add_training_only_features <- function(train_dt, pred_dt) {
  # Historical risk summaries must be computed from train_dt only.
  train_dt <- copy(train_dt)
  pred_dt <- copy(pred_dt)

  hist_state_week <- train_dt[, .(
    hist_mean_cases = as.numeric(mean(cases, na.rm = TRUE)),
    hist_median_cases = as.numeric(median(cases, na.rm = TRUE)),
    hist_q75_cases = as.numeric(quantile(cases, 0.75, na.rm = TRUE, type = 8)),
    hist_q90_cases = as.numeric(quantile(cases, 0.90, na.rm = TRUE, type = 8)),
    hist_zero_rate = as.numeric(mean(cases == 0, na.rm = TRUE)),
    hist_outbreak_rate = as.numeric(mean(
      incidence_per_100k >= quantile(incidence_per_100k, 0.90, na.rm = TRUE),
      na.rm = TRUE
    ))
  ), by = .(uf, week)]

  hist_state <- train_dt[, .(
    state_mean_cases = as.numeric(mean(cases, na.rm = TRUE)),
    state_median_cases = as.numeric(median(cases, na.rm = TRUE)),
    state_q90_cases = as.numeric(quantile(cases, 0.90, na.rm = TRUE, type = 8)),
    state_outbreak_threshold = as.numeric(quantile(incidence_per_100k, 0.90, na.rm = TRUE, type = 8)),
    state_volatility = as.numeric(sd(incidence_per_100k, na.rm = TRUE))
  ), by = uf]

  hist_region_week <- train_dt[, .(
    region_week_mean_cases = as.numeric(mean(cases, na.rm = TRUE)),
    region_week_q90_cases = as.numeric(quantile(cases, 0.90, na.rm = TRUE, type = 8))
  ), by = .(macroregion_name, week)]

  # Climate climatology from training years only. These are known-at-forecast
  # "normal climate" summaries, not future observations.
  climate_clim <- train_dt[, .(
    clim_temp_med = as.numeric(mean(obs_temp_med, na.rm = TRUE)),
    clim_precip_med = as.numeric(mean(obs_precip_med, na.rm = TRUE)),
    clim_humid_med = as.numeric(mean(obs_rel_humid_med, na.rm = TRUE)),
    clim_rainy_days = as.numeric(mean(obs_rainy_days, na.rm = TRUE)),
    clim_thermal_range = as.numeric(mean(obs_thermal_range, na.rm = TRUE))
  ), by = .(uf, week)]

  # Forecast-safe climate lags from the state-week climatology. These do not use
  # future observed climate; they say what the typical climate was 4/8/12 weeks
  # before this epidemiological week in the training period.
  setorder(climate_clim, uf, week)
  climate_base_cols <- c(
    "clim_temp_med", "clim_precip_med", "clim_humid_med",
    "clim_rainy_days", "clim_thermal_range"
  )
  for (cc in climate_base_cols) {
    for (ll in c(4L, 8L, 12L)) {
      new_col <- paste0(cc, "_lag", ll, "w")
      climate_clim[
        ,
        (new_col) := {
          v <- get(cc)
          n <- .N
          if (n == 0) numeric() else v[((seq_len(n) - ll - 1L) %% n) + 1L]
        },
        by = uf
      ]
    }
  }
  climate_clim[
    ,
    clim_precip_roll4w := rowMeans(.SD, na.rm = TRUE),
    .SDcols = c("clim_precip_med", "clim_precip_med_lag4w")
  ]
  climate_clim[
    ,
    clim_precip_roll8w := rowMeans(.SD, na.rm = TRUE),
    .SDcols = c("clim_precip_med", "clim_precip_med_lag4w", "clim_precip_med_lag8w")
  ]
  climate_clim[
    ,
    clim_humid_roll4w := rowMeans(.SD, na.rm = TRUE),
    .SDcols = c("clim_humid_med", "clim_humid_med_lag4w")
  ]
  climate_clim[
    ,
    clim_humid_roll8w := rowMeans(.SD, na.rm = TRUE),
    .SDcols = c("clim_humid_med", "clim_humid_med_lag4w", "clim_humid_med_lag8w")
  ]
  climate_clim[
    ,
    clim_temp_roll4w := rowMeans(.SD, na.rm = TRUE),
    .SDcols = c("clim_temp_med", "clim_temp_med_lag4w")
  ]
  climate_clim[
    ,
    clim_temp_roll8w := rowMeans(.SD, na.rm = TRUE),
    .SDcols = c("clim_temp_med", "clim_temp_med_lag4w", "clim_temp_med_lag8w")
  ]

  # Forecast-safe spatial synchrony features. These summarize whether the
  # macroregion historically has outbreak activity in the same season or in
  # weeks preceding the target week. They use only training-period outbreak
  # labels, not current/future cases from neighboring states.
  outbreak_train <- merge(
    train_dt,
    hist_state[, .(uf, state_outbreak_threshold)],
    by = "uf",
    all.x = TRUE
  )
  outbreak_train[, outbreak_flag := as.integer(incidence_per_100k >= state_outbreak_threshold)]
  sync_region_week <- outbreak_train[, .(
    sync_region_outbreak_rate = as.numeric(mean(outbreak_flag, na.rm = TRUE)),
    sync_region_mean_incidence = as.numeric(mean(incidence_per_100k, na.rm = TRUE))
  ), by = .(macroregion_name, week)]
  setorder(sync_region_week, macroregion_name, week)
  for (cc in c("sync_region_outbreak_rate", "sync_region_mean_incidence")) {
    for (ll in c(1L, 2L, 4L)) {
      new_col <- paste0(cc, "_lag", ll, "w")
      sync_region_week[
        ,
        (new_col) := {
          v <- get(cc)
          n <- .N
          if (n == 0) numeric() else v[((seq_len(n) - ll - 1L) %% n) + 1L]
        },
        by = macroregion_name
      ]
    }
  }

  join_all <- function(dt) {
    dt <- merge(dt, hist_state_week, by = c("uf", "week"), all.x = TRUE)
    dt <- merge(dt, hist_state, by = "uf", all.x = TRUE)
    dt <- merge(dt, hist_region_week, by = c("macroregion_name", "week"), all.x = TRUE)
    dt <- merge(dt, climate_clim, by = c("uf", "week"), all.x = TRUE)
    dt <- merge(dt, sync_region_week, by = c("macroregion_name", "week"), all.x = TRUE)

    # Backfill new or sparse cells with state-level/global summaries.
    fill_cols <- c(
      "hist_mean_cases", "hist_median_cases", "hist_q75_cases", "hist_q90_cases",
      "hist_zero_rate", "hist_outbreak_rate", "region_week_mean_cases",
      "region_week_q90_cases", "clim_temp_med", "clim_precip_med",
      "clim_humid_med", "clim_rainy_days", "clim_thermal_range",
      names(climate_clim)[grepl("_lag(4|8|12)w$|_roll(4|8)w$", names(climate_clim))],
      names(sync_region_week)[grepl("^sync_", names(sync_region_week))]
    )
    for (cc in fill_cols) {
      if (cc %in% names(dt)) {
        fallback <- median(dt[[cc]], na.rm = TRUE)
        if (!is.finite(fallback)) fallback <- mean(train_dt$cases, na.rm = TRUE)
        if (!is.finite(fallback)) fallback <- 0
        dt[is.na(get(cc)), (cc) := fallback]
      }
    }
    dt[]
  }

  list(train = join_all(train_dt), pred = join_all(pred_dt))
}

use_forecast_climate_for_future <- function(pred_dt, fc_state, forecast_origin, split_id = NA_integer_) {
  # Optional: replace climate climatology with actual forecast climate for
  # future months when a matching reference month exists.
  pred_dt <- copy(pred_dt)
  if (is.null(forecast_origin) || is.na(forecast_origin)) return(pred_dt)
  if (nrow(fc_state) == 0) return(pred_dt)

  if ("round" %in% names(fc_state)) {
    round_id <- paste0("round_", as.integer(split_id))
    fc <- fc_state[round == round_id]
    if (nrow(fc) == 0) return(pred_dt)

    fc_cols <- c(
      "uf", "date",
      "fc_temp_med", "fc_humid_med", "fc_precip_med",
      "fc_rainy_days", "fc_thermal_range",
      "clim_temp_med_lag4w", "clim_temp_med_lag8w", "clim_temp_med_lag12w",
      "clim_precip_med_lag4w", "clim_precip_med_lag8w", "clim_precip_med_lag12w",
      "clim_humid_med_lag4w", "clim_humid_med_lag8w", "clim_humid_med_lag12w",
      "clim_rainy_days_lag4w", "clim_rainy_days_lag8w", "clim_rainy_days_lag12w",
      "clim_thermal_range_lag4w", "clim_thermal_range_lag8w", "clim_thermal_range_lag12w",
      "clim_precip_roll4w", "clim_precip_roll8w",
      "clim_humid_roll4w", "clim_humid_roll8w",
      "clim_temp_roll4w", "clim_temp_roll8w"
    )
    fc_cols <- intersect(fc_cols, names(fc))
    pred_dt <- merge(pred_dt, fc[, ..fc_cols], by = c("uf", "date"), all.x = TRUE, suffixes = c("", "_fc"))

    replacement_map <- c(
      fc_temp_med = "clim_temp_med",
      fc_humid_med = "clim_humid_med",
      fc_precip_med = "clim_precip_med",
      fc_rainy_days = "clim_rainy_days",
      fc_thermal_range = "clim_thermal_range"
    )
    for (src in names(replacement_map)) {
      dst <- unname(replacement_map[[src]])
      if (src %in% names(pred_dt)) pred_dt[!is.na(get(src)), (dst) := get(src)]
    }
    for (cc in climate_feature_cols) {
      fc_cc <- paste0(cc, "_fc")
      if (fc_cc %in% names(pred_dt)) {
        pred_dt[!is.na(get(fc_cc)), (cc) := get(fc_cc)]
      }
    }
    pred_dt[, forecast_climate_round := round_id]
    pred_dt[, forecast_climate_used := !is.na(fc_temp_med)]
    return(pred_dt[])
  }

  ref_month <- month_floor(forecast_origin)
  fc <- fc_state[reference_month == ref_month]
  if (nrow(fc) == 0) return(pred_dt)

  setorder(fc, uf, forecast_month)
  fc[
    ,
    `:=`(
      fc_temp_med_lag1m = shift(fc_temp_med, 1L),
      fc_temp_med_lag2m = shift(fc_temp_med, 2L),
      fc_humid_med_lag1m = shift(fc_humid_med, 1L),
      fc_humid_med_lag2m = shift(fc_humid_med, 2L),
      fc_precip_tot_lag1m = shift(fc_precip_tot, 1L),
      fc_precip_tot_lag2m = shift(fc_precip_tot, 2L)
    ),
    by = uf
  ]
  fc[
    ,
    `:=`(
      fc_temp_roll2m = rowMeans(cbind(fc_temp_med, fc_temp_med_lag1m), na.rm = TRUE),
      fc_temp_roll3m = rowMeans(cbind(fc_temp_med, fc_temp_med_lag1m, fc_temp_med_lag2m), na.rm = TRUE),
      fc_humid_roll2m = rowMeans(cbind(fc_humid_med, fc_humid_med_lag1m), na.rm = TRUE),
      fc_humid_roll3m = rowMeans(cbind(fc_humid_med, fc_humid_med_lag1m, fc_humid_med_lag2m), na.rm = TRUE),
      fc_precip_roll2m = rowMeans(cbind(fc_precip_tot, fc_precip_tot_lag1m), na.rm = TRUE),
      fc_precip_roll3m = rowMeans(cbind(fc_precip_tot, fc_precip_tot_lag1m, fc_precip_tot_lag2m), na.rm = TRUE)
    )
  ]

  pred_dt[, forecast_month := month_floor(date)]
  pred_dt <- merge(
    pred_dt,
    fc[, .(
      uf, forecast_month, fc_temp_med, fc_humid_med, fc_precip_tot,
      fc_temp_med_lag1m, fc_temp_med_lag2m,
      fc_humid_med_lag1m, fc_humid_med_lag2m,
      fc_precip_tot_lag1m, fc_precip_tot_lag2m,
      fc_temp_roll2m, fc_temp_roll3m,
      fc_humid_roll2m, fc_humid_roll3m,
      fc_precip_roll2m, fc_precip_roll3m,
      forecast_months_ahead
    )],
    by = c("uf", "forecast_month"),
    all.x = TRUE
  )
  pred_dt[!is.na(fc_temp_med), clim_temp_med := fc_temp_med]
  pred_dt[!is.na(fc_humid_med), clim_humid_med := fc_humid_med]
  pred_dt[!is.na(fc_precip_tot), clim_precip_med := fc_precip_tot]
  pred_dt[!is.na(fc_temp_med_lag1m), clim_temp_med_lag4w := fc_temp_med_lag1m]
  pred_dt[!is.na(fc_temp_med_lag2m), clim_temp_med_lag8w := fc_temp_med_lag2m]
  pred_dt[!is.na(fc_humid_med_lag1m), clim_humid_med_lag4w := fc_humid_med_lag1m]
  pred_dt[!is.na(fc_humid_med_lag2m), clim_humid_med_lag8w := fc_humid_med_lag2m]
  pred_dt[!is.na(fc_precip_tot_lag1m), clim_precip_med_lag4w := fc_precip_tot_lag1m]
  pred_dt[!is.na(fc_precip_tot_lag2m), clim_precip_med_lag8w := fc_precip_tot_lag2m]
  pred_dt[!is.na(fc_temp_roll2m), clim_temp_roll4w := fc_temp_roll2m]
  pred_dt[!is.na(fc_temp_roll3m), clim_temp_roll8w := fc_temp_roll3m]
  pred_dt[!is.na(fc_humid_roll2m), clim_humid_roll4w := fc_humid_roll2m]
  pred_dt[!is.na(fc_humid_roll3m), clim_humid_roll8w := fc_humid_roll3m]
  pred_dt[!is.na(fc_precip_roll2m), clim_precip_roll4w := fc_precip_roll2m]
  pred_dt[!is.na(fc_precip_roll3m), clim_precip_roll8w := fc_precip_roll3m]
  pred_dt[]
}

# ----------------------------- Expert models ----------------------------- #

base_feature_cols <- c(
  "year", "week", "month",
  "sin52_1", "cos52_1", "sin52_2", "cos52_2",
  "log_population", "uf", "macroregion_name",
  "koppen_mode", "biome_mode",
  "prop_amazon", "prop_cerrado", "prop_mata_atlantica",
  "n_municipalities",
  "enso", "iod", "pdo"
)

hist_feature_cols <- c(
  "hist_mean_cases", "hist_median_cases", "hist_q75_cases", "hist_q90_cases",
  "hist_zero_rate", "hist_outbreak_rate",
  "state_mean_cases", "state_median_cases", "state_q90_cases",
  "state_volatility", "region_week_mean_cases", "region_week_q90_cases"
)

climate_feature_cols <- c(
  "clim_temp_med", "clim_precip_med", "clim_humid_med",
  "clim_rainy_days", "clim_thermal_range",
  "clim_temp_med_lag4w", "clim_temp_med_lag8w", "clim_temp_med_lag12w",
  "clim_precip_med_lag4w", "clim_precip_med_lag8w", "clim_precip_med_lag12w",
  "clim_humid_med_lag4w", "clim_humid_med_lag8w", "clim_humid_med_lag12w",
  "clim_rainy_days_lag4w", "clim_rainy_days_lag8w", "clim_rainy_days_lag12w",
  "clim_thermal_range_lag4w", "clim_thermal_range_lag8w", "clim_thermal_range_lag12w",
  "clim_precip_roll4w", "clim_precip_roll8w",
  "clim_humid_roll4w", "clim_humid_roll8w",
  "clim_temp_roll4w", "clim_temp_roll8w"
)

spatial_feature_cols <- c(
  "sync_region_outbreak_rate",
  "sync_region_outbreak_rate_lag1w",
  "sync_region_outbreak_rate_lag2w",
  "sync_region_outbreak_rate_lag4w",
  "sync_region_mean_incidence",
  "sync_region_mean_incidence_lag1w",
  "sync_region_mean_incidence_lag2w",
  "sync_region_mean_incidence_lag4w"
)

all_model_features <- c(base_feature_cols, hist_feature_cols, climate_feature_cols, spatial_feature_cols)

complete_model_frame <- function(dt, features = all_model_features) {
  dt <- copy(dt)
  for (cc in features) {
    if (!cc %in% names(dt)) dt[, (cc) := NA]
  }
  char_cols <- names(dt)[vapply(dt, is.character, logical(1))]
  for (cc in char_cols) dt[, (cc) := factor(get(cc))]
  dt[]
}

# Expert 1: Seasonal negative-binomial GAM
#   Purpose: Learn smooth annual cycles and state-specific baseline differences.
#   Uses: epiweek harmonics/splines, state and macroregion factors, population.
fit_expert_seasonal_gam <- function(train_dt) {
  train_dt <- complete_model_frame(train_dt)
  mgcv::gam(
    cases ~
      offset(log_population) +
      s(week, bs = "cc", k = 20) +
      s(year, k = 8) +
      s(uf, bs = "re") +
      s(macroregion_name, bs = "re"),
    family = nb(),
    data = train_dt,
    method = "REML"
  )
}

predict_expert_seasonal_gam <- function(model, train_dt, pred_dt) {
  pred_dt <- complete_model_frame(pred_dt)
  mu <- as.numeric(predict(model, newdata = pred_dt, type = "response"))
  theta <- model$family$getTheta(TRUE)
  samples <- lapply(mu, negative_binomial_samples, size = theta)
  list(mu = mu, samples = samples)
}

# Expert 2: Historical risk lookup
#   Purpose: Strong year-ahead prior using only previous training years.
#   Uses: state-week historical mean/median/upper quantiles and population scale.
fit_expert_historical_risk <- function(train_dt) {
  theta <- tryCatch(
    MASS::theta.ml(y = train_dt$cases, mu = pmax(train_dt$hist_mean_cases, 1e-6)),
    error = function(e) 10
  )
  list(theta = theta)
}

predict_expert_historical_risk <- function(model, train_dt, pred_dt) {
  mu <- pmax(pred_dt$hist_mean_cases, 0)
  # Blend mean and median to reduce domination by extreme outbreak years.
  mu <- 0.7 * mu + 0.3 * pmax(pred_dt$hist_median_cases, 0)
  samples <- lapply(mu, negative_binomial_samples, size = model$theta)
  list(mu = mu, samples = samples)
}

# Expert 3: Climate-suitability random forest
#   Purpose: Capture nonlinear relations between forecast-available climate,
#   climate climatology, oscillations, static environment, and dengue burden.
#   Uses no previous cases from the forecast year.
fit_expert_climate_rf <- function(train_dt) {
  train_dt <- complete_model_frame(train_dt)
  f <- as.formula(paste(
    "safe_log1p(cases) ~",
    paste(c(base_feature_cols, climate_feature_cols, hist_feature_cols), collapse = " + ")
  ))
  ranger::ranger(
    formula = f,
    data = train_dt,
    num.trees = 600,
    mtry = max(2, floor(sqrt(length(all_model_features)))),
    min.node.size = 8,
    importance = "permutation",
    seed = 20260616
  )
}

predict_expert_climate_rf <- function(model, train_dt, pred_dt) {
  pred_dt <- complete_model_frame(pred_dt)
  pred_log <- as.numeric(predict(model, data = pred_dt)$predictions)
  mu <- inv_log1p(pred_log)
  train_pred <- as.numeric(predict(model, data = complete_model_frame(train_dt))$predictions)
  sigma <- sd(safe_log1p(train_dt$cases) - train_pred, na.rm = TRUE)
  samples <- lapply(mu, lognormal_count_samples, sigma = sigma)
  list(mu = mu, samples = samples)
}

# Expert 4: Outbreak/anomaly expert
#   Purpose: Separate high-incidence regime probability from magnitude.
#   Definition: outbreak week = state-specific incidence >= 90th percentile in
#   the training period. This threshold is recomputed inside each split only.
fit_expert_outbreak <- function(train_dt) {
  train_dt <- complete_model_frame(train_dt)
  train_dt[, outbreak := as.integer(incidence_per_100k >= state_outbreak_threshold)]

  cls_formula <- as.formula(paste(
    "as.factor(outbreak) ~",
    paste(c(base_feature_cols, climate_feature_cols, hist_feature_cols), collapse = " + ")
  ))
  cls <- ranger::ranger(
    formula = cls_formula,
    data = train_dt,
    probability = TRUE,
    num.trees = 500,
    min.node.size = 10,
    seed = 20260617
  )

  mag_dt <- train_dt[outbreak == 1]
  if (nrow(mag_dt) < 30) mag_dt <- train_dt
  mag_formula <- as.formula(paste(
    "safe_log1p(cases) ~",
    paste(c(base_feature_cols, climate_feature_cols, hist_feature_cols), collapse = " + ")
  ))
  mag <- ranger::ranger(
    formula = mag_formula,
    data = mag_dt,
    num.trees = 500,
    min.node.size = 5,
    seed = 20260618
  )

  base_mu <- mean(train_dt$cases, na.rm = TRUE)
  list(cls = cls, mag = mag, base_mu = base_mu)
}

predict_expert_outbreak <- function(model, train_dt, pred_dt) {
  pred_dt <- complete_model_frame(pred_dt)
  prob <- predict(model$cls, data = pred_dt)$predictions
  p_out <- if (is.matrix(prob) && "1" %in% colnames(prob)) prob[, "1"] else rep(mean(train_dt$cases > 0), nrow(pred_dt))
  mag_log <- as.numeric(predict(model$mag, data = pred_dt)$predictions)
  mag_mu <- inv_log1p(mag_log)
  quiet_mu <- pmax(pred_dt$hist_median_cases, 0)
  mu <- p_out * mag_mu + (1 - p_out) * quiet_mu
  sigma <- sd(safe_log1p(train_dt$cases) - safe_log1p(pmax(train_dt$hist_mean_cases, 0)), na.rm = TRUE)
  samples <- lapply(seq_along(mu), function(i) {
    is_out <- rbinom(N_SAMPLES, 1, p_out[i])
    quiet <- lognormal_count_samples(quiet_mu[i], sigma = 0.55, n = N_SAMPLES)
    high <- lognormal_count_samples(mag_mu[i], sigma = max(0.75, sigma), n = N_SAMPLES)
    ifelse(is_out == 1, high, quiet)
  })
  list(mu = mu, samples = samples)
}

# Optional Expert 4b: Peak-only specialist
#   Purpose: Model the magnitude of genuine peak weeks more aggressively.
#   Definition: peak week = state-specific incidence >= 90th percentile in the
#   training period. The classifier sees all weeks; the magnitude model is fit
#   only to peak weeks. This expert is intentionally not conservative, so it
#   should be used through Top-3 selection rather than forced into every week.
fit_expert_peak_only <- function(train_dt) {
  train_dt <- complete_model_frame(train_dt)
  train_dt[, peak_week := as.integer(incidence_per_100k >= state_outbreak_threshold)]

  cls_formula <- as.formula(paste(
    "as.factor(peak_week) ~",
    paste(c(base_feature_cols, climate_feature_cols, hist_feature_cols, spatial_feature_cols), collapse = " + ")
  ))
  cls <- ranger::ranger(
    formula = cls_formula,
    data = train_dt,
    probability = TRUE,
    num.trees = 700,
    min.node.size = 8,
    seed = 20260624
  )

  peak_dt <- train_dt[peak_week == 1]
  # If a small pilot or early split has too few peak rows, relax to the upper
  # quartile so the expert still learns high-incidence magnitude.
  if (nrow(peak_dt) < 80) {
    fallback_threshold <- quantile(train_dt$incidence_per_100k, 0.75, na.rm = TRUE, type = 8)
    peak_dt <- train_dt[incidence_per_100k >= fallback_threshold]
  }
  if (nrow(peak_dt) < 30) peak_dt <- train_dt

  mag_formula <- as.formula(paste(
    "safe_log1p(cases) ~",
    paste(c(base_feature_cols, climate_feature_cols, hist_feature_cols, spatial_feature_cols), collapse = " + ")
  ))
  mag <- ranger::ranger(
    formula = mag_formula,
    data = peak_dt,
    num.trees = 700,
    min.node.size = 4,
    quantreg = TRUE,
    seed = 20260625
  )

  train_pred <- as.numeric(predict(mag, data = peak_dt)$predictions)
  sigma <- sd(safe_log1p(peak_dt$cases) - train_pred, na.rm = TRUE)
  if (!is.finite(sigma) || sigma <= 0) sigma <- 0.9

  list(cls = cls, mag = mag, sigma = sigma)
}

predict_expert_peak_only <- function(model, train_dt, pred_dt) {
  pred_dt <- complete_model_frame(pred_dt)
  prob <- predict(model$cls, data = pred_dt)$predictions
  p_peak <- if (is.matrix(prob) && "1" %in% colnames(prob)) {
    prob[, "1"]
  } else {
    rep(mean(train_dt$incidence_per_100k >= train_dt$state_outbreak_threshold, na.rm = TRUE), nrow(pred_dt))
  }

  mag_log <- as.numeric(predict(model$mag, data = pred_dt)$predictions)
  peak_mu <- inv_log1p(mag_log)
  baseline_mu <- pmax(pred_dt$hist_q75_cases, pred_dt$hist_mean_cases, 0)

  # Keep the point forecast peak-oriented, but soften it by peak probability so
  # it does not explode completely in quiet weeks.
  mu <- pmax(
    baseline_mu,
    p_peak * peak_mu + (1 - p_peak) * pmax(pred_dt$hist_q75_cases, 0)
  )

  samples <- lapply(seq_along(mu), function(i) {
    peak_draw <- rbinom(N_SAMPLES, 1, p_peak[i])
    base <- lognormal_count_samples(baseline_mu[i], sigma = 0.65, n = N_SAMPLES)
    peak <- lognormal_count_samples(peak_mu[i], sigma = max(0.9, model$sigma), n = N_SAMPLES)
    ifelse(peak_draw == 1, peak, pmax(base, rpois(N_SAMPLES, lambda = pmax(mu[i], 1e-6))))
  })

  list(mu = mu, samples = samples)
}

# Expert 5: Low-incidence hurdle expert
#   Purpose: Avoid overpredicting in quiet states/weeks. It models whether a
#   week is above a low threshold, then predicts positive magnitude separately.
fit_expert_low_incidence <- function(train_dt) {
  train_dt <- complete_model_frame(train_dt)
  threshold <- pmax(1, quantile(train_dt$cases, 0.25, na.rm = TRUE, type = 8))
  train_dt[, above_low := as.integer(cases > threshold)]

  cls_formula <- as.formula(paste(
    "as.factor(above_low) ~",
    paste(c(base_feature_cols, climate_feature_cols, hist_feature_cols), collapse = " + ")
  ))
  cls <- ranger::ranger(
    formula = cls_formula,
    data = train_dt,
    probability = TRUE,
    num.trees = 500,
    min.node.size = 12,
    seed = 20260619
  )

  pos_dt <- train_dt[cases > threshold]
  if (nrow(pos_dt) < 30) pos_dt <- train_dt[cases > 0]
  if (nrow(pos_dt) < 30) pos_dt <- train_dt
  mag <- mgcv::gam(
    cases ~
      offset(log_population) +
      s(week, bs = "cc", k = 16) +
      s(clim_temp_med, k = 8) +
      s(clim_humid_med, k = 8) +
      s(clim_precip_med, k = 8) +
      s(uf, bs = "re"),
    family = nb(),
    data = pos_dt,
    method = "REML"
  )

  list(cls = cls, mag = mag, threshold = threshold)
}

predict_expert_low_incidence <- function(model, train_dt, pred_dt) {
  pred_dt <- complete_model_frame(pred_dt)
  prob <- predict(model$cls, data = pred_dt)$predictions
  p_above <- if (is.matrix(prob) && "1" %in% colnames(prob)) prob[, "1"] else rep(mean(train_dt$cases > model$threshold), nrow(pred_dt))
  mag_mu <- as.numeric(predict(model$mag, newdata = pred_dt, type = "response"))
  low_mu <- pmin(pred_dt$hist_median_cases, model$threshold)
  mu <- p_above * mag_mu + (1 - p_above) * low_mu
  theta <- model$mag$family$getTheta(TRUE)
  samples <- lapply(seq_along(mu), function(i) {
    above <- rbinom(N_SAMPLES, 1, p_above[i])
    low <- rpois(N_SAMPLES, lambda = pmax(low_mu[i], 1e-6))
    high <- negative_binomial_samples(mag_mu[i], size = theta, n = N_SAMPLES)
    ifelse(above == 1, high, low)
  })
  list(mu = mu, samples = samples)
}

# Expert 6: Spatial/historical synchrony expert
#   Purpose: Use shared regional seasonality and historical synchrony without
#   borrowing current-year cases from neighbors. For state-level forecasts this
#   is a macroregion/state-cluster prior.
fit_expert_spatial <- function(train_dt) {
  train_dt <- complete_model_frame(train_dt)
  mgcv::gam(
    cases ~
      offset(log_population) +
      s(week, bs = "cc", k = 18) +
	      s(macroregion_name, bs = "re") +
	      s(uf, bs = "re") +
	      s(region_week_mean_cases, k = 8) +
	      s(region_week_q90_cases, k = 8) +
	      sync_region_outbreak_rate +
	      sync_region_outbreak_rate_lag1w +
	      sync_region_outbreak_rate_lag2w +
	      sync_region_outbreak_rate_lag4w +
	      sync_region_mean_incidence +
	      sync_region_mean_incidence_lag1w +
	      sync_region_mean_incidence_lag2w +
	      sync_region_mean_incidence_lag4w,
    family = nb(),
    data = train_dt,
    method = "REML"
  )
}

predict_expert_spatial <- function(model, train_dt, pred_dt) {
  pred_dt <- complete_model_frame(pred_dt)
  mu <- as.numeric(predict(model, newdata = pred_dt, type = "response"))
  theta <- model$family$getTheta(TRUE)
  samples <- lapply(mu, negative_binomial_samples, size = theta)
  list(mu = mu, samples = samples)
}

expert_registry <- list(
  seasonal_gam = list(fit = fit_expert_seasonal_gam, predict = predict_expert_seasonal_gam),
  historical_risk = list(fit = fit_expert_historical_risk, predict = predict_expert_historical_risk),
  climate_rf = list(fit = fit_expert_climate_rf, predict = predict_expert_climate_rf),
  outbreak = list(fit = fit_expert_outbreak, predict = predict_expert_outbreak),
  low_incidence = list(fit = fit_expert_low_incidence, predict = predict_expert_low_incidence),
  spatial = list(fit = fit_expert_spatial, predict = predict_expert_spatial)
)

if (INCLUDE_PEAK_EXPERT) {
  expert_registry$peak_only <- list(fit = fit_expert_peak_only, predict = predict_expert_peak_only)
}

# ------------------------------- Bridge ---------------------------------- #

make_expert_prediction_table <- function(pred_dt, expert_name, pred_obj) {
  qdt <- rbindlist(lapply(pred_obj$samples, function(s) as.data.table(quantile_table(s))))
  out <- cbind(
    pred_dt[, .(row_id, uf, uf_code, date, epiweek, year, week, cases, population, macroregion_name)],
    data.table(expert = expert_name, expert_pred = pred_obj$mu),
    qdt
  )
  out[]
}

fit_bridge_error_models <- function(expert_oof) {
  # Dynamic bridge: for each expert, learn expected absolute error from context.
  # At prediction time, convert predicted errors to softmax-like inverse-error
  # weights. This avoids using future observed cases and lets weights change by
  # state, season, climate, historical risk, and model uncertainty.
  dt <- copy(expert_oof)
  dt[, abs_error := abs(cases - expert_pred)]
  dt[, log_abs_error := log1p(abs_error)]
  dt <- dt[is.finite(log_abs_error) & is.finite(expert_pred)]
  if (nrow(dt) == 0) {
    stop("Bridge training has no finite expert errors. Check expert predictions for all-NA outputs.")
  }
  dt <- complete_model_frame(dt, features = c(all_model_features, "expert_pred"))

  models <- list()
  for (ee in unique(dt$expert)) {
    sub <- dt[expert == ee]
    sub <- sub[is.finite(log_abs_error)]
    if (nrow(sub) < 20) {
      warning("Too few clean bridge rows for expert ", ee, "; using pooled bridge rows as fallback.")
      sub <- dt
    }
    f <- as.formula(paste(
      "log_abs_error ~ expert_pred +",
      paste(c(base_feature_cols, hist_feature_cols, climate_feature_cols), collapse = " + ")
    ))
    models[[ee]] <- ranger::ranger(
      formula = f,
      data = sub,
      num.trees = 400,
      min.node.size = 8,
      seed = 20260620
    )
  }
  models
}

predict_bridge_weights <- function(error_models, expert_pred_long, temperature = 0.75) {
  dt <- copy(expert_pred_long)
  dt <- complete_model_frame(dt, features = c(all_model_features, "expert_pred"))
  dt[, pred_log_error := NA_real_]

  for (ee in names(error_models)) {
    idx <- dt$expert == ee
    if (any(idx)) {
      dt[idx, pred_log_error := as.numeric(predict(error_models[[ee]], data = dt[idx])$predictions)]
    }
  }

  dt[is.na(pred_log_error), pred_log_error := median(pred_log_error, na.rm = TRUE)]
  if (any(!is.finite(dt$pred_log_error))) {
    fallback <- median(dt$pred_log_error[is.finite(dt$pred_log_error)], na.rm = TRUE)
    if (!is.finite(fallback)) fallback <- 1
    dt[!is.finite(pred_log_error), pred_log_error := fallback]
  }
  dt[, inv_error_score := exp(-pred_log_error / temperature)]
  dt[, bridge_weight := inv_error_score / sum(inv_error_score), by = .(uf, date)]
  dt[!is.finite(bridge_weight), bridge_weight := 1 / .N, by = .(uf, date)]
  dt[]
}

apply_topk_bridge_weights <- function(weighted_long, k = BRIDGE_TOP_K, temperature = BRIDGE_TEMPERATURE) {
  dt <- copy(weighted_long)
  dt[, error_rank := frank(pred_log_error, ties.method = "first"), by = .(uf, date)]
  dt[error_rank > k, bridge_weight := 0]
  dt[error_rank <= k, topk_score := exp(-pred_log_error / temperature)]
  dt[
    error_rank <= k,
    bridge_weight := topk_score / sum(topk_score),
    by = .(uf, date)
  ]
  dt[!is.finite(bridge_weight), bridge_weight := 0]
  dt[, topk_score := NULL]
  dt[]
}

combine_experts_with_weights <- function(weighted_long, expert_samples) {
  # expert_samples is a named list. Each element contains row-aligned samples.
  keys <- unique(weighted_long[, .(row_id, uf, uf_code, date, epiweek, year, week, cases, population, macroregion_name)])
  setorder(keys, row_id)

  combined <- vector("list", nrow(keys))
  for (i in seq_len(nrow(keys))) {
    rid <- keys$row_id[i]
    row_weights <- weighted_long[row_id == rid, .(expert, bridge_weight)]
    row_weights <- row_weights[bridge_weight > 0]
    if (nrow(row_weights) == 0) {
      combined[[i]] <- rep(NA_real_, N_SAMPLES)
      next
    }
    chosen <- sample(row_weights$expert, size = N_SAMPLES, replace = TRUE, prob = row_weights$bridge_weight)
    draws <- numeric(N_SAMPLES)
            for (ee in row_weights$expert) {
      idx <- which(chosen == ee)
      if (length(idx) > 0) {
        draws[idx] <- sample(expert_samples[[ee]][[rid]], size = length(idx), replace = TRUE)
      }
    }
    combined[[i]] <- draws
  }

  qdt <- rbindlist(lapply(combined, function(s) as.data.table(quantile_table(s))))
  cbind(keys, qdt)
}

season_label_from_split <- function(split_id) {
  split_id <- as.integer(split_id)
  start_year <- 2021L + split_id
  paste0(start_year, "-", start_year + 1L)
}

export_validation_season_csvs <- function(pred_dt, prefix, out_dir = file.path(OUT_DIR, "validation_predictions")) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  dt <- copy(pred_dt)
  if (!"split_id" %in% names(dt)) stop("Validation export needs split_id.")
  dt[, season := season_label_from_split(split_id)]
  export_cols <- c(
    "uf", "season", "pred",
    "lower_50", "upper_50",
    "lower_80", "upper_80",
    "lower_90", "upper_90",
    "lower_95", "upper_95",
    "date"
  )
  missing_cols <- setdiff(export_cols, names(dt))
  if (length(missing_cols) > 0) stop("Missing export columns: ", paste(missing_cols, collapse = ", "))

  files <- character()
  for (ss in sort(unique(dt$season))) {
    out <- dt[season == ss, ..export_cols]
    setorder(out, uf, date)
    file_out <- file.path(out_dir, paste0(prefix, "_validation_", ss, ".csv"))
    fwrite(out, file_out)
    files <- c(files, file_out)
  }
  files
}

# ------------------------- Rolling validation loop ------------------------ #

run_split <- function(panel, fc_state, split_id) {
  train_col <- paste0("train_", split_id)
  target_col <- paste0("target_", split_id)
  if (!all(c(train_col, target_col) %in% names(panel))) {
    stop("Missing split columns: ", train_col, ", ", target_col)
  }

  message_step("Running split ", split_id)
  train_raw <- panel[get(train_col) == TRUE]
  pred_raw <- panel[get(target_col) == TRUE]

  # If train/target overlap in the raw file, remove training rows from target
  # for validation scoring. Keep only dates later than the max training date.
  max_train_date <- max(train_raw$date, na.rm = TRUE)
  train_raw <- mask_training_seasons(train_raw, context = paste0("split ", split_id, " training"))
  pred_raw <- pred_raw[date > max_train_date]
  if (nrow(pred_raw) == 0) {
    warning("No validation rows for split ", split_id)
    return(NULL)
  }

  engineered <- add_training_only_features(train_raw, pred_raw)
  train_dt <- engineered$train
  pred_dt <- engineered$pred
  pred_dt <- use_forecast_climate_for_future(
    pred_dt,
    fc_state,
    forecast_origin = max_train_date,
    split_id = split_id
  )
  pred_dt[, row_id := .I]

  expert_tables <- list()
  expert_samples <- list()

  for (ee in names(expert_registry)) {
    message_step("  fitting expert: ", ee)
    fit_fun <- expert_registry[[ee]]$fit
    pred_fun <- expert_registry[[ee]]$predict
    model <- fit_fun(train_dt)
    pred <- pred_fun(model, train_dt, pred_dt)
    expert_tables[[ee]] <- make_expert_prediction_table(pred_dt, ee, pred)
    expert_samples[[ee]] <- pred$samples
  }

  expert_long <- rbindlist(expert_tables, fill = TRUE)
  expert_long[, split_id := split_id]

  # Add bridge covariates back onto long table.
  covar_cols <- unique(c("uf", "date", all_model_features))
  covars <- pred_dt[, covar_cols, with = FALSE]
  expert_long <- merge(expert_long, covars, by = c("uf", "date"), all.x = TRUE)

  list(expert_long = expert_long, expert_samples = expert_samples, pred_dt = pred_dt)
}

run_oof_pipeline <- function(panel, fc_state) {
  split_results <- list()
  all_expert_oof <- list()

  for (sid in SPLITS) {
    res <- run_split(panel, fc_state, sid)
    if (!is.null(res)) {
      split_results[[as.character(sid)]] <- res
      all_expert_oof[[as.character(sid)]] <- res$expert_long
    }
  }

  expert_oof <- rbindlist(all_expert_oof, fill = TRUE)
  fwrite(expert_oof, file.path(OUT_DIR, "expert_oof_predictions.csv"))

  bridge_outputs <- list()
  top3_bridge_outputs <- list()
  bridge_weight_outputs <- list()
  top3_bridge_weight_outputs <- list()
  for (sid in names(split_results)) {
    res <- split_results[[sid]]
    message_step("Fitting bridge for held-out split ", sid)
    bridge_train <- expert_oof[split_id != as.integer(sid)]
    if (nrow(bridge_train) == 0) bridge_train <- expert_oof
    bridge_models_sid <- fit_bridge_error_models(bridge_train)
    weighted <- predict_bridge_weights(bridge_models_sid, res$expert_long)
    bridge_weight_outputs[[sid]] <- weighted[, .(
      split_id = as.integer(sid),
      row_id, uf, uf_code, date, epiweek, expert,
      expert_pred, pred_log_error, bridge_weight
    )]
    bridge <- combine_experts_with_weights(weighted, res$expert_samples)
    bridge[, split_id := as.integer(sid)]
    bridge_outputs[[sid]] <- bridge

    weighted_top3 <- apply_topk_bridge_weights(weighted, k = BRIDGE_TOP_K, temperature = BRIDGE_TEMPERATURE)
    top3_bridge_weight_outputs[[sid]] <- weighted_top3[, .(
      split_id = as.integer(sid),
      row_id, uf, uf_code, date, epiweek, expert,
      expert_pred, pred_log_error, bridge_weight, error_rank
    )]
    top3_bridge <- combine_experts_with_weights(weighted_top3, res$expert_samples)
    top3_bridge[, split_id := as.integer(sid)]
    top3_bridge_outputs[[sid]] <- top3_bridge
  }

  bridge_oof <- rbindlist(bridge_outputs, fill = TRUE)
  top3_bridge_oof <- rbindlist(top3_bridge_outputs, fill = TRUE)
  bridge_weights_oof <- rbindlist(bridge_weight_outputs, fill = TRUE)
  top3_bridge_weights_oof <- rbindlist(top3_bridge_weight_outputs, fill = TRUE)
  validate_forecast_rules(bridge_oof, group_cols = c("uf", "split_id"))
  validate_forecast_rules(top3_bridge_oof, group_cols = c("uf", "split_id"))
  fwrite(bridge_oof, file.path(OUT_DIR, "bridge_oof_predictions.csv"))
  fwrite(top3_bridge_oof, file.path(OUT_DIR, "top3_bridge_oof_predictions.csv"))
  fwrite(bridge_weights_oof, file.path(OUT_DIR, "bridge_oof_weights.csv"))
  fwrite(top3_bridge_weights_oof, file.path(OUT_DIR, "top3_bridge_oof_weights.csv"))
  export_validation_season_csvs(top3_bridge_oof, prefix = "Top3_bridge")

  message_step("Fitting final bridge error models from all out-of-fold expert predictions")
  bridge_models <- fit_bridge_error_models(expert_oof)

  list(
    expert_oof = expert_oof,
    bridge_oof = bridge_oof,
    top3_bridge_oof = top3_bridge_oof,
    bridge_weights_oof = bridge_weights_oof,
    top3_bridge_weights_oof = top3_bridge_weights_oof,
    bridge_models = bridge_models
  )
}

# ------------------------- Final forecast training ------------------------ #

fit_all_experts_for_final <- function(panel, fc_state, split_id = max(SPLITS)) {
  # Uses the selected train split and produces a forecast for its target period.
  # For a real competition submission, adjust this function to the exact
  # forecast origin and target dates requested by Mosqlimate.
  train_col <- paste0("train_", split_id)
  target_col <- paste0("target_", split_id)
  train_raw <- panel[get(train_col) == TRUE]
  pred_raw <- panel[get(target_col) == TRUE]
  max_train_date <- max(train_raw$date, na.rm = TRUE)
  train_raw <- mask_training_seasons(train_raw, context = "final training")
  pred_raw <- pred_raw[date > max_train_date]

  engineered <- add_training_only_features(train_raw, pred_raw)
  train_dt <- engineered$train
  pred_dt <- use_forecast_climate_for_future(
    engineered$pred,
    fc_state,
    forecast_origin = max_train_date,
    split_id = split_id
  )
  pred_dt[, row_id := .I]

  expert_tables <- list()
  expert_samples <- list()
  for (ee in names(expert_registry)) {
    message_step("Final fit expert: ", ee)
    model <- expert_registry[[ee]]$fit(train_dt)
    pred <- expert_registry[[ee]]$predict(model, train_dt, pred_dt)
    expert_tables[[ee]] <- make_expert_prediction_table(pred_dt, ee, pred)
    expert_samples[[ee]] <- pred$samples
  }
  expert_long <- rbindlist(expert_tables, fill = TRUE)
  covar_cols <- unique(c("uf", "date", all_model_features))
  covars <- pred_dt[, covar_cols, with = FALSE]
  expert_long <- merge(expert_long, covars, by = c("uf", "date"), all.x = TRUE)
  list(expert_long = expert_long, expert_samples = expert_samples, pred_dt = pred_dt)
}

# --------------------------------- Main ----------------------------------- #

main <- function() {
  inputs <- read_inputs(DATA_DIR)
  prepared <- prepare_state_week_panel(inputs)
  panel <- prepared$panel
  fc_state <- prepared$forecast_climate_state

  message_step("Panel rows: ", nrow(panel), "; states: ", paste(sort(unique(panel$uf)), collapse = ", "))
  message_step("Date range: ", min(panel$date), " to ", max(panel$date))

  oof <- run_oof_pipeline(panel, fc_state)

  # Summary validation.
  bridge <- oof$bridge_oof
  if (nrow(bridge) > 0) {
    message_step("Bridge OOF MAE: ", round(score_mae(bridge$cases, bridge$pred), 3))
    message_step("Bridge OOF RMSE: ", round(score_rmse(bridge$cases, bridge$pred), 3))
    bridge[, wis := mapply(function(y, i) wis_single(y, bridge[i]), cases, seq_len(.N))]
    message_step("Bridge OOF mean WIS: ", round(mean(bridge$wis, na.rm = TRUE), 3))
    fwrite(
      bridge[, .(
        mae = mean(abs(cases - pred), na.rm = TRUE),
        rmse = sqrt(mean((cases - pred)^2, na.rm = TRUE)),
        wis = mean(wis, na.rm = TRUE)
      ), by = .(uf, split_id)],
      file.path(OUT_DIR, "bridge_oof_scores_by_state.csv")
    )
  }

  # Final template from last split target period.
  final_res <- fit_all_experts_for_final(panel, fc_state, split_id = max(SPLITS))
  weighted_final <- predict_bridge_weights(oof$bridge_models, final_res$expert_long)
  final_forecast <- combine_experts_with_weights(weighted_final, final_res$expert_samples)
  weighted_final_top3 <- apply_topk_bridge_weights(weighted_final, k = BRIDGE_TOP_K, temperature = BRIDGE_TEMPERATURE)
  final_forecast_top3 <- combine_experts_with_weights(weighted_final_top3, final_res$expert_samples)
  validate_forecast_rules(final_forecast, group_cols = "uf")
  validate_forecast_rules(final_forecast_top3, group_cols = "uf")
  fwrite(final_forecast, file.path(OUT_DIR, "final_forecast_template.csv"))
  fwrite(final_forecast_top3, file.path(OUT_DIR, "top3_final_forecast_template.csv"))
  fwrite(
    weighted_final[, .(row_id, uf, uf_code, date, epiweek, expert, expert_pred, pred_log_error, bridge_weight)],
    file.path(OUT_DIR, "final_bridge_weights.csv")
  )
  fwrite(
    weighted_final_top3[, .(row_id, uf, uf_code, date, epiweek, expert, expert_pred, pred_log_error, bridge_weight, error_rank)],
    file.path(OUT_DIR, "top3_final_bridge_weights.csv")
  )

  message_step("Done. Outputs written to: ", normalizePath(OUT_DIR))
}

if (identical(environment(), globalenv())) {
  main()
}
