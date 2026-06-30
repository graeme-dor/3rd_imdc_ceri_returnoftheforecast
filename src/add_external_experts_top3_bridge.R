# Add Graeme and Carlin validation predictions as extra experts
# -------------------------------------------------------------
# This script starts from an existing internal experiment folder, appends
# external validation predictions as extra experts, then reruns the Top-3
# bridge over the combined expert set.
#
# Default:
#   input:  output_peak_masked_irregular/
#   output: output_peak_masked_irregular_external_experts/
#
# Run:
#   Rscript add_external_experts_top3_bridge.R
#
# Then regenerate plots:
#   OUT_DIR=output_peak_masked_irregular_external_experts Rscript plot_top3_final_validation_outputs.R
#   OUT_DIR=output_peak_masked_irregular_external_experts Rscript plot_expert_model_validation_lines.R

options(stringsAsFactors = FALSE)

DATA_DIR <- "."
BASE_OUT_DIR <- Sys.getenv("BASE_OUT_DIR", "output_peak_masked_irregular")
OUT_DIR <- Sys.getenv("OUT_DIR", "output_peak_masked_irregular_external_experts")
CARLIN_DIR <- Sys.getenv("CARLIN_DIR", "Carlin_GNN")
GRAEME_DIR <- Sys.getenv("GRAEME_DIR", "Graeme_bayesian_nb_glmm_thermal")
GRAEME_ZIP <- Sys.getenv("GRAEME_ZIP", "Graeme_bayesian_nb_glmm_thermal.zip")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUT_DIR, "validation_predictions"), showWarnings = FALSE, recursive = TRUE)

BRIDGE_TOP_K <- 3
BRIDGE_TEMPERATURE <- 0.75
N_SAMPLES <- 400
set.seed(20260624)

required_pkgs <- c("data.table", "ranger")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    "Install missing packages before running:\n  install.packages(c(",
    paste(sprintf('"%s"', missing_pkgs), collapse = ", "),
    "))"
  )
}

library(data.table)
library(ranger)

message_step <- function(...) {
  message(format(Sys.time(), "%H:%M:%S"), " | ", ...)
}

quantile_probs <- c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
quantile_names <- c(
  "lower_95", "lower_90", "lower_80", "lower_50", "pred",
  "upper_50", "upper_80", "upper_90", "upper_95"
)

base_feature_cols <- c(
  "year", "week", "month",
  "sin52_1", "cos52_1", "sin52_2", "cos52_2",
  "log_population",
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

bridge_feature_cols <- c(base_feature_cols, hist_feature_cols, climate_feature_cols)

normalize_internal_names <- function(dt) {
  dt <- copy(dt)
  if ("year.x" %in% names(dt) && !"year" %in% names(dt)) setnames(dt, "year.x", "year")
  if ("week.x" %in% names(dt) && !"week" %in% names(dt)) setnames(dt, "week.x", "week")
  if ("macroregion_name.x" %in% names(dt) && !"macroregion_name" %in% names(dt)) {
    setnames(dt, "macroregion_name.x", "macroregion_name")
  }
  dt[]
}

complete_model_frame <- function(dt, features = bridge_feature_cols) {
  dt <- copy(dt)
  for (cc in features) {
    if (!cc %in% names(dt)) dt[, (cc) := NA]
  }
  char_cols <- names(dt)[vapply(dt, is.character, logical(1))]
  for (cc in char_cols) dt[, (cc) := factor(get(cc))]
  dt[]
}

make_sample_from_quantiles <- function(row, n = N_SAMPLES) {
  qs <- as.numeric(row[, ..quantile_names])
  qs[!is.finite(qs)] <- NA_real_
  if (all(is.na(qs))) return(rep(NA_real_, n))
  qs <- zoo_fill(qs)
  qs <- cummax(pmax(qs, 0))
  u <- runif(n)
  pmax(round(approx(quantile_probs, qs, xout = u, rule = 2, ties = "ordered")$y), 0)
}

zoo_fill <- function(x) {
  if (all(is.na(x))) return(x)
  ok <- which(!is.na(x))
  approx(ok, x[ok], xout = seq_along(x), rule = 2)$y
}

quantile_table <- function(samples) {
  qs <- stats::quantile(samples, probs = quantile_probs, na.rm = TRUE, names = FALSE, type = 8)
  setNames(as.list(qs), quantile_names)
}

season_label_from_split <- function(split_id) {
  split_id <- as.integer(split_id)
  start_year <- 2021L + split_id
  paste0(start_year, "-", start_year + 1L)
}

read_internal_experts <- function() {
  path <- file.path(BASE_OUT_DIR, "expert_oof_predictions.csv")
  if (!file.exists(path)) stop("Missing internal expert file: ", path)
  dt <- normalize_internal_names(fread(path))
  dt[, date := as.Date(date)]
  dt[]
}

make_row_lookup <- function(internal) {
  non_unique <- internal[, .N, by = .(split_id, uf, date)][N == 0]
  if (nrow(non_unique) > 0) stop("Unexpected missing row lookup keys.")
  id_cols <- c("split_id", "row_id", "uf", "uf_code", "date", "epiweek", "year", "week",
               "cases", "population", "macroregion_name")
  feature_cols <- intersect(bridge_feature_cols, names(internal))
  lookup_cols <- unique(c(id_cols, feature_cols))
  lookup <- unique(internal[, lookup_cols, with = FALSE], by = c("split_id", "uf", "date"))
  lookup[]
}

read_carlin_expert <- function(row_lookup) {
  files <- list.files(CARLIN_DIR, pattern = "\\.csv$", full.names = TRUE)
  if (length(files) == 0) {
    warning("No Carlin CSV files found in ", CARLIN_DIR)
    return(NULL)
  }
  dt <- rbindlist(lapply(files, fread), fill = TRUE)
  dt[, date := as.Date(date)]
  out <- dt[, .(
    split_id = as.integer(split_id),
    uf,
    date,
    expert = "carlin_gnn",
    expert_pred = as.numeric(q_0.5),
    lower_95 = as.numeric(q_0.025),
    lower_90 = as.numeric(q_0.05),
    lower_80 = as.numeric(q_0.1),
    lower_50 = as.numeric(q_0.25),
    pred = as.numeric(q_0.5),
    upper_50 = as.numeric(q_0.75),
    upper_80 = as.numeric(q_0.9),
    upper_90 = as.numeric(q_0.95),
    upper_95 = as.numeric(q_0.975)
  )]
  out <- merge(row_lookup, out, by = c("split_id", "uf", "date"))
  message_step("Carlin external expert rows aligned: ", nrow(out))
  out[]
}

read_graeme_expert <- function(row_lookup) {
  files <- if (dir.exists(GRAEME_DIR)) {
    list.files(GRAEME_DIR, pattern = "validation_round_.*\\.csv$", full.names = TRUE)
  } else {
    character()
  }
  if (length(files) == 0 && file.exists(GRAEME_ZIP)) {
    td <- tempfile("graeme_unzip_")
    dir.create(td)
    unzip(GRAEME_ZIP, exdir = td)
    files <- list.files(file.path(td, "bayesian_nb_glmm_thermal"), pattern = "validation_round_.*\\.csv$", full.names = TRUE)
  }
  if (length(files) == 0) {
    warning("No Graeme validation_round CSVs found in ", GRAEME_DIR, " or ", GRAEME_ZIP)
    return(NULL)
  }
  dt <- rbindlist(lapply(files, function(f) {
    x <- fread(f)
    x[, split_id := as.integer(sub(".*validation_round_(\\d+)\\.csv$", "\\1", basename(f)))]
    x
  }), fill = TRUE)
  dt[, date := as.Date(date)]
  code_map <- unique(row_lookup[, .(uf_code, uf)])
  dt <- merge(dt, code_map, by.x = "adm_1", by.y = "uf_code", all.x = TRUE)
  out <- dt[!is.na(uf), .(
    split_id = as.integer(split_id),
    uf,
    date,
    expert = "graeme_bayesian_nb_glmm_thermal",
    expert_pred = as.numeric(pred),
    lower_95 = as.numeric(lower_95),
    lower_90 = as.numeric(lower_90),
    lower_80 = as.numeric(lower_80),
    lower_50 = as.numeric(lower_50),
    pred = as.numeric(pred),
    upper_50 = as.numeric(upper_50),
    upper_80 = as.numeric(upper_80),
    upper_90 = as.numeric(upper_90),
    upper_95 = as.numeric(upper_95)
  )]
  out <- merge(row_lookup, out, by = c("split_id", "uf", "date"))
  message_step("Graeme external expert rows aligned: ", nrow(out))
  out[]
}

fit_bridge_error_models <- function(expert_oof) {
  dt <- copy(expert_oof)
  dt[, abs_error := abs(cases - expert_pred)]
  dt[, log_abs_error := log1p(abs_error)]
  dt <- dt[is.finite(log_abs_error) & is.finite(expert_pred)]
  dt <- complete_model_frame(dt, features = c(bridge_feature_cols, "expert_pred"))

  features <- intersect(bridge_feature_cols, names(dt))
  features <- features[vapply(dt[, ..features], function(x) is.numeric(x) || is.factor(x), logical(1))]
  f <- as.formula(paste("log_abs_error ~ expert_pred +", paste(features, collapse = " + ")))

  models <- list()
  for (ee in sort(unique(as.character(dt$expert)))) {
    sub <- dt[as.character(expert) == ee]
    if (nrow(sub) < 20) sub <- dt
    models[[ee]] <- ranger::ranger(
      formula = f,
      data = sub,
      num.trees = 400,
      min.node.size = 8,
      seed = 20260624
    )
  }
  models
}

predict_bridge_weights <- function(error_models, expert_pred_long, temperature = BRIDGE_TEMPERATURE) {
  dt <- complete_model_frame(expert_pred_long, features = c(bridge_feature_cols, "expert_pred"))
  dt[, pred_log_error := NA_real_]

  for (ee in names(error_models)) {
    idx <- as.character(dt$expert) == ee
    if (any(idx)) {
      dt[idx, pred_log_error := as.numeric(predict(error_models[[ee]], data = dt[idx])$predictions)]
    }
  }
  fallback <- median(dt$pred_log_error[is.finite(dt$pred_log_error)], na.rm = TRUE)
  if (!is.finite(fallback)) fallback <- 1
  dt[!is.finite(pred_log_error), pred_log_error := fallback]

  dt[, error_rank := frank(pred_log_error, ties.method = "first"), by = .(split_id, uf, date)]
  dt[error_rank > BRIDGE_TOP_K, bridge_weight := 0]
  dt[error_rank <= BRIDGE_TOP_K, topk_score := exp(-pred_log_error / temperature)]
  dt[error_rank <= BRIDGE_TOP_K, bridge_weight := topk_score / sum(topk_score), by = .(split_id, uf, date)]
  dt[!is.finite(bridge_weight), bridge_weight := 0]
  dt[, topk_score := NULL]
  dt[]
}

combine_weighted_quantile_samples <- function(weighted_long) {
  keys <- unique(weighted_long[, .(split_id, row_id, uf, uf_code, date, epiweek, year, week, cases, population, macroregion_name)])
  setorder(keys, split_id, row_id)
  out <- vector("list", nrow(keys))

  for (ii in seq_len(nrow(keys))) {
    key <- keys[ii]
    rows <- weighted_long[split_id == key$split_id & row_id == key$row_id & bridge_weight > 0]
    if (nrow(rows) == 0) {
      out[[ii]] <- cbind(key, as.data.table(as.list(setNames(rep(NA_real_, length(quantile_names)), quantile_names))))
      next
    }
    chosen <- sample(seq_len(nrow(rows)), size = N_SAMPLES, replace = TRUE, prob = rows$bridge_weight)
    draws <- numeric(N_SAMPLES)
    for (jj in seq_len(nrow(rows))) {
      idx <- which(chosen == jj)
      if (length(idx) > 0) {
        draws[idx] <- sample(make_sample_from_quantiles(rows[jj], n = N_SAMPLES), size = length(idx), replace = TRUE)
      }
    }
    out[[ii]] <- cbind(key, as.data.table(quantile_table(draws)))
  }
  rbindlist(out, fill = TRUE)
}

export_validation_season_csvs <- function(pred_dt, prefix = "Top3_bridge") {
  dt <- copy(pred_dt)
  dt[, season := season_label_from_split(split_id)]
  export_cols <- c(
    "uf", "season", "pred",
    "lower_50", "upper_50",
    "lower_80", "upper_80",
    "lower_90", "upper_90",
    "lower_95", "upper_95",
    "date"
  )
  for (ss in sort(unique(dt$season))) {
    out <- dt[season == ss, ..export_cols]
    setorder(out, uf, date)
    fwrite(out, file.path(OUT_DIR, "validation_predictions", paste0(prefix, "_validation_", ss, ".csv")))
  }
}

main <- function() {
  internal <- read_internal_experts()
  row_lookup <- make_row_lookup(internal)
  carlin <- read_carlin_expert(row_lookup)
  graeme <- read_graeme_expert(row_lookup)

  combined <- rbindlist(list(internal, carlin, graeme), fill = TRUE)
  combined[, date := as.Date(date)]
  combined <- combined[is.finite(expert_pred)]
  fwrite(combined, file.path(OUT_DIR, "expert_oof_predictions.csv"))

  message_step("Combined expert rows: ", nrow(combined))
  message_step("Experts: ", paste(sort(unique(combined$expert)), collapse = ", "))

  pred_list <- list()
  weight_list <- list()
  for (sid in sort(unique(combined$split_id))) {
    message_step("Fitting external-inclusive bridge for held-out split ", sid)
    train <- combined[split_id != sid]
    test <- combined[split_id == sid]
    models <- fit_bridge_error_models(train)
    weighted <- predict_bridge_weights(models, test)
    weight_list[[as.character(sid)]] <- weighted[, .(
      split_id, row_id, uf, uf_code, date, epiweek, expert, expert_pred,
      pred_log_error, bridge_weight, error_rank
    )]
    pred_list[[as.character(sid)]] <- combine_weighted_quantile_samples(weighted)
  }

  top3_pred <- rbindlist(pred_list, fill = TRUE)
  top3_weights <- rbindlist(weight_list, fill = TRUE)
  fwrite(top3_pred, file.path(OUT_DIR, "top3_bridge_oof_predictions.csv"))
  fwrite(top3_weights, file.path(OUT_DIR, "top3_bridge_oof_weights.csv"))
  export_validation_season_csvs(top3_pred)

  message_step("Done. External-inclusive Top-3 bridge written to: ", normalizePath(OUT_DIR))
}

if (identical(environment(), globalenv())) {
  main()
}
