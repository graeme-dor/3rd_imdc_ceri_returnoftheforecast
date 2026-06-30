# Final Top-9 bridge, temperature 0.4
# -----------------------------------
# Creates the final validation outputs for the selected bridge:
#   - all external-inclusive experts
#   - Top-9 selection, i.e. all available experts
#   - softmax temperature = 0.4
#   - original mixture-style interval aggregation
#
# Run:
#   Rscript create_final_top9_temp04_outputs.R
#   OUT_DIR=output_final_top9_temp04 MODEL_LABEL="Top-9 bridge, temperature 0.4" Rscript plot_top3_final_validation_outputs.R

options(stringsAsFactors = FALSE)

BASE_OUT_DIR <- Sys.getenv("BASE_OUT_DIR", "output_peak_masked_irregular_external_experts")
OUT_DIR <- Sys.getenv("OUT_DIR", "output_final_top9_temp04")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUT_DIR, "validation_predictions"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUT_DIR, "validation_predictions_scored"), showWarnings = FALSE, recursive = TRUE)

required_pkgs <- c("data.table")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Install missing packages: install.packages(c(", paste(sprintf('\"%s\"', missing_pkgs), collapse = ", "), "))")
}

library(data.table)

set.seed(20260625)

K <- 9L
TEMPERATURE <- 0.4
N_SAMPLES <- 2000L

quantile_probs <- c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
quantile_cols <- c("lower_95", "lower_90", "lower_80", "lower_50", "pred",
                   "upper_50", "upper_80", "upper_90", "upper_95")

score_wis <- function(y, dt) {
  intervals <- list(
    list(alpha = 0.50, lo = dt$lower_50, hi = dt$upper_50),
    list(alpha = 0.20, lo = dt$lower_80, hi = dt$upper_80),
    list(alpha = 0.10, lo = dt$lower_90, hi = dt$upper_90),
    list(alpha = 0.05, lo = dt$lower_95, hi = dt$upper_95)
  )
  median_score <- abs(y - dt$pred)
  interval_scores <- vapply(intervals, function(z) {
    (z$hi - z$lo) +
      (2 / z$alpha) * (z$lo - y) * (y < z$lo) +
      (2 / z$alpha) * (y - z$hi) * (y > z$hi)
  }, numeric(1))
  (0.5 * median_score + sum((c(0.50, 0.20, 0.10, 0.05) / 2) * interval_scores)) /
    (0.5 + sum(c(0.50, 0.20, 0.10, 0.05) / 2))
}

message_step <- function(...) message(format(Sys.time(), "%H:%M:%S"), " | ", ...)

zoo_fill <- function(x) {
  if (all(is.na(x))) return(x)
  ok <- which(!is.na(x))
  approx(ok, x[ok], xout = seq_along(x), rule = 2)$y
}

make_sample_from_quantiles <- function(row, n = N_SAMPLES) {
  qs <- as.numeric(row[, ..quantile_cols])
  qs[!is.finite(qs)] <- NA_real_
  if (all(is.na(qs))) return(rep(NA_real_, n))
  qs <- zoo_fill(qs)
  qs <- cummax(pmax(qs, 0))
  u <- runif(n)
  pmax(round(approx(quantile_probs, qs, xout = u, rule = 2, ties = "ordered")$y), 0)
}

quantile_table <- function(samples) {
  qs <- stats::quantile(samples, probs = quantile_probs, na.rm = TRUE, names = FALSE, type = 8)
  as.data.table(as.list(setNames(qs, quantile_cols)))
}

season_label_from_split <- function(split_id) {
  start_year <- 2021L + as.integer(split_id)
  paste0(start_year, "-", start_year + 1L)
}

read_weighted_experts <- function() {
  expert <- fread(file.path(BASE_OUT_DIR, "expert_oof_predictions.csv"))
  weights <- fread(file.path(BASE_OUT_DIR, "top3_bridge_oof_weights.csv"))
  expert[, date := as.Date(date)]
  weights[, date := as.Date(date)]
  merge(
    expert,
    weights[, .(split_id, row_id, uf, date, expert, pred_log_error, error_rank)],
    by = c("split_id", "row_id", "uf", "date", "expert"),
    all.x = FALSE,
    allow.cartesian = TRUE
  )[error_rank <= K]
}

key_and_covariate_cols <- function(dt) {
  excluded <- c(
    "expert", "expert_pred",
    "lower_95", "lower_90", "lower_80", "lower_50", "pred",
    "upper_50", "upper_80", "upper_90", "upper_95",
    "pred_log_error", "error_rank", "topk_score", "bridge_weight"
  )
  keep <- setdiff(names(dt), excluded)
  keep <- keep[!grepl("\\.(x|y)$", keep)]
  unique(c(
    "split_id", "row_id", "uf", "uf_code", "date", "epiweek", "week",
    "cases", "population", "macroregion_name",
    keep
  ))
}

collapse_covariates <- function(dt) {
  key_cols <- c("split_id", "row_id", "uf", "date")
  cov_cols <- setdiff(intersect(key_and_covariate_cols(dt), names(dt)), key_cols)
  cov_cols <- setdiff(cov_cols, c("cases", "uf_code", "epiweek", "week", "population", "macroregion_name"))
  if (length(cov_cols) == 0) return(unique(dt[, ..key_cols]))
  first_non_missing <- function(v) {
    ok <- which(!is.na(v))
    if (length(ok) == 0) return(v[NA_integer_][1])
    v[ok[1]]
  }
  collapsed <- dt[, lapply(.SD, first_non_missing), by = key_cols, .SDcols = cov_cols]
  collapsed[]
}

combine_final_bridge <- function(weighted) {
  dt <- copy(weighted)
  dt[, topk_score := exp(-pred_log_error / TEMPERATURE)]
  dt[, bridge_weight := topk_score / sum(topk_score), by = .(split_id, row_id, uf, date)]
  dt[!is.finite(bridge_weight), bridge_weight := 0]

  keys <- unique(dt[, .(split_id, row_id, uf, uf_code, date, epiweek, week, cases, population, macroregion_name)])
  setorder(keys, split_id, row_id)
  out <- vector("list", nrow(keys))

  for (ii in seq_len(nrow(keys))) {
    if (ii %% 500 == 0) message_step("Combining row ", ii, " / ", nrow(keys))
    key <- keys[ii]
    rows <- dt[split_id == key$split_id & row_id == key$row_id & bridge_weight > 0]
    if (nrow(rows) == 0) next
    chosen <- sample(seq_len(nrow(rows)), size = N_SAMPLES, replace = TRUE, prob = rows$bridge_weight)
    draws <- numeric(N_SAMPLES)
    for (jj in seq_len(nrow(rows))) {
      idx <- which(chosen == jj)
      if (length(idx) > 0) {
        draws[idx] <- sample(make_sample_from_quantiles(rows[jj], n = N_SAMPLES), size = length(idx), replace = TRUE)
      }
    }
    out[[ii]] <- cbind(key, quantile_table(draws))
  }

  pred <- rbindlist(out, fill = TRUE)
  covars <- collapse_covariates(dt)
  pred <- merge(pred, covars, by = c("split_id", "row_id", "uf", "date"), all.x = TRUE)
  pred[, forecast_climate_round := paste0("round_", split_id)]
  pred[, forecast_climate_used := TRUE]
  pred[, model := "top9_temp04_bridge"]
  weights <- dt[, .(
    split_id, row_id, uf, uf_code, date, epiweek, expert, expert_pred,
    pred_log_error, bridge_weight, error_rank
  )]
  list(predictions = pred, weights = weights)
}

add_row_scores <- function(pred_dt) {
  dt <- copy(pred_dt)
  dt[, actual_cases := cases]
  dt[, mae := abs(actual_cases - pred)]
  dt[, rmse := sqrt((actual_cases - pred)^2)]
  dt[, wis := vapply(seq_len(.N), function(i) score_wis(actual_cases[i], dt[i]), numeric(1))]
  dt[, normalized_wis := wis / pmax(abs(actual_cases), 1)]
  dt[]
}

write_score_summaries <- function(pred_dt) {
  dt <- copy(pred_dt)
  dt[, season := season_label_from_split(split_id)]
  overall <- dt[, .(
    mae = mean(mae, na.rm = TRUE),
    rmse = sqrt(mean((actual_cases - pred)^2, na.rm = TRUE)),
    wis = mean(wis, na.rm = TRUE),
    normalized_wis = mean(normalized_wis, na.rm = TRUE)
  )]
  by_split <- dt[, .(
    mae = mean(mae, na.rm = TRUE),
    rmse = sqrt(mean((actual_cases - pred)^2, na.rm = TRUE)),
    wis = mean(wis, na.rm = TRUE),
    normalized_wis = mean(normalized_wis, na.rm = TRUE)
  ), by = .(split_id, season)]
  by_state_split <- dt[, .(
    mae = mean(mae, na.rm = TRUE),
    rmse = sqrt(mean((actual_cases - pred)^2, na.rm = TRUE)),
    wis = mean(wis, na.rm = TRUE),
    normalized_wis = mean(normalized_wis, na.rm = TRUE)
  ), by = .(uf, split_id, season)]
  fwrite(overall, file.path(OUT_DIR, "top9_temp04_bridge_scores_overall.csv"))
  fwrite(overall, file.path(OUT_DIR, "final_top9_temp04_scores_overall.csv"))
  fwrite(by_split, file.path(OUT_DIR, "top9_temp04_bridge_scores_by_split.csv"))
  fwrite(by_state_split, file.path(OUT_DIR, "top9_temp04_bridge_scores_by_state_season.csv"))

  # Compatibility names used by existing plotting scripts.
  fwrite(overall, file.path(OUT_DIR, "top3_bridge_scores_overall.csv"))
  fwrite(by_state_split, file.path(OUT_DIR, "top3_bridge_scores_by_state_season.csv"))
  fwrite(
    by_state_split[, .(
      mae = mean(mae, na.rm = TRUE),
      rmse = sqrt(mean(rmse^2, na.rm = TRUE)),
      wis = mean(wis, na.rm = TRUE),
      normalized_wis = mean(normalized_wis, na.rm = TRUE)
    ), by = uf],
    file.path(OUT_DIR, "top3_bridge_scores_by_state.csv")
  )
}

export_validation_csvs <- function(pred_dt) {
  dt <- copy(pred_dt)
  dt[, season := season_label_from_split(split_id)]
  dt[, state := uf]
  covariate_cols <- grep(
    "^(clim_|hist_|state_|region_|sync_|sin52_|cos52_|prop_|n_municipalities|enso|iod|pdo|log_population|population|week|epiweek|forecast_climate_)",
    names(dt),
    value = TRUE
  )
  covariate_cols <- setdiff(covariate_cols, c("state", "state.x", "state.y"))
  export_cols <- c(
    "state", "date", "season", "split_id", "actual_cases",
    covariate_cols,
    "pred",
    "lower_50", "upper_50",
    "lower_80", "upper_80",
    "lower_90", "upper_90",
    "lower_95", "upper_95",
    "wis", "mae", "rmse"
  )
  export_cols <- unique(intersect(export_cols, names(dt)))
  for (ss in sort(unique(dt$season))) {
    out <- dt[season == ss, ..export_cols]
    setorder(out, state, date)
    fwrite(out, file.path(OUT_DIR, "validation_predictions", paste0("Top9_temp04_bridge_validation_", ss, ".csv")))
    fwrite(out, file.path(OUT_DIR, "validation_predictions_scored", paste0("Top9_temp04_bridge_validation_scored_", ss, ".csv")))
  }
}

main <- function() {
  weighted <- read_weighted_experts()
  message_step("Experts included: ", paste(sort(unique(weighted$expert)), collapse = ", "))
  final <- combine_final_bridge(weighted)
  final$predictions <- add_row_scores(final$predictions)

  fwrite(weighted, file.path(OUT_DIR, "expert_oof_predictions.csv"))
  fwrite(final$predictions, file.path(OUT_DIR, "top9_temp04_bridge_oof_predictions.csv"))
  fwrite(final$predictions, file.path(OUT_DIR, "final_top9_temp04_predictions.csv"))
  fwrite(final$weights, file.path(OUT_DIR, "top9_temp04_bridge_oof_weights.csv"))
  fwrite(final$weights, file.path(OUT_DIR, "final_top9_temp04_weights.csv"))

  # Compatibility copies for the existing plotting/scoring script.
  fwrite(final$predictions, file.path(OUT_DIR, "top3_bridge_oof_predictions.csv"))
  fwrite(final$weights, file.path(OUT_DIR, "top3_bridge_oof_weights.csv"))

  write_score_summaries(final$predictions)
  export_validation_csvs(final$predictions)
  message_step("Done. Final Top-9 temp 0.4 outputs written to: ", normalizePath(OUT_DIR))
}

if (identical(environment(), globalenv())) main()
