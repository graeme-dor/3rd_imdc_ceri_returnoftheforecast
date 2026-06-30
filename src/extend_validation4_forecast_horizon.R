# Extend the final 2025-2026 validation forecast to the full season horizon.
#
# Current internal Top-9 validation rows stop on 2026-03-08 because dengue.csv
# target_4 and forecasting_climate_delta_adjusted_weekly.csv round_4 stop there.
# Carlin and Graeme provide full-horizon split-4 predictions through 2026-10-04.
#
# Extension rule:
# - Keep existing Top-9 rows through 2026-03-08 unchanged.
# - For 2026-03-15 onward, combine Carlin and Graeme using each state's mean
#   Top-9 bridge weights from the observed split-4 overlap, renormalized across
#   those two external experts.
# - ES is not present in Carlin/Graeme, so ES future rows use Betty's Bayesian
#   2025-2026 forecast as an explicit fallback. A diagnostic source file is
#   written so this is not hidden.

options(stringsAsFactors = FALSE)

OUT_DIR <- Sys.getenv("OUT_DIR", "output_final_top9_temp04_forecast_climate")
SUBMISSION_DIR <- file.path(OUT_DIR, "validation_predictions_submission_format")
CURRENT_FILE <- file.path(SUBMISSION_DIR, "Top9_temp04_bridge_validation_2025-2026.csv")
BACKUP_FILE <- file.path(SUBMISSION_DIR, "Top9_temp04_bridge_validation_2025-2026_to_2026-03-08_backup.csv")
DIAG_FILE <- file.path(SUBMISSION_DIR, "Top9_temp04_bridge_validation_2025-2026_extended_sources.csv")

CARLIN_FILE <- "Carlin_GNN/model11_fc_best_evaluation_split_4.csv"
GRAEME_FILE <- "Graeme_bayesian_nb_glmm_thermal/validation_round_4.csv"
BETTY_ES_FALLBACK_FILE <- "Betty_predictions/Bayesian_forecast_2025_2026.csv"

N_SAMPLES <- 2000L
set.seed(20260630)

required_pkgs <- c("data.table")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Install missing packages: install.packages(c(", paste(sprintf('\"%s\"', missing_pkgs), collapse = ", "), "))")
}

library(data.table)

submission_cols <- c(
  "uf", "date", "pred",
  "lower_50", "upper_50",
  "lower_80", "upper_80",
  "lower_90", "upper_90",
  "lower_95", "upper_95"
)

quantile_probs <- c(0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
quantile_cols <- c(
  "lower_95", "lower_90", "lower_80", "lower_50", "pred",
  "upper_50", "upper_80", "upper_90", "upper_95"
)

message_step <- function(...) message(format(Sys.time(), "%H:%M:%S"), " | ", ...)

fill_quantiles <- function(x) {
  if (all(is.na(x))) return(x)
  ok <- which(!is.na(x))
  approx(ok, x[ok], xout = seq_along(x), rule = 2)$y
}

sample_from_quantiles <- function(row, prefix = "", n = N_SAMPLES) {
  cols <- paste0(prefix, quantile_cols)
  qs <- as.numeric(row[, ..cols])
  qs[!is.finite(qs)] <- NA_real_
  qs <- fill_quantiles(qs)
  qs <- cummax(pmax(qs, 0))
  u <- runif(n)
  pmax(round(approx(quantile_probs, qs, xout = u, rule = 2, ties = "ordered")$y), 0)
}

quantile_table <- function(samples) {
  qs <- stats::quantile(samples, probs = quantile_probs, na.rm = TRUE, names = FALSE, type = 8)
  as.list(setNames(qs, quantile_cols))
}

validate_submission <- function(dt) {
  missing <- setdiff(submission_cols, names(dt))
  if (length(missing) > 0) stop("Missing columns: ", paste(missing, collapse = ", "))
  if (!all(as.POSIXlt(dt$date)$wday == 0)) stop("Non-Sunday dates found.")
  if (any(dt$pred < 0, na.rm = TRUE)) stop("Negative predictions found.")
  bad <- dt[
    lower_95 > lower_90 |
      lower_90 > lower_80 |
      lower_80 > lower_50 |
      lower_50 > pred |
      pred > upper_50 |
      upper_50 > upper_80 |
      upper_80 > upper_90 |
      upper_90 > upper_95
  ]
  if (nrow(bad) > 0) stop("Non-nested intervals found.")
  invisible(TRUE)
}

read_carlin <- function() {
  dt <- fread(CARLIN_FILE)
  dt[, date := as.Date(date)]
  dt[, .(
    uf, date,
    carlin_lower_95 = as.numeric(q_0.025),
    carlin_lower_90 = as.numeric(q_0.05),
    carlin_lower_80 = as.numeric(q_0.1),
    carlin_lower_50 = as.numeric(q_0.25),
    carlin_pred = as.numeric(q_0.5),
    carlin_upper_50 = as.numeric(q_0.75),
    carlin_upper_80 = as.numeric(q_0.9),
    carlin_upper_90 = as.numeric(q_0.95),
    carlin_upper_95 = as.numeric(q_0.975)
  )]
}

read_graeme <- function() {
  dt <- fread(GRAEME_FILE)
  dt[, date := as.Date(date)]
  code_map <- unique(fread("map_regional_health.csv")[, .(uf_code, uf)])
  dt <- merge(dt, code_map, by.x = "adm_1", by.y = "uf_code", all.x = TRUE)
  dt[!is.na(uf), .(
    uf, date,
    graeme_lower_95 = as.numeric(lower_95),
    graeme_lower_90 = as.numeric(lower_90),
    graeme_lower_80 = as.numeric(lower_80),
    graeme_lower_50 = as.numeric(lower_50),
    graeme_pred = as.numeric(pred),
    graeme_upper_50 = as.numeric(upper_50),
    graeme_upper_80 = as.numeric(upper_80),
    graeme_upper_90 = as.numeric(upper_90),
    graeme_upper_95 = as.numeric(upper_95)
  )]
}

state_external_weights <- function() {
  w <- fread(file.path(OUT_DIR, "top9_temp04_bridge_oof_weights.csv"))
  w <- w[split_id == 4 & expert %in% c("carlin_gnn", "graeme_bayesian_nb_glmm_thermal")]
  state_w <- w[, .(mean_weight = mean(bridge_weight, na.rm = TRUE)), by = .(uf, expert)]
  state_w <- dcast(state_w, uf ~ expert, value.var = "mean_weight")
  setnames(state_w, c("carlin_gnn", "graeme_bayesian_nb_glmm_thermal"), c("w_carlin", "w_graeme"))
  global <- state_w[, .(
    w_carlin = mean(w_carlin, na.rm = TRUE),
    w_graeme = mean(w_graeme, na.rm = TRUE)
  )]
  state_w[is.na(w_carlin), w_carlin := global$w_carlin]
  state_w[is.na(w_graeme), w_graeme := global$w_graeme]
  state_w[, weight_sum := w_carlin + w_graeme]
  state_w[weight_sum <= 0 | !is.finite(weight_sum), `:=`(w_carlin = 0.5, w_graeme = 0.5, weight_sum = 1)]
  state_w[, `:=`(w_carlin = w_carlin / weight_sum, w_graeme = w_graeme / weight_sum)]
  state_w[, weight_sum := NULL]
  state_w[]
}

combine_external_rows <- function(dt) {
  out <- vector("list", nrow(dt))
  for (ii in seq_len(nrow(dt))) {
    row <- dt[ii]
    carlin_draws <- sample_from_quantiles(row, prefix = "carlin_")
    graeme_draws <- sample_from_quantiles(row, prefix = "graeme_")
    choose_carlin <- rbinom(N_SAMPLES, 1, row$w_carlin)
    draws <- ifelse(choose_carlin == 1, carlin_draws, graeme_draws)
    out[[ii]] <- cbind(row[, .(uf, date)], as.data.table(quantile_table(draws)))
  }
  rbindlist(out)
}

main <- function() {
  current <- fread(CURRENT_FILE)
  current[, date := as.Date(date)]
  last_current_date <- max(current$date)
  if (!file.exists(BACKUP_FILE)) fwrite(current, BACKUP_FILE)

  carlin <- read_carlin()
  graeme <- read_graeme()
  weights <- state_external_weights()

  future_external <- merge(carlin, graeme, by = c("uf", "date"))
  future_external <- merge(future_external, weights, by = "uf", all.x = TRUE)
  future_external <- future_external[date > last_current_date]
  future_external <- combine_external_rows(future_external)
  future_external[, source := "carlin_graeme_bridge_weighted_extension"]

  betty_es <- fread(BETTY_ES_FALLBACK_FILE)
  betty_es[, date := as.Date(date)]
  betty_es <- betty_es[uf == "ES" & date > last_current_date, ..submission_cols]
  if (nrow(betty_es) > 0) {
    betty_es[, source := "betty_bayesian_fallback_for_ES"]
  }

  current_diag <- copy(current)
  current_diag[, source := "original_top9_temp04_bridge"]

  extended_diag <- rbindlist(list(
    current_diag,
    future_external[, c(submission_cols, "source"), with = FALSE],
    betty_es[, c(submission_cols, "source"), with = FALSE]
  ), fill = TRUE)
  setorder(extended_diag, uf, date)

  extended <- extended_diag[, ..submission_cols]
  validate_submission(extended)

  fwrite(extended, CURRENT_FILE)
  fwrite(extended_diag, DIAG_FILE)

  message_step("Backed up original file to: ", normalizePath(BACKUP_FILE))
  message_step("Extended validation-4 file written to: ", normalizePath(CURRENT_FILE))
  message_step("Source diagnostic written to: ", normalizePath(DIAG_FILE))
  message_step("New date range: ", min(extended$date), " to ", max(extended$date), "; rows: ", nrow(extended))
}

if (identical(environment(), globalenv())) main()

