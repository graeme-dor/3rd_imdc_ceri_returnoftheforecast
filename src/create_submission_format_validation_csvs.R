# Create clean validation CSVs in the exact prediction format:
# date,pred,lower_50,upper_50,lower_80,upper_80,lower_90,upper_90,lower_95,upper_95
#
# The final Top-9 validation files contain all states plus covariates/scores.
# This script writes one submission-style file per validation season, with the
# state/UF column first.

options(stringsAsFactors = FALSE)

OUT_DIR <- Sys.getenv("OUT_DIR", "output_final_top9_temp04_forecast_climate")
IN_DIR <- file.path(OUT_DIR, "validation_predictions")
SUBMISSION_DIR <- file.path(OUT_DIR, "validation_predictions_submission_format")

dir.create(SUBMISSION_DIR, showWarnings = FALSE, recursive = TRUE)

required_pkgs <- c("data.table")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Install missing packages: install.packages(c(", paste(sprintf('\"%s\"', missing_pkgs), collapse = ", "), "))")
}

library(data.table)

message_step <- function(...) message(format(Sys.time(), "%H:%M:%S"), " | ", ...)

submission_cols <- c(
  "uf", "date", "pred",
  "lower_50", "upper_50",
  "lower_80", "upper_80",
  "lower_90", "upper_90",
  "lower_95", "upper_95"
)

validate_submission_format <- function(dt, file_label) {
  missing_cols <- setdiff(submission_cols, names(dt))
  if (length(missing_cols) > 0) {
    stop(file_label, " is missing columns: ", paste(missing_cols, collapse = ", "))
  }

  if (any(is.na(dt$date))) stop(file_label, " has missing dates.")
  if (!all(as.POSIXlt(dt$date)$wday == 0)) stop(file_label, " has non-Sunday dates.")
  if (any(dt$pred < 0, na.rm = TRUE)) stop(file_label, " has negative predictions.")

  bad_nested <- dt[
    lower_95 > lower_90 |
      lower_90 > lower_80 |
      lower_80 > lower_50 |
      lower_50 > pred |
      pred > upper_50 |
      upper_50 > upper_80 |
      upper_80 > upper_90 |
      upper_90 > upper_95
  ]
  if (nrow(bad_nested) > 0) {
    stop(file_label, " has non-nested prediction intervals.")
  }

  invisible(TRUE)
}

main <- function() {
  files <- list.files(IN_DIR, pattern = "^Top9_temp04_bridge_validation_.*\\.csv$", full.names = TRUE)
  if (length(files) == 0) stop("No Top-9 validation files found in: ", IN_DIR)

  written <- character()
  for (ff in files) {
    dt <- fread(ff)
    dt[, date := as.Date(date)]
    if (!"state" %in% names(dt)) stop("Expected a state column in ", ff)
    season <- sub("^Top9_temp04_bridge_validation_(.*)\\.csv$", "\\1", basename(ff))
    dt[, uf := state]

    out <- dt[, ..submission_cols]
    setorder(out, uf, date)
    validate_submission_format(out, season)

    file_out <- file.path(
      SUBMISSION_DIR,
      paste0("Top9_temp04_bridge_validation_", season, ".csv")
    )
    fwrite(out, file_out)
    written <- c(written, file_out)
  }

  manifest <- data.table(
    file = basename(written),
    path = written
  )
  fwrite(manifest, file.path(SUBMISSION_DIR, "manifest.csv"))
  message_step("Wrote ", length(written), " submission-format validation files to: ", normalizePath(SUBMISSION_DIR))
}

if (identical(environment(), globalenv())) main()
