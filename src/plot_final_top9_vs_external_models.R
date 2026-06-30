# Final Top-9 bridge versus external models
# -----------------------------------------
# Compares the selected final bridge:
#   Top-9 bridge, temperature = 0.4
# against Carlin GNN and Graeme Bayesian NB GLMM.
#
# Outputs:
#   output_final_top9_temp04/plots/external_comparison/
#     final_top9_vs_external_overall.png/pdf
#     final_top9_vs_external_by_validation_split.png/pdf
#     final_top9_vs_external_scores_overall.csv
#     final_top9_vs_external_scores_by_split.csv
#
# Run:
#   Rscript plot_final_top9_vs_external_models.R

options(stringsAsFactors = FALSE)

FINAL_DIR <- Sys.getenv("FINAL_DIR", "output_final_top9_temp04")
CARLIN_DIR <- Sys.getenv("CARLIN_DIR", "Carlin_GNN")
GRAEME_DIR <- Sys.getenv("GRAEME_DIR", "Graeme_bayesian_nb_glmm_thermal")
GRAEME_ZIP <- Sys.getenv("GRAEME_ZIP", "Graeme_bayesian_nb_glmm_thermal.zip")
PLOT_DIR <- file.path(FINAL_DIR, "plots", "external_comparison")
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

required_pkgs <- c("data.table", "ggplot2", "scales")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Install missing packages: install.packages(c(", paste(sprintf('\"%s\"', missing_pkgs), collapse = ", "), "))")
}

library(data.table)
library(ggplot2)
library(scales)

interval_score <- function(y, lower, upper, alpha) {
  width <- upper - lower
  width +
    (2 / alpha) * (lower - y) * (y < lower) +
    (2 / alpha) * (y - upper) * (y > upper)
}

wis_vector <- function(dt) {
  y <- dt$cases
  median_score <- abs(y - dt$pred)
  interval_scores <- cbind(
    interval_score(y, dt$lower_50, dt$upper_50, alpha = 0.50),
    interval_score(y, dt$lower_80, dt$upper_80, alpha = 0.20),
    interval_score(y, dt$lower_90, dt$upper_90, alpha = 0.10),
    interval_score(y, dt$lower_95, dt$upper_95, alpha = 0.05)
  )
  weights <- c(0.50, 0.20, 0.10, 0.05) / 2
  (0.5 * median_score + as.numeric(interval_scores %*% weights)) /
    (0.5 + sum(weights))
}

read_final_bridge <- function() {
  path <- file.path(FINAL_DIR, "final_top9_temp04_predictions.csv")
  if (!file.exists(path)) stop("Missing final Top-9 predictions: ", path)
  dt <- fread(path)
  dt[, date := as.Date(date)]
  dt[, model := "Top-9 bridge temp 0.4"]
  dt[, .(
    model,
    split_id,
    uf,
    uf_code,
    date,
    cases = as.numeric(cases),
    pred = as.numeric(pred),
    lower_50 = as.numeric(lower_50),
    upper_50 = as.numeric(upper_50),
    lower_80 = as.numeric(lower_80),
    upper_80 = as.numeric(upper_80),
    lower_90 = as.numeric(lower_90),
    upper_90 = as.numeric(upper_90),
    lower_95 = as.numeric(lower_95),
    upper_95 = as.numeric(upper_95)
  )]
}

read_carlin <- function() {
  files <- list.files(CARLIN_DIR, pattern = "\\.csv$", full.names = TRUE)
  if (length(files) == 0) stop("No Carlin CSV files found in ", CARLIN_DIR)
  dt <- rbindlist(lapply(files, fread), fill = TRUE)
  dt[, date := as.Date(date)]
  dt[, model := "Carlin GNN"]
  dt[, .(
    model,
    split_id = as.integer(split_id),
    uf,
    date,
    cases = as.numeric(casos),
    pred = as.numeric(q_0.5),
    lower_50 = as.numeric(q_0.25),
    upper_50 = as.numeric(q_0.75),
    lower_80 = as.numeric(q_0.1),
    upper_80 = as.numeric(q_0.9),
    lower_90 = as.numeric(q_0.05),
    upper_90 = as.numeric(q_0.95),
    lower_95 = as.numeric(q_0.025),
    upper_95 = as.numeric(q_0.975)
  )]
}

read_graeme <- function(observed_lookup) {
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
  if (length(files) == 0) stop("No Graeme validation files found in ", GRAEME_DIR, " or ", GRAEME_ZIP)

  dt <- rbindlist(lapply(files, function(f) {
    x <- fread(f)
    x[, split_id := as.integer(sub(".*validation_round_(\\d+)\\.csv$", "\\1", basename(f)))]
    x
  }), fill = TRUE)
  dt[, date := as.Date(date)]

  code_map <- unique(observed_lookup[, .(uf_code, uf)])
  dt <- merge(dt, code_map, by.x = "adm_1", by.y = "uf_code", all.x = TRUE)
  dt <- merge(
    dt,
    observed_lookup[, .(split_id, uf, date, cases)],
    by = c("split_id", "uf", "date"),
    all.x = FALSE
  )
  dt[, model := "Graeme Bayesian NB GLMM"]
  dt[, .(
    model,
    split_id = as.integer(split_id),
    uf,
    date,
    cases = as.numeric(cases),
    pred = as.numeric(pred),
    lower_50 = as.numeric(lower_50),
    upper_50 = as.numeric(upper_50),
    lower_80 = as.numeric(lower_80),
    upper_80 = as.numeric(upper_80),
    lower_90 = as.numeric(lower_90),
    upper_90 = as.numeric(upper_90),
    lower_95 = as.numeric(lower_95),
    upper_95 = as.numeric(upper_95)
  )]
}

score_overall <- function(dt) {
  dt <- copy(dt)
  dt[, wis := wis_vector(.SD), by = model]
  dt[, .(
    n = .N,
    n_states = uniqueN(uf),
    mae = mean(abs(cases - pred), na.rm = TRUE),
    rmse = sqrt(mean((cases - pred)^2, na.rm = TRUE)),
    mean_wis = mean(wis, na.rm = TRUE),
    normalized_wis = mean(wis, na.rm = TRUE) / pmax(mean(cases, na.rm = TRUE), 1),
    coverage_50 = mean(cases >= lower_50 & cases <= upper_50, na.rm = TRUE),
    coverage_80 = mean(cases >= lower_80 & cases <= upper_80, na.rm = TRUE),
    coverage_90 = mean(cases >= lower_90 & cases <= upper_90, na.rm = TRUE),
    coverage_95 = mean(cases >= lower_95 & cases <= upper_95, na.rm = TRUE)
  ), by = model][order(mean_wis)]
}

score_by_split <- function(dt) {
  dt <- copy(dt)
  dt[, wis := wis_vector(.SD), by = model]
  dt[, .(
    n = .N,
    n_states = uniqueN(uf),
    mae = mean(abs(cases - pred), na.rm = TRUE),
    rmse = sqrt(mean((cases - pred)^2, na.rm = TRUE)),
    mean_wis = mean(wis, na.rm = TRUE),
    coverage_95 = mean(cases >= lower_95 & cases <= upper_95, na.rm = TRUE)
  ), by = .(model, split_id)][order(split_id, mean_wis)]
}

plot_overall <- function(score_dt) {
  long <- melt(
    score_dt,
    id.vars = c("model", "n", "n_states"),
    measure.vars = c("mae", "rmse", "mean_wis", "normalized_wis"),
    variable.name = "metric",
    value.name = "score"
  )
  long[, metric := factor(
    metric,
    levels = c("mae", "rmse", "mean_wis", "normalized_wis"),
    labels = c("MAE", "RMSE", "Mean WIS", "Normalized WIS")
  )]
  long[, model := factor(model, levels = score_dt[order(mean_wis), model])]
  long[, label := ifelse(metric == "Normalized WIS", number(score, accuracy = 0.001), comma(score, accuracy = 1))]

  p <- ggplot(long, aes(x = model, y = score, fill = model)) +
    geom_col(width = 0.68, show.legend = FALSE) +
    geom_text(aes(label = label), hjust = -0.08, size = 4.2, fontface = "bold") +
    coord_flip(clip = "off") +
    facet_wrap(~ metric, scales = "free_x", ncol = 2) +
    scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.24))) +
    labs(
      title = "Final Model Comparison",
      subtitle = "Top-9 bridge, temperature 0.4 versus external models. Lower scores are better.",
      x = NULL,
      y = "Validation score"
    ) +
    theme_minimal(base_size = 15) +
    theme(
      plot.title = element_text(face = "bold", size = 22),
      plot.subtitle = element_text(size = 14),
      strip.text = element_text(face = "bold", size = 14),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 55, 10, 10)
    )

  ggsave(file.path(PLOT_DIR, "final_top9_vs_external_overall.png"), p, width = 15, height = 9, dpi = 500)
  ggsave(file.path(PLOT_DIR, "final_top9_vs_external_overall.pdf"), p, width = 15, height = 9, device = grDevices::pdf)
}

plot_by_split <- function(split_dt) {
  split_dt[, split_label := factor(paste0("Validation ", split_id), levels = paste0("Validation ", sort(unique(split_id))))]
  split_dt[, model := factor(model, levels = split_dt[, .(overall = mean(mean_wis)), by = model][order(overall), model])]

  p <- ggplot(split_dt, aes(x = split_label, y = mean_wis, fill = model)) +
    geom_col(position = position_dodge(width = 0.78), width = 0.68) +
    geom_text(
      aes(label = comma(mean_wis, accuracy = 1)),
      position = position_dodge(width = 0.78),
      vjust = -0.25,
      size = 4,
      fontface = "bold"
    ) +
    scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.20))) +
    labs(
      title = "Mean WIS By Validation Split",
      subtitle = "Top-9 bridge, temperature 0.4 versus external models. Lower WIS is better.",
      x = NULL,
      y = "Mean WIS",
      fill = "Model"
    ) +
    theme_minimal(base_size = 15) +
    theme(
      plot.title = element_text(face = "bold", size = 22),
      plot.subtitle = element_text(size = 14),
      axis.text.x = element_text(face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(size = 13),
      legend.text = element_text(size = 12),
      panel.grid.minor = element_blank()
    )

  ggsave(file.path(PLOT_DIR, "final_top9_vs_external_by_validation_split.png"), p, width = 14, height = 8, dpi = 500)
  ggsave(file.path(PLOT_DIR, "final_top9_vs_external_by_validation_split.pdf"), p, width = 14, height = 8, device = grDevices::pdf)
}

main <- function() {
  final_bridge <- read_final_bridge()
  observed_lookup <- unique(final_bridge[, .(split_id, uf, uf_code, date, cases)])
  carlin <- read_carlin()
  graeme <- read_graeme(observed_lookup)

  all_dt <- rbindlist(list(final_bridge, carlin, graeme), fill = TRUE)
  overall <- score_overall(all_dt)
  by_split <- score_by_split(all_dt)

  fwrite(overall, file.path(PLOT_DIR, "final_top9_vs_external_scores_overall.csv"))
  fwrite(by_split, file.path(PLOT_DIR, "final_top9_vs_external_scores_by_split.csv"))
  plot_overall(overall)
  plot_by_split(by_split)

  print(overall)
  print(by_split)
}

if (identical(environment(), globalenv())) main()
