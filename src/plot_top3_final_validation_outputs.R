# Plot final Top-3 bridge validation outputs
# ------------------------------------------
# Run this after dengue_dynamic_expert_ensemble.R has produced:
#   output/top3_bridge_oof_predictions.csv
#   output/top3_bridge_oof_weights.csv
#   output/validation_predictions/Top3_bridge_validation_*.csv
#
# The script does not refit any model. It only reads saved predictions/weights
# and creates plots for presentation/checking.

options(stringsAsFactors = FALSE)

DATA_DIR <- "."
OUT_DIR <- Sys.getenv("OUT_DIR", file.path(DATA_DIR, "output"))
PLOT_DIR <- file.path(OUT_DIR, "plots", "top3_final")
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

PLOT_BASE_SIZE <- 20

save_plot <- function(file_out, plot_obj, width, height, dpi = 600) {
  ggsave(file_out, plot_obj, width = width, height = height, dpi = dpi, limitsize = FALSE)
  pdf_out <- sub("\\.png$", ".pdf", file_out)
  ggsave(pdf_out, plot_obj, width = width, height = height, device = grDevices::pdf, limitsize = FALSE)
  message_step("Saved: ", normalizePath(file_out))
  message_step("Saved: ", normalizePath(pdf_out))
}

required_pkgs <- c("data.table", "ggplot2", "scales")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    "Install missing packages before running:\n  install.packages(c(",
    paste(sprintf('"%s"', missing_pkgs), collapse = ", "),
    "))"
  )
}

library(data.table)
library(ggplot2)
library(scales)

message_step <- function(...) {
  message(format(Sys.time(), "%H:%M:%S"), " | ", ...)
}

season_label_from_split <- function(split_id) {
  split_id <- as.integer(split_id)
  start_year <- 2021L + split_id
  paste0(start_year, "-", start_year + 1L)
}

read_top3_oof <- function() {
  path <- file.path(OUT_DIR, "top3_bridge_oof_predictions.csv")
  if (!file.exists(path)) {
    stop("Missing ", path, "\nRun dengue_dynamic_expert_ensemble.R first.")
  }
  dt <- fread(path)
  dt[, date := as.Date(date)]
  dt[, season := season_label_from_split(split_id)]
  dt[]
}

read_top3_weights <- function() {
  path <- file.path(OUT_DIR, "top3_bridge_oof_weights.csv")
  if (!file.exists(path)) {
    warning("Missing ", path, "; skipping weight plots.")
    return(NULL)
  }
  dt <- fread(path)
  dt[, date := as.Date(date)]
  dt[, season := season_label_from_split(split_id)]
  dt[]
}

score_by_state <- function(pred_dt) {
  pred_dt[, .(
    n = .N,
    mae = mean(abs(cases - pred), na.rm = TRUE),
    rmse = sqrt(mean((cases - pred)^2, na.rm = TRUE)),
    wis = mean(wis_vector(.SD), na.rm = TRUE),
    normalized_wis = mean(wis_vector(.SD), na.rm = TRUE) / pmax(mean(cases, na.rm = TRUE), 1),
    coverage_50 = mean(cases >= lower_50 & cases <= upper_50, na.rm = TRUE),
    coverage_80 = mean(cases >= lower_80 & cases <= upper_80, na.rm = TRUE),
    coverage_90 = mean(cases >= lower_90 & cases <= upper_90, na.rm = TRUE),
    coverage_95 = mean(cases >= lower_95 & cases <= upper_95, na.rm = TRUE)
  ), by = uf][order(mae)]
}

score_by_state_season <- function(pred_dt) {
  pred_dt[, .(
    n = .N,
    mae = mean(abs(cases - pred), na.rm = TRUE),
    rmse = sqrt(mean((cases - pred)^2, na.rm = TRUE)),
    wis = mean(wis_vector(.SD), na.rm = TRUE),
    normalized_wis = mean(wis_vector(.SD), na.rm = TRUE) / pmax(mean(cases, na.rm = TRUE), 1),
    coverage_50 = mean(cases >= lower_50 & cases <= upper_50, na.rm = TRUE),
    coverage_80 = mean(cases >= lower_80 & cases <= upper_80, na.rm = TRUE),
    coverage_90 = mean(cases >= lower_90 & cases <= upper_90, na.rm = TRUE),
    coverage_95 = mean(cases >= lower_95 & cases <= upper_95, na.rm = TRUE)
  ), by = .(uf, season)][order(season, mae)]
}

score_overall <- function(pred_dt) {
  pred_dt[, .(
    n = .N,
    mae = mean(abs(cases - pred), na.rm = TRUE),
    rmse = sqrt(mean((cases - pred)^2, na.rm = TRUE)),
    mean_wis = mean(wis_vector(.SD), na.rm = TRUE),
    normalized_wis = mean(wis_vector(.SD), na.rm = TRUE) / pmax(mean(cases, na.rm = TRUE), 1),
    coverage_50 = mean(cases >= lower_50 & cases <= upper_50, na.rm = TRUE),
    coverage_80 = mean(cases >= lower_80 & cases <= upper_80, na.rm = TRUE),
    coverage_90 = mean(cases >= lower_90 & cases <= upper_90, na.rm = TRUE),
    coverage_95 = mean(cases >= lower_95 & cases <= upper_95, na.rm = TRUE)
  )]
}

interval_score <- function(y, lower, upper, alpha) {
  width <- upper - lower
  below <- y < lower
  above <- y > upper
  width +
    (2 / alpha) * (lower - y) * below +
    (2 / alpha) * (y - upper) * above
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
  # Approximate WIS using the available central intervals. Smaller is better.
  (0.5 * median_score + rowSums(interval_scores) / 4) / 5
}

plot_prediction_lines <- function(pred_dt) {
  p <- ggplot(pred_dt, aes(x = date)) +
    geom_ribbon(aes(ymin = lower_80, ymax = upper_80), fill = "#8fb7ff", alpha = 0.25) +
    geom_line(aes(y = pred), colour = "#1455d9", linewidth = 0.75) +
    geom_line(aes(y = cases), colour = "black", linetype = "dashed", linewidth = 0.75) +
    facet_wrap(~ uf, scales = "free_y", nrow = 4) +
    scale_y_continuous(labels = comma) +
    scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
    labs(
      title = "Top-3 Bridge Validation Predictions By State",
      subtitle = "Blue line = Top-3 bridge prediction; shaded band = 80% interval; dashed black = observed cases.",
      x = NULL,
      y = "Weekly dengue cases"
    ) +
    theme_minimal(base_size = PLOT_BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", size = 22),
      plot.subtitle = element_text(size = 15),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = 14),
      axis.title = element_text(size = 15),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
      axis.text.y = element_text(size = 11)
    )

  file_out <- file.path(PLOT_DIR, "top3_validation_predictions_by_state.png")
  save_plot(file_out, p, width = 34, height = 22)
}

plot_prediction_lines_by_season <- function(pred_dt) {
  p <- ggplot(pred_dt, aes(x = date)) +
    geom_ribbon(aes(ymin = lower_80, ymax = upper_80), fill = "#8fb7ff", alpha = 0.22) +
    geom_line(aes(y = pred), colour = "#1455d9", linewidth = 0.65) +
    geom_line(aes(y = cases), colour = "black", linetype = "dashed", linewidth = 0.7) +
    facet_grid(uf ~ season, scales = "free_y") +
    scale_y_continuous(labels = comma) +
    scale_x_date(date_breaks = "3 months", date_labels = "%b\n%Y") +
    labs(
      title = "Top-3 Bridge Predictions Across Validation Seasons",
      subtitle = "Each panel is one state-season validation target.",
      x = NULL,
      y = "Weekly dengue cases"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 22),
      plot.subtitle = element_text(size = 14),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = 10),
      axis.title = element_text(size = 14),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      axis.text.y = element_text(size = 8)
    )

  file_out <- file.path(PLOT_DIR, "top3_validation_predictions_by_state_and_season.png")
  save_plot(file_out, p, width = 34, height = 48)
}

plot_prediction_lines_one_file_per_season <- function(pred_dt) {
  for (ss in sort(unique(pred_dt$season))) {
    plot_dt <- pred_dt[season == ss]
    state_metrics <- plot_dt[, .(
      mae = mean(abs(cases - pred), na.rm = TRUE),
      rmse = sqrt(mean((cases - pred)^2, na.rm = TRUE)),
      wis = mean(wis_vector(.SD), na.rm = TRUE)
    ), by = uf]
    state_metrics[, facet_label := sprintf(
      "%s\nMAE = %s    |    RMSE = %s\nWIS = %s",
      uf,
      comma(mae, accuracy = 1),
      comma(rmse, accuracy = 1),
      comma(wis, accuracy = 1)
    )]
    state_metrics[, facet_label := factor(facet_label, levels = state_metrics[order(uf), facet_label])]
    plot_dt <- merge(plot_dt, state_metrics[, .(uf, facet_label)], by = "uf", all.x = TRUE)

    p <- ggplot(plot_dt, aes(x = date)) +
      geom_ribbon(aes(ymin = lower_80, ymax = upper_80), fill = "#8fb7ff", alpha = 0.25) +
      geom_line(aes(y = pred), colour = "#1455d9", linewidth = 0.75) +
      geom_line(aes(y = cases), colour = "black", linetype = "dashed", linewidth = 0.75) +
      facet_wrap(~ facet_label, scales = "free_y", nrow = 4) +
      scale_y_continuous(labels = comma) +
      scale_x_date(date_breaks = "2 months", date_labels = "%b\n%Y") +
      labs(
        title = paste0("Top-3 Bridge Validation Predictions: ", ss),
        subtitle = "Blue line = Top-3 bridge prediction; shaded band = 80% interval; dashed black = observed cases. State labels show validation MAE, RMSE, and WIS.",
        x = NULL,
        y = "Weekly dengue cases"
      ) +
      theme_minimal(base_size = PLOT_BASE_SIZE) +
      theme(
        plot.title = element_text(face = "bold", size = 30),
        plot.subtitle = element_text(size = 22),
        panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold", size = 18, lineheight = 1.25, margin = margin(8, 4, 8, 4)),
        strip.background = element_blank(),
        axis.title = element_text(size = 24),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 17),
        axis.text.y = element_text(size = 17)
      )

    safe_ss <- gsub("[^0-9A-Za-z_-]+", "_", ss)
    file_out <- file.path(PLOT_DIR, paste0("top3_validation_predictions_by_state_", safe_ss, ".png"))
    save_plot(file_out, p, width = 40, height = 28)
  }
}

plot_state_performance <- function(state_scores) {
  state_scores[, uf := factor(uf, levels = state_scores[order(mae), uf])]

  p <- ggplot(state_scores, aes(x = uf, y = mae)) +
    geom_col(fill = "#1455d9", width = 0.75) +
    coord_flip() +
    scale_y_continuous(labels = comma) +
    labs(
      title = "Top-3 Bridge MAE By State",
      subtitle = "Lower values are better.",
      x = NULL,
      y = "MAE"
    ) +
    theme_minimal(base_size = PLOT_BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", size = 20),
      plot.subtitle = element_text(size = 14),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 12),
      panel.grid.minor = element_blank()
    )

  file_out <- file.path(PLOT_DIR, "top3_mae_by_state.png")
  save_plot(file_out, p, width = 14, height = 11)
}

plot_state_wis <- function(state_scores) {
  state_scores[, uf := factor(uf, levels = state_scores[order(wis), uf])]

  p <- ggplot(state_scores, aes(x = uf, y = wis)) +
    geom_col(fill = "#7c3aed", width = 0.75) +
    coord_flip() +
    scale_y_continuous(labels = comma) +
    labs(
      title = "Top-3 Bridge WIS By State",
      subtitle = "Lower values are better. WIS evaluates both prediction accuracy and interval quality.",
      x = NULL,
      y = "WIS"
    ) +
    theme_minimal(base_size = PLOT_BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", size = 20),
      plot.subtitle = element_text(size = 14),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 12),
      panel.grid.minor = element_blank()
    )

  file_out <- file.path(PLOT_DIR, "top3_wis_by_state.png")
  save_plot(file_out, p, width = 14, height = 11)
}

plot_state_metric_panels <- function(state_scores) {
  metric_long <- melt(
    state_scores,
    id.vars = c("uf", "n"),
    measure.vars = c("mae", "rmse", "wis", "normalized_wis"),
    variable.name = "metric",
    value.name = "score"
  )
  metric_long[, metric := factor(
    metric,
    levels = c("mae", "rmse", "wis", "normalized_wis"),
    labels = c("MAE", "RMSE", "Mean WIS", "Normalized WIS")
  )]
  metric_long[, uf := factor(uf, levels = state_scores[order(mae), uf])]

  p <- ggplot(metric_long, aes(x = uf, y = score, fill = metric)) +
    geom_col(width = 0.75, show.legend = FALSE) +
    coord_flip() +
    facet_wrap(~ metric, scales = "free_x", ncol = 2) +
    scale_y_continuous(labels = comma) +
    labs(
      title = "Top-3 Bridge Validation Metrics By State",
      subtitle = "Lower values are better. Normalized WIS = mean WIS divided by mean observed cases.",
      x = NULL,
      y = "Score"
    ) +
    theme_minimal(base_size = PLOT_BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", size = 20),
      plot.subtitle = element_text(size = 14),
      strip.text = element_text(face = "bold", size = 14),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 11),
      panel.grid.minor = element_blank()
    )

  file_out <- file.path(PLOT_DIR, "top3_validation_metrics_by_state.png")
  save_plot(file_out, p, width = 18, height = 13)
}

plot_state_season_heatmap <- function(state_season_scores) {
  p <- ggplot(state_season_scores, aes(x = season, y = uf, fill = mae)) +
    geom_tile(colour = "white", linewidth = 0.2) +
    scale_fill_viridis_c(option = "magma", labels = comma, direction = -1) +
    labs(
      title = "Top-3 Bridge MAE By State And Validation Season",
      subtitle = "Darker cells indicate higher error.",
      x = NULL,
      y = NULL,
      fill = "MAE"
    ) +
    theme_minimal(base_size = PLOT_BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", size = 20),
      plot.subtitle = element_text(size = 14),
      legend.title = element_text(size = 13),
      legend.text = element_text(size = 12),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 12),
      panel.grid.minor = element_blank()
    )

  file_out <- file.path(PLOT_DIR, "top3_mae_heatmap_by_state_season.png")
  save_plot(file_out, p, width = 14, height = 12)
}

plot_coverage_by_state <- function(state_scores) {
  cov_long <- melt(
    state_scores,
    id.vars = c("uf", "n"),
    measure.vars = c("coverage_50", "coverage_80", "coverage_90", "coverage_95"),
    variable.name = "interval",
    value.name = "coverage"
  )
  cov_long[, interval := factor(
    interval,
    levels = c("coverage_50", "coverage_80", "coverage_90", "coverage_95"),
    labels = c("50%", "80%", "90%", "95%")
  )]
  target <- data.table(
    interval = factor(c("50%", "80%", "90%", "95%"), levels = c("50%", "80%", "90%", "95%")),
    target_coverage = c(0.50, 0.80, 0.90, 0.95)
  )

  p <- ggplot(cov_long, aes(x = uf, y = coverage, fill = interval)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    geom_hline(data = target, aes(yintercept = target_coverage), linetype = "dashed", colour = "grey30") +
    facet_wrap(~ interval, ncol = 1) +
    coord_flip() +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(
      title = "Top-3 Bridge Interval Coverage By State",
      subtitle = "Dashed line is the nominal target coverage for each interval.",
      x = NULL,
      y = "Observed coverage"
    ) +
    theme_minimal(base_size = PLOT_BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", size = 20),
      plot.subtitle = element_text(size = 14),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = 14),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 11),
      legend.position = "none"
    )

  file_out <- file.path(PLOT_DIR, "top3_interval_coverage_by_state.png")
  save_plot(file_out, p, width = 14, height = 18)
}

plot_weight_summary <- function(weight_dt) {
  if (is.null(weight_dt)) return(invisible(NULL))
  weight_dt <- weight_dt[bridge_weight > 0]

  p <- ggplot(weight_dt, aes(x = expert, y = bridge_weight, fill = expert)) +
    stat_summary(fun = mean, geom = "col", width = 0.72, show.legend = FALSE) +
    stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2) +
    facet_wrap(~ season) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, NA)) +
    labs(
      title = "Top-3 Bridge Expert Weights",
      subtitle = "Mean non-zero Top-3 weights, with standard error whiskers.",
      x = NULL,
      y = "Mean Top-3 weight"
    ) +
    theme_minimal(base_size = PLOT_BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", size = 20),
      plot.subtitle = element_text(size = 14),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = 14),
      axis.title = element_text(size = 15),
      axis.text.x = element_text(angle = 30, hjust = 1, size = 12),
      axis.text.y = element_text(size = 12)
    )

  file_out <- file.path(PLOT_DIR, "top3_mean_weights_by_season.png")
  save_plot(file_out, p, width = 16, height = 10)
}

plot_selection_percentage_by_state <- function(weight_dt) {
  if (is.null(weight_dt)) return(invisible(NULL))
  selected <- weight_dt[bridge_weight > 0]
  if (nrow(selected) == 0) return(invisible(NULL))
  pct <- selected[, .N, by = .(uf, expert)]
  totals <- unique(weight_dt[, .(n_weeks = uniqueN(date)), by = uf])
  pct <- merge(pct, totals, by = "uf", all.x = TRUE)
  pct[, selection_pct := 100 * N / pmax(n_weeks, 1)]

  pct[, uf := factor(uf, levels = sort(unique(uf)))]

  p <- ggplot(pct, aes(x = uf, y = selection_pct, fill = expert)) +
    geom_col(width = 0.78, colour = "white", linewidth = 0.25) +
    scale_y_continuous(labels = function(x) paste0(x, "%"), limits = c(0, 300), breaks = seq(0, 300, 50)) +
    labs(
      title = "Expert Selection Percentage By State",
      subtitle = "Stacked percentages show how often each expert was selected into the Top-3 bridge. Totals reach 300% because three experts are selected per state-week.",
      x = NULL,
      y = "Selected weeks",
      fill = "Expert"
    ) +
    theme_minimal(base_size = PLOT_BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", size = 28),
      plot.subtitle = element_text(size = 20),
      legend.position = "bottom",
      legend.title = element_text(size = 20),
      legend.text = element_text(size = 18),
      panel.grid.minor = element_blank(),
      axis.title = element_text(size = 22),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 18, face = "bold"),
      axis.text.y = element_text(size = 18)
    )

  file_out <- file.path(PLOT_DIR, "top3_expert_selection_percentage_by_state.png")
  save_plot(file_out, p, width = 28, height = 16)

  fwrite(pct[order(uf, -selection_pct)], file.path(OUT_DIR, "top3_expert_selection_percentage_by_state.csv"))
}

export_scored_validation_csvs <- function(pred_dt) {
  out_dir <- file.path(OUT_DIR, "validation_predictions_scored")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  scored <- copy(pred_dt)
  scored[, mae := abs(cases - pred)]
  scored[, rmse := sqrt((cases - pred)^2)]
  scored[, wis := wis_vector(.SD)]
  scored[, covered_50 := cases >= lower_50 & cases <= upper_50]
  scored[, covered_80 := cases >= lower_80 & cases <= upper_80]
  scored[, covered_90 := cases >= lower_90 & cases <= upper_90]
  scored[, covered_95 := cases >= lower_95 & cases <= upper_95]

  export_cols <- c(
    "uf", "season", "date", "cases",
    "pred",
    "lower_50", "upper_50",
    "lower_80", "upper_80",
    "lower_90", "upper_90",
    "lower_95", "upper_95",
    "mae", "rmse", "wis",
    "covered_50", "covered_80", "covered_90", "covered_95"
  )

  for (ss in sort(unique(scored$season))) {
    out <- scored[season == ss, ..export_cols]
    setorder(out, uf, date)
    file_out <- file.path(out_dir, paste0("Top3_bridge_validation_scored_", ss, ".csv"))
    fwrite(out, file_out)
    message_step("Saved scored validation CSV: ", normalizePath(file_out))
  }
}

main <- function() {
  pred_dt <- read_top3_oof()
  weight_dt <- read_top3_weights()

  state_scores <- score_by_state(pred_dt)
  state_season_scores <- score_by_state_season(pred_dt)
  overall_scores <- score_overall(pred_dt)

  fwrite(overall_scores, file.path(OUT_DIR, "top3_bridge_scores_overall.csv"))
  fwrite(state_scores, file.path(OUT_DIR, "top3_bridge_scores_by_state.csv"))
  fwrite(state_season_scores, file.path(OUT_DIR, "top3_bridge_scores_by_state_season.csv"))

  plot_prediction_lines(pred_dt)
  plot_prediction_lines_by_season(pred_dt)
  plot_prediction_lines_one_file_per_season(pred_dt)
  plot_state_performance(state_scores)
  plot_state_wis(state_scores)
  plot_state_metric_panels(state_scores)
  plot_state_season_heatmap(state_season_scores)
  plot_coverage_by_state(state_scores)
  plot_weight_summary(weight_dt)
  plot_selection_percentage_by_state(weight_dt)
  export_scored_validation_csvs(pred_dt)

  message_step("Done. Top-3 plots written to: ", normalizePath(PLOT_DIR))
}

if (identical(environment(), globalenv())) {
  main()
}
