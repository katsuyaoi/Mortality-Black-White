# ================================================================
# COMPARATIVE MORTALITY MODELS
#   1) Restrictive baseline
#   2) Spline-age model
#   3) Race × spline(age) model
#
# Sex-stratified RE-logit discrete-time hazard models
# Full data only
# Full-data evaluations only
# No calibration plots
#
# Edits:
#   - year_4 and year_5 collapsed into year_4_5
#   - spline df controlled by SPLINE_DF
#   - all outputs saved into one compact folder
#
# Graphing:
#   - Age range: 50 to 90
#   - Age grid: 5-year intervals
#   - Bands: fixed-effect 95% Wald confidence intervals
# ================================================================
windowsFonts(TNR = windowsFont("Times New Roman"))

packages_needed <- c(
  "dplyr",
  "lme4",
  "ggplot2",
  "tidyr",
  "splines",
  "pROC",
  "PRROC"
)

to_install <- packages_needed[!packages_needed %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(packages_needed, library, character.only = TRUE))

set.seed(142)
setwd("yourdatalocation")

# ================================================================
# 0) Settings
# ================================================================
SPLINE_DF <- 3

age_min <- 50
age_max <- 90
age_by  <- 5

age_grid <- seq(age_min, age_max, by = age_by)
ci_level <- 0.95

model_label_base      <- "Logged age"
model_label_spline    <- "Spline"
model_label_spline_rx <- "Spline  × NH-Black"

out_dir <- "regression_hazard_model_results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

safe_name <- function(x) {
  x <- gsub("×", "x", x)
  x <- gsub("[^A-Za-z0-9]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  tolower(x)
}

# ================================================================
# 1) Load data, clean types, preserve raw age for plotting
# ================================================================
df0 <- read.csv("hrs_survival_post.csv", header = TRUE)

df0 <- df0 %>%
  mutate(
    id     = as.factor(id),
    died_  = as.integer(died_),
    female = as.integer(female)
  ) %>%
  mutate(
    across(.cols = -c(id, died_), .fns = ~ suppressWarnings(as.numeric(.x)))
  )

stopifnot(all(df0$died_ %in% c(0L, 1L)))
stopifnot(all(df0$female %in% c(0L, 1L)))
stopifnot("age_" %in% names(df0))
stopifnot("age2_" %in% names(df0))
stopifnot(all(c("year_4", "year_5") %in% names(df0)))

df0 <- df0 %>%
  mutate(
    age_raw_plot = as.numeric(age_),
    year_4_5 = ifelse(year_4 == 1 | year_5 == 1, 1, 0)
  )

# ================================================================
# 2) Standardize continuous predictors EXCEPT age2_ and age_raw_plot
# ================================================================
standardize_continuous <- function(df,
                                   exclude = c("age2_", "age_raw_plot"),
                                   y_col = "died_",
                                   id_col = "id") {
  df2 <- df
  
  num_vars <- names(df2)[sapply(df2, is.numeric)]
  num_vars <- setdiff(num_vars, c(y_col, exclude))
  if (id_col %in% num_vars) num_vars <- setdiff(num_vars, id_col)
  
  cont_vars <- num_vars[sapply(num_vars, function(v) {
    x <- df2[[v]]
    ux <- unique(x[!is.na(x)])
    length(ux) > 2
  })]
  
  if (length(cont_vars) > 0) {
    mu  <- sapply(cont_vars, function(v) mean(df2[[v]], na.rm = TRUE))
    sdv <- sapply(cont_vars, function(v) sd(df2[[v]], na.rm = TRUE))
    sdv[sdv == 0 | is.na(sdv)] <- 1
    
    for (v in cont_vars) {
      df2[[v]] <- (df2[[v]] - mu[[v]]) / sdv[[v]]
    }
  }
  
  list(df = df2, cont_vars = cont_vars)
}

std_out <- standardize_continuous(
  df0,
  exclude = c("age2_", "age_raw_plot")
)

df <- std_out$df

cat("\nStandardized continuous variables (excluded: age2_, age_raw_plot):\n")
print(std_out$cont_vars)

write.csv(
  data.frame(standardized_continuous_variables = std_out$cont_vars),
  file.path(out_dir, "standardized_continuous_variables.csv"),
  row.names = FALSE
)

# ================================================================
# 3) Model formulas
# ================================================================
spline_term <- paste0("ns(age_raw_plot, df = ", SPLINE_DF, ")")

rhs_common <- paste(
  "year_2 + year_3 + year_4_5 +",
  "mothered_2 + mothered_3 + mothered_4 + mothered_5 +",
  "ed_2 + ed_3 + ed_4 + ed_5 +",
  "black + others + hispanic +",
  "sayret_ + cesd_ + shlt_ + cancre_ + diabe_ + hearte_ + mobila_ + adl5a_"
)

form_base <- as.formula(
  paste("died_ ~ age2_ +", rhs_common, "+ (1 | id)")
)

form_spline <- as.formula(
  paste("died_ ~", spline_term, "+", rhs_common, "+ (1 | id)")
)

form_spline_rx <- as.formula(
  paste(
    "died_ ~", spline_term, "* black +",
    "others + hispanic +",
    "year_2 + year_3 + year_4_5 +",
    "mothered_2 + mothered_3 + mothered_4 + mothered_5 +",
    "ed_2 + ed_3 + ed_4 + ed_5 +",
    "sayret_ + cesd_ + shlt_ + cancre_ + diabe_ + hearte_ + mobila_ + adl5a_ +",
    "(1 | id)"
  )
)

formula_table <- data.frame(
  model = c(
    model_label_base,
    model_label_spline,
    model_label_spline_rx
  ),
  formula = c(
    paste(deparse(form_base), collapse = " "),
    paste(deparse(form_spline), collapse = " "),
    paste(deparse(form_spline_rx), collapse = " ")
  ),
  spline_df = c(NA_integer_, SPLINE_DF, SPLINE_DF),
  year_collapse = "year_4 and year_5 collapsed into year_4_5",
  stringsAsFactors = FALSE
)

write.csv(
  formula_table,
  file.path(out_dir, "model_formulas.csv"),
  row.names = FALSE
)

# ================================================================
# 4) Fit sex-stratified models
# ================================================================
fit_glmer_safe <- function(formula_obj, data_in) {
  glmer(
    formula_obj,
    data   = data_in,
    family = binomial,
    nAGQ   = 0,
    control = glmerControl(
      optimizer   = "bobyqa",
      calc.derivs = FALSE,
      optCtrl     = list(maxfun = 5e4)
    )
  )
}

df_f <- df %>% filter(female == 1)
df_m <- df %>% filter(female == 0)

mod_base_f      <- fit_glmer_safe(form_base,      df_f)
mod_spline_f    <- fit_glmer_safe(form_spline,    df_f)
mod_spline_rx_f <- fit_glmer_safe(form_spline_rx, df_f)

mod_base_m      <- fit_glmer_safe(form_base,      df_m)
mod_spline_m    <- fit_glmer_safe(form_spline,    df_m)
mod_spline_rx_m <- fit_glmer_safe(form_spline_rx, df_m)

cat("\n--- Female restrictive baseline ---\n")
print(summary(mod_base_f))

cat("\n--- Female spline(age) ---\n")
print(summary(mod_spline_f))

cat("\n--- Female spline(age) × black ---\n")
print(summary(mod_spline_rx_f))

cat("\n--- Male restrictive baseline ---\n")
print(summary(mod_base_m))

cat("\n--- Male spline(age) ---\n")
print(summary(mod_spline_m))

cat("\n--- Male spline(age) × black ---\n")
print(summary(mod_spline_rx_m))

saveRDS(
  list(
    mod_base_f      = mod_base_f,
    mod_spline_f    = mod_spline_f,
    mod_spline_rx_f = mod_spline_rx_f,
    mod_base_m      = mod_base_m,
    mod_spline_m    = mod_spline_m,
    mod_spline_rx_m = mod_spline_rx_m
  ),
  file.path(out_dir, "regression_model_objects_all.rds")
)

capture.output(
  list(
    female_restrictive_baseline = summary(mod_base_f),
    female_spline_age           = summary(mod_spline_f),
    female_spline_age_x_black   = summary(mod_spline_rx_f),
    male_restrictive_baseline   = summary(mod_base_m),
    male_spline_age             = summary(mod_spline_m),
    male_spline_age_x_black     = summary(mod_spline_rx_m)
  ),
  file = file.path(out_dir, "regression_model_summaries_all.txt")
)

# ================================================================
# 5) Save coefficient tables for all models
# ================================================================
extract_coef_table <- function(model, sex_label, model_label) {
  cf <- coef(summary(model))
  
  data.frame(
    sex       = sex_label,
    model     = model_label,
    term      = rownames(cf),
    estimate  = cf[, "Estimate"],
    std_error = cf[, "Std. Error"],
    z_value   = cf[, "z value"],
    p_value   = cf[, "Pr(>|z|)"],
    row.names = NULL,
    check.names = FALSE
  ) %>%
    mutate(
      conf_low_logit  = estimate - 1.96 * std_error,
      conf_high_logit = estimate + 1.96 * std_error,
      odds_ratio      = exp(estimate),
      conf_low_or     = exp(conf_low_logit),
      conf_high_or    = exp(conf_high_logit)
    )
}

coef_all_models <- bind_rows(
  extract_coef_table(mod_base_f,      "Female", model_label_base),
  extract_coef_table(mod_spline_f,    "Female", model_label_spline),
  extract_coef_table(mod_spline_rx_f, "Female", model_label_spline_rx),
  extract_coef_table(mod_base_m,      "Male",   model_label_base),
  extract_coef_table(mod_spline_m,    "Male",   model_label_spline),
  extract_coef_table(mod_spline_rx_m, "Male",   model_label_spline_rx)
)

write.csv(
  coef_all_models,
  file.path(out_dir, "coefficient_table_all_models.csv"),
  row.names = FALSE
)

cat("\n================================================\n")
cat("COEFFICIENT TABLE: ALL REGRESSION MODELS\n")
cat("================================================\n")
print(coef_all_models)

# ================================================================
# 6) Race-effect testing for spline interaction models
# ================================================================
extract_race_terms <- function(model, sex_label) {
  cf <- coef(summary(model))
  rn <- rownames(cf)
  
  keep <- grepl("^black$", rn) | grepl("black", rn, fixed = TRUE)
  
  data.frame(
    sex       = sex_label,
    term      = rn[keep],
    estimate  = cf[keep, "Estimate"],
    std_error = cf[keep, "Std. Error"],
    z_value   = cf[keep, "z value"],
    p_value   = cf[keep, "Pr(>|z|)"],
    row.names = NULL,
    check.names = FALSE
  ) %>%
    mutate(
      conf_low_logit  = estimate - 1.96 * std_error,
      conf_high_logit = estimate + 1.96 * std_error,
      odds_ratio      = exp(estimate),
      conf_low_or     = exp(conf_low_logit),
      conf_high_or    = exp(conf_high_logit)
    )
}

race_terms_all <- bind_rows(
  extract_race_terms(mod_spline_rx_f, "Female"),
  extract_race_terms(mod_spline_rx_m, "Male")
)

cat("\n================================================\n")
cat("Race-related terms in spline interaction models\n")
cat("================================================\n")
print(race_terms_all)

write.csv(
  race_terms_all,
  file.path(out_dir, "race_terms_spline_interaction_models.csv"),
  row.names = FALSE
)

cat("\n================================================\n")
cat("Likelihood-ratio tests: spline vs spline × black\n")
cat("================================================\n")

lrt_f <- as.data.frame(anova(mod_spline_f, mod_spline_rx_f, test = "Chisq"))
lrt_f$model_row <- rownames(lrt_f)
lrt_f$sex <- "Female"
rownames(lrt_f) <- NULL

lrt_m <- as.data.frame(anova(mod_spline_m, mod_spline_rx_m, test = "Chisq"))
lrt_m$model_row <- rownames(lrt_m)
lrt_m$sex <- "Male"
rownames(lrt_m) <- NULL

lrt_all <- bind_rows(lrt_f, lrt_m) %>%
  select(sex, model_row, everything())

cat("\nFemale:\n")
print(lrt_f)

cat("\nMale:\n")
print(lrt_m)

write.csv(
  lrt_all,
  file.path(out_dir, "likelihood_ratio_tests_spline_vs_spline_x_black.csv"),
  row.names = FALSE
)

# ================================================================
# 7) Reference rows by race × sex
# ================================================================
make_ref_row <- function(df, black_val, female_val) {
  df %>%
    filter(
      black == black_val,
      hispanic == 0,
      others == 0,
      female == female_val
    ) %>%
    summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE))) %>%
    mutate(
      id       = factor("ref"),
      black    = black_val,
      hispanic = 0,
      others   = 0,
      female   = female_val
    )
}

ref_w_f <- make_ref_row(df, black_val = 0, female_val = 1)
ref_b_f <- make_ref_row(df, black_val = 1, female_val = 1)

ref_w_m <- make_ref_row(df, black_val = 0, female_val = 0)
ref_b_m <- make_ref_row(df, black_val = 1, female_val = 0)

# ================================================================
# 8) Build raw-age -> age2_ mapper ONCE from df
# ================================================================
make_age2_mapper <- function(df,
                             age_raw_col = "age_raw_plot",
                             age2_col = "age2_") {
  d <- df %>%
    filter(
      !is.na(.data[[age_raw_col]]),
      !is.na(.data[[age2_col]])
    ) %>%
    transmute(
      age_raw = .data[[age_raw_col]],
      age2    = .data[[age2_col]]
    ) %>%
    mutate(age_round = round(age_raw)) %>%
    group_by(age_round) %>%
    summarise(
      age2_med = median(age2, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(age_round)
  
  write.csv(
    d,
    file.path(out_dir, "age_raw_to_age2_mapper_values.csv"),
    row.names = FALSE
  )
  
  function(age_vec) {
    approx(
      x    = d$age_round,
      y    = d$age2_med,
      xout = age_vec,
      rule = 2
    )$y
  }
}

age2_from_raw <- make_age2_mapper(df)

# ================================================================
# 9) Save age grid configuration
# ================================================================
age_grid_config <- data.frame(
  age_min = age_min,
  age_max = age_max,
  age_by = age_by,
  ci_level = ci_level,
  spline_df = SPLINE_DF,
  age_grid = paste(age_grid, collapse = ", "),
  year_collapse = "year_4 and year_5 collapsed into year_4_5"
)

write.csv(
  age_grid_config,
  file.path(out_dir, "age_grid_config.csv"),
  row.names = FALSE
)

# ================================================================
# 10) Fixed-effect confidence interval helper
# ================================================================
fixed_effect_curve_ci <- function(model,
                                  age_raw_vec,
                                  ref_row,
                                  age2_mapper = NULL,
                                  level = 0.95) {
  
  nd <- ref_row[rep(1, length(age_raw_vec)), , drop = FALSE]
  
  if (!"age_raw_plot" %in% names(nd)) {
    stop("age_raw_plot was not found in the reference row.")
  }
  
  nd$age_raw_plot <- age_raw_vec
  
  model_vars <- all.vars(formula(model))
  
  if ("age2_" %in% model_vars) {
    if (is.null(age2_mapper)) {
      stop("age2_mapper must be supplied for models using age2_.")
    }
    
    if (!"age2_" %in% names(nd)) {
      stop("age2_ was not found in the reference row.")
    }
    
    nd$age2_ <- age2_mapper(age_raw_vec)
  }
  
  if (!"id" %in% names(nd)) {
    nd$id <- factor("ref")
  }
  
  eta <- as.numeric(
    predict(
      model,
      newdata = nd,
      type = "link",
      re.form = NA,
      allow.new.levels = TRUE
    )
  )
  
  fixed_terms <- try(
    delete.response(terms(model, fixed.only = TRUE)),
    silent = TRUE
  )
  
  if (inherits(fixed_terms, "try-error")) {
    fixed_formula <- lme4::nobars(formula(model))
    fixed_terms <- delete.response(terms(fixed_formula))
  }
  
  X <- model.matrix(
    fixed_terms,
    data = nd,
    contrasts.arg = attr(lme4::getME(model, "X"), "contrasts")
  )
  
  beta <- lme4::fixef(model)
  V    <- as.matrix(vcov(model))
  
  missing_cols <- setdiff(names(beta), colnames(X))
  
  if (length(missing_cols) > 0) {
    for (cc in missing_cols) {
      X[, cc] <- 0
    }
  }
  
  X <- X[, names(beta), drop = FALSE]
  V <- V[names(beta), names(beta), drop = FALSE]
  
  se_eta <- sqrt(pmax(0, rowSums((X %*% V) * X)))
  zcrit  <- qnorm(1 - (1 - level) / 2)
  
  eta_low  <- eta - zcrit * se_eta
  eta_high <- eta + zcrit * se_eta
  
  data.frame(
    age      = age_raw_vec,
    eta      = eta,
    se_eta   = se_eta,
    eta_low  = eta_low,
    eta_high = eta_high,
    p_mid    = plogis(eta),
    p_low    = plogis(eta_low),
    p_high   = plogis(eta_high)
  )
}

# ================================================================
# 11) Generate curves for all 3 models
# ================================================================
get_curve_df <- function(model,
                         ref_white,
                         ref_black,
                         sex_label,
                         model_label,
                         age2_mapper = NULL,
                         level = ci_level) {
  
  res_w <- fixed_effect_curve_ci(
    model       = model,
    age_raw_vec = age_grid,
    ref_row     = ref_white,
    age2_mapper = age2_mapper,
    level       = level
  )
  
  res_b <- fixed_effect_curve_ci(
    model       = model,
    age_raw_vec = age_grid,
    ref_row     = ref_black,
    age2_mapper = age2_mapper,
    level       = level
  )
  
  bind_rows(
    res_w %>%
      mutate(
        race  = "NH-White",
        sex   = sex_label,
        model = model_label
      ),
    res_b %>%
      mutate(
        race  = "NH-Black",
        sex   = sex_label,
        model = model_label
      )
  )
}

df_curves_all <- bind_rows(
  get_curve_df(
    mod_base_f,
    ref_w_f,
    ref_b_f,
    "Female",
    model_label_base,
    age2_mapper = age2_from_raw
  ),
  get_curve_df(
    mod_spline_f,
    ref_w_f,
    ref_b_f,
    "Female",
    model_label_spline
  ),
  get_curve_df(
    mod_spline_rx_f,
    ref_w_f,
    ref_b_f,
    "Female",
    model_label_spline_rx
  ),
  get_curve_df(
    mod_base_m,
    ref_w_m,
    ref_b_m,
    "Male",
    model_label_base,
    age2_mapper = age2_from_raw
  ),
  get_curve_df(
    mod_spline_m,
    ref_w_m,
    ref_b_m,
    "Male",
    model_label_spline
  ),
  get_curve_df(
    mod_spline_rx_m,
    ref_w_m,
    ref_b_m,
    "Male",
    model_label_spline_rx
  )
)

write.csv(
  df_curves_all,
  file.path(out_dir, "mortality_model_comparison_curves_age50_90_by5_fixed_effect_ci.csv"),
  row.names = FALSE
)

# ================================================================
# 12) Plot comparative fitted hazard curves
# ================================================================
p_comp_models <- ggplot(
  df_curves_all,
  aes(x = age, y = p_mid, color = race, fill = race)
) +
  geom_ribbon(
    aes(ymin = p_low, ymax = p_high),
    alpha = 0.16,
    color = NA
  ) +
  geom_line(linewidth = 1.05) +
  facet_grid(sex ~ model) +
  scale_x_continuous(
    breaks = age_grid,
    limits = c(age_min, age_max)
  ) +
  scale_color_manual(
    values = c(
      "NH-Black" = "#1f78b4",
      "NH-White" = "#e31a1c"
    )
  ) +
  scale_fill_manual(
    values = c(
      "NH-Black" = "#1f78b4",
      "NH-White" = "#e31a1c"
    )
  ) +
  labs(
    x = "Age",
    y = "Predicted discrete-time hazard, fixed-effect estimate",
    color = "",
    fill  = ""
  ) +
  theme_minimal(base_size = 14, base_family = "TNR") +
  theme(
    text = element_text(family = "TNR", face = "plain"),
    axis.title = element_text(family = "TNR", face = "plain"),
    axis.text = element_text(family = "TNR", face = "plain"),
    strip.text = element_text(family = "TNR", face = "plain"),
    legend.text = element_text(family = "TNR", face = "plain"),
    legend.title = element_text(family = "TNR", face = "plain")
  )

png(
  filename = file.path(
    out_dir,
    "mortality_model_comparison_curves_age50_90_by5_fixed_effect_ci.png"
  ),
  width = 12,
  height = 7,
  units = "in",
  res = 600,
  type = "windows"
)

print(p_comp_models)

dev.off()

# ================================================================
# 13) Full-data predictions for all models
# ================================================================
predict_pop_single_model <- function(model, df_new) {
  as.numeric(
    predict(
      model,
      newdata = df_new,
      type = "response",
      re.form = NA,
      allow.new.levels = TRUE
    )
  )
}

p_base_f      <- predict_pop_single_model(mod_base_f,      df_f)
p_spline_f    <- predict_pop_single_model(mod_spline_f,    df_f)
p_spline_rx_f <- predict_pop_single_model(mod_spline_rx_f, df_f)

p_base_m      <- predict_pop_single_model(mod_base_m,      df_m)
p_spline_m    <- predict_pop_single_model(mod_spline_m,    df_m)
p_spline_rx_m <- predict_pop_single_model(mod_spline_rx_m, df_m)

df_eval <- bind_rows(
  df_f %>%
    mutate(
      p_base      = p_base_f,
      p_spline    = p_spline_f,
      p_spline_rx = p_spline_rx_f
    ),
  df_m %>%
    mutate(
      p_base      = p_base_m,
      p_spline    = p_spline_m,
      p_spline_rx = p_spline_rx_m
    )
)

prediction_na_counts <- data.frame(
  variable = c("died_", "p_base", "p_spline", "p_spline_rx"),
  n_missing = as.integer(colSums(is.na(df_eval[, c("died_", "p_base", "p_spline", "p_spline_rx")])))
)

cat("\n================================================\n")
cat("NA counts in full-data predictions\n")
cat("================================================\n")
print(prediction_na_counts)

write.csv(
  prediction_na_counts,
  file.path(out_dir, "prediction_na_counts.csv"),
  row.names = FALSE
)

write.csv(
  df_eval %>%
    select(id, died_, female, black, hispanic, others, age_raw_plot, p_base, p_spline, p_spline_rx),
  file.path(out_dir, "full_data_predictions_all_regression_models.csv"),
  row.names = FALSE
)

# ================================================================
# 14) Full-data evaluation helpers
# ================================================================
safe_clip <- function(p, eps = 1e-15) {
  pmin(pmax(as.numeric(p), eps), 1 - eps)
}

calc_auc <- function(y, p) {
  keep <- is.finite(y) & is.finite(p)
  y <- as.integer(y[keep])
  p <- as.numeric(p[keep])
  as.numeric(pROC::auc(y, p))
}

calc_prauc <- function(y, p) {
  keep <- is.finite(y) & is.finite(p)
  y <- as.integer(y[keep])
  p <- as.numeric(p[keep])
  
  fg <- p[y == 1]
  bg <- p[y == 0]
  
  pr <- PRROC::pr.curve(
    scores.class0 = fg,
    scores.class1 = bg,
    curve = FALSE
  )
  
  as.numeric(pr$auc.integral)
}

calc_brier <- function(y, p) {
  keep <- is.finite(y) & is.finite(p)
  y <- as.numeric(y[keep])
  p <- as.numeric(p[keep])
  mean((p - y)^2)
}

calc_logloss <- function(y, p) {
  keep <- is.finite(y) & is.finite(p)
  y <- as.numeric(y[keep])
  p <- safe_clip(p[keep])
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

metrics_from_prob <- function(y_true, p, thr = 0.50) {
  keep <- is.finite(y_true) & is.finite(p)
  y_true <- as.integer(y_true[keep])
  p      <- as.numeric(p[keep])
  
  yhat <- as.integer(p >= thr)
  
  cm <- table(
    Predicted = factor(yhat, levels = c(0, 1)),
    Actual    = factor(y_true, levels = c(0, 1))
  )
  
  tn <- as.integer(cm["0", "0"])
  tp <- as.integer(cm["1", "1"])
  fn <- as.integer(cm["0", "1"])
  fp <- as.integer(cm["1", "0"])
  
  acc  <- (tp + tn) / sum(cm)
  sens <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  spec <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
  prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  npv  <- if ((tn + fn) > 0) tn / (tn + fn) else NA_real_
  
  f1 <- if (!is.na(prec) && !is.na(sens) && (prec + sens) > 0) {
    2 * prec * sens / (prec + sens)
  } else {
    NA_real_
  }
  
  data.frame(
    n_used    = length(y_true),
    threshold = thr,
    tp        = tp,
    tn        = tn,
    fp        = fp,
    fn        = fn,
    acc       = acc,
    sens      = sens,
    spec      = spec,
    precision = prec,
    npv       = npv,
    f1        = f1
  )
}

subgroup_metrics_rxsex <- function(df, p_col, thr) {
  thr <- as.numeric(thr)
  
  df %>%
    mutate(
      race = case_when(
        black == 1 & hispanic == 0 & others == 0 ~ "NH-Black",
        black == 0 & hispanic == 0 & others == 0 ~ "NH-White",
        TRUE ~ NA_character_
      ),
      sex = ifelse(female == 1, "Female", "Male"),
      p = .data[[p_col]]
    ) %>%
    filter(
      !is.na(race),
      is.finite(died_),
      is.finite(p)
    ) %>%
    group_by(race, sex) %>%
    summarise(
      n  = n(),
      tp = sum(p >= thr & died_ == 1L, na.rm = TRUE),
      tn = sum(p <  thr & died_ == 0L, na.rm = TRUE),
      fp = sum(p >= thr & died_ == 0L, na.rm = TRUE),
      fn = sum(p <  thr & died_ == 1L, na.rm = TRUE),
      acc  = (tp + tn) / (tp + tn + fp + fn),
      sens = if ((tp + fn) > 0) tp / (tp + fn) else NA_real_,
      spec = if ((tn + fp) > 0) tn / (tn + fp) else NA_real_,
      precision = if ((tp + fp) > 0) tp / (tp + fp) else NA_real_,
      .groups = "drop"
    )
}

evaluate_model_full <- function(df_eval, p_col, model_label) {
  y <- as.integer(df_eval$died_)
  p <- as.numeric(df_eval[[p_col]])
  
  keep <- is.finite(y) & is.finite(p)
  y2 <- y[keep]
  p2 <- p[keep]
  
  pi_full <- mean(y2 == 1L, na.rm = TRUE)
  
  tau_prev_full <- as.numeric(
    quantile(
      p2,
      probs = 1 - pi_full,
      na.rm = TRUE,
      type = 7
    )
  )
  
  overall_threshold_free <- data.frame(
    model           = model_label,
    n_used          = length(y2),
    prevalence_full = pi_full,
    tau_prev_full   = tau_prev_full,
    auc             = calc_auc(y2, p2),
    pr_auc          = calc_prauc(y2, p2),
    brier           = calc_brier(y2, p2),
    logloss         = calc_logloss(y2, p2)
  )
  
  overall_thr_050 <- metrics_from_prob(y2, p2, thr = 0.50) %>%
    mutate(
      model = model_label,
      threshold_type = "0.50"
    )
  
  overall_thr_prev <- metrics_from_prob(y2, p2, thr = tau_prev_full) %>%
    mutate(
      model = model_label,
      threshold_type = "prevalence_calibrated_full"
    )
  
  subgroup_050 <- subgroup_metrics_rxsex(
    df_eval,
    p_col = p_col,
    thr = 0.50
  ) %>%
    mutate(
      model = model_label,
      threshold_type = "0.50"
    )
  
  subgroup_prev <- subgroup_metrics_rxsex(
    df_eval,
    p_col = p_col,
    thr = tau_prev_full
  ) %>%
    mutate(
      model = model_label,
      threshold_type = "prevalence_calibrated_full"
    )
  
  list(
    threshold_free       = overall_threshold_free,
    overall_thresholded  = bind_rows(overall_thr_050, overall_thr_prev),
    subgroup_thresholded = bind_rows(subgroup_050, subgroup_prev)
  )
}

# ================================================================
# 15) Run full-data evaluations for all models
# ================================================================
eval_base <- evaluate_model_full(
  df_eval,
  "p_base",
  model_label_base
)

eval_spline <- evaluate_model_full(
  df_eval,
  "p_spline",
  model_label_spline
)

eval_spline_rx <- evaluate_model_full(
  df_eval,
  "p_spline_rx",
  model_label_spline_rx
)

eval_threshold_free_all <- bind_rows(
  eval_base$threshold_free,
  eval_spline$threshold_free,
  eval_spline_rx$threshold_free
)

eval_overall_thresholded_all <- bind_rows(
  eval_base$overall_thresholded,
  eval_spline$overall_thresholded,
  eval_spline_rx$overall_thresholded
)

eval_subgroup_thresholded_all <- bind_rows(
  eval_base$subgroup_thresholded,
  eval_spline$subgroup_thresholded,
  eval_spline_rx$subgroup_thresholded
)

cat("\n================================================\n")
cat("FULL-DATA THRESHOLD-FREE EVALUATION\n")
cat("================================================\n")
print(eval_threshold_free_all)

cat("\n================================================\n")
cat("FULL-DATA THRESHOLD-BASED EVALUATION\n")
cat("0.50 and prevalence-calibrated thresholds\n")
cat("================================================\n")
print(eval_overall_thresholded_all)

cat("\n================================================\n")
cat("FULL-DATA RACE × SEX SUBGROUP EVALUATION\n")
cat("0.50 and prevalence-calibrated thresholds\n")
cat("================================================\n")
print(eval_subgroup_thresholded_all)

write.csv(
  eval_threshold_free_all,
  file.path(out_dir, "eval_threshold_free_full_data.csv"),
  row.names = FALSE
)

write.csv(
  eval_overall_thresholded_all,
  file.path(out_dir, "eval_thresholded_full_data.csv"),
  row.names = FALSE
)

write.csv(
  eval_subgroup_thresholded_all,
  file.path(out_dir, "eval_subgroup_thresholded_full_data.csv"),
  row.names = FALSE
)

# ================================================================
# 15b) Race × sex-specific threshold-free evaluation
# ================================================================
evaluate_model_threshold_free_by_race_sex <- function(df_eval, p_col, model_label) {
  
  df_eval %>%
    mutate(
      race = case_when(
        black == 1 & hispanic == 0 & others == 0 ~ "NH-Black",
        black == 0 & hispanic == 0 & others == 0 ~ "NH-White",
        TRUE ~ NA_character_
      ),
      sex = ifelse(female == 1, "Female", "Male"),
      p = as.numeric(.data[[p_col]]),
      y = as.integer(died_)
    ) %>%
    filter(
      !is.na(race),
      is.finite(y),
      is.finite(p)
    ) %>%
    group_by(race, sex) %>%
    summarise(
      model = model_label,
      n_used = n(),
      prevalence = mean(y == 1L, na.rm = TRUE),
      tau_prev = as.numeric(
        quantile(
          p,
          probs = 1 - mean(y == 1L, na.rm = TRUE),
          na.rm = TRUE,
          type = 7
        )
      ),
      auc = if (length(unique(y)) == 2) calc_auc(y, p) else NA_real_,
      pr_auc = if (length(unique(y)) == 2) calc_prauc(y, p) else NA_real_,
      brier = calc_brier(y, p),
      logloss = calc_logloss(y, p),
      .groups = "drop"
    ) %>%
    select(
      model,
      race,
      sex,
      n_used,
      prevalence,
      tau_prev,
      auc,
      pr_auc,
      brier,
      logloss
    )
}

eval_threshold_free_by_race_sex_all <- bind_rows(
  evaluate_model_threshold_free_by_race_sex(
    df_eval,
    "p_base",
    model_label_base
  ),
  evaluate_model_threshold_free_by_race_sex(
    df_eval,
    "p_spline",
    model_label_spline
  ),
  evaluate_model_threshold_free_by_race_sex(
    df_eval,
    "p_spline_rx",
    model_label_spline_rx
  )
)

cat("\n================================================\n")
cat("FULL-DATA RACE × SEX-SPECIFIC THRESHOLD-FREE EVALUATION\n")
cat("================================================\n")
print(eval_threshold_free_by_race_sex_all)

write.csv(
  eval_threshold_free_by_race_sex_all,
  file.path(out_dir, "eval_threshold_free_full_data_by_race_sex.csv"),
  row.names = FALSE
)

# ================================================================
# 15c) Race × sex-specific thresholded evaluation
# ================================================================
evaluate_model_by_race_sex <- function(df_eval, p_col, model_label) {
  
  df_eval %>%
    mutate(
      race = case_when(
        black == 1 & hispanic == 0 & others == 0 ~ "NH-Black",
        black == 0 & hispanic == 0 & others == 0 ~ "NH-White",
        TRUE ~ NA_character_
      ),
      sex = ifelse(female == 1, "Female", "Male"),
      p = as.numeric(.data[[p_col]]),
      y = as.integer(died_)
    ) %>%
    filter(
      !is.na(race),
      is.finite(y),
      is.finite(p)
    ) %>%
    group_by(race, sex) %>%
    group_modify(~ {
      y2 <- .x$y
      p2 <- .x$p
      
      pi_group <- mean(y2 == 1L, na.rm = TRUE)
      
      tau_prev_group <- as.numeric(
        quantile(
          p2,
          probs = 1 - pi_group,
          na.rm = TRUE,
          type = 7
        )
      )
      
      bind_rows(
        metrics_from_prob(y2, p2, thr = 0.50) %>%
          mutate(
            threshold_type = "0.50",
            prevalence = pi_group,
            tau_prev = tau_prev_group
          ),
        metrics_from_prob(y2, p2, thr = tau_prev_group) %>%
          mutate(
            threshold_type = "prevalence_calibrated_race_sex",
            prevalence = pi_group,
            tau_prev = tau_prev_group
          )
      )
    }) %>%
    ungroup() %>%
    mutate(model = model_label) %>%
    select(
      model,
      race,
      sex,
      threshold_type,
      prevalence,
      tau_prev,
      n_used,
      threshold,
      tp,
      tn,
      fp,
      fn,
      acc,
      sens,
      spec,
      precision,
      npv,
      f1
    )
}

eval_thresholded_by_race_sex_all <- bind_rows(
  evaluate_model_by_race_sex(
    df_eval,
    "p_base",
    model_label_base
  ),
  evaluate_model_by_race_sex(
    df_eval,
    "p_spline",
    model_label_spline
  ),
  evaluate_model_by_race_sex(
    df_eval,
    "p_spline_rx",
    model_label_spline_rx
  )
)

cat("\n================================================\n")
cat("FULL-DATA RACE × SEX-SPECIFIC THRESHOLD-BASED EVALUATION\n")
cat("0.50 and race × sex-specific prevalence-calibrated thresholds\n")
cat("================================================\n")
print(eval_thresholded_by_race_sex_all)

write.csv(
  eval_thresholded_by_race_sex_all,
  file.path(out_dir, "eval_thresholded_full_data_by_race_sex.csv"),
  row.names = FALSE
)

# ================================================================
# 16) Compact fit comparison table
# ================================================================
model_fit_table <- bind_rows(
  data.frame(
    sex = "Female",
    model = c(
      model_label_base,
      model_label_spline,
      model_label_spline_rx
    ),
    AIC = c(
      AIC(mod_base_f),
      AIC(mod_spline_f),
      AIC(mod_spline_rx_f)
    ),
    BIC = c(
      BIC(mod_base_f),
      BIC(mod_spline_f),
      BIC(mod_spline_rx_f)
    ),
    logLik = c(
      as.numeric(logLik(mod_base_f)),
      as.numeric(logLik(mod_spline_f)),
      as.numeric(logLik(mod_spline_rx_f))
    )
  ),
  data.frame(
    sex = "Male",
    model = c(
      model_label_base,
      model_label_spline,
      model_label_spline_rx
    ),
    AIC = c(
      AIC(mod_base_m),
      AIC(mod_spline_m),
      AIC(mod_spline_rx_m)
    ),
    BIC = c(
      BIC(mod_base_m),
      BIC(mod_spline_m),
      BIC(mod_spline_rx_m)
    ),
    logLik = c(
      as.numeric(logLik(mod_base_m)),
      as.numeric(logLik(mod_spline_m)),
      as.numeric(logLik(mod_spline_rx_m))
    )
  )
)

cat("\n================================================\n")
cat("MODEL FIT TABLE\n")
cat("================================================\n")
print(model_fit_table)

write.csv(
  model_fit_table,
  file.path(out_dir, "model_fit_table_mortality_models.csv"),
  row.names = FALSE
)

# ================================================================
# 17) Save compact output index and workspace
# ================================================================
output_index <- data.frame(
  file = c(
    "standardized_continuous_variables.csv",
    "model_formulas.csv",
    "regression_model_objects_all.rds",
    "regression_model_summaries_all.txt",
    "coefficient_table_all_models.csv",
    "race_terms_spline_interaction_models.csv",
    "likelihood_ratio_tests_spline_vs_spline_x_black.csv",
    "age_raw_to_age2_mapper_values.csv",
    "age_grid_config.csv",
    "mortality_model_comparison_curves_age50_90_by5_fixed_effect_ci.csv",
    "mortality_model_comparison_curves_age50_90_by5_fixed_effect_ci.png",
    "prediction_na_counts.csv",
    "full_data_predictions_all_regression_models.csv",
    "eval_threshold_free_full_data.csv",
    "eval_thresholded_full_data.csv",
    "eval_subgroup_thresholded_full_data.csv",
    "eval_threshold_free_full_data_by_race_sex.csv",
    "eval_thresholded_full_data_by_race_sex.csv",
    "model_fit_table_mortality_models.csv",
    "regression_hazard_model_workspace.RData"
  ),
  description = c(
    "Continuous variables standardized before model fitting",
    "Model formulas, spline df, and year-collapse setting",
    "All six fitted glmer model objects",
    "Text summaries for all six fitted models",
    "Combined coefficient table for all six models",
    "Race-related terms from race-specific spline models",
    "Likelihood-ratio tests comparing spline vs race-specific spline models",
    "Mapper used to recover age2_ values from raw age for baseline curves",
    "Age grid, CI level, spline df, and year-collapse setting",
    "Predicted fixed-effect mortality curves from age 50 to 90 by 5-year grid",
    "Figure of fitted age-specific hazards with fixed-effect 95% Wald CIs",
    "Missingness check for predicted probabilities",
    "Full-data predicted probabilities from all three model types",
    "Threshold-free evaluation metrics",
    "Thresholded evaluation metrics using 0.50 and prevalence-calibrated thresholds",
    "Race × sex subgroup classification metrics using full-sample prevalence-calibrated threshold",
    "Race × sex-specific threshold-free evaluation metrics",
    "Race × sex-specific thresholded evaluation metrics using 0.50 and race × sex-specific prevalence-calibrated thresholds",
    "AIC, BIC, and log-likelihood by sex and model",
    "Full R workspace after script completion"
  )
)

write.csv(
  output_index,
  file.path(out_dir, "00_output_index.csv"),
  row.names = FALSE
)

save.image(
  file = file.path(out_dir, "regression_hazard_model_workspace.RData")
)

cat("\n================================================\n")
cat("All regression hazard model outputs saved to:\n")
cat(normalizePath(out_dir), "\n")
cat("================================================\n")
print(output_index)