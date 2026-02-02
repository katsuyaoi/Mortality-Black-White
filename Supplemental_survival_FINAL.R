# ================================================================
# BASELINE COMPARATIVE GRAPHS (RE-LOGIT)
# - Sex-stratified glmer hazard models
# - Standardize all continuous predictors EXCEPT age2_
# - Preserve TRUE raw age for plotting (avoid collapse to 40)
# - 2-year binned hazard curves, NH-White vs NH-Black, by Sex
# ================================================================

library(dplyr)
library(lme4)
library(ggplot2)
library(caret)
library(tidyr)

set.seed(142)
setwd("datalocation")

df0 <- read.csv("hrs_survival_post.csv", header = TRUE)

# ================================================================
# 1) Clean types + preserve raw age for plotting
# ================================================================
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

# --- preserve TRUE raw age for plotting BEFORE any scaling ---
# Assumes your true raw age column is age_. If not, change here.
stopifnot("age_" %in% names(df0))
df0 <- df0 %>% mutate(age_raw_plot = as.numeric(age_))

# ================================================================
# 2) Standardize continuous predictors EXCEPT age2_ and age_raw_plot
#    - leaves 0/1 and dummy vars alone (<=2 unique values)
# ================================================================
standardize_continuous <- function(df, exclude = c("age2_", "age_raw_plot"),
                                   y_col = "died_", id_col = "id") {
  df2 <- df
  
  num_vars <- names(df2)[sapply(df2, is.numeric)]
  num_vars <- setdiff(num_vars, c(y_col, exclude))
  # keep id out if it became numeric somehow
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
    
    for (v in cont_vars) df2[[v]] <- (df2[[v]] - mu[[v]]) / sdv[[v]]
  }
  list(df = df2, cont_vars = cont_vars)
}

std_out <- standardize_continuous(df0, exclude = c("age2_", "age_raw_plot"))
df <- std_out$df

cat("\nStandardized continuous variables (excluded: age2_, age_raw_plot):\n")
print(std_out$cont_vars)

# ================================================================
# 3) Fit RE logit by sex (FULL DATA)
# ================================================================
form_base <- died_ ~ age2_ +
  year_2 + year_3 + year_4 + year_5 +
  mothered_2 + mothered_3 + mothered_4 + mothered_5 +
  ed_2 + ed_3 + ed_4 + ed_5 +
  black + others + hispanic +
  sayret_ + cesd_ + shlt_ + cancre_ + diabe_ + hearte_ + mobila_ + adl5a_ +
  (1 | id)

re_logit_female <- glmer(
  form_base,
  data   = df[df$female == 1, ],
  family = binomial,
  nAGQ   = 0,
  control = glmerControl(
    optimizer   = "bobyqa",
    calc.derivs = FALSE,
    optCtrl     = list(maxfun = 5e4)
  )
)

re_logit_male <- glmer(
  form_base,
  data   = df[df$female == 0, ],
  family = binomial,
  nAGQ   = 0,
  control = glmerControl(
    optimizer   = "bobyqa",
    calc.derivs = FALSE,
    optCtrl     = list(maxfun = 5e4)
  )
)

cat("\n--- Female model summary ---\n"); print(summary(re_logit_female))
cat("\n--- Male model summary ---\n");   print(summary(re_logit_male))

# ================================================================
# 4) Population-level prediction (re.form = NA), sex-aware
# ================================================================
predict_pop_by_sex <- function(df_new, model_female, model_male) {
  p <- rep(NA_real_, nrow(df_new))
  idx_f <- which(df_new$female == 1)
  idx_m <- which(df_new$female == 0)
  
  if (length(idx_f) > 0) {
    p[idx_f] <- predict(model_female, newdata = df_new[idx_f, , drop = FALSE],
                        type = "response", re.form = NA)
  }
  if (length(idx_m) > 0) {
    p[idx_m] <- predict(model_male, newdata = df_new[idx_m, , drop = FALSE],
                        type = "response", re.form = NA)
  }
  as.numeric(p)
}

p_full <- predict_pop_by_sex(df, re_logit_female, re_logit_male)

library(dplyr)
library(lme4)
library(ggplot2)
library(MASS)   # mvrnorm
library(tidyr)

set.seed(142)
setwd("C:/Users/oikat/Box/temporary workspace/HSR_al_embedding")

# ----------------------------
# 0) Load + clean
# ----------------------------
df0 <- read.csv("hrs_survival_post.csv", header = TRUE) %>%
  mutate(
    id     = as.factor(id),
    died_  = as.integer(died_),
    female = as.integer(female)
  ) %>%
  mutate(across(.cols = -c(id, died_), .fns = ~ suppressWarnings(as.numeric(.x))))

stopifnot(all(df0$died_ %in% c(0L, 1L)))
stopifnot(all(df0$female %in% c(0L, 1L)))
stopifnot("age_" %in% names(df0))
stopifnot("age2_" %in% names(df0))

# Preserve TRUE raw age for plotting
df0 <- df0 %>% mutate(age_raw_plot = as.numeric(age_))

# ----------------------------
# 1) Standardize continuous predictors EXCEPT age2_ and age_raw_plot
# ----------------------------
standardize_continuous <- function(df, exclude = c("age2_", "age_raw_plot"),
                                   y_col = "died_", id_col = "id") {
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
    
    for (v in cont_vars) df2[[v]] <- (df2[[v]] - mu[[v]]) / sdv[[v]]
  }
  list(df = df2, cont_vars = cont_vars)
}

std_out <- standardize_continuous(df0, exclude = c("age2_", "age_raw_plot"))
df <- std_out$df

# ----------------------------
# 2) Fit RE-logit by sex
# ----------------------------
form_base <- died_ ~ age2_ +
  year_2 + year_3 + year_4 + year_5 +
  mothered_2 + mothered_3 + mothered_4 + mothered_5 +
  ed_2 + ed_3 + ed_4 + ed_5 +
  black + others + hispanic +
  sayret_ + cesd_ + shlt_ + cancre_ + diabe_ + hearte_ + mobila_ + adl5a_ +
  (1 | id)

re_logit_female <- glmer(
  form_base,
  data   = df[df$female == 1, ],
  family = binomial,
  nAGQ   = 0,
  control = glmerControl(optimizer="bobyqa", calc.derivs=FALSE, optCtrl=list(maxfun=5e4))
)

re_logit_male <- glmer(
  form_base,
  data   = df[df$female == 0, ],
  family = binomial,
  nAGQ   = 0,
  control = glmerControl(optimizer="bobyqa", calc.derivs=FALSE, optCtrl=list(maxfun=5e4))
)

# ----------------------------
# 3) CRITICAL FIX: map raw age -> the model’s age2_ scale
#    (so you NEVER guess age2_ = age^2 again)
# ----------------------------
make_age2_mapper <- function(df, age_raw_col = "age_raw_plot", age2_col = "age2_") {
  d <- df %>%
    filter(!is.na(.data[[age_raw_col]]), !is.na(.data[[age2_col]])) %>%
    transmute(age_raw = .data[[age_raw_col]], age2 = .data[[age2_col]])
  
  # Collapse to a smooth-ish mapping using median age2 per rounded age
  d2 <- d %>%
    mutate(age_round = round(age_raw)) %>%
    group_by(age_round) %>%
    summarise(age2_med = median(age2, na.rm = TRUE), .groups = "drop") %>%
    arrange(age_round)
  
  # Return a function: age_raw -> age2_ (using linear interpolation)
  function(age_vec) {
    approx(x = d2$age_round, y = d2$age2_med, xout = age_vec, rule = 2)$y
  }
}

age2_from_raw <- make_age2_mapper(df)

# quick sanity check (optional):
# plot(round(df$age_raw_plot), df$age2_, pch=16, cex=.2); lines(40:90, age2_from_raw(40:90), col=2, lwd=2)

# ----------------------------
# 4) Parametric fixed-effect bootstrap bands (population-level)
# ----------------------------
bootstrap_curve_fixef <- function(model, age_raw_vec, ref_row, age2_mapper,
                                  B = 500, alpha = 0.05) {
  
  # Build newdata: replicate reference row and set age2_ according to mapper
  nd <- ref_row[rep(1, length(age_raw_vec)), , drop = FALSE]
  nd$age_raw_plot <- age_raw_vec                      # for your own tracking
  nd$age2_ <- age2_mapper(age_raw_vec)                # <-- the whole point
  # Make sure id exists (glmer formula has (1|id), even if re.form=NA)
  if (!("id" %in% names(nd))) nd$id <- factor("ref")
  if (is.factor(model@frame$id)) nd$id <- factor(nd$id, levels = levels(model@frame$id))
  
  # Base design matrix for fixed effects (drop random effect parts)
  X <- model.matrix(lme4::nobars(formula(model)), data = nd)
  beta_hat <- fixef(model)
  V <- as.matrix(vcov(model))
  
  # Align columns (just in case)
  X <- X[, names(beta_hat), drop = FALSE]
  
  # Draw betas
  betas <- MASS::mvrnorm(n = B, mu = beta_hat, Sigma = V)
  
  # Linear predictor + inverse link (population-level)
  eta_mat <- X %*% t(betas)
  p_mat <- plogis(eta_mat)
  
  p_mid  <- rowMeans(p_mat)
  p_low  <- apply(p_mat, 1, quantile, probs = alpha/2,     na.rm = TRUE, type = 7)
  p_high <- apply(p_mat, 1, quantile, probs = 1 - alpha/2, na.rm = TRUE, type = 7)
  
  list(p_mid = p_mid, p_low = p_low, p_high = p_high, newdata = nd)
}

# ----------------------------
# 5) Build reference rows (means within each race×sex stratum)
#    IMPORTANT: ref rows must include all predictors used in model
# ----------------------------
make_ref_row <- function(df, black_val, female_val) {
  df %>%
    filter(black == black_val, hispanic == 0, others == 0, female == female_val) %>%
    summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE))) %>%
    mutate(
      id = factor("ref"),
      # ensure indicators are exact for group definition
      black = black_val,
      hispanic = 0,
      others = 0,
      female = female_val
    )
}

ref_w_f <- make_ref_row(df, black_val = 0, female_val = 1)
ref_b_f <- make_ref_row(df, black_val = 1, female_val = 1)
ref_w_m <- make_ref_row(df, black_val = 0, female_val = 0)
ref_b_m <- make_ref_row(df, black_val = 1, female_val = 0)

# ----------------------------
# 6) Compute curves (2-year bins)
# ----------------------------
age_grid <- seq(40, 90, by = 2)

res_wf <- bootstrap_curve_fixef(re_logit_female, age_grid, ref_w_f, age2_from_raw, B = 500)
res_bf <- bootstrap_curve_fixef(re_logit_female, age_grid, ref_b_f, age2_from_raw, B = 500)
res_wm <- bootstrap_curve_fixef(re_logit_male,   age_grid, ref_w_m, age2_from_raw, B = 500)
res_bm <- bootstrap_curve_fixef(re_logit_male,   age_grid, ref_b_m, age2_from_raw, B = 500)

df_curve_wf <- data.frame(age = age_grid, p_mid=res_wf$p_mid, p_low=res_wf$p_low, p_high=res_wf$p_high,
                          race="NH-White", sex="Female")
df_curve_bf <- data.frame(age = age_grid, p_mid=res_bf$p_mid, p_low=res_bf$p_low, p_high=res_bf$p_high,
                          race="NH-Black", sex="Female")
df_curve_wm <- data.frame(age = age_grid, p_mid=res_wm$p_mid, p_low=res_wm$p_low, p_high=res_wm$p_high,
                          race="NH-White", sex="Male")
df_curve_bm <- data.frame(age = age_grid, p_mid=res_bm$p_mid, p_low=res_bm$p_low, p_high=res_bm$p_high,
                          race="NH-Black", sex="Male")

df_curves_all <- bind_rows(df_curve_wf, df_curve_bf, df_curve_wm, df_curve_bm)

# ----------------------------
# 7) Plot
# ----------------------------
p_comp_infer95 <- ggplot(
  df_curves_all,
  aes(x = age, color = race, fill = race)
) +
  geom_ribbon(
    aes(ymin = p_low, ymax = p_high),
    alpha = 0.18,
    color = NA
  ) +
  geom_line(aes(y = p_mid), linewidth = 1.1) +
  facet_wrap(~ sex, nrow = 1) +
  scale_x_continuous(breaks = c(50, 50, 60, 70, 80, 90)) +
  
  ## >>> THIS IS THE ONLY ADDITION <<<
  scale_color_manual(
    values = c(
      "NH-Black" = "#1f78b4",  # blue
      "NH-White" = "#e31a1c"   # red
    )
  ) +
  scale_fill_manual(
    values = c(
      "NH-Black" = "#1f78b4",
      "NH-White" = "#e31a1c"
    )
  ) +
  
  labs(
    x = "Age (2-year bins)",
    y = "Predicted discrete-time hazard (population-level)",
    title = "",
    color = "",
    fill  = ""
  ) +
  theme_minimal(base_size = 15)

print(p_comp_infer95)

## ================================================================
## Helper: metrics from probabilities + threshold
## ================================================================
metrics_from_prob <- function(y_true, p, thr = 0.50) {
  y_true <- as.integer(y_true)
  p      <- as.numeric(p)
  stopifnot(all(y_true %in% c(0L, 1L)))
  
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
  
  list(
    thr = thr,
    cm  = cm,
    tp = tp, tn = tn, fp = fp, fn = fn,
    acc = acc, sens = sens, spec = spec
  )
}


## ================================================================
## Eq.6: Prevalence-calibrated thresholding for classification
##   - τ chosen so that P(p_hat >= τ) ≈ π
##   - Implementation: τ = quantile(p_hat_train, 1 - π_train)
##   - Evaluation only (does not change regression estimates)
## ================================================================

#--- Train/Test split (canonical data object is df)
set.seed(142)
tr_idx   <- createDataPartition(df$died_, p = 0.80, list = FALSE)
df_train <- df[tr_idx, , drop = FALSE]
df_test  <- df[-tr_idx, , drop = FALSE]

#--- predicted probabilities (population-level) for TRAIN/TEST
p_train <- predict_pop_by_sex(df_train, re_logit_female, re_logit_male)
p_test  <- predict_pop_by_sex(df_test,  re_logit_female, re_logit_male)

y_train <- as.integer(df_train$died_)
y_test  <- as.integer(df_test$died_)

#--- π = observed mortality prevalence (person–interval)
pi_train <- mean(y_train == 1L, na.rm = TRUE)

#--- τ: choose threshold so predicted positive rate matches π
tau_prev <- as.numeric(quantile(p_train, probs = 1 - pi_train, na.rm = TRUE, type = 7))

# sanity check: should match pi_train closely
pred_pos_rate_train <- mean(p_train >= tau_prev, na.rm = TRUE)

cat(sprintf("\nEq.6 prevalence calibration on TRAIN:\n"))
cat(sprintf("  pi_train (observed prevalence) = %.6f\n", pi_train))
cat(sprintf("  tau_prev (1 - pi quantile)     = %.6f\n", tau_prev))
cat(sprintf("  P(p_hat >= tau_prev) on TRAIN  = %.6f\n\n", pred_pos_rate_train))

#--- Evaluate TEST at both thresholds
m_test_50   <- metrics_from_prob(y_test, p_test, thr = 0.50)
m_test_prev <- metrics_from_prob(y_test, p_test, thr = tau_prev)

cat("TEST @ thr=0.50\n")
print(m_test_50$cm)
cat(sprintf("acc=%.3f sens=%.3f spec=%.3f (TP=%d TN=%d FP=%d FN=%d)\n\n",
            m_test_50$acc, m_test_50$sens, m_test_50$spec,
            m_test_50$tp, m_test_50$tn, m_test_50$fp, m_test_50$fn))

cat("TEST @ prevalence-calibrated tau_prev\n")
print(m_test_prev$cm)
cat(sprintf("acc=%.3f sens=%.3f spec=%.3f (TP=%d TN=%d FP=%d FN=%d)\n\n",
            m_test_prev$acc, m_test_prev$sens, m_test_prev$spec,
            m_test_prev$tp, m_test_prev$tn, m_test_prev$fp, m_test_prev$fn))

#--- Optional: FULL sample evaluation (same tau_prev; evaluation only)
p_full <- predict_pop_by_sex(df, re_logit_female, re_logit_male)

m_full_50   <- metrics_from_prob(df$died_, p_full, thr = 0.50)
m_full_prev <- metrics_from_prob(df$died_, p_full, thr = tau_prev)

cat("FULL @ thr=0.50\n")
print(m_full_50$cm)
cat(sprintf("acc=%.3f sens=%.3f spec=%.3f\n\n",
            m_full_50$acc, m_full_50$sens, m_full_50$spec))

cat("FULL @ prevalence-calibrated tau_prev\n")
print(m_full_prev$cm)
cat(sprintf("acc=%.3f sens=%.3f spec=%.3f\n\n",
            m_full_prev$acc, m_full_prev$sens, m_full_prev$spec))


## ================================================================
## Eq.6 — Race × Sex subgroup performance (population-level)
##   NH-White / NH-Black × Male / Female
## ================================================================
subgroup_metrics_rxsex <- function(df, thr) {
  thr <- as.numeric(thr)
  
  df %>%
    mutate(
      race = case_when(
        black == 1 & hispanic == 0 & others == 0 ~ "NH-Black",
        black == 0 & hispanic == 0 & others == 0 ~ "NH-White",
        TRUE ~ NA_character_
      ),
      sex = ifelse(female == 1, "Female", "Male")
    ) %>%
    filter(!is.na(race)) %>%
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
      .groups = "drop"
    )
}

## ================================================================
## Build df_eval_rxsex (if you don't already have it)
##   Needs: df, p_full
## ================================================================
df_eval_rxsex <- df %>%
  mutate(p = as.numeric(p_full))

cat("\n==============================\n")
cat("Race × Sex metrics @ thr = 0.50\n")
cat("==============================\n")
print(subgroup_metrics_rxsex(df_eval_rxsex, thr = 0.50))

cat("\n===========================================\n")
cat("Race × Sex metrics @ prevalence-calibrated τ\n")
cat(sprintf("τ_prev = %.6f\n", tau_prev))
cat("===========================================\n")
print(subgroup_metrics_rxsex(df_eval_rxsex, thr = tau_prev))

