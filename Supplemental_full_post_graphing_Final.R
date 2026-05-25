###############################################################################
## POST-EVALUATION TABLES — FULL MODEL ONLY
## 10 ID-level splits × 20 training seeds
##
## NO ROW-LEVEL PREDICTION FILE SAVED.
##
## Outputs compact:
##   1) Race × sex performance summaries
##   2) Age-bin × race × sex performance summaries
##   3) Observed predicted-risk summaries
##   4) Black–White risk gaps and risk ratios
##   5) Inference-time ablation summaries:
##        Observed
##        AL = 0
##        Personality = 0
##        AL + Personality = 0
##   6) Scenario-vs-observed risk/performance changes
##   7) Scenario-vs-observed Black–White gap/RR changes
##   8) Representational Probe 2:
##        Age-specific observed / held-constant predicted-risk ratios
##
## Interpretation:
##   These are inference-time ablations. They are NOT causal effects.
##   They show whether AL/personality domains are vital sources of predictive
##   information when trained parameters are held fixed.
###############################################################################
windowsFonts(TNR = windowsFont("Times New Roman"))

packages_needed <- c("torch", "dplyr", "tidyr", "readr", "ggplot2")

to_install <- packages_needed[!packages_needed %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(packages_needed, library, character.only = TRUE))

setwd("yourdatalocation")

root_ckpt_dir <- "checkpoints_fullspec_IDsplit_multisplit"
split_meta_dir <- file.path(root_ckpt_dir, "split_meta")

out_dir <- file.path(root_ckpt_dir, "post_eval_tables_full_only_compact")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

###############################################################################
## AGE-BIN SETTINGS — EDIT HERE FOR AGE-SPECIFIC TABLES / PLOTS
###############################################################################
AGE_BIN_MIN <- 50
AGE_BIN_WIDTH <- 5
AGE_BIN_TOPCODE <- 90

## If TRUE, observations below AGE_BIN_MIN get assigned to the first bin.
## If FALSE, they are excluded from age-bin summaries.
AGE_BIN_INCLUDE_BELOW_MIN <- FALSE

age_bin_breaks <- seq(AGE_BIN_MIN, AGE_BIN_TOPCODE, by = AGE_BIN_WIDTH)

age_bin_config <- data.frame(
  age_bin_min = AGE_BIN_MIN,
  age_bin_width = AGE_BIN_WIDTH,
  age_bin_topcode = AGE_BIN_TOPCODE,
  include_below_min = AGE_BIN_INCLUDE_BELOW_MIN
)

write_csv(
  age_bin_config,
  file.path(out_dir, "00_age_bin_config.csv")
)

###############################################################################
## 0) Load global training image so model class exists
###############################################################################
global_image <- file.path(root_ckpt_dir, "TRAINING_fullspec_IDsplit_multisplit_10x20.RData")

if (!file.exists(global_image)) {
  stop("Global training image not found: ", global_image)
}

load(global_image)
stopifnot(exists("hazard_mlp_full"))

###############################################################################
## 1) General helpers
###############################################################################
is_torch_tensor <- function(x) {
  if (exists("torch_is_tensor", mode = "function")) return(torch_is_tensor(x))
  inherits(x, "torch_tensor")
}

tN <- function(x) as.integer(x$size()[1])
tP <- function(x) as.integer(x$size()[2])

torch_to_array <- function(x) {
  as_array(x$to(device = torch_device("cpu")))
}

safe_mean <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

safe_sd <- function(x) {
  if (sum(!is.na(x)) <= 1) return(NA_real_)
  sd(x, na.rm = TRUE)
}

safe_min <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  min(x, na.rm = TRUE)
}

safe_max <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  max(x, na.rm = TRUE)
}

safe_file <- function(path_from_registry, split_name, seed) {
  if (!is.na(path_from_registry) && file.exists(path_from_registry)) {
    return(path_from_registry)
  }
  
  alt <- file.path(root_ckpt_dir, split_name, sprintf("seed_%03d.pt", as.integer(seed)))
  if (file.exists(alt)) return(alt)
  
  stop("Checkpoint not found for ", split_name, " seed ", seed)
}

make_age_bin <- function(age_raw) {
  if (isTRUE(AGE_BIN_INCLUDE_BELOW_MIN)) {
    case_when(
      is.na(age_raw) ~ NA_integer_,
      age_raw >= AGE_BIN_TOPCODE ~ AGE_BIN_TOPCODE,
      age_raw < AGE_BIN_MIN ~ AGE_BIN_MIN,
      TRUE ~ as.integer(
        floor((age_raw - AGE_BIN_MIN) / AGE_BIN_WIDTH) * AGE_BIN_WIDTH + AGE_BIN_MIN
      )
    )
  } else {
    case_when(
      is.na(age_raw) ~ NA_integer_,
      age_raw < AGE_BIN_MIN ~ NA_integer_,
      age_raw >= AGE_BIN_TOPCODE ~ AGE_BIN_TOPCODE,
      TRUE ~ as.integer(
        floor((age_raw - AGE_BIN_MIN) / AGE_BIN_WIDTH) * AGE_BIN_WIDTH + AGE_BIN_MIN
      )
    )
  }
}

age_bin_label <- function(age_bin) {
  case_when(
    is.na(age_bin) ~ NA_character_,
    age_bin == AGE_BIN_TOPCODE ~ paste0(AGE_BIN_TOPCODE, "+"),
    TRUE ~ paste0(age_bin, "-", age_bin + AGE_BIN_WIDTH - 1)
  )
}

metrics_counts <- function(pred01, truth01) {
  pred01  <- as.integer(pred01)
  truth01 <- as.integer(truth01)
  
  tn <- sum(pred01 == 0L & truth01 == 0L, na.rm = TRUE)
  tp <- sum(pred01 == 1L & truth01 == 1L, na.rm = TRUE)
  fn <- sum(pred01 == 0L & truth01 == 1L, na.rm = TRUE)
  fp <- sum(pred01 == 1L & truth01 == 0L, na.rm = TRUE)
  
  n <- tp + tn + fp + fn
  events <- tp + fn
  nonevents <- tn + fp
  
  acc  <- if (n > 0) (tp + tn) / n else NA_real_
  sens <- if (events > 0) tp / events else NA_real_
  spec <- if (nonevents > 0) tn / nonevents else NA_real_
  fnr  <- if (events > 0) fn / events else NA_real_
  fpr  <- if (nonevents > 0) fp / nonevents else NA_real_
  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  
  data.frame(
    n = n,
    events = events,
    nonevents = nonevents,
    tp = tp,
    tn = tn,
    fp = fp,
    fn = fn,
    acc = acc,
    sens = sens,
    spec = spec,
    fnr = fnr,
    fpr = fpr,
    precision = precision
  )
}

summarise_seed_to_split <- function(df, group_vars, value_vars) {
  df %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      across(
        all_of(value_vars),
        list(
          mean = ~ safe_mean(.x),
          sd   = ~ safe_sd(.x)
        ),
        .names = "{.col}_{.fn}"
      ),
      n_seeds = n_distinct(seed),
      .groups = "drop"
    )
}

summarise_split_to_headline <- function(df, group_vars, value_vars) {
  df %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      n_splits = n_distinct(split_name),
      across(
        all_of(value_vars),
        list(
          mean = ~ safe_mean(.x),
          sd   = ~ safe_sd(.x),
          min  = ~ safe_min(.x),
          max  = ~ safe_max(.x)
        ),
        .names = "{.col}_{.fn}"
      ),
      .groups = "drop"
    )
}

predict_logits <- function(model, x_t, mask_t, id_t,
                           use_alpers = TRUE,
                           batch_size = 1024) {
  model$eval()
  
  if (!is_torch_tensor(x_t)) {
    x_t <- torch_tensor(x_t, dtype = torch_float())
  }
  if (!is_torch_tensor(mask_t)) {
    mask_t <- torch_tensor(mask_t, dtype = torch_float())
  }
  if (!is_torch_tensor(id_t)) {
    id_t <- torch_tensor(id_t, dtype = torch_long())
  }
  
  n_all <- tN(x_t)
  n_batches <- ceiling(n_all / batch_size)
  out_list <- vector("list", n_batches)
  
  for (b in seq_len(n_batches)) {
    idx <- ((b - 1L) * batch_size + 1L):min(b * batch_size, n_all)
    
    with_no_grad({
      out_list[[b]] <- model(
        x_t[idx, , drop = FALSE],
        mask_t[idx, , drop = FALSE],
        id_t[idx],
        use_alpers = use_alpers
      )
    })
  }
  
  torch_cat(out_list, dim = 1L)
}

make_zero_cf <- function(x_t, mask_t, cols_idx, zero_mask_also = TRUE) {
  if (!is_torch_tensor(x_t)) {
    x_t <- torch_tensor(x_t, dtype = torch_float())
  }
  if (!is_torch_tensor(mask_t)) {
    mask_t <- torch_tensor(mask_t, dtype = torch_float())
  }
  
  x_cf <- x_t$clone()
  m_cf <- mask_t$clone()
  
  cols_idx <- as.integer(cols_idx)
  cols_idx <- cols_idx[!is.na(cols_idx)]
  cols_idx <- cols_idx[cols_idx >= 1L & cols_idx <= tP(x_cf)]
  
  if (length(cols_idx) > 0L) {
    x_cf[, cols_idx] <- 0
    if (isTRUE(zero_mask_also)) {
      m_cf[, cols_idx] <- 0
    }
  }
  
  list(x = x_cf, mask = m_cf, cols = cols_idx)
}

get_age_like_cols <- function(all_cols) {
  which(grepl(
    "^age(_|p_)?$|^age_raw$|^age_[0-9]+$|^age_raw_[0-9]+$|^agep_[0-9]+$|^agep_raw_[0-9]+$",
    all_cols
  ))
}

strip_age_from_zero <- function(cols_idx, all_cols, verbose = TRUE) {
  age_like_idx <- get_age_like_cols(all_cols)
  
  cols_idx <- as.integer(cols_idx)
  cols_idx <- cols_idx[!is.na(cols_idx)]
  
  bad <- intersect(cols_idx, age_like_idx)
  
  if (verbose && length(bad) > 0L) {
    cat("WARNING: removing age-like columns from ZERO set:\n")
    print(all_cols[bad])
  }
  
  setdiff(cols_idx, age_like_idx)
}

make_model_from_env <- function(e) {
  with(e, {
    hazard_mlp_full(
      static_idx       = static_idx,
      tv_idx           = tv_idx,
      n_steps          = n_steps,
      tv_step_feat_dim = tv_step_feat_dim,
      age_idx          = age_idx,
      age_raw_idx      = age_raw_idx,
      al1_idx          = al1_idx,
      al3_idx          = al3_idx,
      al5_idx          = al5_idx,
      al_age_idx       = al_age_idx,
      al_age_raw_idx   = al_age_raw_idx,
      pers1_idx        = pers1_idx,
      pers3_idx        = pers3_idx,
      pers5_idx        = pers5_idx,
      pers_age_idx     = pers_age_idx,
      pers_age_raw_idx = pers_age_raw_idx,
      num_ids          = num_ids,
      id_emb_dim       = id_emb_dim,
      h_static         = 64,
      h_tv             = 64,
      h_al             = 64,
      h_pers           = 64,
      mix_dim          = 32,
      output_dim       = output_dim,
      drop_p           = 0.25
    )
  })
}

get_age_raw_from_env <- function(e, test_row_ids) {
  d <- e$data_surv_al
  
  if ("age_raw" %in% names(d)) {
    return(as.numeric(d[test_row_ids, "age_raw"]))
  }
  
  if ("age_" %in% names(d)) {
    ## In training, age_ was logged after preserving age_raw.
    return(exp(as.numeric(d[test_row_ids, "age_"])))
  }
  
  stop("Could not recover raw age from data_surv_al.")
}

###############################################################################
## 2) Find split states
###############################################################################
split_state_paths <- list.files(
  split_meta_dir,
  pattern = "^split_[0-9]+_training_state\\.RData$",
  full.names = TRUE
)

split_state_paths <- sort(split_state_paths)

if (length(split_state_paths) == 0L) {
  stop("No split-level training state files found in: ", split_meta_dir)
}

###############################################################################
## 3) Initialize compact summary lists
###############################################################################
metrics_race_sex_seed_list <- list()
metrics_agebin_seed_list   <- list()
risk_agebin_seed_list      <- list()

cf_risk_seed_list <- list()
cf_metrics_race_sex_seed_list <- list()
cf_risk_race_sex_seed_list <- list()

###############################################################################
## 4) Loop over split states and checkpoints
###############################################################################
for (split_path in split_state_paths) {
  
  e <- new.env(parent = globalenv())
  load(split_path, envir = e)
  
  cat("\n====================================================\n")
  cat("Processing:", e$split_name, "| split seed:", e$split_seed, "\n")
  cat("====================================================\n")
  
  all_cols <- e$all_cols
  
  ## ------------------------------------------------
  ## Safe counterfactual zero sets
  ## ------------------------------------------------
  al_zero_idx_safe <- strip_age_from_zero(
    e$al_zero_idx,
    all_cols,
    verbose = TRUE
  )
  
  pers_zero_idx_safe <- strip_age_from_zero(
    e$pers_zero_idx,
    all_cols,
    verbose = TRUE
  )
  
  both_zero_idx_safe <- strip_age_from_zero(
    e$both_zero_idx,
    all_cols,
    verbose = TRUE
  )
  
  cf_defs <- list(
    Observed = list(cols = integer(0), zero_mask_also = TRUE),
    `AL=0` = list(cols = al_zero_idx_safe, zero_mask_also = TRUE),
    `Personality=0` = list(cols = pers_zero_idx_safe, zero_mask_also = TRUE),
    `AL+Personality=0` = list(cols = both_zero_idx_safe, zero_mask_also = TRUE)
  )
  
  ## ------------------------------------------------
  ## Rebuild test tensors fresh
  ## ------------------------------------------------
  te <- e$te
  core_row_ids <- which(e$keep_rows)
  test_row_ids <- core_row_ids[te]
  
  xt_t <- torch_tensor(e$X_scaled[te, , drop = FALSE], dtype = torch_float())
  mt_t <- torch_tensor(e$na_mask_mat[te, , drop = FALSE], dtype = torch_float())
  id_t <- torch_tensor(e$id_index_core[te], dtype = torch_long())
  
  truth01 <- as.integer(e$df_core$died_[te])
  
  age_raw <- get_age_raw_from_env(e, test_row_ids)
  
  meta_test <- e$data_surv_al[test_row_ids, , drop = FALSE] %>%
    transmute(
      id = id,
      race = case_when(
        black == 1 & hispanic == 0 & others == 0 ~ "NH-Black",
        black == 0 & hispanic == 0 & others == 0 ~ "NH-White",
        TRUE ~ "Other"
      ),
      sex = if_else(female == 1, "Female", "Male"),
      age_raw = as.numeric(age_raw),
      age_bin = make_age_bin(age_raw),
      age_bin_label = age_bin_label(age_bin)
    )
  
  stopifnot(nrow(meta_test) == length(truth01))
  
  results_df <- e$results_df %>%
    arrange(seed)
  
  ## ------------------------------------------------
  ## Loop over seeds/checkpoints
  ## ------------------------------------------------
  for (rr in seq_len(nrow(results_df))) {
    
    seed <- results_df$seed[rr]
    ckpt <- safe_file(
      path_from_registry = results_df$ckpt_path[rr],
      split_name = e$split_name,
      seed = seed
    )
    
    cat("  Loading seed:", seed, "\n")
    
    model <- make_model_from_env(e)
    model$load_state_dict(torch_load(ckpt))
    model$eval()
    
    ## ============================================================
    ## Observed prediction
    ## ============================================================
    logits_obs_t <- predict_logits(
      model = model,
      x_t = xt_t,
      mask_t = mt_t,
      id_t = id_t,
      use_alpers = TRUE
    )
    
    logits_obs <- as.matrix(torch_to_array(logits_obs_t))
    if (ncol(logits_obs) != 2L) {
      stop("Expected binary logits with 2 columns.")
    }
    
    probs_obs <- as.matrix(torch_to_array(nnf_softmax(logits_obs_t, dim = 2L)))
    p_death <- as.numeric(probs_obs[, 2])
    
    ## Argmax rule: event predicted when logit1 >= logit0
    pred01 <- as.integer(logits_obs[, 2] >= logits_obs[, 1])
    
    pred_df <- meta_test %>%
      mutate(
        split_name = e$split_name,
        split_seed = e$split_seed,
        seed = seed,
        truth01 = truth01,
        pred01 = pred01,
        p_death = p_death
      )
    
    ## ------------------------------------------------------------
    ## 1) Race × sex test metrics
    ## ------------------------------------------------------------
    metrics_race_sex_seed_list[[length(metrics_race_sex_seed_list) + 1L]] <-
      pred_df %>%
      filter(race %in% c("NH-White", "NH-Black")) %>%
      group_by(split_name, split_seed, seed, race, sex) %>%
      summarise(
        metrics_counts(pred01, truth01),
        .groups = "drop"
      )
    
    ## ------------------------------------------------------------
    ## 2) Age-bin × race × sex test metrics
    ## ------------------------------------------------------------
    metrics_agebin_seed_list[[length(metrics_agebin_seed_list) + 1L]] <-
      pred_df %>%
      filter(
        race %in% c("NH-White", "NH-Black"),
        !is.na(age_bin)
      ) %>%
      group_by(split_name, split_seed, seed, age_bin, age_bin_label, race, sex) %>%
      summarise(
        metrics_counts(pred01, truth01),
        .groups = "drop"
      )
    
    ## ------------------------------------------------------------
    ## 3) Age-bin × race × sex observed predicted risk
    ## ------------------------------------------------------------
    risk_agebin_seed_list[[length(risk_agebin_seed_list) + 1L]] <-
      pred_df %>%
      filter(
        race %in% c("NH-White", "NH-Black"),
        !is.na(age_bin)
      ) %>%
      group_by(split_name, split_seed, seed, age_bin, age_bin_label, race, sex) %>%
      summarise(
        n = n(),
        events = sum(truth01 == 1L, na.rm = TRUE),
        observed_event_rate = mean(truth01 == 1L, na.rm = TRUE),
        mean_predicted_risk = mean(p_death, na.rm = TRUE),
        median_predicted_risk = median(p_death, na.rm = TRUE),
        .groups = "drop"
      )
    
    ## ============================================================
    ## Inference-time ablation / counterfactual scenarios
    ## ============================================================
    for (scenario in names(cf_defs)) {
      
      def <- cf_defs[[scenario]]
      
      if (length(def$cols) == 0L) {
        logits_cf_t <- logits_obs_t
        probs_cf <- probs_obs
      } else {
        cf <- make_zero_cf(
          x_t = xt_t,
          mask_t = mt_t,
          cols_idx = def$cols,
          zero_mask_also = def$zero_mask_also
        )
        
        logits_cf_t <- predict_logits(
          model = model,
          x_t = cf$x,
          mask_t = cf$mask,
          id_t = id_t,
          use_alpers = TRUE
        )
        
        probs_cf <- as.matrix(torch_to_array(nnf_softmax(logits_cf_t, dim = 2L)))
      }
      
      logits_cf <- as.matrix(torch_to_array(logits_cf_t))
      p_cf <- as.numeric(probs_cf[, 2])
      
      ## Argmax rule under scenario
      pred01_cf <- as.integer(logits_cf[, 2] >= logits_cf[, 1])
      
      cf_pred_df <- meta_test %>%
        mutate(
          split_name = e$split_name,
          split_seed = e$split_seed,
          seed = seed,
          scenario = scenario,
          truth01 = truth01,
          pred01 = pred01_cf,
          p_death = p_cf
        )
      
      ## ------------------------------------------------------------
      ## Scenario × race × sex predicted risk
      ## ------------------------------------------------------------
      cf_risk_race_sex_seed_list[[length(cf_risk_race_sex_seed_list) + 1L]] <-
        cf_pred_df %>%
        filter(race %in% c("NH-White", "NH-Black")) %>%
        group_by(split_name, split_seed, seed, scenario, race, sex) %>%
        summarise(
          n = n(),
          events = sum(truth01 == 1L, na.rm = TRUE),
          observed_event_rate = mean(truth01 == 1L, na.rm = TRUE),
          mean_predicted_risk = mean(p_death, na.rm = TRUE),
          median_predicted_risk = median(p_death, na.rm = TRUE),
          .groups = "drop"
        )
      
      ## ------------------------------------------------------------
      ## Scenario × race × sex classification performance
      ## ------------------------------------------------------------
      cf_metrics_race_sex_seed_list[[length(cf_metrics_race_sex_seed_list) + 1L]] <-
        cf_pred_df %>%
        filter(race %in% c("NH-White", "NH-Black")) %>%
        group_by(split_name, split_seed, seed, scenario, race, sex) %>%
        summarise(
          metrics_counts(pred01, truth01),
          .groups = "drop"
        )
      
      ## ------------------------------------------------------------
      ## Scenario × age-bin × race × sex predicted risk
      ## ------------------------------------------------------------
      cf_risk_seed_list[[length(cf_risk_seed_list) + 1L]] <-
        cf_pred_df %>%
        filter(
          race %in% c("NH-White", "NH-Black"),
          !is.na(age_bin)
        ) %>%
        group_by(split_name, split_seed, seed, scenario, age_bin, age_bin_label, race, sex) %>%
        summarise(
          n = n(),
          events = sum(truth01 == 1L, na.rm = TRUE),
          observed_event_rate = mean(truth01 == 1L, na.rm = TRUE),
          mean_predicted_risk = mean(p_death, na.rm = TRUE),
          median_predicted_risk = median(p_death, na.rm = TRUE),
          .groups = "drop"
        )
    }
    
    rm(model)
    gc()
  }
}

###############################################################################
## 5) Bind compact per-seed summaries
###############################################################################
metrics_race_sex_seed <- bind_rows(metrics_race_sex_seed_list)
metrics_agebin_seed   <- bind_rows(metrics_agebin_seed_list)
risk_agebin_seed      <- bind_rows(risk_agebin_seed_list)

cf_risk_seed <- bind_rows(cf_risk_seed_list)
cf_risk_race_sex_seed <- bind_rows(cf_risk_race_sex_seed_list)
cf_metrics_race_sex_seed <- bind_rows(cf_metrics_race_sex_seed_list)

write_csv(
  metrics_race_sex_seed,
  file.path(out_dir, "01a_seed_metrics_race_sex.csv")
)

write_csv(
  metrics_agebin_seed,
  file.path(out_dir, "02a_seed_metrics_agebin_race_sex.csv")
)

write_csv(
  risk_agebin_seed,
  file.path(out_dir, "03a_seed_observed_risk_agebin_race_sex.csv")
)

write_csv(
  cf_risk_seed,
  file.path(out_dir, "05a_seed_counterfactual_risk_agebin_race_sex_scenario.csv")
)

write_csv(
  cf_risk_race_sex_seed,
  file.path(out_dir, "09a_seed_counterfactual_risk_race_sex.csv")
)

write_csv(
  cf_metrics_race_sex_seed,
  file.path(out_dir, "09b_seed_counterfactual_metrics_race_sex.csv")
)

###############################################################################
## 6) Race × sex metrics: seed -> split -> headline
###############################################################################
metrics_race_sex_split <- summarise_seed_to_split(
  df = metrics_race_sex_seed,
  group_vars = c("split_name", "split_seed", "race", "sex"),
  value_vars = c("n", "events", "tp", "tn", "fp", "fn", "acc", "sens", "spec", "fnr", "fpr", "precision")
)

metrics_race_sex_headline <- summarise_split_to_headline(
  df = metrics_race_sex_split,
  group_vars = c("race", "sex"),
  value_vars = c(
    "n_mean", "events_mean",
    "tp_mean", "tn_mean", "fp_mean", "fn_mean",
    "acc_mean", "sens_mean", "spec_mean", "fnr_mean", "fpr_mean", "precision_mean"
  )
)

write_csv(
  metrics_race_sex_split,
  file.path(out_dir, "01b_split_metrics_race_sex_seed_averaged.csv")
)

write_csv(
  metrics_race_sex_headline,
  file.path(out_dir, "01c_headline_metrics_race_sex_across_splits.csv")
)

###############################################################################
## 7) Age-bin × race × sex metrics
###############################################################################
metrics_agebin_split <- summarise_seed_to_split(
  df = metrics_agebin_seed,
  group_vars = c("split_name", "split_seed", "age_bin", "age_bin_label", "race", "sex"),
  value_vars = c("n", "events", "tp", "tn", "fp", "fn", "acc", "sens", "spec", "fnr", "fpr", "precision")
)

metrics_agebin_headline <- summarise_split_to_headline(
  df = metrics_agebin_split,
  group_vars = c("age_bin", "age_bin_label", "race", "sex"),
  value_vars = c(
    "n_mean", "events_mean",
    "tp_mean", "tn_mean", "fp_mean", "fn_mean",
    "acc_mean", "sens_mean", "spec_mean", "fnr_mean", "fpr_mean", "precision_mean"
  )
)

write_csv(
  metrics_agebin_split,
  file.path(out_dir, "02b_split_metrics_agebin_race_sex_seed_averaged.csv")
)

write_csv(
  metrics_agebin_headline,
  file.path(out_dir, "02c_headline_metrics_agebin_race_sex_across_splits.csv")
)

###############################################################################
## 8) Observed predicted risk by age bin × race × sex
###############################################################################
risk_agebin_split <- summarise_seed_to_split(
  df = risk_agebin_seed,
  group_vars = c("split_name", "split_seed", "age_bin", "age_bin_label", "race", "sex"),
  value_vars = c("n", "events", "observed_event_rate", "mean_predicted_risk", "median_predicted_risk")
)

risk_agebin_headline <- summarise_split_to_headline(
  df = risk_agebin_split,
  group_vars = c("age_bin", "age_bin_label", "race", "sex"),
  value_vars = c(
    "n_mean",
    "events_mean",
    "observed_event_rate_mean",
    "mean_predicted_risk_mean",
    "median_predicted_risk_mean"
  )
)

write_csv(
  risk_agebin_split,
  file.path(out_dir, "03b_split_observed_risk_agebin_race_sex_seed_averaged.csv")
)

write_csv(
  risk_agebin_headline,
  file.path(out_dir, "03c_headline_observed_risk_agebin_race_sex_across_splits.csv")
)

###############################################################################
## 9) Observed Black–White risk gap and RR by age bin × sex
###############################################################################
eps <- 1e-12

risk_gap_rr_split <- risk_agebin_split %>%
  select(
    split_name,
    split_seed,
    age_bin,
    age_bin_label,
    sex,
    race,
    mean_predicted_risk_mean,
    median_predicted_risk_mean,
    observed_event_rate_mean,
    n_mean,
    events_mean
  ) %>%
  pivot_wider(
    names_from = race,
    values_from = c(
      mean_predicted_risk_mean,
      median_predicted_risk_mean,
      observed_event_rate_mean,
      n_mean,
      events_mean
    ),
    names_sep = "__"
  ) %>%
  mutate(
    predicted_risk_gap_black_minus_white =
      `mean_predicted_risk_mean__NH-Black` - `mean_predicted_risk_mean__NH-White`,
    
    predicted_risk_ratio_black_white =
      pmax(`mean_predicted_risk_mean__NH-Black`, eps) /
      pmax(`mean_predicted_risk_mean__NH-White`, eps),
    
    observed_rate_gap_black_minus_white =
      `observed_event_rate_mean__NH-Black` - `observed_event_rate_mean__NH-White`,
    
    observed_rate_ratio_black_white =
      pmax(`observed_event_rate_mean__NH-Black`, eps) /
      pmax(`observed_event_rate_mean__NH-White`, eps)
  )

risk_gap_rr_headline <- summarise_split_to_headline(
  df = risk_gap_rr_split,
  group_vars = c("age_bin", "age_bin_label", "sex"),
  value_vars = c(
    "predicted_risk_gap_black_minus_white",
    "predicted_risk_ratio_black_white",
    "observed_rate_gap_black_minus_white",
    "observed_rate_ratio_black_white"
  )
)

write_csv(
  risk_gap_rr_split,
  file.path(out_dir, "04a_split_observed_risk_gap_rr_agebin_sex.csv")
)

write_csv(
  risk_gap_rr_headline,
  file.path(out_dir, "04b_headline_observed_risk_gap_rr_agebin_sex.csv")
)

###############################################################################
## 10) Counterfactual risk by scenario × age bin × race × sex
###############################################################################
cf_risk_split <- summarise_seed_to_split(
  df = cf_risk_seed,
  group_vars = c("split_name", "split_seed", "scenario", "age_bin", "age_bin_label", "race", "sex"),
  value_vars = c("n", "events", "observed_event_rate", "mean_predicted_risk", "median_predicted_risk")
)

cf_risk_headline <- summarise_split_to_headline(
  df = cf_risk_split,
  group_vars = c("scenario", "age_bin", "age_bin_label", "race", "sex"),
  value_vars = c(
    "n_mean",
    "events_mean",
    "observed_event_rate_mean",
    "mean_predicted_risk_mean",
    "median_predicted_risk_mean"
  )
)

write_csv(
  cf_risk_split,
  file.path(out_dir, "05b_split_counterfactual_risk_agebin_race_sex_scenario.csv")
)

write_csv(
  cf_risk_headline,
  file.path(out_dir, "05c_headline_counterfactual_risk_agebin_race_sex_scenario.csv")
)

###############################################################################
## 11) Counterfactual risk change vs observed, by age bin × race × sex
###############################################################################
cf_obs_risk_split <- cf_risk_split %>%
  filter(scenario == "Observed") %>%
  select(
    split_name,
    split_seed,
    age_bin,
    age_bin_label,
    race,
    sex,
    observed_mean_predicted_risk = mean_predicted_risk_mean,
    observed_median_predicted_risk = median_predicted_risk_mean
  )

cf_risk_change_split <- cf_risk_split %>%
  filter(scenario != "Observed") %>%
  left_join(
    cf_obs_risk_split,
    by = c("split_name", "split_seed", "age_bin", "age_bin_label", "race", "sex")
  ) %>%
  mutate(
    mean_risk_change_vs_observed =
      mean_predicted_risk_mean - observed_mean_predicted_risk,
    
    mean_risk_reduction_from_observed =
      observed_mean_predicted_risk - mean_predicted_risk_mean,
    
    pct_mean_risk_reduction =
      ifelse(
        abs(observed_mean_predicted_risk) > eps,
        mean_risk_reduction_from_observed / abs(observed_mean_predicted_risk),
        NA_real_
      ),
    
    mean_risk_ratio_observed_over_held =
      pmax(observed_mean_predicted_risk, eps) /
      pmax(mean_predicted_risk_mean, eps),
    
    mean_risk_ratio_held_over_observed =
      pmax(mean_predicted_risk_mean, eps) /
      pmax(observed_mean_predicted_risk, eps),
    
    median_risk_change_vs_observed =
      median_predicted_risk_mean - observed_median_predicted_risk,
    
    median_risk_reduction_from_observed =
      observed_median_predicted_risk - median_predicted_risk_mean,
    
    median_risk_ratio_observed_over_held =
      pmax(observed_median_predicted_risk, eps) /
      pmax(median_predicted_risk_mean, eps)
  )

cf_risk_change_headline <- summarise_split_to_headline(
  df = cf_risk_change_split,
  group_vars = c("scenario", "age_bin", "age_bin_label", "race", "sex"),
  value_vars = c(
    "mean_risk_change_vs_observed",
    "mean_risk_reduction_from_observed",
    "pct_mean_risk_reduction",
    "mean_risk_ratio_observed_over_held",
    "mean_risk_ratio_held_over_observed",
    "median_risk_change_vs_observed",
    "median_risk_reduction_from_observed",
    "median_risk_ratio_observed_over_held"
  )
)

write_csv(
  cf_risk_change_split,
  file.path(out_dir, "05d_split_counterfactual_risk_change_vs_observed_agebin_race_sex.csv")
)

write_csv(
  cf_risk_change_headline,
  file.path(out_dir, "05e_headline_counterfactual_risk_change_vs_observed_agebin_race_sex.csv")
)

###############################################################################
## 11b) Representational Probe 2:
##      Age-specific observed / held-constant predicted-risk ratios
###############################################################################

probe2_agebin_riskdiff_plot <- cf_risk_change_headline %>%
  mutate(
    scenario_label = case_when(
      scenario == "AL+Personality=0" ~ "AL + Personality held constant",
      scenario == "AL=0" ~ "AL held constant",
      scenario == "Personality=0" ~ "Personality held constant",
      TRUE ~ scenario
    ),
    
    ## Probability difference: observed - held constant
    risk_diff_pp = 100 * mean_risk_reduction_from_observed_mean,
    risk_diff_sd = 100 * mean_risk_reduction_from_observed_sd,
    risk_diff_low = risk_diff_pp - risk_diff_sd,
    risk_diff_high = risk_diff_pp + risk_diff_sd
  ) %>%
  arrange(scenario_label, race, sex, age_bin)

write_csv(
  probe2_agebin_riskdiff_plot,
  file.path(out_dir, "18_probe2_agebin_observed_minus_heldconstant_risk_difference.csv")
)

p_probe2_diff <- ggplot(
  probe2_agebin_riskdiff_plot,
  aes(
    x = age_bin,
    y = risk_diff_pp,
    color = scenario_label,
    fill = scenario_label,
    group = scenario_label
  )
) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.4
  ) +
  geom_ribbon(
    aes(ymin = risk_diff_low, ymax = risk_diff_high),
    alpha = 0.14,
    color = NA
  ) +
  geom_line(linewidth = 1.05) +
  geom_point(size = 1.8) +
  facet_grid(sex ~ race) +
  scale_x_continuous(
    breaks = sort(unique(probe2_agebin_riskdiff_plot$age_bin)),
    labels = sort(unique(probe2_agebin_riskdiff_plot$age_bin_label))
  ) +
  labs(
    x = "Age",
    y = "Predicted mortality-risk difference\n percentage points",
    color = "",
    fill = ""
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p_probe2_diff)

ggsave(
  filename = file.path(out_dir, "18_probe2_agebin_observed_minus_heldconstant_risk_difference.png"),
  plot = p_probe2_diff,
  width = 11,
  height = 7,
  dpi = 600
)

###############################################################################
## 12) Counterfactual Black–White gap and RR by scenario × age bin × sex
###############################################################################
cf_gap_rr_split <- cf_risk_split %>%
  select(
    split_name,
    split_seed,
    scenario,
    age_bin,
    age_bin_label,
    sex,
    race,
    mean_predicted_risk_mean,
    median_predicted_risk_mean,
    observed_event_rate_mean,
    n_mean,
    events_mean
  ) %>%
  pivot_wider(
    names_from = race,
    values_from = c(
      mean_predicted_risk_mean,
      median_predicted_risk_mean,
      observed_event_rate_mean,
      n_mean,
      events_mean
    ),
    names_sep = "__"
  ) %>%
  mutate(
    predicted_risk_gap_black_minus_white =
      `mean_predicted_risk_mean__NH-Black` - `mean_predicted_risk_mean__NH-White`,
    
    predicted_risk_ratio_black_white =
      pmax(`mean_predicted_risk_mean__NH-Black`, eps) /
      pmax(`mean_predicted_risk_mean__NH-White`, eps),
    
    median_risk_gap_black_minus_white =
      `median_predicted_risk_mean__NH-Black` - `median_predicted_risk_mean__NH-White`,
    
    median_risk_ratio_black_white =
      pmax(`median_predicted_risk_mean__NH-Black`, eps) /
      pmax(`median_predicted_risk_mean__NH-White`, eps),
    
    observed_rate_gap_black_minus_white =
      `observed_event_rate_mean__NH-Black` - `observed_event_rate_mean__NH-White`,
    
    observed_rate_ratio_black_white =
      pmax(`observed_event_rate_mean__NH-Black`, eps) /
      pmax(`observed_event_rate_mean__NH-White`, eps)
  )

cf_gap_rr_headline <- summarise_split_to_headline(
  df = cf_gap_rr_split,
  group_vars = c("scenario", "age_bin", "age_bin_label", "sex"),
  value_vars = c(
    "predicted_risk_gap_black_minus_white",
    "predicted_risk_ratio_black_white",
    "median_risk_gap_black_minus_white",
    "median_risk_ratio_black_white",
    "observed_rate_gap_black_minus_white",
    "observed_rate_ratio_black_white"
  )
)

write_csv(
  cf_gap_rr_split,
  file.path(out_dir, "06a_split_counterfactual_gap_rr_agebin_sex_scenario.csv")
)

write_csv(
  cf_gap_rr_headline,
  file.path(out_dir, "06b_headline_counterfactual_gap_rr_agebin_sex_scenario.csv")
)

###############################################################################
## 13) Counterfactual Black–White gap/RR change vs observed
###############################################################################
obs_gap_rr_split <- cf_gap_rr_split %>%
  filter(scenario == "Observed") %>%
  select(
    split_name,
    split_seed,
    age_bin,
    age_bin_label,
    sex,
    observed_predicted_gap = predicted_risk_gap_black_minus_white,
    observed_predicted_rr  = predicted_risk_ratio_black_white,
    observed_median_gap    = median_risk_gap_black_minus_white,
    observed_median_rr     = median_risk_ratio_black_white
  )

cf_gap_rr_change_split <- cf_gap_rr_split %>%
  filter(scenario != "Observed") %>%
  left_join(
    obs_gap_rr_split,
    by = c("split_name", "split_seed", "age_bin", "age_bin_label", "sex")
  ) %>%
  mutate(
    predicted_gap_change_vs_observed =
      predicted_risk_gap_black_minus_white - observed_predicted_gap,
    
    predicted_gap_reduction_from_observed =
      observed_predicted_gap - predicted_risk_gap_black_minus_white,
    
    pct_predicted_gap_reduction =
      ifelse(
        abs(observed_predicted_gap) > eps,
        predicted_gap_reduction_from_observed / abs(observed_predicted_gap),
        NA_real_
      ),
    
    predicted_rr_change_vs_observed =
      predicted_risk_ratio_black_white - observed_predicted_rr,
    
    predicted_rr_reduction_from_observed =
      observed_predicted_rr - predicted_risk_ratio_black_white,
    
    median_gap_change_vs_observed =
      median_risk_gap_black_minus_white - observed_median_gap,
    
    median_gap_reduction_from_observed =
      observed_median_gap - median_risk_gap_black_minus_white,
    
    median_rr_change_vs_observed =
      median_risk_ratio_black_white - observed_median_rr,
    
    median_rr_reduction_from_observed =
      observed_median_rr - median_risk_ratio_black_white
  )

cf_gap_rr_change_headline <- summarise_split_to_headline(
  df = cf_gap_rr_change_split,
  group_vars = c("scenario", "age_bin", "age_bin_label", "sex"),
  value_vars = c(
    "predicted_gap_change_vs_observed",
    "predicted_gap_reduction_from_observed",
    "pct_predicted_gap_reduction",
    "predicted_rr_change_vs_observed",
    "predicted_rr_reduction_from_observed",
    "median_gap_change_vs_observed",
    "median_gap_reduction_from_observed",
    "median_rr_change_vs_observed",
    "median_rr_reduction_from_observed"
  )
)

write_csv(
  cf_gap_rr_change_split,
  file.path(out_dir, "06c_split_counterfactual_gap_rr_change_vs_observed_agebin_sex.csv")
)

write_csv(
  cf_gap_rr_change_headline,
  file.path(out_dir, "06d_headline_counterfactual_gap_rr_change_vs_observed_agebin_sex.csv")
)

###############################################################################
## 14) Race × sex performance gaps: Black - NH-White
###############################################################################
metrics_gap_race_sex_split <- metrics_race_sex_split %>%
  select(
    split_name,
    split_seed,
    race,
    sex,
    acc_mean,
    sens_mean,
    spec_mean,
    fnr_mean,
    fpr_mean,
    precision_mean
  ) %>%
  pivot_wider(
    names_from = race,
    values_from = c(acc_mean, sens_mean, spec_mean, fnr_mean, fpr_mean, precision_mean),
    names_sep = "__"
  ) %>%
  mutate(
    acc_gap_black_minus_white =
      `acc_mean__NH-Black` - `acc_mean__NH-White`,
    
    sens_gap_black_minus_white =
      `sens_mean__NH-Black` - `sens_mean__NH-White`,
    
    spec_gap_black_minus_white =
      `spec_mean__NH-Black` - `spec_mean__NH-White`,
    
    fnr_gap_black_minus_white =
      `fnr_mean__NH-Black` - `fnr_mean__NH-White`,
    
    fpr_gap_black_minus_white =
      `fpr_mean__NH-Black` - `fpr_mean__NH-White`,
    
    precision_gap_black_minus_white =
      `precision_mean__NH-Black` - `precision_mean__NH-White`
  )

metrics_gap_race_sex_headline <- summarise_split_to_headline(
  df = metrics_gap_race_sex_split,
  group_vars = c("sex"),
  value_vars = c(
    "acc_gap_black_minus_white",
    "sens_gap_black_minus_white",
    "spec_gap_black_minus_white",
    "fnr_gap_black_minus_white",
    "fpr_gap_black_minus_white",
    "precision_gap_black_minus_white"
  )
)

write_csv(
  metrics_gap_race_sex_split,
  file.path(out_dir, "07a_split_performance_gap_race_sex_black_minus_white.csv")
)

write_csv(
  metrics_gap_race_sex_headline,
  file.path(out_dir, "07b_headline_performance_gap_race_sex_black_minus_white.csv")
)

###############################################################################
## 15) Age-bin performance gaps: Black - NH-White
###############################################################################
metrics_gap_agebin_split <- metrics_agebin_split %>%
  select(
    split_name,
    split_seed,
    age_bin,
    age_bin_label,
    race,
    sex,
    acc_mean,
    sens_mean,
    spec_mean,
    fnr_mean,
    fpr_mean,
    precision_mean,
    n_mean,
    events_mean
  ) %>%
  pivot_wider(
    names_from = race,
    values_from = c(
      acc_mean, sens_mean, spec_mean, fnr_mean, fpr_mean, precision_mean,
      n_mean, events_mean
    ),
    names_sep = "__"
  ) %>%
  mutate(
    acc_gap_black_minus_white =
      `acc_mean__NH-Black` - `acc_mean__NH-White`,
    
    sens_gap_black_minus_white =
      `sens_mean__NH-Black` - `sens_mean__NH-White`,
    
    spec_gap_black_minus_white =
      `spec_mean__NH-Black` - `spec_mean__NH-White`,
    
    fnr_gap_black_minus_white =
      `fnr_mean__NH-Black` - `fnr_mean__NH-White`,
    
    fpr_gap_black_minus_white =
      `fpr_mean__NH-Black` - `fpr_mean__NH-White`,
    
    precision_gap_black_minus_white =
      `precision_mean__NH-Black` - `precision_mean__NH-White`
  )

metrics_gap_agebin_headline <- summarise_split_to_headline(
  df = metrics_gap_agebin_split,
  group_vars = c("age_bin", "age_bin_label", "sex"),
  value_vars = c(
    "acc_gap_black_minus_white",
    "sens_gap_black_minus_white",
    "spec_gap_black_minus_white",
    "fnr_gap_black_minus_white",
    "fpr_gap_black_minus_white",
    "precision_gap_black_minus_white"
  )
)

write_csv(
  metrics_gap_agebin_split,
  file.path(out_dir, "08a_split_performance_gap_agebin_sex_black_minus_white.csv")
)

write_csv(
  metrics_gap_agebin_headline,
  file.path(out_dir, "08b_headline_performance_gap_agebin_sex_black_minus_white.csv")
)

###############################################################################
## 16) Counterfactual compact table: scenario × race × sex risk
###############################################################################
cf_risk_race_sex_split <- summarise_seed_to_split(
  df = cf_risk_race_sex_seed,
  group_vars = c("split_name", "split_seed", "scenario", "race", "sex"),
  value_vars = c(
    "n",
    "events",
    "observed_event_rate",
    "mean_predicted_risk",
    "median_predicted_risk"
  )
)

cf_risk_race_sex_headline <- summarise_split_to_headline(
  df = cf_risk_race_sex_split,
  group_vars = c("scenario", "race", "sex"),
  value_vars = c(
    "n_mean",
    "events_mean",
    "observed_event_rate_mean",
    "mean_predicted_risk_mean",
    "median_predicted_risk_mean"
  )
)

write_csv(
  cf_risk_race_sex_split,
  file.path(out_dir, "09c_split_counterfactual_risk_race_sex.csv")
)

write_csv(
  cf_risk_race_sex_headline,
  file.path(out_dir, "09d_headline_counterfactual_risk_race_sex.csv")
)

###############################################################################
## 17) Counterfactual compact table: scenario × race × sex performance
###############################################################################
cf_metrics_race_sex_split <- summarise_seed_to_split(
  df = cf_metrics_race_sex_seed,
  group_vars = c("split_name", "split_seed", "scenario", "race", "sex"),
  value_vars = c(
    "n",
    "events",
    "tp",
    "tn",
    "fp",
    "fn",
    "acc",
    "sens",
    "spec",
    "fnr",
    "fpr",
    "precision"
  )
)

cf_metrics_race_sex_headline <- summarise_split_to_headline(
  df = cf_metrics_race_sex_split,
  group_vars = c("scenario", "race", "sex"),
  value_vars = c(
    "n_mean",
    "events_mean",
    "tp_mean",
    "tn_mean",
    "fp_mean",
    "fn_mean",
    "acc_mean",
    "sens_mean",
    "spec_mean",
    "fnr_mean",
    "fpr_mean",
    "precision_mean"
  )
)

write_csv(
  cf_metrics_race_sex_split,
  file.path(out_dir, "10a_split_counterfactual_metrics_race_sex.csv")
)

write_csv(
  cf_metrics_race_sex_headline,
  file.path(out_dir, "10b_headline_counterfactual_metrics_race_sex.csv")
)

###############################################################################
## 18) Scenario-vs-observed change: race × sex predicted risk
###############################################################################
obs_cf_risk_race_sex_split <- cf_risk_race_sex_split %>%
  filter(scenario == "Observed") %>%
  select(
    split_name,
    split_seed,
    race,
    sex,
    observed_mean_predicted_risk = mean_predicted_risk_mean,
    observed_median_predicted_risk = median_predicted_risk_mean
  )

cf_risk_race_sex_change_split <- cf_risk_race_sex_split %>%
  filter(scenario != "Observed") %>%
  left_join(
    obs_cf_risk_race_sex_split,
    by = c("split_name", "split_seed", "race", "sex")
  ) %>%
  mutate(
    mean_risk_change_vs_observed =
      mean_predicted_risk_mean - observed_mean_predicted_risk,
    
    mean_risk_reduction_from_observed =
      observed_mean_predicted_risk - mean_predicted_risk_mean,
    
    pct_mean_risk_reduction =
      ifelse(
        abs(observed_mean_predicted_risk) > eps,
        mean_risk_reduction_from_observed / abs(observed_mean_predicted_risk),
        NA_real_
      ),
    
    mean_risk_ratio_observed_over_held =
      pmax(observed_mean_predicted_risk, eps) /
      pmax(mean_predicted_risk_mean, eps),
    
    median_risk_change_vs_observed =
      median_predicted_risk_mean - observed_median_predicted_risk,
    
    median_risk_reduction_from_observed =
      observed_median_predicted_risk - median_predicted_risk_mean,
    
    median_risk_ratio_observed_over_held =
      pmax(observed_median_predicted_risk, eps) /
      pmax(median_predicted_risk_mean, eps)
  )

cf_risk_race_sex_change_headline <- summarise_split_to_headline(
  df = cf_risk_race_sex_change_split,
  group_vars = c("scenario", "race", "sex"),
  value_vars = c(
    "mean_risk_change_vs_observed",
    "mean_risk_reduction_from_observed",
    "pct_mean_risk_reduction",
    "mean_risk_ratio_observed_over_held",
    "median_risk_change_vs_observed",
    "median_risk_reduction_from_observed",
    "median_risk_ratio_observed_over_held"
  )
)

write_csv(
  cf_risk_race_sex_change_split,
  file.path(out_dir, "11a_split_counterfactual_risk_change_vs_observed_race_sex.csv")
)

write_csv(
  cf_risk_race_sex_change_headline,
  file.path(out_dir, "11b_headline_counterfactual_risk_change_vs_observed_race_sex.csv")
)

###############################################################################
## 19) Scenario-vs-observed change: race × sex performance
###############################################################################
obs_cf_metrics_race_sex_split <- cf_metrics_race_sex_split %>%
  filter(scenario == "Observed") %>%
  select(
    split_name,
    split_seed,
    race,
    sex,
    observed_acc = acc_mean,
    observed_sens = sens_mean,
    observed_spec = spec_mean,
    observed_fnr = fnr_mean,
    observed_fpr = fpr_mean,
    observed_precision = precision_mean
  )

cf_metrics_race_sex_change_split <- cf_metrics_race_sex_split %>%
  filter(scenario != "Observed") %>%
  left_join(
    obs_cf_metrics_race_sex_split,
    by = c("split_name", "split_seed", "race", "sex")
  ) %>%
  mutate(
    acc_change_vs_observed = acc_mean - observed_acc,
    sens_change_vs_observed = sens_mean - observed_sens,
    spec_change_vs_observed = spec_mean - observed_spec,
    fnr_change_vs_observed = fnr_mean - observed_fnr,
    fpr_change_vs_observed = fpr_mean - observed_fpr,
    precision_change_vs_observed = precision_mean - observed_precision
  )

cf_metrics_race_sex_change_headline <- summarise_split_to_headline(
  df = cf_metrics_race_sex_change_split,
  group_vars = c("scenario", "race", "sex"),
  value_vars = c(
    "acc_change_vs_observed",
    "sens_change_vs_observed",
    "spec_change_vs_observed",
    "fnr_change_vs_observed",
    "fpr_change_vs_observed",
    "precision_change_vs_observed"
  )
)

write_csv(
  cf_metrics_race_sex_change_split,
  file.path(out_dir, "12a_split_counterfactual_metrics_change_vs_observed_race_sex.csv")
)

write_csv(
  cf_metrics_race_sex_change_headline,
  file.path(out_dir, "12b_headline_counterfactual_metrics_change_vs_observed_race_sex.csv")
)

###############################################################################
## 20) Manuscript-style race × sex table
###############################################################################
manuscript_race_sex_table <- metrics_race_sex_headline %>%
  transmute(
    Model = "Full",
    Race = race,
    Sex = sex,
    
    `Avg. Person-Intervals` = round(n_mean_mean, 0),
    `SD Person-Intervals`  = round(n_mean_sd, 1),
    
    `Avg. Events` = round(events_mean_mean, 1),
    `SD Events`  = round(events_mean_sd, 1),
    
    `Avg. TP` = round(tp_mean_mean, 1),
    `SD TP`  = round(tp_mean_sd, 1),
    
    `Avg. TN` = round(tn_mean_mean, 1),
    `SD TN`  = round(tn_mean_sd, 1),
    
    `Avg. FP` = round(fp_mean_mean, 1),
    `SD FP`  = round(fp_mean_sd, 1),
    
    `Avg. FN` = round(fn_mean_mean, 1),
    `SD FN`  = round(fn_mean_sd, 1),
    
    `Avg. Accuracy` = round(acc_mean_mean, 3),
    `SD Accuracy`  = round(acc_mean_sd, 3),
    
    `Avg. Sensitivity` = round(sens_mean_mean, 3),
    `SD Sensitivity`  = round(sens_mean_sd, 3),
    
    `Avg. Specificity` = round(spec_mean_mean, 3),
    `SD Specificity`  = round(spec_mean_sd, 3),
    
    `Avg. FNR` = round(fnr_mean_mean, 3),
    `SD FNR`  = round(fnr_mean_sd, 3),
    
    `Avg. FPR` = round(fpr_mean_mean, 3),
    `SD FPR`  = round(fpr_mean_sd, 3)
  ) %>%
  arrange(Race, Sex)

write_csv(
  manuscript_race_sex_table,
  file.path(out_dir, "13_manuscript_full_model_race_sex_summary.csv")
)

###############################################################################
## 21) Manuscript-ready compact ablation table
###############################################################################
manuscript_counterfactual_predictive_table <- cf_risk_race_sex_change_headline %>%
  transmute(
    Scenario = scenario,
    Race = race,
    Sex = sex,
    
    `Risk change vs observed` =
      round(mean_risk_change_vs_observed_mean, 4),
    
    `Risk reduction from observed` =
      round(mean_risk_reduction_from_observed_mean, 4),
    
    `Risk ratio observed / held` =
      round(mean_risk_ratio_observed_over_held_mean, 3),
    
    `Percent risk reduction` =
      round(pct_mean_risk_reduction_mean, 3)
  ) %>%
  left_join(
    cf_metrics_race_sex_change_headline %>%
      transmute(
        Scenario = scenario,
        Race = race,
        Sex = sex,
        
        `Accuracy change` =
          round(acc_change_vs_observed_mean, 3),
        
        `Sensitivity change` =
          round(sens_change_vs_observed_mean, 3),
        
        `Specificity change` =
          round(spec_change_vs_observed_mean, 3),
        
        `FNR change` =
          round(fnr_change_vs_observed_mean, 3),
        
        `FPR change` =
          round(fpr_change_vs_observed_mean, 3)
      ),
    by = c("Scenario", "Race", "Sex")
  ) %>%
  arrange(Scenario, Race, Sex)

write_csv(
  manuscript_counterfactual_predictive_table,
  file.path(out_dir, "14_manuscript_counterfactual_predictive_ablation_table.csv")
)

###############################################################################
## 22) Compact scenario-specific predicted-risk table
###############################################################################
manuscript_counterfactual_risk_table <- cf_risk_race_sex_headline %>%
  transmute(
    Scenario = scenario,
    Race = race,
    Sex = sex,
    `Avg. Person-Intervals` = round(n_mean_mean, 0),
    `Avg. Events` = round(events_mean_mean, 1),
    `Observed event rate` = round(observed_event_rate_mean_mean, 4),
    `Mean predicted risk` = round(mean_predicted_risk_mean_mean, 4),
    `SD predicted risk across splits` = round(mean_predicted_risk_mean_sd, 4),
    `Median predicted risk` = round(median_predicted_risk_mean_mean, 4)
  ) %>%
  arrange(Scenario, Race, Sex)

write_csv(
  manuscript_counterfactual_risk_table,
  file.path(out_dir, "15_manuscript_counterfactual_scenario_risk_table.csv")
)

###############################################################################
## 23) Compact age-bin table for plotting later
###############################################################################
plot_ready_agebin_observed <- risk_gap_rr_headline %>%
  arrange(sex, age_bin)

plot_ready_agebin_counterfactual <- cf_gap_rr_change_headline %>%
  arrange(scenario, sex, age_bin)

write_csv(
  plot_ready_agebin_observed,
  file.path(out_dir, "16_plot_ready_observed_agebin_gap_rr.csv")
)

write_csv(
  plot_ready_agebin_counterfactual,
  file.path(out_dir, "17_plot_ready_counterfactual_agebin_gap_rr_change.csv")
)

###############################################################################
## 24) Manuscript-ready Probe 2 compact table
###############################################################################
manuscript_probe2_agebin_rr_table <- probe2_agebin_rr_plot %>%
  transmute(
    Scenario = scenario_label,
    Race = race,
    Sex = sex,
    `Age bin` = age_bin_label,
    `Observed / held-constant RR` = round(rr_mean, 3),
    `SD across splits` = round(rr_sd, 3),
    `RR low` = round(rr_low, 3),
    `RR high` = round(rr_high, 3)
  ) %>%
  arrange(Scenario, Race, Sex, `Age bin`)

write_csv(
  manuscript_probe2_agebin_rr_table,
  file.path(out_dir, "19_manuscript_probe2_agebin_observed_over_heldconstant_rr_table.csv")
)

###############################################################################
## 25) Print key outputs
###############################################################################
cat("\n====================================================\n")
cat("FULL MODEL: RACE × SEX MANUSCRIPT TABLE\n")
cat("Seed-averaged within split; summarized across 10 splits\n")
cat("====================================================\n")
print(manuscript_race_sex_table)

cat("\n====================================================\n")
cat("COUNTERFACTUAL / ABLATION PREDICTIVE TABLE\n")
cat("Scenario-vs-observed, not causal effects\n")
cat("====================================================\n")
print(manuscript_counterfactual_predictive_table)

cat("\n====================================================\n")
cat("COUNTERFACTUAL SCENARIO RISK TABLE\n")
cat("Predicted risk under observed and ablated inputs\n")
cat("====================================================\n")
print(manuscript_counterfactual_risk_table)

cat("\n====================================================\n")
cat("REPRESENTATIONAL PROBE 2 AGE-BIN RISK-RATIO TABLE\n")
cat("Observed predicted risk / held-constant predicted risk\n")
cat("====================================================\n")
print(manuscript_probe2_agebin_rr_table)

cat("\n====================================================\n")
cat("AGE BIN SETTINGS USED\n")
cat("====================================================\n")
print(age_bin_config)

cat("\n====================================================\n")
cat("Output directory:\n")
cat(out_dir, "\n")
cat("====================================================\n")