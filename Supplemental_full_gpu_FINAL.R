###############################################################################
## TRAIN + SAVE CHECKPOINTS (GPU-READY) — MULTI-SPLIT VERSION
## - 10 ID-level random splits
## - 20 seeds per split
## - Saves per-seed checkpoints nested inside split folders
## - Saves split-specific RData images for later post-estimation evaluation
## - Saves master registry for cross-split graphing / robustness work
## - No counterfactual inference inside this script
###############################################################################

## ================================================================
## 0. Libraries + setup
## ================================================================
packages_needed <- c("torch", "coro", "caret", "dplyr", "tidyr", "ggplot2")
to_install <- packages_needed[!packages_needed %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(packages_needed, library, character.only = TRUE))

torch_manual_seed(142)

setwd("yourdatalocation")

device <- torch_device(if (torch::cuda_is_available()) "cuda" else "cpu")
cat("Using device:", device$type, "\n")
cat("CUDA available:", torch::cuda_is_available(), "\n")
cat("CUDA devices:", torch::cuda_device_count(), "\n")

root_ckpt_dir <- "checkpoints_fullspec_IDsplit_multisplit"
dir.create(root_ckpt_dir, showWarnings = FALSE, recursive = TRUE)

split_meta_dir <- file.path(root_ckpt_dir, "split_meta")
dir.create(split_meta_dir, showWarnings = FALSE, recursive = TRUE)

to_cpu_array <- function(x) {
  as.array(x$to(device = torch_device("cpu")))
}

## ================================================================
## Helper: compute metrics (expects truth/preds in 0/1 integers)
## ================================================================
calc_metrics_01 <- function(truth01, preds01) {
  truth01 <- as.integer(truth01)
  preds01 <- as.integer(preds01)
  
  tn <- sum(preds01 == 0L & truth01 == 0L, na.rm = TRUE)
  tp <- sum(preds01 == 1L & truth01 == 1L, na.rm = TRUE)
  fn <- sum(preds01 == 0L & truth01 == 1L, na.rm = TRUE)
  fp <- sum(preds01 == 1L & truth01 == 0L, na.rm = TRUE)
  
  acc  <- (tp + tn) / max(1, (tp + tn + fp + fn))
  sens <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  spec <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
  
  c(acc = acc, sens = sens, spec = spec, n = (tp + tn + fp + fn))
}

calc_sens_spec <- function(truth_lbl, pred_lbl) {
  cm <- table(Predicted = pred_lbl, Actual = truth_lbl)
  
  if (!("0" %in% rownames(cm))) cm <- rbind(cm, "0" = 0)
  if (!("1" %in% rownames(cm))) cm <- rbind(cm, "1" = 0)
  if (!("0" %in% colnames(cm))) cm <- cbind(cm, "0" = 0)
  if (!("1" %in% colnames(cm))) cm <- cbind(cm, "1" = 0)
  
  cm <- cm[c("0", "1"), c("0", "1"), drop = FALSE]
  
  tn <- cm["0", "0"]; tp <- cm["1", "1"]
  fn <- cm["0", "1"]; fp <- cm["1", "0"]
  
  sens <- if ((tp + fn) > 0) as.numeric(tp / (tp + fn)) else NA_real_
  spec <- if ((tn + fp) > 0) as.numeric(tn / (tn + fp)) else NA_real_
  acc  <- if (sum(cm) > 0) as.numeric((tp + tn) / sum(cm)) else NA_real_
  
  list(cm = cm, acc = acc, sens = sens, spec = spec,
       tn = tn, fp = fp, fn = fn, tp = tp)
}

eval_by_group <- function(truth_idx, pred_idx, group_vec,
                          classes = c("0", "1"),
                          groups_keep = c("NH-White", "Black")) {
  
  truth_lbl <- factor(truth_idx, levels = seq_along(classes), labels = classes)
  pred_lbl  <- factor(pred_idx,  levels = seq_along(classes), labels = classes)
  
  out <- lapply(groups_keep, function(g) {
    ii <- which(group_vec == g)
    m  <- calc_sens_spec(truth_lbl[ii], pred_lbl[ii])
    data.frame(
      group = g,
      n = length(ii),
      acc  = m$acc,
      sens = m$sens,
      spec = m$spec,
      tn = as.numeric(m$tn), fp = as.numeric(m$fp),
      fn = as.numeric(m$fn), tp = as.numeric(m$tp)
    )
  })
  
  dplyr::bind_rows(out)
}

## ================================================================
## 1. Load and merge data (survival + AL/personality)
## ================================================================
data_surv <- read.csv("hrs_survival.csv", header = TRUE)
data_al   <- read.csv("hrs_al.csv",    header = TRUE)

overlap <- intersect(names(data_surv), names(data_al))
overlap <- setdiff(overlap, "id")

data_al_clean <- dplyr::select(data_al, -dplyr::all_of(overlap))
data_surv_al <- data_surv %>%
  semi_join(data_al_clean, by = "id") %>%
  left_join(data_al_clean,  by = "id")

id_index_all <- as.integer(factor(data_surv_al$id))

## ================================================================
## 2. Outcome, ages, static, AL, personality, TV vars
## ================================================================
data_surv_al <- data_surv_al %>%
  mutate(
    age_raw = age_,
    age_    = log(age_),
    
    age_raw_1 = age_1,
    age_raw_3 = age_3,
    age_raw_5 = age_5,
    age_1 = log(age_1),
    age_3 = log(age_3),
    age_5 = log(age_5),
    
    agep_raw_1 = agep_1,
    agep_raw_3 = agep_3,
    agep_raw_5 = agep_5,
    agep_1 = log(agep_1),
    agep_3 = log(agep_3),
    agep_5 = log(agep_5)
  )

event_var <- "died_"
time_vars <- "age_"

static_covars <- c(
  "ed_2", "ed_3", "ed_4", "ed_5",
  "black", "others", "hispanic", "female",
  "tage", "year_2", "year_3", "year_4", "year_5", "year_6"
)

other_covars <- c("mothered_2", "mothered_3", "mothered_4", "mothered_5")

al_predictors <- c(
  "age_1","tochol_1","cysc_1","hgb_1","hdl_1","crp_1","ldl_1",
  "fev_1","hbp_1","hbp_s_1","hbp_d_1","pulse_1","waist_1","bmi_1","grip_1",
  
  "age_3","tochol_3","cysc_3","hgb_3","hdl_3","crp_3","ldl_3",
  "fev_3","hbp_3","hbp_s_3","hbp_d_3","pulse_3","waist_3","bmi_3","grip_3",
  
  "age_5","tochol_5","cysc_5","hgb_5","hdl_5","crp_5","ldl_5",
  "fev_5","hbp_5","hbp_s_5","hbp_d_5","pulse_5","waist_5","bmi_5","grip_5"
)

personality_predictors <- c(
  "open_1","open_3","open_5",
  "cons_1","cons_3","cons_5",
  "extra_1","extra_3","extra_5",
  "neuro_1","neuro_3","neuro_5",
  "opt_1","opt_3","opt_5",
  "control_1","control_3","control_5",
  "master_1","master_3","master_5",
  "agep_1","agep_3","agep_5"
)

tv_bases <- c(
  "sayret","mwid","cesd","shlt","cancre","diabe","hearte","mobila","adl5a"
)
waves <- 1:15

tv_vars <- unlist(
  lapply(tv_bases, function(b) {
    cols <- paste0(b, "_", waves)
    cols[cols %in% names(data_surv_al)]
  })
)

age_raw_vars <- c(
  "age_raw",
  "age_raw_1", "age_raw_3", "age_raw_5",
  "agep_raw_1", "agep_raw_3", "agep_raw_5"
)

predictors <- unique(c(
  time_vars,
  static_covars,
  other_covars,
  al_predictors,
  personality_predictors,
  tv_vars,
  age_raw_vars
))

## ================================================================
## 3. Build df_core, X, y, ID indices
## ================================================================
df_core <- data_surv_al[, c(event_var, predictors)]

keep_rows <- !is.na(df_core[[event_var]])
df_core   <- df_core[keep_rows, , drop = FALSE]

id_index_core <- id_index_all[keep_rows]
stopifnot(length(id_index_core) == nrow(df_core))

y_fac   <- factor(df_core[[event_var]], levels = c(0, 1))
classes <- levels(y_fac)
y_idx   <- as.integer(y_fac)

X_df <- df_core[, predictors, drop = FALSE]

X_df <- data.frame(
  lapply(X_df, function(v) {
    if (is.factor(v)) {
      suppressWarnings(as.numeric(as.character(v)))
    } else if (is.character(v)) {
      suppressWarnings(as.numeric(v))
    } else {
      v
    }
  }),
  check.names = FALSE
)

X <- as.matrix(X_df)

cat("df_core N:", nrow(df_core), "\n")
cat("X N:", nrow(X), "\n")
stopifnot(nrow(X) == length(y_idx))

id_core <- data_surv_al$id[keep_rows]
stopifnot(length(id_core) == nrow(df_core))

group_core <- data_surv_al[keep_rows, ] %>%
  dplyr::transmute(
    group = dplyr::case_when(
      black == 1 ~ "Black",
      black == 0 & hispanic == 0 & others == 0 ~ "NH-White",
      TRUE ~ "Other"
    )
  ) %>%
  dplyr::pull(group)

stopifnot(length(group_core) == nrow(df_core))

## ================================================================
## 4. NA mask, binary detection, scaling (NA-aware)
## ================================================================
na_mask_mat <- ifelse(is.na(X), 1, 0)

is_binary_col <- function(v) {
  u <- unique(v); u <- u[!is.na(u)]
  all(u %in% c(0, 1))
}
bin_mask <- apply(X, 2, is_binary_col)

age_names <- c(
  "age_", "age_raw",
  "age_1", "age_3", "age_5",
  "age_raw_1", "age_raw_3", "age_raw_5",
  "agep_1", "agep_3", "agep_5",
  "agep_raw_1", "agep_raw_3", "agep_raw_5"
)
age_cols <- which(colnames(X) %in% age_names)
if (length(age_cols) > 0L) bin_mask[age_cols] <- FALSE

X_scaled <- X
non_age_non_bin <- setdiff(which(!bin_mask), age_cols)
if (length(non_age_non_bin) > 0L) {
  X_scaled[, non_age_non_bin] <- scale(X[, non_age_non_bin])
}
X_scaled[is.na(X_scaled)] <- 0

cat("Rows X_scaled:", nrow(X_scaled), " Cols:", ncol(X_scaled), "\n")
all_cols <- colnames(X_scaled)

## ================================================================
## 5. Shared column indices (same across all splits)
## ================================================================
static_names <- c(time_vars, static_covars, other_covars)
static_idx   <- match(static_names, all_cols)
if (any(is.na(static_idx))) {
  stop("Some static columns not found: ",
       paste(static_names[is.na(static_idx)], collapse = ", "))
}
static_idx <- as.integer(static_idx)

wave_idx_list <- lapply(waves, function(t) {
  wave_vars <- paste0(tv_bases, "_", t)
  wave_vars <- wave_vars[wave_vars %in% all_cols]
  idx <- match(wave_vars, all_cols)
  idx[!is.na(idx)]
})
wave_lengths <- sapply(wave_idx_list, length)
cat("Raw TV wave feature sizes:\n"); print(wave_lengths)

non_empty <- which(wave_lengths > 0)
if (length(non_empty) == 0L) stop("No non-empty TV waves found; check tv_bases.")
if (length(non_empty) < length(wave_idx_list)) {
  cat("Dropping waves with 0 TV variables at positions:",
      paste(setdiff(seq_along(wave_idx_list), non_empty), collapse = ", "), "\n")
}
wave_idx_list <- wave_idx_list[non_empty]
wave_lengths  <- wave_lengths[non_empty]
if (length(unique(wave_lengths)) != 1L) stop("Remaining TV waves do not have equal feature count.")

tv_step_feat_dim <- as.integer(wave_lengths[1])
n_steps          <- as.integer(length(wave_idx_list))
tv_idx           <- as.integer(unlist(wave_idx_list))

cat("Final TV wave count:", n_steps, "\n")
cat("Per-step TV feature dim:", tv_step_feat_dim, "\n")

al1_names <- c(
  "age_1","tochol_1","cysc_1","hgb_1","hdl_1","crp_1","ldl_1",
  "fev_1","hbp_1","hbp_s_1","hbp_d_1","pulse_1","waist_1","bmi_1","grip_1"
)
al3_names <- c(
  "age_3","tochol_3","cysc_3","hgb_3","hdl_3","crp_3","ldl_3",
  "fev_3","hbp_3","hbp_s_3","hbp_d_3","pulse_3","waist_3","bmi_3","grip_3"
)
al5_names <- c(
  "age_5","tochol_5","cysc_5","hgb_5","hdl_5","crp_5","ldl_5",
  "fev_5","hbp_5","hbp_s_5","hbp_d_5","pulse_5","waist_5","bmi_5","grip_5"
)

al1_idx <- match(al1_names, all_cols)
al3_idx <- match(al3_names, all_cols)
al5_idx <- match(al5_names, all_cols)
if (any(is.na(c(al1_idx, al3_idx, al5_idx)))) stop("Some AL columns not found.")
al_feat_dim <- length(al1_idx)

pers1_names <- c("agep_1", "open_1","cons_1","extra_1","neuro_1","opt_1","control_1","master_1")
pers3_names <- c("agep_3", "open_3","cons_3","extra_3","neuro_3","opt_3","control_3","master_3")
pers5_names <- c("agep_5", "open_5","cons_5","extra_5","neuro_5","opt_5","control_5","master_5")

pers1_idx <- match(pers1_names, all_cols)
pers3_idx <- match(pers3_names, all_cols)
pers5_idx <- match(pers5_names, all_cols)
if (any(is.na(c(pers1_idx, pers3_idx, pers5_idx)))) stop("Some personality columns not found.")
pers_feat_dim <- length(pers1_idx)

cat("AL feat dim per wave:", al_feat_dim, "\n")
cat("Personality feat dim per wave:", pers_feat_dim, "\n")

age_idx     <- match("age_", all_cols);      if (is.na(age_idx)) stop("age_ not found")
age_raw_idx <- match("age_raw", all_cols);   if (is.na(age_raw_idx)) stop("age_raw not found")

al_age_names <- c("age_1", "age_3", "age_5")
al_age_idx   <- match(al_age_names, all_cols)
if (any(is.na(al_age_idx))) stop("Some AL log age cols missing.")

al_age_raw_names <- c("age_raw_1", "age_raw_3", "age_raw_5")
al_age_raw_idx   <- match(al_age_raw_names, all_cols)
if (any(is.na(al_age_raw_idx))) stop("Some AL raw age cols missing.")

pers_age_names <- c("agep_1", "agep_3", "agep_5")
pers_age_idx   <- match(pers_age_names, all_cols)
if (any(is.na(pers_age_idx))) stop("Some personality log age cols missing.")

pers_age_raw_names <- c("agep_raw_1", "agep_raw_3", "agep_raw_5")
pers_age_raw_idx   <- match(pers_age_raw_names, all_cols)
if (any(is.na(pers_age_raw_idx))) stop("Some personality raw age cols missing.")

## ================================================================
## 6. Counterfactual ZERO sets (saved for later scripts only)
## ================================================================
al_all_idx  <- unique(c(al1_idx, al3_idx, al5_idx))
al_zero_idx <- setdiff(al_all_idx, al_age_idx)

pers_all_idx  <- unique(c(pers1_idx, pers3_idx, pers5_idx))
pers_zero_idx <- setdiff(pers_all_idx, pers_age_idx)

both_zero_idx <- sort(unique(c(al_zero_idx, pers_zero_idx)))

cat("AL zero idx count:", length(al_zero_idx), "\n")
cat("Pers zero idx count:", length(pers_zero_idx), "\n")
cat("Both zero idx count:", length(both_zero_idx), "\n")

## ================================================================
## 7. Full NA-aware MLP with ID random effect (baseline + AL/Pers)
## ================================================================
output_dim <- length(classes)
num_ids    <- max(id_index_core)
id_emb_dim <- 1L

hazard_mlp_full <- nn_module(
  "hazard_mlp_full",
  initialize = function(static_idx,
                        tv_idx,
                        n_steps,
                        tv_step_feat_dim,
                        age_idx,
                        age_raw_idx,
                        al1_idx, al3_idx, al5_idx,
                        al_age_idx,
                        al_age_raw_idx,
                        pers1_idx, pers3_idx, pers5_idx,
                        pers_age_idx,
                        pers_age_raw_idx,
                        num_ids,
                        id_emb_dim = 1L,
                        h_static = 64,
                        h_tv     = 64,
                        h_al     = 64,
                        h_pers   = 64,
                        mix_dim  = 32,
                        output_dim = 2,
                        drop_p   = 0.25) {
    
    self$static_idx <- as.integer(static_idx)
    self$tv_idx     <- as.integer(tv_idx)
    
    self$age_idx      <- as.integer(age_idx)
    self$age_raw_idx  <- as.integer(age_raw_idx)
    
    self$al1_idx    <- as.integer(al1_idx)
    self$al3_idx    <- as.integer(al3_idx)
    self$al5_idx    <- as.integer(al5_idx)
    self$al_age_idx     <- as.integer(al_age_idx)
    self$al_age_raw_idx <- as.integer(al_age_raw_idx)
    
    self$pers1_idx    <- as.integer(pers1_idx)
    self$pers3_idx    <- as.integer(pers3_idx)
    self$pers5_idx    <- as.integer(pers5_idx)
    self$pers_age_idx     <- as.integer(pers_age_idx)
    self$pers_age_raw_idx <- as.integer(pers_age_raw_idx)
    
    self$h_tv_dim   <- as.integer(h_tv)
    self$h_al_dim   <- as.integer(h_al)
    self$h_pers_dim <- as.integer(h_pers)
    
    self$al_feat_dim   <- length(self$al1_idx)
    self$pers_feat_dim <- length(self$pers1_idx)
    
    self$n_steps          <- as.integer(n_steps)
    self$tv_step_feat_dim <- as.integer(tv_step_feat_dim)
    
    self$num_ids    <- as.integer(num_ids)
    self$id_emb_dim <- as.integer(id_emb_dim)
    
    self$id_embed <- nn_embedding(
      num_embeddings = self$num_ids + 1L,
      embedding_dim  = self$id_emb_dim
    )
    self$alpha_id <- nn_parameter(torch_tensor(0, dtype = torch_float()))
    
    self$fc_static1 <- nn_linear(length(self$static_idx), h_static)
    self$fc_static2 <- nn_linear(h_static, h_static)
    
    tv_step_input_dim <- 2L * self$tv_step_feat_dim
    self$fc_tv1 <- nn_linear(tv_step_input_dim, h_tv)
    self$fc_tv2 <- nn_linear(h_tv, h_tv)
    self$att_tv <- nn_linear(h_tv, 1)
    
    self$al_step_input_dim <- 2L * self$al_feat_dim + 1L
    self$al_fc1 <- nn_linear(self$al_step_input_dim, h_al)
    self$al_fc2 <- nn_linear(h_al, h_al)
    self$al_att <- nn_linear(h_al, 1)
    
    self$pers_step_input_dim <- 2L * self$pers_feat_dim + 1L
    self$pers_fc1 <- nn_linear(self$pers_step_input_dim, h_pers)
    self$pers_fc2 <- nn_linear(h_pers, h_pers)
    self$pers_att <- nn_linear(h_pers, 1)
    
    self$fc_mix <- nn_linear(h_static + h_tv + h_al + h_pers, mix_dim)
    
    self$fc_base   <- nn_linear(h_static + h_tv, output_dim)
    self$fc_alpers <- nn_linear(h_al + h_pers + mix_dim, output_dim)
    
    self$drop <- nn_dropout(p = drop_p)
  },
  
  forward = function(x, mask, id_idx, use_alpers = TRUE) {
    device <- x$device
    B <- as.integer(x$size()[1])
    
    if (id_idx$device != device) id_idx <- id_idx$to(device = device)
    
    static_idx_t <- torch_tensor(self$static_idx, dtype = torch_long(), device = device)
    tv_idx_t     <- torch_tensor(self$tv_idx,     dtype = torch_long(), device = device)
    age_raw_idx_t <- torch_tensor(self$age_raw_idx, dtype = torch_long(), device = device)
    
    al1_idx_t <- torch_tensor(self$al1_idx, dtype = torch_long(), device = device)
    al3_idx_t <- torch_tensor(self$al3_idx, dtype = torch_long(), device = device)
    al5_idx_t <- torch_tensor(self$al5_idx, dtype = torch_long(), device = device)
    al_age_raw_idx_t <- torch_tensor(self$al_age_raw_idx, dtype = torch_long(), device = device)
    
    p1_idx_t <- torch_tensor(self$pers1_idx, dtype = torch_long(), device = device)
    p3_idx_t <- torch_tensor(self$pers3_idx, dtype = torch_long(), device = device)
    p5_idx_t <- torch_tensor(self$pers5_idx, dtype = torch_long(), device = device)
    pers_age_raw_idx_t <- torch_tensor(self$pers_age_raw_idx, dtype = torch_long(), device = device)
    
    x_static <- x$index_select(2L, static_idx_t)
    h_static <- x_static %>%
      self$fc_static1() %>% nnf_relu() %>% self$drop() %>%
      self$fc_static2() %>% nnf_relu()
    
    x_tv    <- x$index_select(2L, tv_idx_t)
    mask_tv <- mask$index_select(2L, tv_idx_t)
    
    T_steps <- as.integer(self$n_steps)
    F_step  <- as.integer(self$tv_step_feat_dim)
    
    x_seq <- x_tv$view(c(B, T_steps, F_step))
    m_seq <- mask_tv$view(c(B, T_steps, F_step))
    
    tv_seq_in  <- torch_cat(list(x_seq, m_seq), dim = 3L)
    tv_flat_in <- tv_seq_in$view(c(as.integer(B * T_steps), as.integer(2L * F_step)))
    
    h_tv_flat <- tv_flat_in %>%
      self$fc_tv1() %>% nnf_relu() %>% self$drop() %>%
      self$fc_tv2() %>% nnf_relu() %>% self$drop()
    
    h_tv_seq <- h_tv_flat$view(c(B, T_steps, as.integer(self$h_tv_dim)))
    
    att_scores <- self$att_tv(h_tv_seq)$squeeze(3L)
    alpha      <- nnf_softmax(att_scores, dim = 2L)$unsqueeze(3L)
    h_tv       <- (h_tv_seq * alpha)$sum(dim = 2L)
    
    age_curr_raw <- x$index_select(2L, age_raw_idx_t)
    al_ages_raw  <- x$index_select(2L, al_age_raw_idx_t)
    ps_ages_raw  <- x$index_select(2L, pers_age_raw_idx_t)
    
    age_curr_rep <- age_curr_raw$expand(c(B, 3L))
    dage_al   <- age_curr_rep - al_ages_raw
    dage_pers <- age_curr_rep - ps_ages_raw
    
    al1 <- x$index_select(2L, al1_idx_t)
    al3 <- x$index_select(2L, al3_idx_t)
    al5 <- x$index_select(2L, al5_idx_t)
    
    m1 <- mask$index_select(2L, al1_idx_t)
    m3 <- mask$index_select(2L, al3_idx_t)
    m5 <- mask$index_select(2L, al5_idx_t)
    
    al_step1_in <- torch_cat(list(al1, m1, dage_al[,1,drop=FALSE]), dim = 2L)
    al_step3_in <- torch_cat(list(al3, m3, dage_al[,2,drop=FALSE]), dim = 2L)
    al_step5_in <- torch_cat(list(al5, m5, dage_al[,3,drop=FALSE]), dim = 2L)
    
    al_steps <- torch_stack(list(al_step1_in, al_step3_in, al_step5_in), dim = 2L)
    al_flat  <- al_steps$view(c(as.integer(B * 3L), as.integer(self$al_step_input_dim)))
    
    h_al_flat <- al_flat %>%
      self$al_fc1() %>% nnf_relu() %>% self$drop() %>%
      self$al_fc2() %>% nnf_relu() %>% self$drop()
    h_al_seq <- h_al_flat$view(c(B, 3L, as.integer(self$h_al_dim)))
    
    al_scores <- self$al_att(h_al_seq)$squeeze(3L)
    al_alpha  <- nnf_softmax(al_scores, dim = 2L)$unsqueeze(3L)
    h_al      <- (h_al_seq * al_alpha)$sum(dim = 2L)
    
    p1 <- x$index_select(2L, p1_idx_t)
    p3 <- x$index_select(2L, p3_idx_t)
    p5 <- x$index_select(2L, p5_idx_t)
    
    mp1 <- mask$index_select(2L, p1_idx_t)
    mp3 <- mask$index_select(2L, p3_idx_t)
    mp5 <- mask$index_select(2L, p5_idx_t)
    
    p_step1 <- torch_cat(list(p1, mp1, dage_pers[,1,drop=FALSE]), dim = 2L)
    p_step3 <- torch_cat(list(p3, mp3, dage_pers[,2,drop=FALSE]), dim = 2L)
    p_step5 <- torch_cat(list(p5, mp5, dage_pers[,3,drop=FALSE]), dim = 2L)
    
    p_steps <- torch_stack(list(p_step1, p_step3, p_step5), dim = 2L)
    p_flat  <- p_steps$view(c(as.integer(B * 3L), as.integer(self$pers_step_input_dim)))
    
    h_p_flat <- p_flat %>%
      self$pers_fc1() %>% nnf_relu() %>% self$drop() %>%
      self$pers_fc2() %>% nnf_relu() %>% self$drop()
    h_p_seq <- h_p_flat$view(c(B, 3L, as.integer(self$h_pers_dim)))
    
    p_scores <- self$pers_att(h_p_seq)$squeeze(3L)
    p_alpha  <- nnf_softmax(p_scores, dim = 2L)$unsqueeze(3L)
    h_pers   <- (h_p_seq * p_alpha)$sum(dim = 2L)
    
    logits_base <- self$fc_base(torch_cat(list(h_static, h_tv), dim = 2L))
    
    mix_input <- torch_cat(list(h_static, h_tv, h_al, h_pers), dim = 2L)
    z_mix <- self$fc_mix(mix_input) %>% nnf_relu()
    
    if (use_alpers) {
      logits_alpers <- self$fc_alpers(torch_cat(list(h_al, h_pers, z_mix), dim = 2L))
    } else {
      logits_alpers <- torch_zeros_like(logits_base)
    }
    
    logits_no_id <- logits_base + logits_alpers
    
    id_vec   <- self$id_embed(id_idx)
    id_shift <- self$alpha_id * id_vec
    
    logits <- logits_no_id$clone()
    logits[, 2] <- logits[, 2] + id_shift$squeeze(2L)
    
    logits
  }
)

## ================================================================
## 8. Generate and save fixed split list
## ================================================================
make_id_split <- function(id_core, y_idx, split_seed, p_train = 0.80) {
  set.seed(split_seed)
  
  person_df <- data.frame(
    id = id_core,
    y_interval = y_idx,
    stringsAsFactors = FALSE
  ) %>%
    dplyr::group_by(id) %>%
    dplyr::summarise(
      y_person = as.integer(any(y_interval == 2L)),
      .groups = "drop"
    )
  
  id_tr <- caret::createDataPartition(person_df$y_person, p = p_train, list = FALSE)
  train_ids <- person_df$id[id_tr]
  test_ids  <- setdiff(person_df$id, train_ids)
  
  tr <- which(id_core %in% train_ids)
  te <- which(id_core %in% test_ids)
  
  stopifnot(length(intersect(id_core[tr], id_core[te])) == 0L)
  
  list(
    split_seed = split_seed,
    train_ids = train_ids,
    test_ids  = test_ids,
    tr = tr,
    te = te,
    train_person_rate = mean(person_df$y_person[person_df$id %in% train_ids]),
    test_person_rate  = mean(person_df$y_person[person_df$id %in% test_ids]),
    n_train_person = length(train_ids),
    n_test_person  = length(test_ids),
    n_train_rows = length(tr),
    n_test_rows  = length(te)
  )
}

n_splits <- 10
split_seeds <- 1001:(1000 + n_splits)
split_names <- paste0("split_", sprintf("%02d", seq_len(n_splits)))

split_list <- vector("list", length = n_splits)
names(split_list) <- split_names

for (i in seq_along(split_names)) {
  split_list[[i]] <- make_id_split(
    id_core = id_core,
    y_idx = y_idx,
    split_seed = split_seeds[i],
    p_train = 0.80
  )
}

saveRDS(split_list, file = file.path(split_meta_dir, "fixed_split_list.rds"))

split_registry <- bind_rows(lapply(seq_along(split_list), function(i) {
  sp <- split_list[[i]]
  data.frame(
    split_name = names(split_list)[i],
    split_seed = sp$split_seed,
    n_train_rows = sp$n_train_rows,
    n_test_rows = sp$n_test_rows,
    n_train_person = sp$n_train_person,
    n_test_person = sp$n_test_person,
    train_person_rate = sp$train_person_rate,
    test_person_rate = sp$test_person_rate
  )
}))

write.csv(split_registry, file.path(split_meta_dir, "fixed_split_registry.csv"), row.names = FALSE)
saveRDS(split_registry,  file.path(split_meta_dir, "fixed_split_registry.rds"))

cat("\nSaved fixed split list and split registry.\n")

## ================================================================
## 9. Split-specific tensors / data objects
## ================================================================
build_split_objects <- function(split_obj, X_scaled, y_idx, na_mask_mat, id_index_core,
                                group_core, classes, device) {
  tr <- split_obj$tr
  te <- split_obj$te
  
  Xtr <- X_scaled[tr, , drop = FALSE];  ytr <- y_idx[tr]
  Xte <- X_scaled[te, , drop = FALSE];  yte <- y_idx[te]
  
  mask_tr <- na_mask_mat[tr, , drop = FALSE]
  mask_te <- na_mask_mat[te, , drop = FALSE]
  
  idtr <- id_index_core[tr]
  idte <- id_index_core[te]
  
  train_counts <- as.numeric(table(factor(ytr, levels = seq_along(classes))))
  cw <- 1 / train_counts
  cw <- cw / mean(cw)
  
  list(
    tr = tr, te = te,
    Xtr = Xtr, ytr = ytr,
    Xte = Xte, yte = yte,
    mask_tr = mask_tr, mask_te = mask_te,
    idtr = idtr, idte = idte,
    group_test = group_core[te],
    class_weights = cw,
    
    xtr_t     = torch_tensor(Xtr,        dtype = torch_float(), device = device),
    mask_tr_t = torch_tensor(mask_tr,    dtype = torch_float(), device = device),
    ytr_t     = torch_tensor(ytr,        dtype = torch_long(),  device = device),
    idtr_t    = torch_tensor(idtr,       dtype = torch_long(),  device = device),
    
    xt_t      = torch_tensor(Xte,        dtype = torch_float(), device = device),
    mask_te_t = torch_tensor(mask_te,    dtype = torch_float(), device = device),
    idte_t    = torch_tensor(idte,       dtype = torch_long(),  device = device)
  )
}

## ================================================================
## 10. Seed runner for a given split
## ================================================================
run_alpers_seed_for_split <- function(torch_seed,
                                      split_name,
                                      split_obj,
                                      split_data,
                                      epochs_phase1 = 50,
                                      epochs_phase2 = 100,
                                      batch_size    = 1024,
                                      lr            = 1e-3,
                                      weight_decay  = 1e-3,
                                      lambda_fp     = 0.5,
                                      save_ckpt     = TRUE,
                                      ckpt_dir      = ".") {
  
  cat("\n==============================\n")
  cat("Split:", split_name, "| Seed:", torch_seed, "\n")
  cat("==============================\n")
  
  torch_manual_seed(torch_seed)
  
  model <- hazard_mlp_full(
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
  )$to(device = device)
  
  class_weights <- torch_tensor(split_data$class_weights, dtype = torch_float(), device = device)
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)
  
  n_train   <- as.integer(nrow(split_data$Xtr))
  n_batches <- as.integer(ceiling(n_train / batch_size))
  
  ## -------------------------
  ## Phase 1 (baseline-only)
  ## -------------------------
  for (epoch in 1:epochs_phase1) {
    model$train()
    idx_all <- sample.int(n_train)
    running_loss <- 0
    
    for (b in 1:n_batches) {
      i1 <- as.integer((b - 1L) * batch_size + 1L)
      i2 <- as.integer(min(b * batch_size, n_train))
      idx <- idx_all[i1:i2]
      
      x_batch  <- split_data$xtr_t[idx, , drop = FALSE]
      m_batch  <- split_data$mask_tr_t[idx, , drop = FALSE]
      y_batch  <- split_data$ytr_t[idx]
      id_batch <- split_data$idtr_t[idx]
      
      optimizer$zero_grad()
      logits <- model(x_batch, m_batch, id_batch, use_alpers = FALSE)
      loss   <- nnf_cross_entropy(logits, y_batch, weight = class_weights)
      loss$backward()
      optimizer$step()
      
      running_loss <- running_loss + loss$item()
    }
    
    if (epoch %% 10 == 0) {
      cat(sprintf("[Split %s] [Seed %d] [Phase 1] Epoch %3d | avg train loss: %.4f\n",
                  split_name, torch_seed, epoch, running_loss / n_batches))
    }
  }
  
  ## -------------------------
  ## Phase 2 (AL + Pers on)
  ## -------------------------
  for (epoch in 1:epochs_phase2) {
    model$train()
    idx_all <- sample.int(n_train)
    running_loss <- 0
    
    for (b in 1:n_batches) {
      i1 <- as.integer((b - 1L) * batch_size + 1L)
      i2 <- as.integer(min(b * batch_size, n_train))
      idx <- idx_all[i1:i2]
      
      x_batch  <- split_data$xtr_t[idx, , drop = FALSE]
      m_batch  <- split_data$mask_tr_t[idx, , drop = FALSE]
      y_batch  <- split_data$ytr_t[idx]
      id_batch <- split_data$idtr_t[idx]
      
      optimizer$zero_grad()
      logits <- model(x_batch, m_batch, id_batch, use_alpers = TRUE)
      
      loss_ce <- nnf_cross_entropy(logits, y_batch, weight = class_weights)
      
      probs   <- nnf_softmax(logits, dim = 2L)
      p_death <- probs[, 2]
      
      surv_mask <- (y_batch == 1L)
      if (surv_mask$any()$item()) {
        p_death_surv <- p_death[surv_mask]
        fp_penalty   <- torch_mean(p_death_surv * p_death_surv)
      } else {
        fp_penalty <- torch_tensor(0, dtype = torch_float(), device = logits$device)
      }
      
      loss <- loss_ce + lambda_fp * fp_penalty
      
      loss$backward()
      optimizer$step()
      
      running_loss <- running_loss + loss$item()
    }
    
    if (epoch %% 10 == 0) {
      cat(sprintf("[Split %s] [Seed %d] [Phase 2] Epoch %3d | avg train+FP loss: %.4f\n",
                  split_name, torch_seed, epoch, running_loss / n_batches))
    }
  }
  
  ## -------------------------
  ## Test evaluation
  ## -------------------------
  model$eval()
  
  n_test        <- as.integer(nrow(split_data$Xte))
  batch_size_te <- 1024L
  n_batches_te  <- as.integer(ceiling(n_test / batch_size_te))
  
  logits_list <- vector("list", n_batches_te)
  
  for (b in 1:n_batches_te) {
    i1 <- as.integer((b - 1L) * batch_size_te + 1L)
    i2 <- as.integer(min(b * batch_size_te, n_test))
    idx <- i1:i2
    
    with_no_grad({
      logits_list[[b]] <- model(
        split_data$xt_t[idx, , drop = FALSE],
        split_data$mask_te_t[idx, , drop = FALSE],
        split_data$idte_t[idx],
        use_alpers = TRUE
      )
    })
  }
  
  logits_test <- torch_cat(logits_list, dim = 1L)
  
  preds_test <- as.integer(to_cpu_array(torch_argmax(logits_test, dim = 2L)))
  truth_test <- as.integer(split_data$yte)
  
  pred_lbl_test <- factor(preds_test, levels = seq_along(classes), labels = classes)
  true_lbl_test <- factor(truth_test, levels = seq_along(classes), labels = classes)
  
  acc_test <- mean(pred_lbl_test == true_lbl_test)
  cm_test  <- table(Predicted = pred_lbl_test, Actual = true_lbl_test)
  
  tn <- if ("0" %in% rownames(cm_test) && "0" %in% colnames(cm_test)) cm_test["0","0"] else 0
  tp <- if ("1" %in% rownames(cm_test) && "1" %in% colnames(cm_test)) cm_test["1","1"] else 0
  fn <- if ("0" %in% rownames(cm_test) && "1" %in% colnames(cm_test)) cm_test["0","1"] else 0
  fp <- if ("1" %in% rownames(cm_test) && "0" %in% colnames(cm_test)) cm_test["1","0"] else 0
  
  sens_test <- if ((tp + fn) > 0) as.numeric(tp / (tp + fn)) else NA_real_
  spec_test <- if ((tn + fp) > 0) as.numeric(tn / (tn + fp)) else NA_real_
  
  grp_test_df <- eval_by_group(
    truth_idx = truth_test,
    pred_idx  = preds_test,
    group_vec = split_data$group_test,
    classes   = classes,
    groups_keep = c("NH-White", "Black")
  )
  
  cat(sprintf("Split %s | Seed %d | Test accuracy: %.3f | Sens: %.3f | Spec: %.3f\n",
              split_name, torch_seed, acc_test, sens_test, spec_test))
  print(grp_test_df)
  
  ckpt_path <- file.path(ckpt_dir, sprintf("seed_%03d.pt", torch_seed))
  if (isTRUE(save_ckpt)) {
    sd_cpu <- lapply(model$state_dict(), function(t) t$to(device = torch_device("cpu")))
    torch_save(sd_cpu, ckpt_path)
    cat("Saved checkpoint:", ckpt_path, "\n")
  }
  
  list(
    split_name = split_name,
    split_seed = split_obj$split_seed,
    seed = torch_seed,
    ckpt_path = ckpt_path,
    acc_test  = acc_test,
    sens_test = sens_test,
    spec_test = spec_test,
    grp_test  = grp_test_df
  )
}

## ================================================================
## 11. Run all splits × seeds
## ================================================================
seeds_within_split <- 1:20

master_results_list <- list()
master_results_df_list <- list()
master_grp_df_list <- list()

for (i in seq_along(split_list)) {
  split_name <- names(split_list)[i]
  split_obj  <- split_list[[i]]
  
  cat("\n############################################################\n")
  cat("NOW RUNNING:", split_name, "| split_seed =", split_obj$split_seed, "\n")
  cat("############################################################\n")
  
  split_dir <- file.path(root_ckpt_dir, split_name)
  dir.create(split_dir, showWarnings = FALSE, recursive = TRUE)
  
  split_data <- build_split_objects(
    split_obj = split_obj,
    X_scaled = X_scaled,
    y_idx = y_idx,
    na_mask_mat = na_mask_mat,
    id_index_core = id_index_core,
    group_core = group_core,
    classes = classes,
    device = device
  )
  
  split_results <- lapply(seeds_within_split, function(sd) {
    run_alpers_seed_for_split(
      torch_seed = sd,
      split_name = split_name,
      split_obj = split_obj,
      split_data = split_data,
      epochs_phase1 = 50,
      epochs_phase2 = 100,
      batch_size = 1024,
      lr = 1e-3,
      weight_decay = 1e-3,
      lambda_fp = 0.5,
      save_ckpt = TRUE,
      ckpt_dir = split_dir
    )
  })
  
  results_df_split <- bind_rows(lapply(split_results, function(res) {
    data.frame(
      split_name = res$split_name,
      split_seed = res$split_seed,
      seed       = res$seed,
      ckpt_path  = res$ckpt_path,
      acc_test   = res$acc_test,
      sens_test  = res$sens_test,
      spec_test  = res$spec_test,
      stringsAsFactors = FALSE
    )
  }))
  
  grp_df_split <- bind_rows(lapply(split_results, function(res) {
    df <- res$grp_test
    df$split_name <- res$split_name
    df$split_seed <- res$split_seed
    df$seed       <- res$seed
    df
  }))
  
  master_results_list[[split_name]] <- split_results
  master_results_df_list[[split_name]] <- results_df_split
  master_grp_df_list[[split_name]] <- grp_df_split
  
  ## save one split-level image for later post-evaluation continuity
  split_save_path <- file.path(split_meta_dir, paste0(split_name, "_training_state.RData"))
  
  results_list <- split_results
  results_df   <- results_df_split
  grp_test_all <- grp_df_split
  
  tr <- split_obj$tr
  te <- split_obj$te
  train_ids <- split_obj$train_ids
  test_ids  <- split_obj$test_ids
  split_seed <- split_obj$split_seed
  
  save(
    split_name, split_seed,
    tr, te, train_ids, test_ids,
    results_list, results_df, grp_test_all,
    keep_rows, df_core, data_surv_al,
    X_scaled, na_mask_mat, id_index_core, id_core, group_core,
    static_idx, tv_idx, n_steps, tv_step_feat_dim,
    age_idx, age_raw_idx,
    al1_idx, al3_idx, al5_idx,
    al_age_idx, al_age_raw_idx,
    pers1_idx, pers3_idx, pers5_idx,
    pers_age_idx, pers_age_raw_idx,
    al_zero_idx, pers_zero_idx, both_zero_idx,
    all_cols, classes, output_dim, num_ids, id_emb_dim,
    file = split_save_path
  )
  
  cat("Saved split-level training state:", split_save_path, "\n")
}

## ================================================================
## 12. Master registries and summaries
## ================================================================
master_results_df <- bind_rows(master_results_df_list)
master_grp_df     <- bind_rows(master_grp_df_list)

write.csv(master_results_df,
          file.path(root_ckpt_dir, "master_checkpoint_registry.csv"),
          row.names = FALSE)
saveRDS(master_results_df,
        file.path(root_ckpt_dir, "master_checkpoint_registry.rds"))

write.csv(master_grp_df,
          file.path(root_ckpt_dir, "master_group_test_metrics.csv"),
          row.names = FALSE)
saveRDS(master_grp_df,
        file.path(root_ckpt_dir, "master_group_test_metrics.rds"))

cat("\nSaved master registries.\n")

cat("\n====================================================\n")
cat("MASTER TEST METRICS BY RUN\n")
cat("====================================================\n")
print(master_results_df)

summary_overall <- master_results_df %>%
  summarise(
    acc_mean  = mean(acc_test,  na.rm = TRUE),
    acc_sd    = sd(acc_test,    na.rm = TRUE),
    sens_mean = mean(sens_test, na.rm = TRUE),
    sens_sd   = sd(sens_test,   na.rm = TRUE),
    spec_mean = mean(spec_test, na.rm = TRUE),
    spec_sd   = sd(spec_test,   na.rm = TRUE)
  )

cat("\n====================================================\n")
cat("OVERALL SUMMARY ACROSS SPLITS × SEEDS\n")
cat("====================================================\n")
print(summary_overall)

summary_by_split <- master_results_df %>%
  group_by(split_name) %>%
  summarise(
    acc_mean  = mean(acc_test,  na.rm = TRUE),
    acc_sd    = sd(acc_test,    na.rm = TRUE),
    sens_mean = mean(sens_test, na.rm = TRUE),
    sens_sd   = sd(sens_test,   na.rm = TRUE),
    spec_mean = mean(spec_test, na.rm = TRUE),
    spec_sd   = sd(spec_test,   na.rm = TRUE),
    .groups = "drop"
  )

cat("\n====================================================\n")
cat("SUMMARY BY SPLIT\n")
cat("====================================================\n")
print(summary_by_split)

summarise_group_runs <- function(df) {
  df %>%
    group_by(split_name, group) %>%
    summarise(
      acc_mean  = mean(acc,  na.rm = TRUE),
      acc_sd    = sd(acc,    na.rm = TRUE),
      sens_mean = mean(sens, na.rm = TRUE),
      sens_sd   = sd(sens,   na.rm = TRUE),
      spec_mean = mean(spec, na.rm = TRUE),
      spec_sd   = sd(spec,   na.rm = TRUE),
      
      tp_mean = mean(tp, na.rm = TRUE),
      tp_sd   = sd(tp,   na.rm = TRUE),
      tn_mean = mean(tn, na.rm = TRUE),
      tn_sd   = sd(tn,   na.rm = TRUE),
      fp_mean = mean(fp, na.rm = TRUE),
      fp_sd   = sd(fp,   na.rm = TRUE),
      fn_mean = mean(fn, na.rm = TRUE),
      fn_sd   = sd(fn,   na.rm = TRUE),
      
      n_mean  = mean(n, na.rm = TRUE),
      n_sd    = sd(n,   na.rm = TRUE),
      .groups = "drop"
    )
}

group_summary_by_split <- summarise_group_runs(master_grp_df)

cat("\n====================================================\n")
cat("GROUP SUMMARY BY SPLIT\n")
cat("====================================================\n")
print(group_summary_by_split)

write.csv(summary_by_split,
          file.path(root_ckpt_dir, "summary_by_split.csv"),
          row.names = FALSE)
write.csv(group_summary_by_split,
          file.path(root_ckpt_dir, "group_summary_by_split.csv"),
          row.names = FALSE)

## ================================================================
## 99. Save global end-of-training image
## ================================================================
save.image(file.path(root_ckpt_dir, "TRAINING_fullspec_IDsplit_multisplit_10x20.RData"))
cat("\nSaved global end-of-training image:",
    file.path(root_ckpt_dir, "TRAINING_fullspec_IDsplit_multisplit_10x20.RData"), "\n")