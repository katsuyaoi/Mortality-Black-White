###############################################################################
## BASELINE (Observed-only) TRAINING SCRIPT w/ CHECKPOINTS — FULL, SELF-CONTAINED
## - GPU-ready (CUDA if available; else CPU)
## - 100 seeds (configurable)
## - STRATIFIED 80/20 split (row-level; NOTE: not ID-level grouping)
## - Saves per-seed checkpoints: checkpoints_base_observed/seed_###.pt
## - Saves a SMALL “slim” RData for later eval scripts:
##     TRAINING_base_observed_100seeds_slim.RData
## - NO plots, NO tables, NO counterfactuals here
###############################################################################

## ================================================================
## 0) Libraries + setup
## ================================================================
suppressPackageStartupMessages({
  library(torch)
  library(caret)    # stratified split
  library(dplyr)
  library(tidyr)
})

torch_manual_seed(142)

setwd("datalocation")

## ================================================================
## 0.1) Device (GPU if available)
## ================================================================
device <- torch_device(if (torch::cuda_is_available()) "cuda" else "cpu")
cat("Using device:", device$type, "\n")
if (device$type == "cuda") {
  cat("CUDA devices:", torch::cuda_device_count(), "\n")
  cat("CUDA current device:", torch::cuda_current_device(), "\n")
}

## ================================================================
## 0.2) Checkpoint directory + helpers
## ================================================================
ckpt_dir <- "checkpoints_base_observed"
dir.create(ckpt_dir, showWarnings = FALSE, recursive = TRUE)

save_ckpt <- function(model, path) {
  sd_cpu <- lapply(model$state_dict(), function(t) t$to(device = torch_device("cpu")))
  torch_save(sd_cpu, path)
}

load_ckpt <- function(model, path, device) {
  sd <- torch_load(path)
  model$load_state_dict(sd)
  model$to(device = device)
  invisible(model)
}

to_cpu_array <- function(x) {
  as.array(x$to(device = torch_device("cpu")))
}

## ================================================================
## 1) Metrics helpers
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
  
  list(acc = acc, sens = sens, spec = spec, tn = tn, tp = tp, fn = fn, fp = fp,
       n = (tp + tn + fp + fn))
}

eval_by_group_simple <- function(truth01, preds01, group_vec,
                                 groups_keep = c("NH-White","NH-Black")) {
  df <- data.frame(
    truth01 = as.integer(truth01),
    preds01 = as.integer(preds01),
    group   = as.character(group_vec),
    stringsAsFactors = FALSE
  ) %>%
    filter(!is.na(group), group %in% groups_keep)
  
  df %>%
    group_by(group) %>%
    summarise(
      n  = n(),
      tp = sum(preds01 == 1L & truth01 == 1L, na.rm = TRUE),
      tn = sum(preds01 == 0L & truth01 == 0L, na.rm = TRUE),
      fp = sum(preds01 == 1L & truth01 == 0L, na.rm = TRUE),
      fn = sum(preds01 == 0L & truth01 == 1L, na.rm = TRUE),
      acc  = (tp + tn) / max(1, (tp + tn + fp + fn)),
      sens = if ((tp + fn) > 0) tp / (tp + fn) else NA_real_,
      spec = if ((tn + fp) > 0) tn / (tn + fp) else NA_real_,
      .groups = "drop"
    )
}

## ================================================================
## 2) Load + merge data (survival + AL/personality)
##     NOTE: merged for comparability; baseline predictors below
## ================================================================
data_surv <- read.csv("hrs_survival.csv", header = TRUE)
data_al   <- read.csv("hrs_al.csv",    header = TRUE)

overlap <- intersect(names(data_surv), names(data_al))
overlap <- setdiff(overlap, "id")

data_al_clean <- data_al %>% select(-all_of(overlap))

data_surv_al <- data_surv %>%
  semi_join(data_al_clean, by = "id") %>%
  left_join(data_al_clean,  by = "id")

## ID index for random effect
id_index_all <- as.integer(factor(data_surv_al$id))

## ================================================================
## 3) Outcome + predictors (BASELINE ONLY)
## ================================================================
data_surv_al <- data_surv_al %>%
  mutate(
    age_raw   = age_,
    age_raw_1 = age_1,
    age_raw_3 = age_3,
    age_raw_5 = age_5,
    
    age_  = log(age_),
    age_1 = log(age_1),
    age_3 = log(age_3),
    age_5 = log(age_5)
  )

event_var <- "died_"
time_vars <- "age_"

static_covars <- c(
  "ed_2","ed_3","ed_4","ed_5",
  "black","others","hispanic","female",
  "tage", "year_2", "year_3", "year_4", "year_5", "year_6"
)

other_covars <- c("mothered_2","mothered_3","mothered_4","mothered_5")

tv_bases <- c("sayret","mwid","cesd","shlt","cancre","diabe","hearte","mobila","adl5a")
waves <- 1:15

tv_vars <- unlist(lapply(tv_bases, function(b) {
  cols <- paste0(b, "_", waves)
  cols[cols %in% names(data_surv_al)]
}))

age_raw_vars <- c("age_raw", "age_raw_1", "age_raw_3", "age_raw_5")

predictors <- unique(c(time_vars, static_covars, other_covars, tv_vars, age_raw_vars))

## ================================================================
## 4) Build df_core, X, y, ID indices
## ================================================================
df_core <- data_surv_al[, c(event_var, predictors)]

keep_rows <- !is.na(df_core[[event_var]])
df_core   <- df_core[keep_rows, , drop = FALSE]

id_index_core <- id_index_all[keep_rows]
stopifnot(length(id_index_core) == nrow(df_core))

y_fac   <- factor(df_core[[event_var]], levels = c(0, 1))
classes <- levels(y_fac)       # c("0","1")
y_idx   <- as.integer(y_fac)   # 1/2 indices for torch CE

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

cat("Core rows:", nrow(df_core), "\n")
cat("X dim:", nrow(X), "x", ncol(X), "\n")
stopifnot(nrow(X) == length(y_idx))

## ================================================================
## 5) NA mask, binary detection, scaling (NA-aware)
## ================================================================
na_mask_mat <- ifelse(is.na(X), 1, 0)

is_binary_col <- function(v) {
  u <- unique(v); u <- u[!is.na(u)]
  all(u %in% c(0, 1))
}
bin_mask <- apply(X, 2, is_binary_col)

age_names <- c("age_", "age_1", "age_3", "age_5",
               "age_raw","age_raw_1","age_raw_3","age_raw_5")
age_cols <- which(colnames(X) %in% age_names)
if (length(age_cols) > 0L) bin_mask[age_cols] <- FALSE

X_scaled <- X
if (any(!bin_mask)) {
  non_age_non_bin <- setdiff(which(!bin_mask), age_cols)
  if (length(non_age_non_bin) > 0L) {
    X_scaled[, non_age_non_bin] <- scale(X[, non_age_non_bin])
  }
}
X_scaled[is.na(X_scaled)] <- 0

cat("Scaled X dim:", nrow(X_scaled), "x", ncol(X_scaled), "\n")

## ================================================================
## 6) Stratified train/test split (row-level)
## ================================================================
set.seed(123)
tr_index <- createDataPartition(y_idx, p = 0.80, list = FALSE)
tr <- as.vector(tr_index)
te <- setdiff(seq_len(nrow(X_scaled)), tr)

Xtr <- X_scaled[tr, , drop = FALSE];  ytr <- y_idx[tr]
Xte <- X_scaled[te, , drop = FALSE];  yte <- y_idx[te]

mask_tr <- na_mask_mat[tr, , drop = FALSE]
mask_te <- na_mask_mat[te, , drop = FALSE]

idtr <- id_index_core[tr]
idte <- id_index_core[te]

all_cols <- colnames(Xtr)

cat("Train/Test sizes:", nrow(Xtr), nrow(Xte), "\n")
cat("Class counts (train):", paste(table(factor(ytr, levels = 1:2)), collapse=" "), "\n")
cat("Class counts (test): ", paste(table(factor(yte, levels = 1:2)), collapse=" "), "\n")

## ================================================================
## 7) Build feature index lists: static + TV waves
## ================================================================
static_names <- c(time_vars, static_covars, other_covars)
static_idx   <- match(static_names, all_cols)
if (any(is.na(static_idx))) stop("Missing static cols: ",
                                 paste(static_names[is.na(static_idx)], collapse=", "))
static_idx <- as.integer(static_idx)

wave_idx_list <- lapply(waves, function(t) {
  wave_vars <- paste0(tv_bases, "_", t)
  wave_vars <- wave_vars[wave_vars %in% all_cols]
  idx <- match(wave_vars, all_cols)
  idx[!is.na(idx)]
})

wave_lengths <- sapply(wave_idx_list, length)
non_empty <- which(wave_lengths > 0)
if (length(non_empty) == 0L) stop("No non-empty TV waves found; check tv_bases.")
wave_idx_list <- wave_idx_list[non_empty]
wave_lengths  <- wave_lengths[non_empty]

if (length(unique(wave_lengths)) != 1L) {
  stop("Remaining TV waves do not have equal feature count.")
}

tv_step_feat_dim <- as.integer(wave_lengths[1])
n_steps          <- as.integer(length(wave_idx_list))
tv_idx           <- as.integer(unlist(wave_idx_list))

cat("TV steps:", n_steps, " | per-step dim:", tv_step_feat_dim, "\n")

## ================================================================
## 8) Baseline NA-aware MLP with ID random effect
## ================================================================
output_dim <- length(classes)
num_ids    <- max(id_index_core)
id_emb_dim <- 1L

hazard_mlp_base <- nn_module(
  "hazard_mlp_base",
  initialize = function(static_idx,
                        tv_idx,
                        n_steps,
                        tv_step_feat_dim,
                        num_ids,
                        id_emb_dim = 1L,
                        h_static  = 64,
                        h_tv      = 64,
                        mix_dim   = 16,
                        output_dim = 2,
                        drop_p    = 0.25) {
    
    self$static_idx <- as.integer(static_idx)
    self$tv_idx     <- as.integer(tv_idx)
    
    self$h_tv_dim        <- as.integer(h_tv)
    self$n_steps         <- as.integer(n_steps)
    self$tv_step_feat_dim<- as.integer(tv_step_feat_dim)
    
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
    
    self$fc_mix  <- nn_linear(h_static + h_tv, mix_dim)
    self$fc_base <- nn_linear(mix_dim, output_dim)
    
    self$drop <- nn_dropout(p = drop_p)
  },
  
  forward = function(x, mask, id_idx) {
    device <- x$device
    B <- as.integer(x$size()[1])
    
    if (id_idx$device != device) id_idx <- id_idx$to(device = device)
    
    static_idx_t <- torch_tensor(self$static_idx, dtype = torch_long(), device = device)
    tv_idx_t     <- torch_tensor(self$tv_idx,     dtype = torch_long(), device = device)
    
    ## static
    x_static <- x$index_select(2L, static_idx_t)
    h_static <- x_static %>%
      self$fc_static1() %>% nnf_relu() %>% self$drop() %>%
      self$fc_static2() %>% nnf_relu()
    
    ## tv
    x_tv    <- x$index_select(2L, tv_idx_t)
    mask_tv <- mask$index_select(2L, tv_idx_t)
    
    T_steps <- as.integer(self$n_steps)
    F_step  <- as.integer(self$tv_step_feat_dim)
    
    x_seq <- x_tv$view(c(B, T_steps, F_step))
    m_seq <- mask_tv$view(c(B, T_steps, F_step))
    
    tv_seq_in <- torch_cat(list(x_seq, m_seq), dim = 3L)  # (B,T,2F)
    tv_flat   <- tv_seq_in$view(c(as.integer(B * T_steps), as.integer(2L * F_step)))
    
    h_tv_flat <- tv_flat %>%
      self$fc_tv1() %>% nnf_relu() %>% self$drop() %>%
      self$fc_tv2() %>% nnf_relu() %>% self$drop()
    
    h_tv_seq <- h_tv_flat$view(c(B, T_steps, as.integer(self$h_tv_dim)))
    
    att_scores <- self$att_tv(h_tv_seq)$squeeze(3L)          # (B,T)
    alpha      <- nnf_softmax(att_scores, dim = 2L)$unsqueeze(3L)
    h_tv       <- (h_tv_seq * alpha)$sum(dim = 2L)           # (B,h_tv)
    
    ## mix + head
    base_input <- torch_cat(list(h_static, h_tv), dim = 2L)
    z_mix      <- self$fc_mix(base_input) %>% nnf_relu()
    logits     <- self$fc_base(z_mix)
    
    ## ID random intercept on death logit (class "1" = col 2)
    id_vec   <- self$id_embed(id_idx)        # (B,1)
    id_shift <- self$alpha_id * id_vec
    
    logits2 <- logits$clone()
    logits2[, 2] <- logits2[, 2] + id_shift$squeeze(2L)
    
    logits2
  }
)

## ================================================================
## 9) Class weights + tensors on device
## ================================================================
train_counts <- as.numeric(table(factor(ytr, levels = 1:2)))
cw <- 1 / train_counts
cw <- cw / mean(cw)
cat("Class weights:", paste(round(cw, 3), collapse = " "), "\n")

xtr_t     <- torch_tensor(Xtr,        dtype = torch_float(), device = device)
mask_tr_t <- torch_tensor(mask_tr,    dtype = torch_float(), device = device)
ytr_t     <- torch_tensor(ytr,        dtype = torch_long(),  device = device)
idtr_t    <- torch_tensor(idtr,       dtype = torch_long(),  device = device)

xt_t      <- torch_tensor(Xte,        dtype = torch_float(), device = device)
mask_te_t <- torch_tensor(mask_te,    dtype = torch_float(), device = device)
idte_t    <- torch_tensor(idte,       dtype = torch_long(),  device = device)

## Group labels aligned to df_core rows (keep_rows space)
group_core <- data_surv_al[keep_rows, ] %>%
  transmute(group = case_when(
    black == 1 ~ "NH-Black",
    black == 0 & hispanic == 0 & others == 0 ~ "NH-White",
    TRUE ~ NA_character_
  )) %>%
  pull(group)

stopifnot(length(group_core) == nrow(df_core))

## ================================================================
## 10) Seed runner: train-or-load checkpoint, then evaluate TEST + FULL
##     Returns a “rich” list per seed (for later evaluation scripts if you want),
##     but we will NOT build plots/tables here.
## ================================================================
run_baseline_seed <- function(torch_seed,
                              epochs       = 100,
                              batch_size   = 1024,
                              lr           = 1e-3,
                              weight_decay = 1e-3,
                              batch_size_te = 1024,
                              verbose_every = 10,
                              save_ckpt_each_seed = TRUE,
                              ckpt_dir = "checkpoints_base_observed") {
  
  ckpt_path <- file.path(ckpt_dir, sprintf("seed_%03d.pt", torch_seed))
  
  cat("\n==============================\n")
  cat("BASELINE seed:", torch_seed, "\n")
  cat("Checkpoint:", ckpt_path, "\n")
  cat("==============================\n")
  
  torch_manual_seed(torch_seed)
  set.seed(torch_seed)
  
  model <- hazard_mlp_base(
    static_idx       = static_idx,
    tv_idx           = tv_idx,
    n_steps          = n_steps,
    tv_step_feat_dim = tv_step_feat_dim,
    num_ids          = num_ids,
    id_emb_dim       = id_emb_dim,
    h_static         = 64,
    h_tv             = 64,
    mix_dim          = 16,
    output_dim       = output_dim,
    drop_p           = 0.25
  )$to(device = device)
  
  if (file.exists(ckpt_path)) {
    cat(">> Found checkpoint. Loading and skipping training.\n")
    load_ckpt(model, ckpt_path, device)
  } else {
    cat(">> No checkpoint. Training...\n")
    
    class_weights <- torch_tensor(cw, dtype = torch_float(), device = device)
    optimizer     <- optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)
    
    n_train   <- nrow(Xtr)
    n_batches <- ceiling(n_train / batch_size)
    
    for (epoch in 1:epochs) {
      model$train()
      idx_all <- sample.int(n_train)
      running_loss <- 0
      
      for (b in 1:n_batches) {
        i1 <- (b - 1) * batch_size + 1
        i2 <- min(b * batch_size, n_train)
        idx <- idx_all[i1:i2]
        
        x_batch  <- xtr_t[idx, , drop = FALSE]
        m_batch  <- mask_tr_t[idx, , drop = FALSE]
        y_batch  <- ytr_t[idx]
        id_batch <- idtr_t[idx]
        
        optimizer$zero_grad()
        logits <- model(x_batch, m_batch, id_batch)
        loss   <- nnf_cross_entropy(logits, y_batch, weight = class_weights)
        loss$backward()
        optimizer$step()
        
        running_loss <- running_loss + loss$item()
      }
      
      if (epoch %% verbose_every == 0) {
        cat(sprintf("[Seed %d] Epoch %3d | avg train loss: %.4f\n",
                    torch_seed, epoch, running_loss / n_batches))
      }
    }
    
    if (isTRUE(save_ckpt_each_seed)) {
      save_ckpt(model, ckpt_path)
      cat(">> Saved checkpoint.\n")
    }
  }
  
  ## -------------------------
  ## Evaluation: TEST (batched)
  ## -------------------------
  model$eval()
  
  n_test       <- nrow(Xte)
  n_batches_te <- ceiling(n_test / batch_size_te)
  logits_list  <- vector("list", n_batches_te)
  
  for (b in 1:n_batches_te) {
    i1 <- (b - 1) * batch_size_te + 1
    i2 <- min(b * batch_size_te, n_test)
    idx <- i1:i2
    
    with_no_grad({
      logits_list[[b]] <- model(
        xt_t[idx, , drop = FALSE],
        mask_te_t[idx, , drop = FALSE],
        idte_t[idx]
      )
    })
  }
  
  logits_test <- torch_cat(logits_list, dim = 1L)
  stopifnot(dim(logits_test)[1] == n_test)
  
  preds_test_12 <- as.integer(to_cpu_array(torch_argmax(logits_test, dim = 2L))) # 1/2
  truth_test_12 <- as.integer(yte)                                               # 1/2
  
  pred_lbl_test <- factor(preds_test_12, levels = 1:2, labels = classes)
  true_lbl_test <- factor(truth_test_12, levels = 1:2, labels = classes)
  
  acc_test <- mean(pred_lbl_test == true_lbl_test)
  cm_test  <- table(Predicted = pred_lbl_test, Actual = true_lbl_test)
  
  tn <- if ("0" %in% rownames(cm_test) && "0" %in% colnames(cm_test)) cm_test["0","0"] else 0
  tp <- if ("1" %in% rownames(cm_test) && "1" %in% colnames(cm_test)) cm_test["1","1"] else 0
  fn <- if ("0" %in% rownames(cm_test) && "1" %in% colnames(cm_test)) cm_test["0","1"] else 0
  fp <- if ("1" %in% rownames(cm_test) && "0" %in% colnames(cm_test)) cm_test["1","0"] else 0
  
  sens_test <- if ((tp + fn) > 0) as.numeric(tp / (tp + fn)) else NA_real_
  spec_test <- if ((tn + fp) > 0) as.numeric(tn / (tn + fp)) else NA_real_
  
  ## subgroup (BW only) on TEST
  group_test <- group_core[te]
  group_test <- ifelse(group_test %in% c("NH-White","NH-Black"), group_test, NA)
  
  preds01_test <- as.integer(pred_lbl_test == "1")
  truth01_test <- as.integer(true_lbl_test == "1")
  
  bygroup_test <- eval_by_group_simple(truth01_test, preds01_test, group_test)
  
  cat(sprintf("Seed %d | TEST acc=%.3f sens=%.3f spec=%.3f\n",
              torch_seed, acc_test, sens_test, spec_test))
  
  ## -------------------------
  ## OPTIONAL: evaluate on FULL core rows (no saving big arrays here)
  ## - If you truly don’t need it now, set do_full_eval=FALSE.
  ## -------------------------
  do_full_eval <- FALSE
  acc_full <- sens_full <- spec_full <- NA_real_
  bygroup_full <- NULL
  
  if (isTRUE(do_full_eval)) {
    # build full tensors on the fly to avoid pointer issues later
    x_all_t    <- torch_tensor(X_scaled,    dtype = torch_float(), device = device)
    mask_all_t <- torch_tensor(na_mask_mat, dtype = torch_float(), device = device)
    id_all_t   <- torch_tensor(id_index_core, dtype = torch_long(), device = device)
    
    n_all <- nrow(X_scaled)
    batch_all <- 2048
    nb_all <- ceiling(n_all / batch_all)
    logits_all_list <- vector("list", nb_all)
    
    for (b in 1:nb_all) {
      i1 <- (b - 1) * batch_all + 1
      i2 <- min(b * batch_all, n_all)
      idx <- i1:i2
      
      with_no_grad({
        logits_all_list[[b]] <- model(
          x_all_t[idx, , drop = FALSE],
          mask_all_t[idx, , drop = FALSE],
          id_all_t[idx]
        )
      })
    }
    
    logits_all <- torch_cat(logits_all_list, dim = 1L)
    preds_all_12 <- as.integer(to_cpu_array(torch_argmax(logits_all, dim = 2L)))
    truth_all_12 <- as.integer(y_idx)
    
    pred_lbl_all <- factor(preds_all_12, levels = 1:2, labels = classes)
    true_lbl_all <- factor(truth_all_12, levels = 1:2, labels = classes)
    
    acc_full <- mean(pred_lbl_all == true_lbl_all)
    cm_all   <- table(Predicted = pred_lbl_all, Actual = true_lbl_all)
    
    tnA <- if ("0" %in% rownames(cm_all) && "0" %in% colnames(cm_all)) cm_all["0","0"] else 0
    tpA <- if ("1" %in% rownames(cm_all) && "1" %in% colnames(cm_all)) cm_all["1","1"] else 0
    fnA <- if ("0" %in% rownames(cm_all) && "1" %in% colnames(cm_all)) cm_all["0","1"] else 0
    fpA <- if ("1" %in% rownames(cm_all) && "0" %in% colnames(cm_all)) cm_all["1","0"] else 0
    
    sens_full <- if ((tpA + fnA) > 0) as.numeric(tpA / (tpA + fnA)) else NA_real_
    spec_full <- if ((tnA + fpA) > 0) as.numeric(tnA / (tnA + fpA)) else NA_real_
    
    group_full <- group_core
    group_full <- ifelse(group_full %in% c("NH-White","NH-Black"), group_full, NA)
    
    preds01_all <- as.integer(pred_lbl_all == "1")
    truth01_all <- as.integer(true_lbl_all == "1")
    
    bygroup_full <- eval_by_group_simple(truth01_all, preds01_all, group_full)
    
    rm(x_all_t, mask_all_t, id_all_t, logits_all, logits_all_list)
    gc()
  }
  
  list(
    seed        = torch_seed,
    ckpt_path   = ckpt_path,
    acc_test    = acc_test,
    sens_test   = sens_test,
    spec_test   = spec_test,
    bygroup_test= bygroup_test,
    acc_full    = acc_full,
    sens_full   = sens_full,
    spec_full   = spec_full,
    bygroup_full= bygroup_full,
    preds_test_12 = preds_test_12,
    truth_test_12 = truth_test_12
  )
}

## ================================================================
## 11) Run seeds
## ================================================================
seeds <- 1:100
results_list <- lapply(seeds, run_baseline_seed,
                       epochs = 100,
                       save_ckpt_each_seed = TRUE,
                       ckpt_dir = ckpt_dir)

## ================================================================
## 12) Save SLIM training artifact (for later eval-only scripts)
##     - store split indices + model config indices + class weights
##     - store per-seed summary metrics + ckpt paths
##     - (OPTIONAL) store per-seed test preds/truth if you want later
## ================================================================
results_slim <- lapply(results_list, function(res) {
  list(
    seed      = res$seed,
    ckpt_path = res$ckpt_path,
    acc_test  = res$acc_test,
    sens_test = res$sens_test,
    spec_test = res$spec_test
  )
})

save(
  results_slim,
  keep_rows, tr, te,
  predictors, classes,
  cw,
  static_idx, tv_idx, n_steps, tv_step_feat_dim,
  id_index_core,
  file = "TRAINING_base_observed_100seeds_slim.RData"
)

cat("\nSaved: TRAINING_base_observed_100seeds_slim.RData\n")
cat("Checkpoints in:", ckpt_dir, "\n")
###############################################################################
## END
###############################################################################
