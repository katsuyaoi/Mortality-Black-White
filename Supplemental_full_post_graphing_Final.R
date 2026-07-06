###############################################################################
## STANDALONE POST-EVALUATION: AGE x time_ HAZARD SURFACES
##
## Purpose:
##   1) Load already-trained full-model checkpoints.
##   2) Recreate observed and held-constant predictions.
##   3) Append test_row_id, id, time_, age, race, and sex to row-level outputs.
##   4) Produce age x time_ predicted-hazard surfaces.
##   5) Replicate Figure 2 with added time_ dimension:
##        observed predicted risk - held-constant predicted risk
##      summarized by age_bin x time_bin x race x sex x scenario.
##
## Important:
##   - This script does NOT retrain the model.
##   - This script does NOT add time_ as a model input.
##   - time_ is used as post-estimation metadata for summaries/graphs.
##   - Run this after the original 10x20 training script has finished.
###############################################################################

packages_needed <- c(
  "torch", "dplyr", "tidyr", "readr", "ggplot2", "plotly", "htmlwidgets"
)

to_install <- packages_needed[!packages_needed %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(packages_needed, library, character.only = TRUE))

## -----------------------------------------------------------------------------
## USER SETTINGS
## -----------------------------------------------------------------------------

setwd("yourdatalocation")

root_ckpt_dir <- "checkpoints_fullspec_IDsplit_multisplit"
split_meta_dir <- file.path(root_ckpt_dir, "split_meta")

out_dir <- file.path(root_ckpt_dir, "post_eval_time_surface_standalone")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## ------------------------------------------------------------------
## OUTPUT MODULES — EDIT THESE SWITCHES
## ------------------------------------------------------------------

## Keep row-level outputs off unless you explicitly need them. These can be large.
SAVE_OBSERVED_ROWLEVEL <- FALSE
SAVE_SCENARIO_ROWLEVEL <- FALSE

## CSV modules. Keep the Figure-2-equivalent headline table on.
SAVE_OBSERVED_SURFACE_CSV <- FALSE
SAVE_OBSERVED_METRICS_CSV <- FALSE
SAVE_PROBE2_SEED_CSV <- FALSE
SAVE_PROBE2_METRICS_CSV <- FALSE
SAVE_PROBE2_SPLIT_CSV <- FALSE
SAVE_PROBE2_HEADLINE_CSV <- TRUE

## Small text diagnostics for describing where positive displacement concentrates.
SAVE_HOTSPOT_SUMMARY_CSV <- TRUE
SAVE_HOTSPOT_TEXT_CSV <- TRUE

## Graph modules.
SAVE_HEATMAP_PNG <- FALSE
SAVE_OBSERVED_3D_HTML <- FALSE
SAVE_PROBE2_3D_HTML <- TRUE

## Dedicated plot subfolder so this does not clutter the old plots folder.
PLOT_SUBDIR <- "plots_probe2_minimal"
plot_dir <- file.path(out_dir, PLOT_SUBDIR)
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

## ------------------------------------------------------------------
## BINNING — EDIT IF NEEDED
## ------------------------------------------------------------------

## Age bins.
AGE_BIN_MIN <- 50
AGE_BIN_WIDTH <- 5
AGE_BIN_TOPCODE <- 90
AGE_BIN_INCLUDE_BELOW_MIN <- FALSE

## time_ bins.
## If time_ is already integer/serial interval, keep width = 1.
TIME_BIN_WIDTH <- 1

## ------------------------------------------------------------------
## 3D GRAPH SETTINGS — EDIT THESE LABELS / DIMENSIONS
## ------------------------------------------------------------------

## Graph font.
## On Windows, Plotly/PNG output should recognize Times New Roman directly.
GRAPH_FONT_FAMILY <- "Times New Roman"
GRAPH_BASE_SIZE <- 13
GRAPH_TITLE_SIZE <- 15

## Only these groups/scenarios are plotted by default.
PLOT_PROBE2_SCENARIOS <- c("AL + Personality held constant")
PLOT_RACES <- c("NH-Black", "NH-White")
PLOT_SEXES <- c("Female", "Male")

## Output behavior.
OVERWRITE_3D_HTML <- TRUE
SELFCONTAINED_3D_HTML <- TRUE   # TRUE = one HTML file; FALSE = smaller HTML plus dependency folder

## Plotly size and fixed scene proportions.
PLOTLY_WIDTH <- 950
PLOTLY_HEIGHT <- 760
SCENE_ASPECT_X <- 1.25
SCENE_ASPECT_Y <- 1.00
SCENE_ASPECT_Z <- 0.75

## Axis labels.
AXIS_TITLE_X <- "Age"
AXIS_TITLE_Y <- "Interval"
AXIS_TITLE_Z_PROBE2 <- "Mortality Risk Difference in Prediction (%)"
AXIS_TITLE_Z_OBSERVED <- "Predicted mortality risk (%)"

## Titles.
PROBE2_TITLE_PREFIX <- "Figure 2 extension"
OBSERVED_TITLE_PREFIX <- "Observed predicted mortality-risk surface"

## Camera angle.
CAMERA_EYE_X <- 1.65
CAMERA_EYE_Y <- 1.65
CAMERA_EYE_Z <- 0.90

## Use the same z-axis scale across the four probe plots.
LOCK_Z_AXIS_ACROSS_PROBE2_PLOTS <- TRUE

## 3D surface color behavior.
## The surface height still uses the original z values.
## Color uses pmax(z, 0), so all values <= 0 receive the same light-gray baseline color.
## Positive values become darker as recovered risk increases, scaled within each plot/group.
COLOR_3D_NONPOSITIVE_AS_ZERO <- TRUE
COLOR_3D_RELATIVE_WITHIN_PANEL <- TRUE
COLOR_3D_BASELINE <- "#D9D9D9"
COLOR_3D_LOW_POSITIVE <- "#BFD3E6"
COLOR_3D_MID_POSITIVE <- "#8FB6D9"
COLOR_3D_HIGH_POSITIVE <- "#1F2A44"

## 3D surface lighting. Lower ambient darkens ridge shadows.
SURFACE_LIGHT_AMBIENT <- 0.4
SURFACE_LIGHT_DIFFUSE <- 0.80
SURFACE_LIGHT_SPECULAR <- 0.05
SURFACE_LIGHT_ROUGHNESS <- 0.50
SURFACE_LIGHT_FRESNEL <- 0.20
SURFACE_LIGHT_X <- 10
SURFACE_LIGHT_Y <- 100000
SURFACE_LIGHT_Z <- 0

## Hotspot text settings.
## Positive displacement = observed predicted risk - held-constant predicted risk > 0.
## The hotspot is the upper tail of positive displacement cells within each race/sex group.
HOTSPOT_QUANTILE <- 0.90
HOTSPOT_TOP_N_CELLS <- 5
SAVE_HOTSPOT_TOP_CELLS_CSV <- TRUE
SAVE_HOTSPOT_PARAGRAPH_TXT <- TRUE

## -----------------------------------------------------------------------------
## GLOBAL MODEL IMAGE
## -----------------------------------------------------------------------------

global_image <- file.path(root_ckpt_dir, "TRAINING_fullspec_IDsplit_multisplit_10x20.RData")

if (!file.exists(global_image)) {
  stop("Global training image not found: ", global_image)
}

load(global_image)
stopifnot(exists("hazard_mlp_full"))

## -----------------------------------------------------------------------------
## HELPERS
## -----------------------------------------------------------------------------

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
    dplyr::case_when(
      is.na(age_raw) ~ NA_integer_,
      age_raw >= AGE_BIN_TOPCODE ~ AGE_BIN_TOPCODE,
      age_raw < AGE_BIN_MIN ~ AGE_BIN_MIN,
      TRUE ~ as.integer(
        floor((age_raw - AGE_BIN_MIN) / AGE_BIN_WIDTH) * AGE_BIN_WIDTH + AGE_BIN_MIN
      )
    )
  } else {
    dplyr::case_when(
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
  dplyr::case_when(
    is.na(age_bin) ~ NA_character_,
    age_bin == AGE_BIN_TOPCODE ~ paste0(AGE_BIN_TOPCODE, "+"),
    TRUE ~ paste0(age_bin, "-", age_bin + AGE_BIN_WIDTH - 1)
  )
}

make_time_bin <- function(time_raw) {
  time_raw <- as.numeric(time_raw)
  dplyr::case_when(
    is.na(time_raw) ~ NA_real_,
    TRUE ~ floor(time_raw / TIME_BIN_WIDTH) * TIME_BIN_WIDTH
  )
}

time_bin_label <- function(time_bin) {
  ifelse(is.na(time_bin), NA_character_, as.character(time_bin))
}

metrics_counts <- function(pred01, truth01) {
  pred01 <- as.integer(pred01)
  truth01 <- as.integer(truth01)
  
  tn <- sum(pred01 == 0L & truth01 == 0L, na.rm = TRUE)
  tp <- sum(pred01 == 1L & truth01 == 1L, na.rm = TRUE)
  fn <- sum(pred01 == 0L & truth01 == 1L, na.rm = TRUE)
  fp <- sum(pred01 == 1L & truth01 == 0L, na.rm = TRUE)
  
  n <- tp + tn + fp + fn
  events <- tp + fn
  nonevents <- tn + fp
  
  acc <- if (n > 0) (tp + tn) / n else NA_real_
  sens <- if (events > 0) tp / events else NA_real_
  spec <- if (nonevents > 0) tn / nonevents else NA_real_
  fnr <- if (events > 0) fn / events else NA_real_
  fpr <- if (nonevents > 0) fp / nonevents else NA_real_
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
        list(mean = ~ safe_mean(.x), sd = ~ safe_sd(.x)),
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
          sd = ~ safe_sd(.x),
          min = ~ safe_min(.x),
          max = ~ safe_max(.x)
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
    return(exp(as.numeric(d[test_row_ids, "age_"])))
  }
  
  stop("Could not recover raw age from data_surv_al.")
}

make_surface_matrix <- function(df, x_var, y_var, z_var) {
  df <- df %>%
    select(all_of(c(x_var, y_var, z_var))) %>%
    filter(!is.na(.data[[x_var]]), !is.na(.data[[y_var]])) %>%
    group_by(across(all_of(c(x_var, y_var)))) %>%
    summarise(z = mean(.data[[z_var]], na.rm = TRUE), .groups = "drop")
  
  x_vals <- sort(unique(df[[x_var]]))
  y_vals <- sort(unique(df[[y_var]]))
  
  grid <- expand.grid(
    x = x_vals,
    y = y_vals,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  names(grid) <- c(x_var, y_var)
  
  grid <- grid %>%
    left_join(df, by = c(x_var, y_var)) %>%
    arrange(.data[[y_var]], .data[[x_var]])
  
  z_mat <- matrix(grid$z, nrow = length(y_vals), ncol = length(x_vals), byrow = TRUE)
  
  list(x = x_vals, y = y_vals, z = z_mat)
}

safe_plot_stub <- function(x) {
  x %>%
    gsub("NH-", "", .) %>%
    gsub("[^A-Za-z0-9]+", "_", .) %>%
    gsub("^_|_$", "", .) %>%
    tolower()
}

save_surface3d <- function(df,
                           z_var,
                           title,
                           file_path,
                           z_title,
                           z_range = NULL) {

  if (!isTRUE(OVERWRITE_3D_HTML) && file.exists(file_path)) {
    cat("  Skipping existing plot:", file_path, "\n")
    return(invisible(NULL))
  }

  surf <- make_surface_matrix(
    df = df,
    x_var = "age_bin",
    y_var = "time_bin",
    z_var = z_var
  )

  zaxis_args <- list(
    title = list(
      text = z_title,
      font = list(family = GRAPH_FONT_FAMILY)
    ),
    tickfont = list(family = GRAPH_FONT_FAMILY)
  )
  if (!is.null(z_range) && length(z_range) == 2L && all(is.finite(z_range))) {
    zaxis_args$range <- z_range
  }

  ## Surface height uses the original z values.
  z_mat <- surf$z

  ## Color uses a separate matrix. This makes all non-positive cells share
  ## the same baseline color while preserving the original 3D surface shape.
  color_mat <- if (isTRUE(COLOR_3D_NONPOSITIVE_AS_ZERO)) {
    pmax(z_mat, 0)
  } else {
    z_mat
  }

  if (isTRUE(COLOR_3D_RELATIVE_WITHIN_PANEL)) {
    color_max <- suppressWarnings(max(color_mat, na.rm = TRUE))
  } else if (!is.null(z_range) && length(z_range) == 2L && all(is.finite(z_range))) {
    color_max <- max(pmax(z_range, 0), na.rm = TRUE)
  } else {
    color_max <- suppressWarnings(max(color_mat, na.rm = TRUE))
  }
  if (!is.finite(color_max) || color_max <= 0) color_max <- 1

  p <- plot_ly(
    x = surf$x,
    y = surf$y,
    z = z_mat,
    surfacecolor = color_mat,
    cmin = 0,
    cmax = color_max,
    colorscale = list(
      c(0.00, COLOR_3D_BASELINE),
      c(0.20, COLOR_3D_BASELINE),
      c(0.45, COLOR_3D_LOW_POSITIVE),
      c(0.70, COLOR_3D_MID_POSITIVE),
      c(1.00, COLOR_3D_HIGH_POSITIVE)
    ),
    colorbar = list(
      title = "",
      tickfont = list(family = GRAPH_FONT_FAMILY),
      tickvals = pretty(c(0, color_max), n = 5),
      ticktext = pretty(c(0, color_max), n = 5)
    ),
    type = "surface",
    lighting = list(
      ambient = SURFACE_LIGHT_AMBIENT,
      diffuse = SURFACE_LIGHT_DIFFUSE,
      specular = SURFACE_LIGHT_SPECULAR,
      roughness = SURFACE_LIGHT_ROUGHNESS,
      fresnel = SURFACE_LIGHT_FRESNEL
    ),
    lightposition = list(
      x = SURFACE_LIGHT_X,
      y = SURFACE_LIGHT_Y,
      z = SURFACE_LIGHT_Z
    ),
    width = PLOTLY_WIDTH,
    height = PLOTLY_HEIGHT
  ) %>%
    layout(
      font = list(family = GRAPH_FONT_FAMILY),
      title = list(
        text = title,
        font = list(family = GRAPH_FONT_FAMILY, size = GRAPH_TITLE_SIZE)
      ),
      margin = list(l = 60, r = 40, b = 60, t = 70),
      scene = list(
        xaxis = list(
          title = list(
            text = AXIS_TITLE_X,
            font = list(family = GRAPH_FONT_FAMILY)
          ),
          tickfont = list(family = GRAPH_FONT_FAMILY),
          tickmode = "array",
          tickvals = surf$x,
          ticktext = age_bin_label(surf$x)
        ),
        yaxis = list(
          title = list(
            text = AXIS_TITLE_Y,
            font = list(family = GRAPH_FONT_FAMILY)
          ),
          tickfont = list(family = GRAPH_FONT_FAMILY)
        ),
        zaxis = zaxis_args,
        aspectmode = "manual",
        aspectratio = list(
          x = SCENE_ASPECT_X,
          y = SCENE_ASPECT_Y,
          z = SCENE_ASPECT_Z
        ),
        camera = list(
          eye = list(
            x = CAMERA_EYE_X,
            y = CAMERA_EYE_Y,
            z = CAMERA_EYE_Z
          )
        )
      )
    )

  htmlwidgets::saveWidget(
    p,
    file = file_path,
    selfcontained = SELFCONTAINED_3D_HTML
  )

  invisible(p)
}

## -----------------------------------------------------------------------------
## FIND SPLIT STATES
## -----------------------------------------------------------------------------

split_state_paths <- list.files(
  split_meta_dir,
  pattern = "^split_[0-9]+_training_state\\.RData$",
  full.names = TRUE
)

split_state_paths <- sort(split_state_paths)

if (length(split_state_paths) == 0L) {
  stop("No split-level training state files found in: ", split_meta_dir)
}

## -----------------------------------------------------------------------------
## STORAGE
## -----------------------------------------------------------------------------

observed_rowlevel_list <- list()
scenario_rowlevel_list <- list()

observed_surface_seed_list <- list()
probe2_surface_seed_list <- list()

observed_metrics_age_time_seed_list <- list()
probe2_metrics_age_time_seed_list <- list()

## -----------------------------------------------------------------------------
## MAIN LOOP
## -----------------------------------------------------------------------------

for (split_path in split_state_paths) {
  
  e <- new.env(parent = globalenv())
  load(split_path, envir = e)
  
  cat("\n====================================================\n")
  cat("Processing:", e$split_name, "| split seed:", e$split_seed, "\n")
  cat("====================================================\n")
  
  if (!("time_" %in% names(e$data_surv_al))) {
    stop("time_ not found in data_surv_al for ", e$split_name,
         ". This standalone script needs time_ in the saved data.")
  }
  
  all_cols <- e$all_cols
  
  al_zero_idx_safe <- strip_age_from_zero(e$al_zero_idx, all_cols, verbose = TRUE)
  pers_zero_idx_safe <- strip_age_from_zero(e$pers_zero_idx, all_cols, verbose = TRUE)
  both_zero_idx_safe <- strip_age_from_zero(e$both_zero_idx, all_cols, verbose = TRUE)
  
  cf_defs <- list(
    Observed = list(cols = integer(0), zero_mask_also = TRUE),
    `AL=0` = list(cols = al_zero_idx_safe, zero_mask_also = TRUE),
    `Personality=0` = list(cols = pers_zero_idx_safe, zero_mask_also = TRUE),
    `AL+Personality=0` = list(cols = both_zero_idx_safe, zero_mask_also = TRUE)
  )
  
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
      test_row_id = test_row_ids,
      id = id,
      time_ = as.numeric(time_),
      time_bin = make_time_bin(time_),
      time_bin_label = time_bin_label(time_bin),
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
  
  results_df <- e$results_df %>% arrange(seed)
  
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
    
    ## Observed prediction.
    logits_obs_t <- predict_logits(
      model = model,
      x_t = xt_t,
      mask_t = mt_t,
      id_t = id_t,
      use_alpers = TRUE
    )
    
    logits_obs <- as.matrix(torch_to_array(logits_obs_t))
    if (ncol(logits_obs) != 2L) stop("Expected binary logits with 2 columns.")
    
    probs_obs <- as.matrix(torch_to_array(nnf_softmax(logits_obs_t, dim = 2L)))
    p_death <- as.numeric(probs_obs[, 2])
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
    
    if (isTRUE(SAVE_OBSERVED_ROWLEVEL)) {
      observed_rowlevel_list[[length(observed_rowlevel_list) + 1L]] <- pred_df
    }
    
    observed_surface_seed_list[[length(observed_surface_seed_list) + 1L]] <-
      pred_df %>%
      filter(
        race %in% c("NH-White", "NH-Black"),
        !is.na(age_bin),
        !is.na(time_bin)
      ) %>%
      group_by(split_name, split_seed, seed, age_bin, age_bin_label,
               time_bin, time_bin_label, race, sex) %>%
      summarise(
        n = n(),
        events = sum(truth01 == 1L, na.rm = TRUE),
        observed_event_rate = mean(truth01 == 1L, na.rm = TRUE),
        mean_predicted_risk = mean(p_death, na.rm = TRUE),
        median_predicted_risk = median(p_death, na.rm = TRUE),
        .groups = "drop"
      )
    
    observed_metrics_age_time_seed_list[[length(observed_metrics_age_time_seed_list) + 1L]] <-
      pred_df %>%
      filter(
        race %in% c("NH-White", "NH-Black"),
        !is.na(age_bin),
        !is.na(time_bin)
      ) %>%
      group_by(split_name, split_seed, seed, age_bin, age_bin_label,
               time_bin, time_bin_label, race, sex) %>%
      summarise(
        metrics_counts(pred01, truth01),
        .groups = "drop"
      )
    
    ## Scenario predictions: observed plus held-constant projections.
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
      
      if (isTRUE(SAVE_SCENARIO_ROWLEVEL)) {
        scenario_rowlevel_list[[length(scenario_rowlevel_list) + 1L]] <- cf_pred_df
      }
      
      probe2_surface_seed_list[[length(probe2_surface_seed_list) + 1L]] <-
        cf_pred_df %>%
        filter(
          race %in% c("NH-White", "NH-Black"),
          !is.na(age_bin),
          !is.na(time_bin)
        ) %>%
        group_by(split_name, split_seed, seed, scenario, age_bin, age_bin_label,
                 time_bin, time_bin_label, race, sex) %>%
        summarise(
          n = n(),
          events = sum(truth01 == 1L, na.rm = TRUE),
          observed_event_rate = mean(truth01 == 1L, na.rm = TRUE),
          mean_predicted_risk = mean(p_death, na.rm = TRUE),
          median_predicted_risk = median(p_death, na.rm = TRUE),
          .groups = "drop"
        )
      
      probe2_metrics_age_time_seed_list[[length(probe2_metrics_age_time_seed_list) + 1L]] <-
        cf_pred_df %>%
        filter(
          race %in% c("NH-White", "NH-Black"),
          !is.na(age_bin),
          !is.na(time_bin)
        ) %>%
        group_by(split_name, split_seed, seed, scenario, age_bin, age_bin_label,
                 time_bin, time_bin_label, race, sex) %>%
        summarise(
          metrics_counts(pred01, truth01),
          .groups = "drop"
        )
    }
    
    rm(model)
    gc()
  }
}

## -----------------------------------------------------------------------------
## WRITE ROW-LEVEL OUTPUTS
## -----------------------------------------------------------------------------

if (isTRUE(SAVE_OBSERVED_ROWLEVEL)) {
  observed_rowlevel <- bind_rows(observed_rowlevel_list)
  write_csv(
    observed_rowlevel,
    file.path(out_dir, "00_rowlevel_observed_predictions_with_time.csv")
  )
}

if (isTRUE(SAVE_SCENARIO_ROWLEVEL)) {
  scenario_rowlevel <- bind_rows(scenario_rowlevel_list)
  write_csv(
    scenario_rowlevel,
    file.path(out_dir, "00_rowlevel_scenario_predictions_with_time.csv")
  )
}

## -----------------------------------------------------------------------------
## OBSERVED AGE x time_ SURFACE SUMMARIES
## -----------------------------------------------------------------------------

observed_surface_seed <- bind_rows(observed_surface_seed_list)
observed_metrics_age_time_seed <- bind_rows(observed_metrics_age_time_seed_list)

if (isTRUE(SAVE_OBSERVED_SURFACE_CSV)) {
  write_csv(
    observed_surface_seed,
    file.path(out_dir, "20a_seed_observed_hazard_surface_age_time_race_sex.csv")
  )
}

if (isTRUE(SAVE_OBSERVED_METRICS_CSV)) {
  write_csv(
    observed_metrics_age_time_seed,
    file.path(out_dir, "20m_seed_observed_metrics_surface_age_time_race_sex.csv")
  )
}

observed_surface_split <- summarise_seed_to_split(
  df = observed_surface_seed,
  group_vars = c("split_name", "split_seed", "age_bin", "age_bin_label",
                 "time_bin", "time_bin_label", "race", "sex"),
  value_vars = c("n", "events", "observed_event_rate",
                 "mean_predicted_risk", "median_predicted_risk")
)

observed_surface_headline <- summarise_split_to_headline(
  df = observed_surface_split,
  group_vars = c("age_bin", "age_bin_label", "time_bin", "time_bin_label", "race", "sex"),
  value_vars = c("n_mean", "events_mean", "observed_event_rate_mean",
                 "mean_predicted_risk_mean", "median_predicted_risk_mean")
)

if (isTRUE(SAVE_OBSERVED_SURFACE_CSV)) {
  write_csv(
    observed_surface_split,
    file.path(out_dir, "20b_split_observed_hazard_surface_age_time_race_sex.csv")
  )

  write_csv(
    observed_surface_headline,
    file.path(out_dir, "20c_headline_observed_hazard_surface_age_time_race_sex.csv")
  )
}

## -----------------------------------------------------------------------------
## FIGURE 2 REPLICATION WITH ADDED time_ DIMENSION
## observed - held constant risk difference by age_bin x time_bin
## -----------------------------------------------------------------------------

probe2_surface_seed <- bind_rows(probe2_surface_seed_list)
probe2_metrics_age_time_seed <- bind_rows(probe2_metrics_age_time_seed_list)

if (isTRUE(SAVE_PROBE2_SEED_CSV)) {
  write_csv(
    probe2_surface_seed,
    file.path(out_dir, "21a_seed_probe2_risk_surface_age_time_race_sex_scenario.csv")
  )
}

if (isTRUE(SAVE_PROBE2_METRICS_CSV)) {
  write_csv(
    probe2_metrics_age_time_seed,
    file.path(out_dir, "21m_seed_probe2_metrics_surface_age_time_race_sex_scenario.csv")
  )
}

probe2_surface_split <- summarise_seed_to_split(
  df = probe2_surface_seed,
  group_vars = c("split_name", "split_seed", "scenario", "age_bin", "age_bin_label",
                 "time_bin", "time_bin_label", "race", "sex"),
  value_vars = c("n", "events", "observed_event_rate",
                 "mean_predicted_risk", "median_predicted_risk")
)

if (isTRUE(SAVE_PROBE2_SPLIT_CSV)) {
  write_csv(
    probe2_surface_split,
    file.path(out_dir, "21b_split_probe2_risk_surface_age_time_race_sex_scenario.csv")
  )
}

obs_probe2_split <- probe2_surface_split %>%
  filter(scenario == "Observed") %>%
  select(
    split_name,
    split_seed,
    age_bin,
    age_bin_label,
    time_bin,
    time_bin_label,
    race,
    sex,
    observed_mean_predicted_risk = mean_predicted_risk_mean,
    observed_median_predicted_risk = median_predicted_risk_mean
  )

probe2_change_split <- probe2_surface_split %>%
  filter(scenario != "Observed") %>%
  left_join(
    obs_probe2_split,
    by = c("split_name", "split_seed", "age_bin", "age_bin_label",
           "time_bin", "time_bin_label", "race", "sex")
  ) %>%
  mutate(
    mean_risk_change_vs_observed =
      mean_predicted_risk_mean - observed_mean_predicted_risk,
    mean_risk_reduction_from_observed =
      observed_mean_predicted_risk - mean_predicted_risk_mean,
    risk_diff_pp = 100 * mean_risk_reduction_from_observed,
    median_risk_change_vs_observed =
      median_predicted_risk_mean - observed_median_predicted_risk,
    median_risk_reduction_from_observed =
      observed_median_predicted_risk - median_predicted_risk_mean,
    median_risk_diff_pp = 100 * median_risk_reduction_from_observed,
    scenario_label = case_when(
      scenario == "AL+Personality=0" ~ "AL + Personality held constant",
      scenario == "AL=0" ~ "AL held constant",
      scenario == "Personality=0" ~ "Personality held constant",
      TRUE ~ scenario
    )
  )

if (isTRUE(SAVE_PROBE2_SPLIT_CSV)) {
  write_csv(
    probe2_change_split,
    file.path(out_dir, "21c_split_probe2_observed_minus_heldconstant_age_time.csv")
  )
}

probe2_change_headline <- summarise_split_to_headline(
  df = probe2_change_split,
  group_vars = c("scenario", "scenario_label", "age_bin", "age_bin_label",
                 "time_bin", "time_bin_label", "race", "sex"),
  value_vars = c("mean_risk_change_vs_observed",
                 "mean_risk_reduction_from_observed",
                 "risk_diff_pp",
                 "median_risk_change_vs_observed",
                 "median_risk_reduction_from_observed",
                 "median_risk_diff_pp")
)

if (isTRUE(SAVE_PROBE2_HEADLINE_CSV)) {
  write_csv(
    probe2_change_headline,
    file.path(out_dir, "21d_headline_probe2_observed_minus_heldconstant_age_time.csv")
  )
}

## -----------------------------------------------------------------------------
## 2D HEATMAPS: SAFER STATIC SURFACE OUTPUTS
## -----------------------------------------------------------------------------

if (isTRUE(SAVE_HEATMAP_PNG)) {
  
  p_obs_heat <- ggplot(
    observed_surface_headline,
    aes(x = age_bin, y = time_bin, fill = 100 * mean_predicted_risk_mean_mean)
  ) +
    geom_tile() +
    facet_grid(sex ~ race) +
    scale_x_continuous(
      breaks = sort(unique(observed_surface_headline$age_bin)),
      labels = sort(unique(observed_surface_headline$age_bin_label))
    ) +
    labs(
      x = "Age",
      y = "time_",
      fill = "Predicted risk\npercentage points",
      title = "Observed predicted mortality-risk surface by age and time_"
    ) +
    theme_minimal(base_size = GRAPH_BASE_SIZE, base_family = GRAPH_FONT_FAMILY) +
    theme(
      text = element_text(family = GRAPH_FONT_FAMILY),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    )
  
  ggsave(
    filename = file.path(plot_dir, "20_observed_hazard_surface_age_time_heatmap.png"),
    plot = p_obs_heat,
    width = 11,
    height = 7,
    dpi = 600
  )
  
  p_probe2_heat <- ggplot(
    probe2_change_headline,
    aes(x = age_bin, y = time_bin, fill = risk_diff_pp_mean)
  ) +
    geom_tile() +
    facet_grid(sex + race ~ scenario_label) +
    scale_x_continuous(
      breaks = sort(unique(probe2_change_headline$age_bin)),
      labels = sort(unique(probe2_change_headline$age_bin_label))
    ) +
    labs(
      x = "Age",
      y = "time_",
      fill = "Observed - held constant\npercentage points",
      title = "Figure 2 extension: predicted mortality-risk difference by age and time_"
    ) +
    theme_minimal(base_size = GRAPH_BASE_SIZE - 2, base_family = GRAPH_FONT_FAMILY) +
    theme(
      text = element_text(family = GRAPH_FONT_FAMILY),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    )
  
  ggsave(
    filename = file.path(plot_dir, "21_probe2_observed_minus_heldconstant_age_time_heatmap.png"),
    plot = p_probe2_heat,
    width = 15,
    height = 9,
    dpi = 600
  )
}

## -----------------------------------------------------------------------------
## TEXT DIAGNOSTIC: POSITIVE DISPLACEMENT HOTSPOTS
## -----------------------------------------------------------------------------

## This is a small descriptive add-on. It does not create plots and does not
## write row-level files. It describes where the positive Figure-2 displacement
## is concentrated over age_bin x time_bin.
##
## risk_diff_pp_mean > 0 means:
##   observed predicted risk > held-constant predicted risk
## for the selected probe scenario.

hotspot_source <- probe2_change_headline %>%
  filter(
    scenario_label %in% PLOT_PROBE2_SCENARIOS,
    race %in% PLOT_RACES,
    sex %in% PLOT_SEXES
  )

if (isTRUE(SAVE_HOTSPOT_SUMMARY_CSV) ||
    isTRUE(SAVE_HOTSPOT_TEXT_CSV) ||
    isTRUE(SAVE_HOTSPOT_TOP_CELLS_CSV) ||
    isTRUE(SAVE_HOTSPOT_PARAGRAPH_TXT)) {

  hotspot_summary <- hotspot_source %>%
    filter(!is.na(risk_diff_pp_mean)) %>%
    group_by(scenario_label, race, sex) %>%
    group_modify(~{

      dat <- .x
      pos_dat <- dat %>% filter(risk_diff_pp_mean > 0)

      if (nrow(pos_dat) == 0L) {
        return(tibble(
          peak_age_bin = NA_character_,
          peak_time_bin = NA_real_,
          peak_risk_diff_pp = NA_real_,
          hotspot_cutoff_pp = NA_real_,
          hotspot_age_range = NA_character_,
          hotspot_time_range = NA_character_,
          hotspot_mean_risk_diff_pp = NA_real_,
          hotspot_median_risk_diff_pp = NA_real_,
          positive_cells = 0L,
          total_cells = nrow(dat),
          positive_cell_share = 0
        ))
      }

      cutoff <- quantile(pos_dat$risk_diff_pp_mean,
                         probs = HOTSPOT_QUANTILE,
                         na.rm = TRUE)

      hotspot <- pos_dat %>% filter(risk_diff_pp_mean >= cutoff)
      peak <- pos_dat %>% slice_max(risk_diff_pp_mean, n = 1, with_ties = FALSE)

      tibble(
        peak_age_bin = as.character(peak$age_bin_label),
        peak_time_bin = peak$time_bin,
        peak_risk_diff_pp = peak$risk_diff_pp_mean,
        hotspot_cutoff_pp = as.numeric(cutoff),
        hotspot_age_range = paste0(
          min(hotspot$age_bin_label, na.rm = TRUE),
          " to ",
          max(hotspot$age_bin_label, na.rm = TRUE)
        ),
        hotspot_time_range = paste0(
          min(hotspot$time_bin, na.rm = TRUE),
          " to ",
          max(hotspot$time_bin, na.rm = TRUE)
        ),
        hotspot_mean_risk_diff_pp = mean(hotspot$risk_diff_pp_mean, na.rm = TRUE),
        hotspot_median_risk_diff_pp = median(hotspot$risk_diff_pp_mean, na.rm = TRUE),
        positive_cells = nrow(pos_dat),
        total_cells = nrow(dat),
        positive_cell_share = nrow(pos_dat) / nrow(dat)
      )
    }) %>%
    ungroup() %>%
    arrange(scenario_label, race, sex)

  hotspot_top_cells <- hotspot_source %>%
    filter(!is.na(risk_diff_pp_mean), risk_diff_pp_mean > 0) %>%
    group_by(scenario_label, race, sex) %>%
    arrange(desc(risk_diff_pp_mean), .by_group = TRUE) %>%
    slice_head(n = HOTSPOT_TOP_N_CELLS) %>%
    ungroup() %>%
    transmute(
      scenario_label,
      race,
      sex,
      age_bin,
      age_bin_label,
      time_bin,
      time_bin_label,
      risk_diff_pp_mean,
      risk_diff_pp_sd,
      risk_diff_pp_min,
      risk_diff_pp_max
    )

  if (isTRUE(SAVE_HOTSPOT_SUMMARY_CSV)) {
    write_csv(
      hotspot_summary,
      file.path(out_dir, "22_probe2_positive_displacement_hotspot_summary.csv")
    )
  }

  if (isTRUE(SAVE_HOTSPOT_TOP_CELLS_CSV)) {
    write_csv(
      hotspot_top_cells,
      file.path(out_dir, "22b_probe2_positive_displacement_top_cells.csv")
    )
  }

  hotspot_text <- hotspot_summary %>%
    mutate(
      sentence = case_when(
        is.na(peak_risk_diff_pp) ~ paste0(
          "For ", race, " ", sex,
          ", there were no positive observed-minus-held-constant cells for ",
          scenario_label, "."
        ),
        TRUE ~ paste0(
          "For ", race, " ", sex,
          ", the strongest positive observed-minus-held-constant displacement under ",
          scenario_label,
          " occurred around age ", peak_age_bin,
          " at time_ interval ", peak_time_bin,
          " (", round(peak_risk_diff_pp, 3),
          " percentage points). The upper-tail positive displacement region was concentrated across ages ",
          hotspot_age_range,
          " and time_ intervals ", hotspot_time_range,
          ", with an average displacement of ",
          round(hotspot_mean_risk_diff_pp, 3),
          " percentage points."
        )
      )
    ) %>%
    select(scenario_label, race, sex, sentence)

  if (isTRUE(SAVE_HOTSPOT_TEXT_CSV)) {
    write_csv(
      hotspot_text,
      file.path(out_dir, "23_probe2_positive_displacement_hotspot_text.csv")
    )
  }

  if (isTRUE(SAVE_HOTSPOT_PARAGRAPH_TXT)) {
    writeLines(
      paste(hotspot_text$sentence, collapse = "\n\n"),
      con = file.path(out_dir, "23_probe2_positive_displacement_hotspot_text.txt")
    )
  }

  cat("\nPositive displacement hotspot summaries written to:\n")
  if (isTRUE(SAVE_HOTSPOT_SUMMARY_CSV)) {
    cat("  ", file.path(out_dir, "22_probe2_positive_displacement_hotspot_summary.csv"), "\n")
  }
  if (isTRUE(SAVE_HOTSPOT_TOP_CELLS_CSV)) {
    cat("  ", file.path(out_dir, "22b_probe2_positive_displacement_top_cells.csv"), "\n")
  }
  if (isTRUE(SAVE_HOTSPOT_TEXT_CSV)) {
    cat("  ", file.path(out_dir, "23_probe2_positive_displacement_hotspot_text.csv"), "\n")
  }
  if (isTRUE(SAVE_HOTSPOT_PARAGRAPH_TXT)) {
    cat("  ", file.path(out_dir, "23_probe2_positive_displacement_hotspot_text.txt"), "\n")
  }
}

## -----------------------------------------------------------------------------
## 3D PLOTLY HTML SURFACES   - You can run again stand alone after loop is done 
## -----------------------------------------------------------------------------

if (isTRUE(SAVE_OBSERVED_3D_HTML)) {

  obs_groups <- observed_surface_headline %>%
    distinct(race, sex) %>%
    filter(race %in% PLOT_RACES, sex %in% PLOT_SEXES)

  for (ii in seq_len(nrow(obs_groups))) {
    rr <- obs_groups$race[ii]
    ss <- obs_groups$sex[ii]

    df_g <- observed_surface_headline %>%
      filter(race == rr, sex == ss) %>%
      mutate(predicted_risk_pp = 100 * mean_predicted_risk_mean_mean)

    safe_name <- paste0(
      "observed_3d_",
      safe_plot_stub(rr), "_",
      safe_plot_stub(ss),
      ".html"
    )

    save_surface3d(
      df = df_g,
      z_var = "predicted_risk_pp",
      title = paste(OBSERVED_TITLE_PREFIX, rr, ss, sep = " | "),
      file_path = file.path(plot_dir, safe_name),
      z_title = AXIS_TITLE_Z_OBSERVED,
      z_range = NULL
    )
  }
}

if (isTRUE(SAVE_PROBE2_3D_HTML)) {

  probe2_for_plot <- probe2_change_headline %>%
    filter(
      scenario_label %in% PLOT_PROBE2_SCENARIOS,
      race %in% PLOT_RACES,
      sex %in% PLOT_SEXES
    )

  z_rng_probe2 <- NULL
  if (isTRUE(LOCK_Z_AXIS_ACROSS_PROBE2_PLOTS)) {
    z_rng_probe2 <- range(probe2_for_plot$risk_diff_pp_mean, na.rm = TRUE)
  }

  probe_groups <- probe2_for_plot %>%
    distinct(scenario_label, race, sex) %>%
    arrange(scenario_label, race, sex)

  for (ii in seq_len(nrow(probe_groups))) {
    sc <- probe_groups$scenario_label[ii]
    rr <- probe_groups$race[ii]
    ss <- probe_groups$sex[ii]

    df_g <- probe2_for_plot %>%
      filter(scenario_label == sc, race == rr, sex == ss) %>%
      mutate(risk_diff_pp_surface = risk_diff_pp_mean)

    safe_name <- paste0(
      "probe2_3d_",
      safe_plot_stub(rr), "_",
      safe_plot_stub(ss),
      ".html"
    )

    save_surface3d(
      df = df_g,
      z_var = "risk_diff_pp_surface",
      title = paste(PROBE2_TITLE_PREFIX, sc, rr, ss, sep = " | "),
      file_path = file.path(plot_dir, safe_name),
      z_title = AXIS_TITLE_Z_PROBE2,
      z_range = z_rng_probe2
    )
  }
}

## -----------------------------------------------------------------------------
## DONE
## -----------------------------------------------------------------------------

cat("\nDONE. Time-surface outputs written to:\n", out_dir, "\n")
cat("Plots written to:\n", plot_dir, "\n")


## -----------------------------------------------------------------------------
## OPTIONAL CONSOLE-ONLY HOTSPOT DESCRIPTION
## Positive and negative concentration.
## No files written. Nothing saved. Prints only.
## -----------------------------------------------------------------------------

RUN_HOTSPOT_DESCRIPTION <- TRUE

HOTSPOT_SCENARIO_LABEL <- "AL + Personality held constant"
HOTSPOT_QUANTILE <- 0.90

if (isTRUE(RUN_HOTSPOT_DESCRIPTION)) {
  
  hotspot_surface <- probe2_change_headline %>%
    filter(
      scenario_label == HOTSPOT_SCENARIO_LABEL,
      race %in% c("NH-Black", "NH-White"),
      sex %in% c("Female", "Male"),
      !is.na(risk_diff_pp_mean)
    )
  
  summarize_displacement_tail <- function(dat, direction, q = 0.90) {
    
    if (direction == "positive") {
      
      tail_dat <- dat %>% filter(risk_diff_pp_mean > 0)
      
      if (nrow(tail_dat) == 0L) {
        return(tibble(
          displacement_direction = "positive",
          peak_age = NA_character_,
          peak_time = NA_real_,
          peak_risk_diff_pp = NA_real_,
          hotspot_cutoff_pp = NA_real_,
          hotspot_age_min = NA_real_,
          hotspot_age_max = NA_real_,
          hotspot_time_min = NA_real_,
          hotspot_time_max = NA_real_,
          hotspot_mean_pp = NA_real_,
          tail_cells = 0L,
          total_cells = nrow(dat),
          tail_cell_share = 0
        ))
      }
      
      cutoff <- quantile(
        tail_dat$risk_diff_pp_mean,
        probs = q,
        na.rm = TRUE
      )
      
      hotspot <- tail_dat %>%
        filter(risk_diff_pp_mean >= cutoff)
      
      peak <- tail_dat %>%
        slice_max(risk_diff_pp_mean, n = 1, with_ties = FALSE)
      
    } else if (direction == "negative") {
      
      tail_dat <- dat %>% filter(risk_diff_pp_mean < 0)
      
      if (nrow(tail_dat) == 0L) {
        return(tibble(
          displacement_direction = "negative",
          peak_age = NA_character_,
          peak_time = NA_real_,
          peak_risk_diff_pp = NA_real_,
          hotspot_cutoff_pp = NA_real_,
          hotspot_age_min = NA_real_,
          hotspot_age_max = NA_real_,
          hotspot_time_min = NA_real_,
          hotspot_time_max = NA_real_,
          hotspot_mean_pp = NA_real_,
          tail_cells = 0L,
          total_cells = nrow(dat),
          tail_cell_share = 0
        ))
      }
      
      cutoff <- quantile(
        tail_dat$risk_diff_pp_mean,
        probs = 1 - q,
        na.rm = TRUE
      )
      
      hotspot <- tail_dat %>%
        filter(risk_diff_pp_mean <= cutoff)
      
      peak <- tail_dat %>%
        slice_min(risk_diff_pp_mean, n = 1, with_ties = FALSE)
      
    } else {
      stop("direction must be 'positive' or 'negative'")
    }
    
    tibble(
      displacement_direction = direction,
      peak_age = as.character(peak$age_bin_label),
      peak_time = peak$time_bin,
      peak_risk_diff_pp = peak$risk_diff_pp_mean,
      hotspot_cutoff_pp = as.numeric(cutoff),
      hotspot_age_min = min(hotspot$age_bin, na.rm = TRUE),
      hotspot_age_max = max(hotspot$age_bin, na.rm = TRUE),
      hotspot_time_min = min(hotspot$time_bin, na.rm = TRUE),
      hotspot_time_max = max(hotspot$time_bin, na.rm = TRUE),
      hotspot_mean_pp = mean(hotspot$risk_diff_pp_mean, na.rm = TRUE),
      tail_cells = nrow(tail_dat),
      total_cells = nrow(dat),
      tail_cell_share = nrow(tail_dat) / nrow(dat)
    )
  }
  
  hotspot_summary <- hotspot_surface %>%
    group_by(race, sex) %>%
    group_modify(~ bind_rows(
      summarize_displacement_tail(.x, "positive", HOTSPOT_QUANTILE),
      summarize_displacement_tail(.x, "negative", HOTSPOT_QUANTILE)
    )) %>%
    ungroup()
  
  hotspot_text <- hotspot_summary %>%
    mutate(
      sentence = case_when(
        
        tail_cells == 0 & displacement_direction == "positive" ~ paste0(
          "For ", race, " ", sex,
          ", there were no positive observed-minus-held-constant cells in the age-by-time surface."
        ),
        
        tail_cells == 0 & displacement_direction == "negative" ~ paste0(
          "For ", race, " ", sex,
          ", there were no negative observed-minus-held-constant cells in the age-by-time surface."
        ),
        
        displacement_direction == "positive" ~ paste0(
          "For ", race, " ", sex,
          ", the strongest positive observed-minus-held-constant displacement occurred around age ",
          peak_age,
          " at time_ interval ",
          peak_time,
          " (",
          round(peak_risk_diff_pp, 3),
          " percentage points). The upper-decile positive displacement region was concentrated across ages ",
          hotspot_age_min,
          " to ",
          hotspot_age_max,
          " and time_ intervals ",
          hotspot_time_min,
          " to ",
          hotspot_time_max,
          ", with an average displacement of ",
          round(hotspot_mean_pp, 3),
          " percentage points."
        ),
        
        displacement_direction == "negative" ~ paste0(
          "For ", race, " ", sex,
          ", the strongest negative observed-minus-held-constant displacement occurred around age ",
          peak_age,
          " at time_ interval ",
          peak_time,
          " (",
          round(peak_risk_diff_pp, 3),
          " percentage points). The lower-decile negative displacement region was concentrated across ages ",
          hotspot_age_min,
          " to ",
          hotspot_age_max,
          " and time_ intervals ",
          hotspot_time_min,
          " to ",
          hotspot_time_max,
          ", with an average displacement of ",
          round(hotspot_mean_pp, 3),
          " percentage points."
        )
      )
    )
  
  cat("\n\nHOTSPOT DESCRIPTION — CONSOLE ONLY\n")
  cat("Scenario:", HOTSPOT_SCENARIO_LABEL, "\n\n")
  cat(paste(hotspot_text$sentence, collapse = "\n\n"))
  cat("\n\n")
}