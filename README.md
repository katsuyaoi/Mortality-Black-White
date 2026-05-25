# README — Supplemental Materials

This repository contains supplemental materials for the manuscript:

**Comparing Surface-Based and Regression-Based Representations of the Black–White Mortality Hazard Gap**

The supplemental files provide the R scripts and input-data structure needed to reproduce the regression-based hazard benchmarks, train the full neural-network hazard model, and generate post-estimation representational probe tables and figures.

## Files included
All data files are included in the zip.file titled 'supplemental data.' Unzip and place them your working directory.

### `Supplemental_full_regression_FINAL.R`

Conventional regression-based discrete-time hazard models using sex-stratified random-intercept logistic regression.

This script estimates three regression specifications:

1. Logged-age model  
2. Three-degree-of-freedom natural spline-age model  
3. Race-specific spline-age model allowing the spline-age function to vary for non-Hispanic Black respondents  

The script also generates fixed-effect predicted mortality curves by age, model-fit statistics, coefficient tables, and classification-performance summaries under both conventional and prevalence-calibrated thresholds.

Required input file:

- `hrs_survival_post.csv`

Place `hrs_survival_post.csv` in the working directory and edit the `setwd("yourdatalocation")` line at the top of the script.

Main output folder:

- `regression_hazard_model_results`

Key outputs include:

- `coefficient_table_all_models.csv`
- `model_fit_table_mortality_models.csv`
- `mortality_model_comparison_curves_age50_90_by5_fixed_effect_ci.csv`
- `mortality_model_comparison_curves_age50_90_by5_fixed_effect_ci.png`  Figure 1 
- `eval_threshold_free_full_data.csv`
- `eval_thresholded_full_data.csv`
- `eval_threshold_free_full_data_by_race_sex.csv`
- `eval_thresholded_full_data_by_race_sex.csv`
- `eval_subgroup_thresholded_full_data.csv`
- `regression_model_objects_all.rds`
- `regression_hazard_model_workspace.RData`

The regression models are estimated on the full analytic person-interval file. Classification results from these models are used as regression-based benchmarks rather than held-out neural-network evaluations.

---

### `Supplemental_full_gpu_FINAL.R`

Full neural-network discrete-time hazard model incorporating demographic, temporal, reported-health, Allostatic Load, and Personality data branches.

This script trains the full neural-network hazard model across:

- 10 respondent-level train/test partitions
- 20 training seeds per partition
- 200 total trained models

Respondents, rather than person-interval rows, are assigned to training and test sets to prevent within-person leakage across partitions.

Required input files:

- `hrs_survival.csv`
- `hrs_al.csv`

Place both files in the same working directory and edit the `setwd("yourdatalocation")` line at the top of the script.

Main output folder:

- `checkpoints_fullspec_IDsplit_multisplit`

Key outputs include:

- Split-specific checkpoint folders
- `split_meta/`
- `fixed_split_list.rds`
- `fixed_split_registry.csv`
- `fixed_split_registry.rds`
- `master_checkpoint_registry.csv`
- `master_checkpoint_registry.rds`
- `master_group_test_metrics.csv`
- `master_group_test_metrics.rds`
- `summary_by_split.csv`
- `group_summary_by_split.csv`
- `TRAINING_fullspec_IDsplit_multisplit_10x20.RData`

The script is GPU-ready. It uses CUDA if available through R `torch`; otherwise, it runs on CPU. GPU acceleration requires a compatible NVIDIA GPU, appropriate drivers, and CUDA support for R `torch`.

The script does not perform counterfactual or held-constant inference. It trains the full model, saves seed-level checkpoints, and stores split-level metadata for post-estimation evaluation.

---

### `Supplemental_full_post_graphing_Final.R`

Post-estimation evaluation and graphing script for the trained full neural-network hazard model.

This script should be run only after `Supplemental_full_gpu_FINAL.R` has successfully completed and the checkpoint directory has been created.

Required existing folder:

- `checkpoints_fullspec_IDsplit_multisplit`

Required existing files inside that folder include:

- `TRAINING_fullspec_IDsplit_multisplit_10x20.RData`
- split-level training state files in `split_meta/`
- seed-level model checkpoints inside the split folders

Main output folder:

- `checkpoints_fullspec_IDsplit_multisplit/post_eval_tables_full_only_compact`

This script produces compact post-estimation summaries and does not save row-level prediction files.

Main outputs include:

- Race × sex performance summaries
- Age-bin × race × sex performance summaries
- Observed predicted-risk summaries
- Black–White risk gaps and risk ratios
- Inference-time held-constant summaries
- Scenario-vs-observed risk and performance changes
- Scenario-vs-observed Black–White gap and risk-ratio changes
- Representational Probe 2 graph and table

Key output files include:

- `00_age_bin_config.csv`
- `01a_seed_metrics_race_sex.csv`
- `01c_headline_metrics_race_sex_across_splits.csv`
- `02c_headline_metrics_agebin_race_sex_across_splits.csv`
- `03c_headline_observed_risk_agebin_race_sex_across_splits.csv`
- `04b_headline_observed_risk_gap_rr_agebin_sex.csv`
- `05e_headline_counterfactual_risk_change_vs_observed_agebin_race_sex.csv`
- `10b_headline_counterfactual_metrics_race_sex.csv`
- `13_manuscript_full_model_race_sex_summary.csv`
- `14_manuscript_counterfactual_predictive_ablation_table.csv`
- `15_manuscript_counterfactual_scenario_risk_table.csv`
- `16_plot_ready_observed_agebin_gap_rr.csv`
- `17_plot_ready_counterfactual_agebin_gap_rr_change.csv`
- `18_probe2_agebin_observed_minus_heldconstant_risk_difference.csv` 
- `18_probe2_agebin_observed_minus_heldconstant_risk_difference.png`   Figure 2

The post-estimation scenarios are:

1. Observed  
2. `AL = 0`  
3. `Personality = 0`  
4. `AL + Personality = 0`  

These are inference-time held-constant probes referenced in the manuscript


## Supplemental files

| File | Purpose | Required input | Main output |
|---|---|---|---|
| `Supplemental_full_regression_FINAL.R` | Estimates regression-based discrete-time hazard benchmarks using sex-stratified random-intercept logistic models | `hrs_survival_post.csv` | `regression_hazard_model_results/` |
| `Supplemental_full_gpu_FINAL.R` | Trains the full neural-network hazard model with demographic, temporal, reported-health, AL, and Personality branches | `hrs_survival.csv`, `hrs_al.csv` | `checkpoints_fullspec_IDsplit_multisplit/` |
| `Supplemental_full_post_graphing_Final.R` | Loads trained full-model checkpoints and generates post-estimation tables, probes, and figures | Completed checkpoint folder from full NN training | `post_eval_tables_full_only_compact/` |

## Recommended run order

| Step | Script | Description |
|---|---|---|
| 1 | `Supplemental_full_regression_FINAL.R` | Fit regression hazard benchmarks and generate fitted age-specific mortality curves |
| 2 | `Supplemental_full_gpu_FINAL.R` | Train full NN hazard models across respondent-level splits and seeds |
| 3 | `Supplemental_full_post_graphing_Final.R` | Generate race–sex performance tables, held-constant probes, and age-bin figures |

## Regression benchmark specifications

| Component | Specification |
|---|---|
| Model type | Sex-stratified random-intercept logistic discrete-time hazard model |
| Outcome | Death in the subsequent interval, `died_` |
| Age model 1 | Logged age |
| Age model 2 | Three-degree-of-freedom natural spline for age |
| Age model 3 | Three-degree-of-freedom natural spline interacted with NH-Black status |
| Random effect | Respondent-level random intercept |
| Evaluation | Full analytic person-interval sample |
| Thresholds | Conventional `0.50` and race–sex-specific prevalence-calibrated threshold |
| Main output folder | `regression_hazard_model_results/` |

## Full neural-network preprocessing

| Component | Specification |
|---|---|
| Outcome | `died_` binary event indicator |
| Hazard age | `age_` log-transformed; `age_raw` retained |
| AL ages | `age_1`, `age_3`, `age_5` log-transformed; raw AL ages retained |
| Personality ages | `agep_1`, `agep_3`, `agep_5` log-transformed; raw personality ages retained |
| Scaling | Z-score scaling for non-binary, non-age variables |
| Missing values | Set to `0` after scaling |
| Missingness mask | Binary NA indicator matrix |
| Mask usage | Values and masks are passed jointly to model branches |
| Person IDs | Encoded for ID embedding/random-intercept component |

## Full neural-network data branches

| Branch | Information represented | Structure |
|---|---|---|
| Static branch | Demographic, education, entry-period, and baseline covariates | Feedforward dense layers |
| Time-varying branch | Reported health and interval-varying measures across up to 15 intervals | Shared per-step MLP + attention |
| AL branch | Biomarkers and anthropometric measures across three waves | Shared wave-level MLP + attention |
| Personality branch | Big Five, control, self-efficacy, optimism across three waves | Shared wave-level MLP + attention |
| ID component | Respondent-specific unobserved heterogeneity | One-dimensional ID embedding added to death logit |

## Full neural-network training design

| Component | Specification |
|---|---|
| Split design | Respondent-level train/test partitions |
| Number of splits | 10 |
| Seeds per split | 20 |
| Total trained models | 200 |
| Train/test ratio | 80/20 |
| Leakage prevention | Respondents, not person-interval rows, assigned to partitions |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-3` |
| Batch size | `1024` |
| Dropout | `0.25` |
| Branch hidden size | `64` |
| Mixing dimension | `32` |
| Classification rule | Argmax over two-class logits |
| Device | CUDA/GPU if available; CPU otherwise |

## Full neural-network saved objects

| Output | Description |
|---|---|
| `checkpoints_fullspec_IDsplit_multisplit/` | Root checkpoint directory |
| `split_meta/` | Split-level training states and metadata |
| `seed_###.pt` | Seed-specific trained model checkpoint within each split folder |
| `master_checkpoint_registry.csv` | Registry of trained checkpoints across splits and seeds |
| `master_group_test_metrics.csv` | Group-level test metrics from training |
| `TRAINING_fullspec_IDsplit_multisplit_10x20.RData` | Global training image used by post-estimation scripts |

## Post-estimation representational probes

| Probe | Description | Main quantities |
|---|---|---|
| Probe 1 | Compares classification under observed profiles and held-constant AL/Personality profiles | TP, FN, accuracy, sensitivity, specificity, precision |
| Probe 2 | Estimates age-specific observed minus held-constant predicted mortality risk | Percentage-point risk difference by age bin, race, sex, and scenario |

## Held-constant scenarios

| Scenario | Description |
|---|---|
| `Observed` | Full model predictions using observed inputs |
| `AL = 0` | Standardized AL inputs set to `0`; trained parameters fixed |
| `Personality = 0` | Standardized Personality inputs set to `0`; trained parameters fixed |
| `AL + Personality = 0` | Both AL and Personality inputs set to `0`; trained parameters fixed |

## Interpretation of held-constant probes

| Point | Interpretation |
|---|---|
| What `0` means | The analytic-sample mean on the standardized scale |
| What is held fixed | All trained model parameters |
| What changes | Selected standardized input domains and their missingness indicators |
| What the probes estimate | Changes in prediction and classification under withheld information |
| What the probes do not estimate | Causal effects of AL, Personality, biomarkers, or psychometric traits |

## Key post-estimation outputs

| File | Description |
|---|---|
| `01c_headline_metrics_race_sex_across_splits.csv` | Race–sex classification performance across held-out partitions |
| `02c_headline_metrics_agebin_race_sex_across_splits.csv` | Age-bin race–sex classification performance |
| `03c_headline_observed_risk_agebin_race_sex_across_splits.csv` | Observed predicted mortality risk by age bin, race, and sex |
| `10b_headline_counterfactual_metrics_race_sex.csv` | Held-constant classification performance by race and sex |
| `13_manuscript_full_model_race_sex_summary.csv` | Manuscript-ready full-model race–sex summary |
| `14_manuscript_counterfactual_predictive_ablation_table.csv` | Manuscript-ready Probe 1 table |
| `18_probe2_agebin_observed_minus_heldconstant_risk_difference.csv` | Probe 2 age-specific risk-difference table |
| `18_probe2_agebin_observed_minus_heldconstant_risk_difference.png` | Probe 2 manuscript figure |

## Software requirements

| Package | Used for |
|---|---|
| `torch` | Neural-network model training and inference |
| `coro` | Torch training loop support |
| `caret` | Train/test partitioning |
| `dplyr` | Data wrangling |
| `tidyr` | Data restructuring |
| `readr` | CSV output |
| `ggplot2` | Figure generation |
| `lme4` | Random-intercept logistic regression |
| `splines` | Natural spline age models |
| `pROC` | ROC/AUC evaluation |
| `PRROC` | Precision-recall evaluation |
