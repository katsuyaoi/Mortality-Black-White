README — Supplemental Materials
Supplemental data.zip will contain three data sets required to run an R program described below.

Supplemental_base_gpu_FINAL
Neural network–based discrete-time hazard model using baseline covariates, estimated across 100 random seeds with optional GPU acceleration.
1.	Place hrs_survival.csv and hrs_al.csv in the same directory.
2.	Edit the setwd() line at the top of the script to point to that directory.
This program will use a graphics processing unit (GPU) if one is available, which can substantially reduce training time. Running GPU-accelerated torch in R typically requires CUDA. As of February 2026, CUDA-based acceleration is supported only on NVIDIA GPUs with the appropriate drivers and CUDA Toolkit installed. CUDA Toolkit downloads are available at: https://developer.nvidia.com/cuda-downloads
For transparency and reproducibility, all hyperparameters are explicitly listed in the script. These parameters define the model architecture and training configuration and may be modified for replication, robustness checks, or adaptation to other research aims. In contrast to common machine-learning practice, hyperparameters are not tuned to maximize predictive performance. Instead, the model uses a fixed, documented configuration to support abductive benchmarking, establishing a stable comparative baseline from which substantive patterns can be evaluated.
Results from 100 random seeds are saved to a dedicated subdirectory via an internal checkpoint function. These seed-level outputs can be reused for post-estimation analyses, robustness checks, and diagnostic evaluation without retraining the model.

Supplemental_full_gpu_FINAL
Neural network–based discrete-time hazard model incorporating physiological and psychometric covariates, estimated across 100 random seeds with optional GPU acceleration.
1.	Place hrs_survival.csv and hrs_al.csv in the same directory.
2.	Edit the setwd() line at the top of the script to point to that directory.
GPU usage, CUDA requirements, hyperparameter documentation, abductive benchmarking logic, and checkpointing behavior are identical to Supplemental_base_gpu_FINAL. Results are saved at the seed level to support reproducibility and post-estimation analyses.

Supplemental_survival_FINAL
Conventional discrete-time survival analysis using multilevel logistic regression on long-format person-period data.
1.	Place hrs_survival_post.csv in the same directory.
2.	Edit the setwd() line at the top of the script to point to that directory.
This file differs from hrs_survival.csv in that survival intervals are stored in long format, enabling multilevel logistic regression. Because computational demands are relatively modest, this program does not implement a checkpointing function.
**Data preprocessing & encoding**
| Component        | Specification                                                           |
| ---------------- | ----------------------------------------------------------------------- |
| Outcome          | `died_` encoded as binary event via `factor(levels = c(0,1))`           |
| Age transforms   | `age_`, `age_1`, `age_3`, `age_5` log-transformed (`log(age_*)`)        |
| Raw age          | `age_raw`, `age_raw_1`, `age_raw_3`, `age_raw_5` retained untransformed |
| Standardization  | Z-score scaling applied to all **non-binary, non-age** variables        |
| Binary detection | Columns restricted to values in `{0,1}`                                 |
| Age exclusions   | All `age_*` and `age_raw_*` variables excluded from scaling             |
| Missing values   | Post-scaling NAs set to zero (`X_scaled[is.na(X_scaled)] <- 0`)         |
| Missingness mask | `na_mask_mat`: 1 if NA, 0 otherwise                                     |
| Mask usage       | Values and masks concatenated at each step: `(x_seq ∥ m_seq)`           |
**Time-varying (TV) structure**
| Component        | Specification                                                                    |
| ---------------- | -------------------------------------------------------------------------------- |
| Intended waves   | `waves <- 1:15`                                                                  |
| Retained waves   | Non-empty waves only                                                             |
| Constraint       | Equal per-step feature counts enforced                                           |
| TV feature bases | `sayret`, `mwid`, `cesd`, `shlt`, `cancre`, `diabe`, `hearte`, `mobila`, `adl5a` |
| Steps            | `n_steps` determined at runtime                                                  |
| Step feature dim | `tv_step_feat_dim`                                                               |
| Step input       | `2 × tv_step_feat_dim` (values + mask)                                           |
**Train/test split**
| Component    | Specification                                 |
| ------------ | --------------------------------------------- |
| Split method | Stratified 80/20 split                        |
| Function     | `caret::createDataPartition(y_idx, p = 0.80)` |
| Unit         | Row-level (not ID-grouped)                    |
**Model architecture**
| Component        | Specification                            |
| ---------------- | ---------------------------------------- |
| Model class      | Multi-component feedforward MLP          |
| Subnetworks      | Static branch + TV branch with attention |
| Mixing           | Concatenation + dense layer              |
| Output           | Two-class logits                         |
| Random intercept | ID embedding added to death logit        |
**Static branch**
| Parameter   | Value           |
| ----------- | --------------- |
| Layers      | 2               |
| Hidden size | `h_static = 64` |
| Activation  | ReLU            |
| Dropout     | `0.25`          |
**Time-varying branch**
| Parameter    | Value                                  |
| ------------ | -------------------------------------- |
| Per-step MLP | 2 layers                               |
| Hidden size  | `h_tv = 64`                            |
| Activation   | ReLU                                   |
| Dropout      | `0.25`                                 |
| Attention    | Softmax-normalized weights across time |
| Output       | Weighted sum over steps                |
**Mixing & output**
| Component     | Specification                               |
| ------------- | ------------------------------------------- |
| Concatenation | `[h_static ∥ h_tv]`                         |
| Mixing layer  | `mix_dim = 16`, ReLU                        |
| Output head   | `mix_dim → 2` logits                        |
| ID embedding  | Dimension = 1                               |
| ID scale      | Learnable scalar `alpha_id` (initialized 0) |
| Application   | Additive shift to death logit (class 1)     |
**Training & optimization**
| Component      | Specification                                |
| -------------- | -------------------------------------------- |
| Seeds          | `1:100`                                      |
| Seeding        | `torch_manual_seed(seed)` + `set.seed(seed)` |
| Epochs         | 100                                          |
| Batch size     | Train = 1024, Test = 1024                    |
| Optimizer      | Adam                                         |
| Learning rate  | `1e-3`                                       |
| Weight decay   | `1e-3`                                       |
| Loss           | Weighted cross-entropy                       |
| Class weights  | Inverse frequency, normalized to mean 1      |
| Classification | Argmax over logits                           |
| Checkpointing  | `checkpoints_base_observed/seed_###.pt`      |
| Device         | CUDA if available                            |





