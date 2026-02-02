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


Hyperparameters For Baseline Model Supplemental_base_gpu_FINAL
A) Data preprocessing + encoding choices
Outcome definition
o	died_ as binary event (classes = 0, 1) via factor(levels=c(0,1))
Age transforms
o	age_, age_1, age_3, age_5 log-transformed (log(age_*))
o	age_raw, age_raw_1, age_raw_3, age_raw_5 retained untransformed (included as predictors)
Standardization
o	Z-score scaling applied to non-binary, non-age columns only:
	Binary columns detected as values in {0,1}
	“Age columns” excluded from scaling:
	age_, age_1, age_3, age_5, age_raw, age_raw_1, age_raw_3, age_raw_5
o	Missing values after scaling are set to 0 (X_scaled[is.na(X_scaled)] <- 0)
Missingness encoding
o	Missingness mask na_mask_mat: 1 if NA else 0
o	For the time-varying sequence, the model concatenates values and masks at each step: (x_seq ∥ m_seq).
Time-varying structure (TV branch)
o	Waves/steps: 15 intended (waves <- 1:15), but n_steps is set to the number of non-empty waves actually found.
o	TV feature bases: sayret, mwid, cesd, shlt, cancre, diabe, hearte, mobila, adl5a
o	Constraint enforced: all retained TV waves must have equal per-step feature counts (tv_step_feat_dim)
Train/test split
o	Stratified 80/20 split using caret::createDataPartition(y_idx, p=0.80)
o	Row-level split (explicitly not ID-level grouping)
B) Representation capacity + architecture choices
Model class
o	Feedforward MLP with:
	Static subnetwork
	Time-varying (sequence) subnetwork with attention pooling
	Mixing layer + output head
	ID embedding random intercept applied to death logit
Static branch
o	Two-layer MLP:
	fc_static1: input = length(static_idx) → h_static
	fc_static2: h_static → h_static
o	Hidden size: h_static = 64
o	Activation: ReLU
o	Dropout: drop_p = 0.25 applied between layers
Time-varying branch (TV)
o	Input per time step is concatenated: values + mask
	Step input dim = 2 * tv_step_feat_dim
o	Two-layer MLP per step:
	fc_tv1: 2F → h_tv
	fc_tv2: h_tv → h_tv
o	Hidden size: h_tv = 64
o	Activation: ReLU
o	Dropout: drop_p = 0.25 applied within TV pathway
o	Attention pooling
	att_tv: h_tv → 1 per time step
	Softmax over time steps to get weights α
	Aggregation: weighted sum to produce h_tv summary vector
Mixing + output head
o	Concatenation: [h_static ∥ h_tv]
o	fc_mix: (h_static + h_tv) → mix_dim
	mix_dim = 16
	Activation: ReLU
o	fc_base: mix_dim → output_dim
	output_dim = 2 (two-class logits)
Random intercept / ID effect
o	id_embed: embedding table size = num_ids + 1
o	Embedding dim: id_emb_dim = 1
o	Learned scalar multiplier: alpha_id (initialized 0)
o	Applied as additive shift to the death logit (class “1”, column 2)

C) Training + optimization settings
Seeds
o	Outer loop: seeds <- 1:100 (100 independent runs)
o	Per-seed: torch_manual_seed(torch_seed) and set.seed(torch_seed)
o	Note: a separate torch_manual_seed(142) is set at script start (global initialization)
Epochs
o	Default in runner: epochs = 100
o	Called with: epochs = 100
Batch sizes
o	Training: batch_size = 1024
o	Test eval batching: batch_size_te = 1024
o	(Optional full eval, if enabled): batch_all = 2048
Optimizer
o	Adam (optim_adam)
o	Learning rate: lr = 1e-3
o	Weight decay: weight_decay = 1e-3
Loss function
o	Weighted cross-entropy: nnf_cross_entropy(logits, y_batch, weight=class_weights)
o	Class weights computed from training class counts:
	cw = (1 / class_counts), then normalized to mean 1: cw <- cw / mean(cw)
Checkpointing
o	Saved per seed to: checkpoints_base_observed/seed_###.pt
o	If checkpoint exists, training is skipped and state is loaded.
Classification rule for metrics
o	Argmax over logits (implicit τ = 0.50 in probability terms; i.e., default MAP class)
Group evaluation
o	Group label restricted to NH-White and NH-Black based on:
	black == 1 ⇒ NH-Black
	black == 0 & hispanic == 0 & others == 0 ⇒ NH-White
o	Metrics computed: accuracy, sensitivity, specificity (+ confusion counts)
Device
o	CUDA , see above instructions.  



Fullspec (Base + AL + Personality) NN-hazard — Hyperparameters Supplemental_full_gpu_FINAL
A) Data preprocessing + encoding choices
Outcome definition
o	died_ as binary event (classes = 0, 1) via factor(levels=c(0,1))
Age transforms
o	Hazard age (current interval):
	age_ log-transformed (log(age_))
	age_raw retained untransformed
o	AL measurement ages:
	age_1, age_3, age_5 log-transformed
	age_raw_1, age_raw_3, age_raw_5 retained untransformed
o	Personality measurement ages:
	agep_1, agep_3, agep_5 log-transformed
	agep_raw_1, agep_raw_3, agep_raw_5 retained untransformed
Standardization
o	Z-score scaling applied to all non-binary, non-age columns:
	Binary columns detected as values in {0,1}
	“Age columns” excluded from scaling:
	age_, age_raw
	age_1,age_3,age_5, age_raw_1,age_raw_3,age_raw_5
	agep_1,agep_3,agep_5, agep_raw_1,agep_raw_3,agep_raw_5
o	Missing values after scaling are set to 0 (X_scaled[is.na(X_scaled)] <- 0)
Missingness encoding
o	Missingness mask na_mask_mat: 1 if NA else 0
o	Masks are concatenated with observed values within each sequence step:
	TV branch: (x_seq ∥ m_seq)
	AL/Pers branches: each wave step input includes (values ∥ mask ∥ Δage)
Time-varying structure (TV branch)
o	Intended waves: waves <- 1:15, but:
	drops waves with 0 available variables
	enforces equal per-wave feature dimension across retained waves
o	TV bases:
	sayret, mwid, cesd, shlt, cancre, diabe, hearte, mobila, adl5a
o	Resulting hyperparameters determined at runtime:
	n_steps = number of retained TV waves
	tv_step_feat_dim = per-wave feature count
	tv_step_input_dim = 2 * tv_step_feat_dim (values + mask)
AL / Personality structure
o	AL waves: fixed at 3 measurement points: 1, 3, 5
	per-wave AL features: al_feat_dim = length(al1_idx) (must match across waves)
o	Personality waves: fixed at 3 measurement points: 1, 3, 5
	per-wave personality features: pers_feat_dim = length(pers1_idx) (must match across waves)
o	Recency encoding (learned feature)
	Uses raw age distances:
	dage_al = age_curr_raw - al_ages_raw (3 values)
	dage_pers = age_curr_raw - ps_ages_raw (3 values)
Train/test split
o	Stratified 80/20 split using caret::createDataPartition(y_idx, p=0.80)
o	Row-level split (not grouped by ID)
Group labels for subgroup metrics
o	black==1 ⇒ "Black"
o	black==0 & hispanic==0 & others==0 ⇒ "NH-White"
o	Else "Other"
o	Subgroup reporting uses groups_keep = c("NH-White","Black")

B) Representation capacity + architecture choices
Model class
o	Multi-branch MLP with:
1.	Static head
2.	TV (15-wave) head with attention pooling
3.	AL (3-wave) head with attention pooling + Δage baseline 
4.	Personality (3-wave) head with attention pooling + Δage baseline
5.	Cross-branch mixing layer
6.	Two additive output heads (base + AL/Pers refinement)
7.	ID embedding random intercept applied to death logit
Static head
o	Two-layer MLP:
	fc_static1: input = length(static_idx) → h_static
	fc_static2: h_static → h_static
o	Hidden size: h_static = 64
o	Activation: ReLU
o	Dropout: drop_p = 0.25 applied between layers
TV head (attention over retained waves)
o	Per-step input: 2 * tv_step_feat_dim (values + mask)
o	Two-layer MLP per step:
	fc_tv1: 2F → h_tv
	fc_tv2: h_tv → h_tv
o	Attention:
	att_tv: h_tv → 1 score per step; softmax over steps
	weighted sum yields h_tv
o	Hidden size: h_tv = 64
o	Dropout: drop_p = 0.25
AL head (3 waves; attention + Δage)
o	Step input includes: values + mask + Δage (one scalar)
	al_step_input_dim = 2 * al_feat_dim + 1
o	Two-layer MLP per AL wave:
	al_fc1: input → h_al
	al_fc2: h_al → h_al
o	Attention:
	al_att: h_al → 1; softmax over 3 waves
	weighted sum yields h_al
o	Hidden size: h_al = 64
o	Dropout: drop_p = 0.25
Personality head (3 waves; attention + Δage)
o	Step input includes: values + mask + Δage (one scalar)
	pers_step_input_dim = 2 * pers_feat_dim + 1
o	Two-layer MLP per personality wave:
	pers_fc1: input → h_pers
	pers_fc2: h_pers → h_pers
o	Attention:
	pers_att: h_pers → 1; softmax over 3 waves
	weighted sum yields h_pers
o	Hidden size: h_pers = 64
o	Dropout: drop_p = 0.25
Cross-branch mixing layer
o	Input: [h_static ∥ h_tv ∥ h_al ∥ h_pers]
o	fc_mix: (h_static + h_tv + h_al + h_pers) → mix_dim
o	mix_dim = 32
o	Activation: ReLU
Two-head additive hazard logits
o	Base head:
	fc_base: (h_static + h_tv) → output_dim
o	Auxiliary (AL+Pers + mix) head:
	fc_alpers: (h_al + h_pers + mix_dim) → output_dim
o	Total logits (pre-ID shift):
	logits_no_id = logits_base + logits_alpers
o	Output dimension: output_dim = 2 (two-class logits)
Random intercept / ID effect
o	id_embed: embedding table size = num_ids + 1
o	Embedding dim: id_emb_dim = 1
o	Learned scalar multiplier: alpha_id (initialized 0)
o	Applied to death logit (column 2)

C) Training + optimization settings
Seeds
o	Outer loop: seeds <- 1:100
o	Per-seed: torch_manual_seed(torch_seed) (note: no explicit set.seed(torch_seed) here)
Two-stage estimation schedule
o	Phase 1 (base-only training):
	epochs_phase1 = 50
	Forward pass uses use_alpers = FALSE
	Loss: weighted cross-entropy only
o	Phase 2 (activate AL+Pers head):
	epochs_phase2 = 100
	Forward pass uses use_alpers = TRUE
	Loss: weighted cross-entropy plus FP penalty term
Batch sizes
o	Training: batch_size = 1024
o	Test eval batching: batch_size_te = 1024
Optimizer
o	Adam (optim_adam)
o	Learning rate: lr = 1e-3
o	Weight decay: weight_decay = 1e-3
Loss function
o	Base loss: weighted cross-entropy
	nnf_cross_entropy(logits, y_batch, weight = class_weights)
o	Class weights computed from training class counts:
	cw = 1 / class_counts, normalized to mean 1 (cw <- cw / mean(cw))
False-positive penalty (Phase 2)
o	Penalty is computed on survival cases only:
	surv_mask <- (y_batch == 1L)
	p_death_surv = p_death[surv_mask]
	fp_penalty = mean(p_death_surv^2)
o	Combined objective:
	loss = loss_ce + lambda_fp * fp_penalty
o	Hyperparameter:
	lambda_fp = 0.5
Classification rule for metrics
o	Argmax over logits (MAP class)
Checkpointing
o	Saved per seed to: checkpoints_fullspec/seed_###.pt
o	Saved as CPU state_dict tensors:
	sd_cpu <- lapply(model$state_dict(), function(t) t$to(cpu))
	torch_save(sd_cpu, ckpt_path)
End-of-training artifact
o	Full workspace image saved:
	save.image("TRAINING_fullspec_100seeds.RData")
Device placement
o	CUDA See above instructions
