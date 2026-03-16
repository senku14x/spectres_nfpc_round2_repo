# NFPC Round 2 — Team Spectres — Code Submission

## Solution Overview

Mule account detection using a 5-model gradient boosting ensemble
(3× LightGBM + XGBoost + CatBoost) over 126 engineered behavioural features,
with walk-forward temporal validation and fold-safe target-encoded MCC risk.

**Final result:** Private AUC 0.999586 | F1 0.951295 | Temporal IoU 0.212756

## Repository Structure

```
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── nfpc_feature_engineering_pipeline_compact.py  # Feature engineering (Blocks 2–16)
├── nfpc_model_inference.py                      # Model loading + inference (Blocks 18–20)
├── colab_verification_snippet.py                # End-to-end Colab test harness
├── nfpc_final_code.py                           # Full final pipeline notebook (exported)
├── nfpc_code_submission_research.ipynb           # Research workflow notebook
├── model_artifacts/                             # Pre-trained model weights
│   ├── lgb_a.txt                                # LightGBM-A (behavioural, seed 42)
│   ├── lgb_b.txt                                # LightGBM-B (behavioural, seed 123)
│   ├── lgb_c.txt                                # LightGBM-C (full + MCC, seed 77)
│   ├── xgb.json                                 # XGBoost (full + MCC)
│   ├── catboost.cbm                             # CatBoost (full + MCC)
│   ├── catboost.json                            # CatBoost (JSON format)
│   ├── v14_final_meta.json                      # Weights, thresholds, config
│   ├── v14_final_behavioural_features.txt       # 125 feature names (LGB-A/B)
│   ├── v14_final_full_features.txt              # 126 feature names (LGB-C/XGB/CAT)
│   └── v14_final_feature_manifest.json          # Full feature manifest
```

### File Roles

| File | Purpose |
|------|---------|
| `nfpc_feature_engineering_pipeline_compact.py` | Standalone feature pipeline. Reads raw competition data, outputs `features_train_v14.parquet` and `features_test_v14.parquet`. |
| `nfpc_model_inference.py` | Loads pre-trained weights from `model_artifacts/`, computes fold-safe `risky_mcc_share`, produces submission CSVs. |
| `colab_verification_snippet.py` | Cell-by-cell Colab script that uploads the ZIP, runs both stages, and validates submission format. |
| `nfpc_final_code.py` | The complete final pipeline as a single exported notebook (Blocks 1–20 including EDA, training, SHAP, and submission assembly). |
| `nfpc_code_submission_research.ipynb` | The full research workflow notebook documenting all experiments from v1 through v14. |

## Environment Setup

**Platform:** Google Colab Pro with NVIDIA A100 GPU (all model building and research was conducted entirely on this platform)

**Python version:** 3.10+ (Colab default)

**Dependencies:**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install lightgbm>=4.1.0 xgboost>=2.0.0 catboost>=1.2.0 duckdb>=0.9.0 pyarrow>=14.0.0 pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 shap>=0.43.0 scipy>=1.11.0 matplotlib>=3.7.0
```

**Hardware requirements:**
- **GPU:** NVIDIA A100 (used for development). Any CUDA-capable GPU works for LightGBM/XGBoost/CatBoost training and inference. CPU fallback is supported but slower.
- **RAM:** ~15 GB minimum for DuckDB feature engineering queries over ~396M transaction rows.
- **Disk:** ~20 GB free for competition data + intermediate parquet files.

## Steps to Reproduce

### Step 1: Prepare Data

Place the competition dataset at `/content/nfpc_data/` (or update `NFPC_DATA` in the scripts):

```
/content/nfpc_data/
├── accounts.parquet
├── accounts-additional.parquet
├── customers.parquet
├── customer_account_linkage.parquet
├── demographics.parquet
├── product_details.parquet
├── branch.parquet
├── train_labels.parquet
├── test_accounts.parquet
├── transactions/batch-*/part_*.parquet      (~396 files, 8.59 GB)
└── transactions_additional/batch-*/part_*.parquet  (~311 files, 8.81 GB)
```

### Step 2: Run Feature Engineering

```bash
python nfpc_feature_engineering_pipeline_compact.py
```

This runs Blocks 2–16 sequentially:
- Loads and parses all static tables
- Builds 8 base feature groups (A–H) via DuckDB over the full transaction history
- Adds Feature Pack v2 (recent windows, fund residence, counterparty novelty)
- Adds Feature Pack v3 (dormancy/reactivation, transaction velocity)
- Adds Feature Pack v4 (digital banking, product holdings, geographic velocity)
- Applies leakage cleanup (removes freeze, balance snapshots, duplicate joins)
- Produces `features_train_v14.parquet` and `features_test_v14.parquet`

**Runtime:** ~45–60 minutes on Colab Pro (dominated by DuckDB queries over ~396M rows).

### Step 3: Run Inference

```bash
python nfpc_model_inference.py
```

This loads the pre-trained models, computes fold-safe `risky_mcc_share` on the
full training set, generates blended predictions, and writes submission CSVs.

**Runtime:** ~5–10 minutes (DuckDB MCC computation + model inference).

### Step 4: Verify (Optional)

Upload the ZIP to Colab and run `colab_verification_snippet.py` cell-by-cell.
It automates Steps 2–3 and validates the output submission format.

### Step 5: Submission

The primary submission file is `submission_v14_final_with_current_windows.csv`
(if a `precise_ts.parquet` window file is available) or
`submission_v14_final_probs_only.csv` (probabilities only).

## Approach Summary

### Feature Engineering (126 features)

125 base behavioural features + 1 fold-safe `risky_mcc_share`:

| Group | Count | Description |
|-------|-------|-------------|
| A. Counterparty network | 7 | Fan asymmetry, CP ratios, branch deviation |
| B. MCC & channel | 5 | Channel entropy, NEFT/IMPS share, ATM share |
| C. Structuring & amounts | 8 | Structuring band, round share, ticket sizes |
| D. Balance & pass-through | 5 | Near-zero share, drain rate, balance CV, turnover |
| E. Temporal & activity | 8 | Weekend/off-hours share, active days, peak-to-avg |
| F. Geographic & IP | 6 | Unique IPs, geo spread, IP sharing |
| G. Account/customer static | 12 | Account age, KYC, scheme risk, product counts |
| H. Branch context | 6 | Branch type, employee density, turnover |
| Pack v2: Recent windows | 15 | w30/w90 concentration, near-zero uplift, credit z-score |
| Pack v2: Fund residence | 12 | Residence hours, same-day drain, drain rate at 1h/6h/24h |
| Pack v2: CP novelty | 12 | New CPs in 30d, CP concentration, bidirectional rate |
| Pack v3: Dormancy | 8 | Inactivity gaps, reactivation volume, burst ratio |
| Pack v3: Velocity | 7 | Max daily/hourly txns, inter-txn gap percentiles, gap CV |
| Pack v4: Demographics | 8 | Digital banking score, product diversity, address recency |
| Pack v4: Geo velocity | 6 | Location entropy, impossible travel, ATM deposit flag |
| Fold-safe MCC | 1 | Top-10 risky mnemonic codes by train-fold lift |
| **Total** | **126** | |

### Model Architecture

Five-model ensemble with two feature views:

- **LGB-A** (seed 42): 125 behavioural features, 500 trees, 63 leaves
- **LGB-B** (seed 123): 125 behavioural features, 500 trees, 63 leaves
- **LGB-C** (seed 77): 126 full features, 400 trees, 31 leaves
- **XGBoost**: 126 full features, 600 trees, depth 6
- **CatBoost**: 126 full features, 500 iterations, depth 6

All models use `scale_pos_weight=34.8` for class imbalance (2,683 mules / 93,408 legit).

**Ensemble weights (F1-optimised):** A=0.6, B=0.0, C=0.0, X=0.2, CAT=0.2

### Validation

3 walk-forward temporal folds on `account_opening_date`:
- Fold 1: train ≤ 2022-06-30, val 2022-07-01 → 2023-06-30
- Fold 2: train ≤ 2023-06-30, val 2023-07-01 → 2024-06-30
- Fold 3: train ≤ 2024-06-30, val 2024-07-01 → 2025-06-30

OOF metrics: AUC 0.929, F1 0.815, PR-AUC 0.734

### Key Design Decisions

1. **Leakage control:** Removed freeze_date, balance snapshots, account_status
   (post-flag bank actions). Rebuilt risky_mcc_share fold-safely per CV split.
2. **Feature-set swap:** v14_final uses 125 base features (vs 137 in exploration).
   Removed 39 sparse MCC/mnemonic taxonomy and fine-grained gap-timing features;
   added 27 broader channel, reactivation, product, and location features.
3. **Behavioural focus:** Strongest signals are counterparty concentration,
   pass-through balance cycling, recent-window acceleration, and dormancy/reactivation.
4. **Window policy:** Suspicious timestamps populated only for accounts scoring ≥ 0.30.
