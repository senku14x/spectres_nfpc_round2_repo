"""
NFPC Round 2 — Model Inference (v14_final)
Team Spectres

Loads the pre-trained v14_final model artifacts and generates
submission predictions from pre-computed feature files.

Prerequisites:
  - Feature files already built by nfpc_feature_engineering_pipeline_compact.py
    (outputs: features_train_v14.parquet, features_test_v14.parquet)
  - Model weights in model_artifacts/ directory
  - Competition data in NFPC_DATA path

Usage:
  python nfpc_model_inference.py

Outputs:
  submission_v14_final_probs_only.csv
  submission_v14_final_with_current_windows.csv  (if window file available)
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, balanced_accuracy_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths for your environment
# ══════════════════════════════════════════════════════════════
NFPC_DATA   = '/content/nfpc_data'       # competition dataset root
OUT_DIR     = '/content'                 # feature parquets + outputs
MODEL_DIR   = 'model_artifacts'          # pre-trained weights
TMP_DIR     = '/tmp'
SEED        = 42
RUN_NAME    = 'v14_final'
WINDOW_SCORE_THRESHOLD = 0.30

np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════
# 1. LOAD META + FEATURE LISTS
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"NFPC v14_final — Model Inference")
print(f"{'='*60}")

meta_path = os.path.join(MODEL_DIR, 'v14_final_meta.json')
with open(meta_path) as f:
    meta = json.load(f)

print(f"Run name          : {meta['run_name']}")
print(f"Base features     : {meta['base_feature_count']}")
print(f"All features      : {meta['all_feature_count']}")
print(f"Best OOF F1       : {meta['best_f1_score']:.5f}")
print(f"Best OOF threshold: {meta['best_oof_threshold']:.4f}")

best_f1_weights = np.array(meta['best_f1_weights'])
best_threshold  = meta['best_oof_threshold']

# Load feature name lists
beh_path = os.path.join(MODEL_DIR, 'v14_final_behavioural_features.txt')
full_path = os.path.join(MODEL_DIR, 'v14_final_full_features.txt')

with open(beh_path) as f:
    behavioural_features = [line.strip() for line in f if line.strip()]
with open(full_path) as f:
    full_features = [line.strip() for line in f if line.strip()]

print(f"Behavioural feats : {len(behavioural_features)}")
print(f"Full feats        : {len(full_features)}")

# ══════════════════════════════════════════════════════════════
# 2. LOAD PRE-COMPUTED FEATURES
# ══════════════════════════════════════════════════════════════
print(f"\nLoading feature files...")

feat_train_path = os.path.join(OUT_DIR, 'features_train_v14.parquet')
feat_test_path  = os.path.join(OUT_DIR, 'features_test_v14.parquet')

feat_train = pd.read_parquet(feat_train_path)
feat_test  = pd.read_parquet(feat_test_path)

labels = pd.read_parquet(os.path.join(NFPC_DATA, 'train_labels.parquet'))
test_ids = pd.read_parquet(os.path.join(NFPC_DATA, 'test_accounts.parquet'))

print(f"  Train features  : {feat_train.shape}")
print(f"  Test features   : {feat_test.shape}")

# Validate features present
base_feat_cols = [c for c in feat_train.columns if c != 'account_id']
missing_beh = [f for f in behavioural_features if f not in base_feat_cols]
if missing_beh:
    print(f"  [WARN] Missing behavioural features: {missing_beh}")

# ══════════════════════════════════════════════════════════════
# 3. FOLD-SAFE RISKY MCC SHARE (for full retrain)
# ══════════════════════════════════════════════════════════════
print(f"\nComputing fold-safe risky_mcc_share for full retrain...")

import duckdb
con = duckdb.connect()
con.execute(f"""
    CREATE OR REPLACE VIEW transactions AS
    SELECT * FROM read_parquet('{NFPC_DATA}/transactions/batch-*/part_*.parquet')
""")
con.execute(f"""
    CREATE OR REPLACE VIEW txn_add AS
    SELECT * FROM read_parquet('{NFPC_DATA}/transactions_additional/batch-*/part_*.parquet')
""")


def compute_fold_safe_mcc(train_account_ids, train_labels_series, all_account_ids):
    """Compute risky_mcc_share using only train-fold labels."""
    train_account_ids = pd.Series(train_account_ids).astype(str).reset_index(drop=True)
    train_labels_series = pd.Series(train_labels_series).reset_index(drop=True)

    mule_ids = train_account_ids[train_labels_series == 1].tolist()
    legit_ids = train_account_ids[train_labels_series == 0].tolist()

    n_mules = len(mule_ids)
    n_legit = len(legit_ids)

    if n_mules == 0 or n_legit == 0:
        return pd.Series(0.0, index=all_account_ids)

    sample_legit = min(n_legit, n_mules * 10)
    mule_id_str = ",".join([f"'{x}'" for x in mule_ids[:5000]])
    legit_id_str = ",".join([f"'{x}'" for x in legit_ids[:sample_legit]])

    risky_codes_df = con.execute(f"""
        WITH mule_codes AS (
            SELECT ta.mnemonic_code,
                   COUNT(*) * 1.0 / {n_mules} AS mule_rate
            FROM txn_add ta
            JOIN transactions t ON ta.transaction_id = t.transaction_id
            WHERE t.account_id IN ({mule_id_str})
              AND ta.mnemonic_code IS NOT NULL
            GROUP BY ta.mnemonic_code
        ),
        legit_codes AS (
            SELECT ta.mnemonic_code,
                   COUNT(*) * 1.0 / {sample_legit} AS legit_rate
            FROM txn_add ta
            JOIN transactions t ON ta.transaction_id = t.transaction_id
            WHERE t.account_id IN ({legit_id_str})
              AND ta.mnemonic_code IS NOT NULL
            GROUP BY ta.mnemonic_code
        )
        SELECT m.mnemonic_code,
               m.mule_rate / (COALESCE(l.legit_rate, 0) + 0.001) AS lift
        FROM mule_codes m
        LEFT JOIN legit_codes l ON m.mnemonic_code = l.mnemonic_code
        ORDER BY lift DESC
        LIMIT 10
    """).df()

    if risky_codes_df.empty:
        return pd.Series(0.0, index=all_account_ids)

    risky_codes = risky_codes_df["mnemonic_code"].astype(str).tolist()
    risky_str = ",".join([f"'{x}'" for x in risky_codes])

    all_account_ids = [str(x) for x in all_account_ids]
    all_id_str = ",".join([f"'{x}'" for x in all_account_ids])

    share_df = con.execute(f"""
        SELECT t.account_id,
               SUM(CASE WHEN ta.mnemonic_code IN ({risky_str}) THEN 1 ELSE 0 END) * 1.0
                   / COUNT(*) AS risky_mcc_share
        FROM transactions t
        JOIN txn_add ta ON t.transaction_id = ta.transaction_id
        WHERE t.account_id IN ({all_id_str})
        GROUP BY t.account_id
    """).df()

    return (
        share_df.set_index("account_id")["risky_mcc_share"]
        .reindex(all_account_ids)
        .fillna(0.0)
    )


# Merge labels for training
feat_train = feat_train.merge(labels[['account_id', 'is_mule']], on='account_id', how='left')

all_train_ids = feat_train['account_id'].astype(str).values.tolist()
all_test_ids  = feat_test['account_id'].astype(str).values.tolist()
y_full = feat_train['is_mule'].values

mcc_final = compute_fold_safe_mcc(
    pd.Series(all_train_ids),
    pd.Series(y_full),
    all_train_ids + all_test_ids,
)

mcc_full_train = mcc_final.reindex(all_train_ids).fillna(0).values
mcc_full_test  = mcc_final.reindex(all_test_ids).fillna(0).values

print(f"  risky_mcc_share computed for {len(all_train_ids) + len(all_test_ids):,} accounts")

# ══════════════════════════════════════════════════════════════
# 4. LOAD PRE-TRAINED MODELS
# ══════════════════════════════════════════════════════════════
print(f"\nLoading pre-trained models from {MODEL_DIR}/...")

final_a = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'lgb_a.txt'))
final_b = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'lgb_b.txt'))
final_c = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'lgb_c.txt'))
final_x = xgb.XGBClassifier()
final_x.load_model(os.path.join(MODEL_DIR, 'xgb.json'))
final_cat = CatBoostClassifier()
final_cat.load_model(os.path.join(MODEL_DIR, 'catboost.cbm'))

print("  All 5 models loaded ✓")

# ══════════════════════════════════════════════════════════════
# 5. GENERATE TEST PREDICTIONS
# ══════════════════════════════════════════════════════════════
print(f"\nGenerating predictions...")

X_test_beh = feat_test[behavioural_features].values
X_test_all = np.column_stack([X_test_beh, mcc_full_test])

test_a   = final_a.predict(X_test_beh)
test_b   = final_b.predict(X_test_beh)
test_c   = final_c.predict(X_test_all)
test_x   = final_x.predict_proba(X_test_all)[:, 1]
test_cat = final_cat.predict_proba(X_test_all)[:, 1]

test_mat = np.column_stack([test_a, test_b, test_c, test_x, test_cat])
blend_scores = test_mat @ best_f1_weights

print(f"  Predictions generated for {len(blend_scores):,} test accounts")
print(f"  Score range: [{blend_scores.min():.6f}, {blend_scores.max():.6f}]")
print(f"  Score > 0.5: {(blend_scores > 0.5).sum():,}")
print(f"  Score > {best_threshold:.3f}: {(blend_scores > best_threshold).sum():,}")

# ══════════════════════════════════════════════════════════════
# 6. BUILD SUBMISSIONS
# ══════════════════════════════════════════════════════════════
print(f"\nBuilding submissions...")

sub = test_ids[['account_id']].copy()
sub['is_mule'] = blend_scores.clip(0, 1)
sub['suspicious_start'] = ''
sub['suspicious_end'] = ''
sub = sub[['account_id', 'is_mule', 'suspicious_start', 'suspicious_end']]

out_probs = os.path.join(OUT_DIR, f'submission_{RUN_NAME}_probs_only.csv')
sub.to_csv(out_probs, index=False)
print(f"  [OK] {out_probs}")

# ── Attach current windows if available ──
def fmt_dt(series, end_of_day=False):
    dt = pd.to_datetime(series, errors="coerce")
    if end_of_day:
        dt = dt.dt.floor("D") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    else:
        dt = dt.dt.floor("D")
    return dt.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")

window_candidates = [
    os.path.join(TMP_DIR, 'precise_ts.parquet'),
    os.path.join(OUT_DIR, 'precise_ts.parquet'),
]
window_path = next((p for p in window_candidates if os.path.exists(p)), None)

if window_path:
    print(f"\n  Found window file: {window_path}")
    w = pd.read_parquet(window_path)

    # Infer start/end columns
    start_col = next((c for c in w.columns if 'start' in c.lower()), None)
    end_col = next((c for c in w.columns if 'end' in c.lower()), None)

    if start_col and end_col:
        w = w.rename(columns={start_col: 'suspicious_start', end_col: 'suspicious_end'})

        sub_w = sub[['account_id', 'is_mule']].merge(
            w[['account_id', 'suspicious_start', 'suspicious_end']],
            on='account_id', how='left',
        )
        sub_w['suspicious_start'] = pd.to_datetime(sub_w['suspicious_start'], errors='coerce')
        sub_w['suspicious_end']   = pd.to_datetime(sub_w['suspicious_end'], errors='coerce')

        # Only populate windows for high-scoring accounts
        low_score = sub_w['is_mule'] < WINDOW_SCORE_THRESHOLD
        sub_w.loc[low_score, ['suspicious_start', 'suspicious_end']] = pd.NaT

        out_w = sub_w[['account_id', 'is_mule']].copy()
        out_w['suspicious_start'] = fmt_dt(sub_w['suspicious_start'], end_of_day=False)
        out_w['suspicious_end']   = fmt_dt(sub_w['suspicious_end'], end_of_day=True)

        out_window = os.path.join(OUT_DIR, f'submission_{RUN_NAME}_with_current_windows.csv')
        out_w.to_csv(out_window, index=False)
        print(f"  [OK] {out_window}")
    else:
        print("  Window columns not found; skipping window attachment.")
else:
    print("\n  No window file found; submitting probabilities only.")

# ══════════════════════════════════════════════════════════════
# 7. OPTIONAL: EVALUATE ON TRAIN (OOF proxy)
# ══════════════════════════════════════════════════════════════
print(f"\nTrain-set sanity check (NOT OOF — for debugging only):")
X_train_beh = feat_train[behavioural_features].values
X_train_all = np.column_stack([X_train_beh, mcc_full_train])

train_a   = final_a.predict(X_train_beh)
train_b   = final_b.predict(X_train_beh)
train_c   = final_c.predict(X_train_all)
train_x   = final_x.predict_proba(X_train_all)[:, 1]
train_cat = final_cat.predict_proba(X_train_all)[:, 1]

train_blend = np.column_stack([train_a, train_b, train_c, train_x, train_cat]) @ best_f1_weights

train_auc = roc_auc_score(y_full, train_blend)
train_f1 = max(f1_score(y_full, train_blend >= t) for t in np.arange(0.1, 0.9, 0.01))
print(f"  Train AUC (resubstitution) : {train_auc:.5f}")
print(f"  Train best F1 (resubst.)   : {train_f1:.5f}")
print(f"  (These are overfit estimates; real OOF was AUC=0.929, F1=0.815)")

print(f"\n{'='*60}")
print(f"INFERENCE COMPLETE ✓")
print(f"{'='*60}")
