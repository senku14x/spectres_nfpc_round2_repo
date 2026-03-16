import os, time, warnings
import numpy  as np
import pandas as pd
warnings.filterwarnings('ignore')
NFPC_DATA = '/content/nfpc_data'
TMP_DIR   = '/tmp'
REF_TS    = pd.Timestamp('2025-06-30')   # competition reference date
print("Loading static tables...")
t0 = time.time()
accounts     = pd.read_parquet(f'{NFPC_DATA}/accounts.parquet')
accounts_add = pd.read_parquet(f'{NFPC_DATA}/accounts-additional.parquet')
customers    = pd.read_parquet(f'{NFPC_DATA}/customers.parquet')
cust_link    = pd.read_parquet(f'{NFPC_DATA}/customer_account_linkage.parquet')
demographics = pd.read_parquet(f'{NFPC_DATA}/demographics.parquet')
product_det  = pd.read_parquet(f'{NFPC_DATA}/product_details.parquet')
branch       = pd.read_parquet(f'{NFPC_DATA}/branch.parquet')
labels       = pd.read_parquet(f'{NFPC_DATA}/train_labels.parquet')
test_acc     = pd.read_parquet(f'{NFPC_DATA}/test_accounts.parquet')
print(f"  Loaded in {time.time()-t0:.1f}s")
for col in ['account_opening_date', 'freeze_date', 'unfreeze_date',
            'last_kyc_date', 'last_mobile_update_date']:
    if col in accounts.columns:
        accounts[col] = pd.to_datetime(accounts[col], errors='coerce')
for col in ['date_of_birth', 'relationship_start_date']:
    if col in customers.columns:
        customers[col] = pd.to_datetime(customers[col], errors='coerce')
if 'address_last_update_date' in demographics.columns:
    demographics['address_last_update_date'] = pd.to_datetime(
        demographics['address_last_update_date'], errors='coerce')
labels['mule_flag_date'] = pd.to_datetime(labels['mule_flag_date'], errors='coerce')
print("Date columns parsed ✓")
YN_COLS_ACCOUNTS = [
    'nomination_flag', 'cheque_allowed', 'cheque_availed',
    'kyc_compliant', 'rural_branch',
]
YN_COLS_CUSTOMERS = [
    'pan_available', 'aadhaar_available', 'passport_available',
    'mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
    'demat_flag', 'credit_card_flag', 'fastag_flag',
]
YN_COLS_DEMOGRAPHICS = ['joint_account_flag', 'nri_flag']
def yn_to_int(df, cols):
    """Convert Y/N string columns to 0/1 integers in-place."""
    for col in cols:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.upper().str.strip() == 'Y').astype(int)
    return df
accounts     = yn_to_int(accounts,     YN_COLS_ACCOUNTS)
customers    = yn_to_int(customers,    YN_COLS_CUSTOMERS)
demographics = yn_to_int(demographics, YN_COLS_DEMOGRAPHICS)
print("Y/N columns converted to 0/1 ✓")
print(f"  accounts    : {YN_COLS_ACCOUNTS}")
print(f"  customers   : {YN_COLS_CUSTOMERS}")
print(f"  demographics: {YN_COLS_DEMOGRAPHICS}")
all_ids = pd.concat([
    labels[['account_id']],
    test_acc[['account_id']]
], ignore_index=True).drop_duplicates()
all_ids.to_parquet(f'{TMP_DIR}/all_ids.parquet', index=False)
print(f"\nall_ids saved → /tmp/all_ids.parquet  ({len(all_ids):,} accounts)")
N_TRAIN   = len(labels)
N_TEST    = len(test_acc)
N_MULES   = int(labels['is_mule'].sum())
N_LEGIT   = N_TRAIN - N_MULES
MULE_RATE = N_MULES / N_TRAIN
SPW       = N_LEGIT / N_MULES   # scale_pos_weight for all tree models
print(f"\n{'='*55}")
print(f"DATASET SUMMARY")
print(f"{'='*55}")
print(f"  Train accounts   : {N_TRAIN:>10,}")
print(f"  Test  accounts   : {N_TEST:>10,}")
print(f"  Total accounts   : {N_TRAIN + N_TEST:>10,}")
print(f"  Mule accounts    : {N_MULES:>10,}  ({MULE_RATE*100:.2f}%)")
print(f"  Legit accounts   : {N_LEGIT:>10,}  ({(1-MULE_RATE)*100:.2f}%)")
print(f"  Imbalance ratio  : 1 : {SPW:.0f}")
print(f"  scale_pos_weight : {SPW:.2f}")
tables = {
    'accounts'    : accounts,
    'accounts_add': accounts_add,
    'customers'   : customers,
    'cust_link'   : cust_link,
    'demographics': demographics,
    'product_det' : product_det,
    'branch'      : branch,
    'labels'      : labels,
    'test_acc'    : test_acc,
}
print(f"\n{'TABLE':<16} {'ROWS':>10} {'COLS':>5}  COLUMNS")
print("-"*95)
for name, df in tables.items():
    print(f"  {name:<14} {len(df):>10,} {len(df.columns):>5}  {list(df.columns)}")
acc_train  = accounts[accounts['account_id'].isin(labels['account_id'])].copy()
acc_train  = acc_train.merge(labels[['account_id','is_mule']], on='account_id')
mule_mask  = acc_train['is_mule'] == 1
print(f"\n{'='*55}")
print("STATIC FIELD STATS — MULE vs LEGIT (train set)")
print(f"{'='*55}")
print(f"\n  account_opening_date range:")
print(f"    min = {accounts['account_opening_date'].min().date()}")
print(f"    max = {accounts['account_opening_date'].max().date()}")
print(f"\n  LEAKAGE FIELDS (excluded from all features):")
freeze_mule  = acc_train.loc[mule_mask,  'freeze_date'].notna().mean()*100
freeze_legit = acc_train.loc[~mule_mask, 'freeze_date'].notna().mean()*100
print(f"    freeze_date present  — mule: {freeze_mule:.1f}%  legit: {freeze_legit:.1f}%")
print(f"    → Post-flag bank action; excluded entirely from feature pipeline")
binary_fields = [
    ('kyc_compliant',   'kyc_compliant'),
    ('rural_branch',    'rural_branch'),
    ('nomination_flag', 'nomination_flag'),
    ('cheque_availed',  'cheque_availed'),
]
print(f"\n  BINARY ACCOUNT FIELDS (after Y/N conversion):")
print(f"    {'FIELD':<22} {'MULE%':>7} {'LEGIT%':>8}  SIGNIFICANT?")
print(f"    {'-'*50}")
for label_str, col in binary_fields:
    if col in acc_train.columns:
        m = acc_train.loc[mule_mask,  col].mean()*100
        l = acc_train.loc[~mule_mask, col].mean()*100
        sig = "YES" if abs(m-l) > 5 else "no"
        print(f"    {label_str:<22} {m:>6.1f}%  {l:>7.1f}%  {sig}")
demo_train = (
    cust_link
    .merge(labels[['account_id','is_mule']], on='account_id')
    .merge(demographics[['customer_id','nri_flag']], on='customer_id', how='left')
    .drop_duplicates('account_id')
)
dm = demo_train['is_mule'] == 1
nri_m = demo_train.loc[dm,  'nri_flag'].mean()*100
nri_l = demo_train.loc[~dm, 'nri_flag'].mean()*100
print(f"    {'nri_flag':<22} {nri_m:>6.1f}%  {nri_l:>7.1f}%  "
      f"{'YES' if abs(nri_m-nri_l) > 5 else 'no'}")
customers['age'] = (REF_TS - customers['date_of_birth']).dt.days / 365.25
cust_train = (
    cust_link
    .merge(labels[['account_id','is_mule']], on='account_id')
    .merge(customers[['customer_id','age']], on='customer_id', how='left')
    .drop_duplicates('account_id')
)
cm = cust_train['is_mule'] == 1
print(f"\n  Customer age  — mule median: {cust_train.loc[cm,'age'].median():.1f} yrs"
      f"  legit median: {cust_train.loc[~cm,'age'].median():.1f} yrs")
print(f"  → Age is NON-SIGNIFICANT (mules recruit demographically normal accounts)")
print(f"\n{'='*55}")
print("MULE FLAG DATE DISTRIBUTION")
print(f"{'='*55}")
mule_labels = labels[labels['is_mule'] == 1].copy()
mule_labels['flag_year'] = mule_labels['mule_flag_date'].dt.year
print(f"\n  Total confirmed mules : {len(mule_labels):,}")
print(f"  Flag date range       : "
      f"{mule_labels['mule_flag_date'].min().date()} → "
      f"{mule_labels['mule_flag_date'].max().date()}")
print(f"\n  By year:")
print(mule_labels['flag_year'].value_counts().sort_index()
      .rename('count').to_string())
null_flags  = mule_labels['mule_flag_date'].isna().sum()
null_reason = (
    labels[labels['is_mule'] == 1]['alert_reason'].isna().sum()
    if 'alert_reason' in labels.columns else 'N/A'
)
print(f"\n  Null mule_flag_date : {null_flags}")
print(f"  Null alert_reason   : {null_reason}")
if isinstance(null_reason, int) and null_reason > 0:
    pct = null_reason / len(mule_labels) * 100
    print(f"  → {pct:.1f}% label noise — null alert_reason accounts are genuine mules"
          f" but with missing administrative annotation")
print("\n" + "="*65)
print("BLOCK 2 COMPLETE ✓")
import os
import time
import warnings
import duckdb
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
NFPC_DATA = '/content/nfpc_data'
OUT_DIR   = '/content'
if 'labels' not in globals():
    labels = pd.read_parquet(f'{NFPC_DATA}/train_labels.parquet')
if 'accounts' not in globals():
    accounts = pd.read_parquet(f'{NFPC_DATA}/accounts.parquet', columns=['account_id', 'branch_code'])
print("Setting up DuckDB views...")
con = duckdb.connect()
con.execute(f"""
    CREATE OR REPLACE VIEW transactions AS
    SELECT * FROM read_parquet('{NFPC_DATA}/transactions/batch-*/part_*.parquet')
""")
con.execute(f"""
    CREATE OR REPLACE VIEW txn_add AS
    SELECT * FROM read_parquet('{NFPC_DATA}/transactions_additional/batch-*/part_*.parquet')
""")
print("DuckDB views ready ✓")
print("\nChecking mcc_code in transactions table...")
mcc_sample = con.execute("""
    SELECT
        mcc_code,
        COUNT(DISTINCT account_id) AS n_accs
    FROM transactions
    GROUP BY mcc_code
    ORDER BY n_accs DESC
    LIMIT 20
""").df()
n_distinct_mcc = con.execute("""
    SELECT COUNT(DISTINCT mcc_code) FROM transactions
""").fetchone()[0]
print(f"  Distinct mcc_code values: {n_distinct_mcc}")
print(mcc_sample.to_string(index=False))
print("\nComputing mule rate per mcc_code (EDA — labels used only here)...")
mcc_lift = con.execute(f"""
    SELECT
        t.mcc_code,
        COUNT(DISTINCT t.account_id) AS n_accounts,
        COUNT(DISTINCT CASE WHEN l.is_mule = 1 THEN t.account_id END) AS n_mule_accs,
        ROUND(
            COUNT(DISTINCT CASE WHEN l.is_mule = 1 THEN t.account_id END) * 100.0
            / NULLIF(COUNT(DISTINCT t.account_id), 0),
            2
        ) AS mule_rate_pct
    FROM transactions t
    JOIN read_parquet('{NFPC_DATA}/train_labels.parquet') l
      ON t.account_id = l.account_id
    GROUP BY t.mcc_code
    HAVING COUNT(DISTINCT t.account_id) >= 50
    ORDER BY mule_rate_pct DESC
    LIMIT 20
""").df()
print("\n  Top mcc_code by mule rate (baseline = 2.79%):")
print(mcc_lift.to_string(index=False))
BASELINE_MULE_RATE = 2.79
RISKY_MCC_THRESHOLD = BASELINE_MULE_RATE * 1.5  # 1.5x lift
risky_mcc_codes = mcc_lift.loc[
    mcc_lift['mule_rate_pct'] >= RISKY_MCC_THRESHOLD,
    'mcc_code'
].tolist()
if risky_mcc_codes:
    print(f"\n  RISKY mcc_codes (≥{RISKY_MCC_THRESHOLD:.1f}% mule rate):")
    print(f"  {risky_mcc_codes}")
    print("  ⚠  These codes come from EDA on full labels.")
    print("     In training (Block 16), risky_mcc_share is recomputed")
    print("     inside each CV fold using train-fold labels only.")
else:
    print(f"\n  No mcc_code shows ≥{RISKY_MCC_THRESHOLD:.1f}% mule rate.")
    print("  risky_mcc_share_raw will be 0 for all accounts.")
RISKY_MCCS_TXN = risky_mcc_codes
print(f"\n{'='*60}")
print("4.2  GROUP A — COUNTERPARTY NETWORK")
print(f"{'='*60}")
t0 = time.time()
feat_A = con.execute("""
    SELECT
        account_id,
        COUNT(DISTINCT counterparty_id) AS unique_counterparties,
        COUNT(DISTINCT CASE WHEN txn_type = 'C' THEN counterparty_id END)
            AS unique_incoming_cp,
        COUNT(DISTINCT CASE WHEN txn_type = 'D' THEN counterparty_id END)
            AS unique_outgoing_cp,
        COUNT(DISTINCT CASE WHEN txn_type = 'C' THEN counterparty_id END) * 1.0
        / (COUNT(DISTINCT CASE WHEN txn_type = 'D' THEN counterparty_id END) + 1)
            AS fan_asymmetry,
        COUNT(DISTINCT CASE WHEN txn_type = 'C' THEN counterparty_id END) * 1.0
        / (COUNT(DISTINCT counterparty_id) + 1)
            AS cp_incoming_ratio,
        COUNT(DISTINCT CASE WHEN txn_type = 'D' THEN counterparty_id END) * 1.0
        / (COUNT(DISTINCT counterparty_id) + 1)
            AS cp_outgoing_ratio
    FROM transactions
    GROUP BY account_id
""").df()
print(f"  Raw cp features: {feat_A.shape}  ({time.time()-t0:.1f}s)")
accounts_branch = accounts[['account_id', 'branch_code']].copy()
feat_A = feat_A.merge(accounts_branch, on='account_id', how='left')
branch_median_cp = (
    feat_A.groupby('branch_code')['unique_counterparties']
    .median()
    .rename('branch_median_cp')
)
feat_A = feat_A.merge(branch_median_cp, on='branch_code', how='left')
feat_A['cp_deviation_from_branch'] = (
    feat_A['unique_counterparties'] - feat_A['branch_median_cp']
)
feat_A = feat_A.drop(columns=['branch_code', 'branch_median_cp'])
print(f"  After cp_deviation_from_branch: {feat_A.shape}")
print("\n  Feature ranges (train mules vs legit):")
check_A = feat_A.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')
for col in [
    'unique_counterparties',
    'cp_deviation_from_branch',
    'unique_incoming_cp',
    'unique_outgoing_cp'
]:
    mule_med = check_A.loc[check_A['is_mule'] == 1, col].median()
    legit_med = check_A.loc[check_A['is_mule'] == 0, col].median()
    print(f"    {col:<32}  mule={mule_med:.1f}  legit={legit_med:.1f}")
feat_A.to_parquet(f'{OUT_DIR}/feat_A.parquet', index=False)
print(f"\n  Saved → {OUT_DIR}/feat_A.parquet  shape={feat_A.shape}")
print(f"\n{'='*60}")
print("4.3  GROUP B — MCC & CHANNEL")
print(f"{'='*60}")
if RISKY_MCCS_TXN:
    risky_list = ", ".join(str(x) for x in RISKY_MCCS_TXN)
    risky_mcc_case = f"CASE WHEN mcc_code IN ({risky_list}) THEN cnt ELSE 0 END"
else:
    risky_mcc_case = "0"
t0 = time.time()
feat_B = con.execute(f"""
    WITH acct_channel_mcc AS (
        SELECT
            account_id,
            channel,
            mcc_code,
            COUNT(*) AS cnt
        FROM transactions
        GROUP BY account_id, channel, mcc_code
    ),
    acct_channel AS (
        SELECT
            account_id,
            channel,
            SUM(cnt) AS channel_cnt
        FROM acct_channel_mcc
        GROUP BY account_id, channel
    ),
    acct_totals AS (
        SELECT
            account_id,
            SUM(channel_cnt) AS total_txns,
            COUNT(DISTINCT channel) AS channel_count,
            MAX(channel_cnt) AS max_channel_cnt
        FROM acct_channel
        GROUP BY account_id
    ),
    channel_entropy_tbl AS (
        SELECT
            c.account_id,
            -SUM(
                (c.channel_cnt * 1.0 / t.total_txns) *
                LN(c.channel_cnt * 1.0 / t.total_txns + 1e-9)
            ) AS channel_entropy
        FROM acct_channel c
        JOIN acct_totals t
          ON c.account_id = t.account_id
        GROUP BY c.account_id
    ),
    channel_share_tbl AS (
        SELECT
            c.account_id,
            SUM(CASE WHEN c.channel IN ('NEFT', 'IMPS', 'NTD', 'IPM')
                     THEN c.channel_cnt ELSE 0 END) * 1.0
                / NULLIF(t.total_txns, 0) AS neft_imps_share,
            SUM(CASE WHEN c.channel = 'ATM'
                     THEN c.channel_cnt ELSE 0 END) * 1.0
                / NULLIF(t.total_txns, 0) AS atm_share
        FROM acct_channel c
        JOIN acct_totals t
          ON c.account_id = t.account_id
        GROUP BY c.account_id, t.total_txns
    ),
    risky_mcc_tbl AS (
        SELECT
            account_id,
            SUM({risky_mcc_case}) * 1.0 / NULLIF(SUM(cnt), 0) AS risky_mcc_share_raw
        FROM acct_channel_mcc
        GROUP BY account_id
    )
    SELECT
        t.account_id,
        e.channel_entropy,
        s.neft_imps_share,
        s.atm_share,
        t.channel_count,
        t.max_channel_cnt * 1.0 / NULLIF(t.total_txns, 0) AS top_channel_share,
        r.risky_mcc_share_raw,
        t.total_txns
    FROM acct_totals t
    LEFT JOIN channel_entropy_tbl e
      ON t.account_id = e.account_id
    LEFT JOIN channel_share_tbl s
      ON t.account_id = s.account_id
    LEFT JOIN risky_mcc_tbl r
      ON t.account_id = r.account_id
""").df()
print(f"  Group B features: {feat_B.shape}  ({time.time()-t0:.1f}s)")
print("\n  Feature ranges (train):")
check_B = feat_B.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')
for col in [
    'channel_entropy',
    'neft_imps_share',
    'atm_share',
    'channel_count',
    'top_channel_share',
    'risky_mcc_share_raw'
]:
    mule_med = check_B.loc[check_B['is_mule'] == 1, col].median()
    legit_med = check_B.loc[check_B['is_mule'] == 0, col].median()
    print(f"    {col:<32}  mule={mule_med:.4f}  legit={legit_med:.4f}")
print("\n  ⚠  risky_mcc_share_raw uses full-dataset label info.")
print("     Block 16 replaces it with fold-safe computation per CV fold.")
feat_B.to_parquet(f'{OUT_DIR}/feat_B.parquet', index=False)
print(f"\n  Saved → {OUT_DIR}/feat_B.parquet  shape={feat_B.shape}")
print(f"\n{'='*60}")
print("4.4  SANITY CHECKS")
print(f"{'='*60}")
for name, df in [('feat_A', feat_A), ('feat_B', feat_B)]:
    null_pct = df.isnull().mean().max() * 100
    n_accs = df['account_id'].nunique()
    print(f"  {name}: {n_accs:,} accounts  max_null={null_pct:.2f}%  cols={list(df.columns)}")
assert feat_A['account_id'].nunique() == 160_153, "feat_A account count mismatch"
assert feat_B['account_id'].nunique() == 160_153, "feat_B account count mismatch"
print("\n  Account count assertions passed ✓")
print("\n" + "="*65)
print("BLOCK 4 COMPLETE ✓")
print("="*65)
print("\nOutputs:")
print(f"  feat_A.parquet   ({feat_A.shape[1]} cols)  → {OUT_DIR}/feat_A.parquet")
print(f"  feat_B.parquet   ({feat_B.shape[1]} cols)  → {OUT_DIR}/feat_B.parquet")
print(f"\nGlobals:")
print(f"  RISKY_MCCS_TXN = {RISKY_MCCS_TXN}")
import time
import numpy as np
import pandas as pd
OUT_DIR = '/content'
print(f"\n{'='*60}")
print("5.1  GROUP C — STRUCTURING & AMOUNTS")
print(f"{'='*60}")
t0 = time.time()
feat_C = con.execute("""
    SELECT
        account_id,
        -- Structuring: just below ₹50K reporting threshold
        AVG(
            CASE
                WHEN ABS(amount) >= 40000 AND ABS(amount) < 50000
                THEN 1.0 ELSE 0.0
            END
        ) AS structuring_share,
        -- Round amounts (multiples of ₹500) — negative signal from EDA
        AVG(
            CASE
                WHEN ABS(amount) > 0 AND ABS(amount) % 500 = 0
                THEN 1.0 ELSE 0.0
            END
        ) AS round_share,
        MEDIAN(ABS(amount)) AS median_txn_amount,
        AVG(ABS(amount))    AS avg_txn_amount,
        -- High-value transactions
        AVG(
            CASE
                WHEN ABS(amount) > 50000
                THEN 1.0 ELSE 0.0
            END
        ) AS high_value_share,
        -- Micro transactions
        AVG(
            CASE
                WHEN ABS(amount) < 100
                THEN 1.0 ELSE 0.0
            END
        ) AS micro_txn_share,
        -- Aggregate amount flows
        SUM(CASE WHEN txn_type = 'C' THEN ABS(amount) ELSE 0 END) AS total_credit_amt,
        SUM(CASE WHEN txn_type = 'D' THEN ABS(amount) ELSE 0 END) AS total_debit_amt,
        COUNT(*) AS txn_count
    FROM transactions
    GROUP BY account_id
""").df()
feat_C['credit_debit_ratio'] = (
    feat_C['total_credit_amt'] / (feat_C['total_debit_amt'] + 1)
).clip(0, 10)
print(f"  Group C features: {feat_C.shape}  ({time.time()-t0:.1f}s)")
print(f"\n{'='*60}")
print("5.2  TRAIN-RANGE CHECKS")
print(f"{'='*60}")
check_C = feat_C.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')
for col in [
    'structuring_share',
    'round_share',
    'median_txn_amount',
    'avg_txn_amount',
    'high_value_share',
    'micro_txn_share',
    'credit_debit_ratio',
]:
    mule_med  = check_C.loc[check_C['is_mule'] == 1, col].median()
    legit_med = check_C.loc[check_C['is_mule'] == 0, col].median()
    print(f"  {col:<24}  mule={mule_med:.4f}  legit={legit_med:.4f}")
feat_C.to_parquet(f'{OUT_DIR}/feat_C.parquet', index=False)
print(f"\n  Saved → {OUT_DIR}/feat_C.parquet  shape={feat_C.shape}")
print(f"\n{'='*60}")
print("5.4  SANITY CHECKS")
print(f"{'='*60}")
null_pct = feat_C.isnull().mean().max() * 100
n_accs   = feat_C['account_id'].nunique()
print(f"  feat_C: {n_accs:,} accounts  max_null={null_pct:.2f}%")
print(f"  cols   : {list(feat_C.columns)}")
assert n_accs == 160_153, "feat_C account count mismatch"
print("\n  Account count assertion passed ✓")
print("\n" + "="*65)
print("BLOCK 5 COMPLETE ✓")
print("="*65)
print("\nOutputs:")
print(f"  feat_C.parquet   ({feat_C.shape[1]} cols)  → {OUT_DIR}/feat_C.parquet")
import time
import numpy as np
import pandas as pd
OUT_DIR = '/content'
print(f"\n{'='*60}")
print("6.1  GROUP D — BALANCE & PASS-THROUGH")
print(f"{'='*60}")
t0 = time.time()
feat_D = con.execute("""
    WITH ordered AS (
        SELECT
            t.account_id,
            t.txn_type,
            ABS(t.amount)                              AS abs_amount,
            ta.balance_after_transaction               AS bal,
            CAST(t.transaction_timestamp AS TIMESTAMP) AS ts,
            LAG(ta.balance_after_transaction) OVER (
                PARTITION BY t.account_id
                ORDER BY t.transaction_timestamp
            ) AS prev_bal,
            LAG(t.txn_type) OVER (
                PARTITION BY t.account_id
                ORDER BY t.transaction_timestamp
            ) AS prev_type,
            DATEDIFF(
                'second',
                LAG(CAST(t.transaction_timestamp AS TIMESTAMP)) OVER (
                    PARTITION BY t.account_id
                    ORDER BY t.transaction_timestamp
                ),
                CAST(t.transaction_timestamp AS TIMESTAMP)
            ) AS secs_since_prev
        FROM transactions t
        JOIN txn_add ta
          ON t.transaction_id = ta.transaction_id
    ),
    per_account AS (
        SELECT
            account_id,
            AVG(ABS(bal))    AS avg_running_balance,
            STDDEV(ABS(bal)) AS std_running_balance,
            -- Near-zero balance after transaction
            AVG(
                CASE
                    WHEN ABS(bal) < 500 THEN 1.0
                    ELSE 0.0
                END
            ) AS near_zero_balance_share,
            -- Debit immediately after credit within 1 hour
            SUM(
                CASE
                    WHEN txn_type = 'D'
                     AND prev_type = 'C'
                     AND secs_since_prev <= 3600
                    THEN 1
                    ELSE 0
                END
            ) AS rapid_drain_raw,
            -- Avg seconds between a credit and the next debit
            AVG(
                CASE
                    WHEN txn_type = 'D' AND prev_type = 'C'
                    THEN secs_since_prev
                    ELSE NULL
                END
            ) AS avg_drain_seconds,
            COUNT(*) AS n_txns
        FROM ordered
        GROUP BY account_id
    )
    SELECT
        account_id,
        avg_running_balance,
        std_running_balance,
        near_zero_balance_share,
        rapid_drain_raw * 1.0 / NULLIF(n_txns, 0) AS rapid_drain_rate,
        avg_drain_seconds,
        CASE
            WHEN avg_running_balance > 0
            THEN std_running_balance / avg_running_balance
            ELSE NULL
        END AS balance_cv
    FROM per_account
""").df()
vol_tmp = con.execute("""
    SELECT
        account_id,
        SUM(ABS(amount)) AS total_volume
    FROM transactions
    GROUP BY account_id
""").df()
feat_D = feat_D.merge(vol_tmp, on='account_id', how='left')
feat_D['turnover_v3'] = feat_D['total_volume'] / (feat_D['avg_running_balance'] + 100)
feat_D.drop(columns=['total_volume'], inplace=True)
print(f"  Group D features: {feat_D.shape}  ({time.time()-t0:.1f}s)")
print(f"\n{'='*60}")
print("6.2  TRAIN-RANGE CHECKS")
print(f"{'='*60}")
check_D = feat_D.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')
for col in [
    'avg_running_balance',
    'std_running_balance',
    'near_zero_balance_share',
    'rapid_drain_rate',
    'avg_drain_seconds',
    'balance_cv',
    'turnover_v3',
]:
    mule_med  = check_D.loc[check_D['is_mule'] == 1, col].median()
    legit_med = check_D.loc[check_D['is_mule'] == 0, col].median()
    print(f"  {col:<26}  mule={mule_med:.4f}  legit={legit_med:.4f}")
feat_D.to_parquet(f'{OUT_DIR}/feat_D.parquet', index=False)
print(f"\n  Saved → {OUT_DIR}/feat_D.parquet  shape={feat_D.shape}")
print(f"\n{'='*60}")
print("6.4  SANITY CHECKS")
print(f"{'='*60}")
null_pct = feat_D.isnull().mean().max() * 100
n_accs   = feat_D['account_id'].nunique()
print(f"  feat_D: {n_accs:,} accounts  max_null={null_pct:.2f}%")
print(f"  cols   : {list(feat_D.columns)}")
assert n_accs == 160_153, "feat_D account count mismatch"
print("\n  Account count assertion passed ✓")
print("\n" + "="*65)
print("BLOCK 6 COMPLETE ✓")
print("="*65)
print("\nOutputs:")
print(f"  feat_D.parquet   ({feat_D.shape[1]} cols)  → {OUT_DIR}/feat_D.parquet")
import time
import numpy as np
import pandas as pd
OUT_DIR = '/content'
print(f"\n{'='*60}")
print("7.1  GROUP E — TEMPORAL & ACTIVITY")
print(f"{'='*60}")
t0 = time.time()
feat_E = con.execute("""
    WITH daily AS (
        SELECT
            account_id,
            CAST(transaction_timestamp AS DATE) AS txn_date,
            COUNT(*) AS daily_count
        FROM transactions
        GROUP BY account_id, txn_date
    ),
    span_stats AS (
        SELECT
            account_id,
            COUNT(DISTINCT txn_date) AS active_days,
            DATEDIFF('day', MIN(txn_date), MAX(txn_date)) + 1 AS calendar_days
        FROM daily
        GROUP BY account_id
    ),
    rolling AS (
        SELECT
            account_id,
            txn_date,
            SUM(daily_count) OVER (
                PARTITION BY account_id
                ORDER BY txn_date
                RANGE BETWEEN INTERVAL 90 DAYS PRECEDING AND CURRENT ROW
            ) AS roll_90d
        FROM daily
    ),
    peak_90 AS (
        SELECT
            account_id,
            MAX(roll_90d) AS peak_90d_count,
            AVG(roll_90d) AS avg_90d_count
        FROM rolling
        GROUP BY account_id
    ),
    time_feats AS (
        SELECT
            account_id,
            -- DuckDB: 0 = Sunday, 6 = Saturday
            AVG(CASE WHEN DAYOFWEEK(CAST(transaction_timestamp AS DATE)) IN (0, 6)
                     THEN 1.0 ELSE 0.0 END) AS weekend_share,
            AVG(CASE WHEN DAYOFWEEK(CAST(transaction_timestamp AS DATE)) = 0
                     THEN 1.0 ELSE 0.0 END) AS sunday_share,
            AVG(CASE WHEN HOUR(CAST(transaction_timestamp AS TIMESTAMP))
                          NOT BETWEEN 6 AND 21
                     THEN 1.0 ELSE 0.0 END) AS offhours_share,
            AVG(CASE WHEN DAY(CAST(transaction_timestamp AS DATE))
                          IN (1,2,3,4,5,25,26,27,28,29,30,31)
                     THEN 1.0 ELSE 0.0 END) AS month_boundary_share,
            MIN(CAST(transaction_timestamp AS TIMESTAMP)) AS first_txn_ts,
            MAX(CAST(transaction_timestamp AS TIMESTAMP)) AS last_txn_ts
        FROM transactions
        GROUP BY account_id
    )
    SELECT
        tf.account_id,
        tf.weekend_share,
        tf.sunday_share,
        tf.offhours_share,
        tf.month_boundary_share,
        tf.first_txn_ts,
        tf.last_txn_ts,
        ss.active_days,
        ss.calendar_days,
        ss.active_days * 1.0 / NULLIF(ss.calendar_days, 0) AS activity_density,
        p.peak_90d_count,
        p.avg_90d_count,
        p.peak_90d_count * 1.0 / NULLIF(p.avg_90d_count, 0) AS peak_to_avg_90d
    FROM time_feats tf
    LEFT JOIN span_stats ss ON tf.account_id = ss.account_id
    LEFT JOIN peak_90     p ON tf.account_id = p.account_id
""").df()
feat_E = feat_E.merge(
    accounts[['account_id', 'account_opening_date']],
    on='account_id',
    how='left'
)
feat_E['first_txn_ts'] = pd.to_datetime(feat_E['first_txn_ts'], errors='coerce')
feat_E['last_txn_ts']  = pd.to_datetime(feat_E['last_txn_ts'],  errors='coerce')
feat_E['days_to_first_txn'] = (
    feat_E['first_txn_ts'] - feat_E['account_opening_date']
).dt.days.clip(lower=0)
feat_E.drop(columns=['account_opening_date'], inplace=True)
print(f"  Group E features: {feat_E.shape}  ({time.time()-t0:.1f}s)")
print(f"\n{'='*60}")
print("7.2  TRAIN-RANGE CHECKS")
print(f"{'='*60}")
check_E = feat_E.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')
for col in [
    'weekend_share',
    'sunday_share',
    'offhours_share',
    'month_boundary_share',
    'active_days',
    'activity_density',
    'peak_to_avg_90d',
    'days_to_first_txn',
]:
    mule_med  = check_E.loc[check_E['is_mule'] == 1, col].median()
    legit_med = check_E.loc[check_E['is_mule'] == 0, col].median()
    print(f"  {col:<24}  mule={mule_med:.4f}  legit={legit_med:.4f}")
feat_E.to_parquet(f'{OUT_DIR}/feat_E.parquet', index=False)
print(f"\n  Saved → {OUT_DIR}/feat_E.parquet  shape={feat_E.shape}")
print(f"\n{'='*60}")
print("7.4  SANITY CHECKS")
print(f"{'='*60}")
null_pct = feat_E.isnull().mean().max() * 100
n_accs   = feat_E['account_id'].nunique()
print(f"  feat_E: {n_accs:,} accounts  max_null={null_pct:.2f}%")
print(f"  cols   : {list(feat_E.columns)}")
assert n_accs == 160_153, "feat_E account count mismatch"
print("\n  Account count assertion passed ✓")
print("\n" + "="*65)
print("BLOCK 7 COMPLETE ✓")
print("="*65)
print("\nOutputs:")
print(f"  feat_E.parquet   ({feat_E.shape[1]} cols)  → {OUT_DIR}/feat_E.parquet")
import time
import numpy as np
import pandas as pd
OUT_DIR = '/content'
TMP_DIR = '/tmp'
print(f"\n{'='*60}")
print("8.1  GROUP F — GEOGRAPHIC & IP")
print(f"{'='*60}")
t0 = time.time()
ip_freq = con.execute("""
    SELECT
        ta.ip_address,
        COUNT(DISTINCT t.account_id) AS ip_account_count
    FROM transactions t
    JOIN txn_add ta
      ON t.transaction_id = ta.transaction_id
    WHERE ta.ip_address IS NOT NULL
      AND ta.ip_address != ''
    GROUP BY ta.ip_address
""").df()
ip_freq.to_parquet(f'{TMP_DIR}/ip_freq.parquet', index=False)
feat_F = con.execute(f"""
    WITH geo AS (
        SELECT
            t.account_id,
            COUNT(DISTINCT ta.ip_address) AS unique_ips,
            STDDEV(ta.latitude)           AS lat_std,
            STDDEV(ta.longitude)          AS lon_std,
            AVG(
                CASE
                    WHEN ta.ip_address IS NULL OR ta.ip_address = ''
                    THEN 1.0 ELSE 0.0
                END
            ) AS ip_null_rate,
            AVG(ipf.ip_account_count) AS avg_ip_sharing
        FROM transactions t
        JOIN txn_add ta
          ON t.transaction_id = ta.transaction_id
        LEFT JOIN read_parquet('{TMP_DIR}/ip_freq.parquet') ipf
          ON ta.ip_address = ipf.ip_address
        GROUP BY t.account_id
    )
    SELECT
        account_id,
        unique_ips,
        lat_std,
        lon_std,
        ip_null_rate,
        avg_ip_sharing,
        SQRT(
            POWER(COALESCE(lat_std, 0), 2) +
            POWER(COALESCE(lon_std, 0), 2)
        ) AS geo_spread
    FROM geo
""").df()
print(f"  Group F features: {feat_F.shape}  ({time.time()-t0:.1f}s)")
print(f"\n{'='*60}")
print("8.2  TRAIN-RANGE CHECKS")
print(f"{'='*60}")
check_F = feat_F.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')
for col in [
    'unique_ips',
    'lat_std',
    'lon_std',
    'ip_null_rate',
    'avg_ip_sharing',
    'geo_spread',
]:
    mule_med  = check_F.loc[check_F['is_mule'] == 1, col].median()
    legit_med = check_F.loc[check_F['is_mule'] == 0, col].median()
    print(f"  {col:<18}  mule={mule_med:.4f}  legit={legit_med:.4f}")
feat_F.to_parquet(f'{OUT_DIR}/feat_F.parquet', index=False)
print(f"\n  Saved → {OUT_DIR}/feat_F.parquet  shape={feat_F.shape}")
print(f"\n{'='*60}")
print("8.4  SANITY CHECKS")
print(f"{'='*60}")
null_pct = feat_F.isnull().mean().max() * 100
n_accs   = feat_F['account_id'].nunique()
print(f"  feat_F: {n_accs:,} accounts  max_null={null_pct:.2f}%")
print(f"  cols   : {list(feat_F.columns)}")
assert n_accs == 160_153, "feat_F account count mismatch"
print("\n  Account count assertion passed ✓")
print("\n" + "="*65)
print("BLOCK 8 COMPLETE ✓")
print("="*65)
print("\nOutputs:")
print(f"  feat_F.parquet   ({feat_F.shape[1]} cols)  → {OUT_DIR}/feat_F.parquet")
import time
import numpy as np
import pandas as pd
OUT_DIR = '/content'
print(f"\n{'='*60}")
print("9.1  GROUP G — ACCOUNT & CUSTOMER STATIC")
print(f"{'='*60}")
t0 = time.time()
feat_G = accounts[
    ['account_id', 'account_opening_date', 'branch_code', 'branch_pin',
     'avg_balance', 'last_mobile_update_date', 'kyc_compliant',
     'rural_branch', 'freeze_date']
].copy()
feat_G = feat_G.merge(cust_link, on='account_id', how='left')
feat_G = feat_G.merge(
    customers[['customer_id', 'date_of_birth', 'relationship_start_date',
               'customer_pin', 'permanent_pin']],
    on='customer_id',
    how='left'
)
feat_G = feat_G.merge(
    product_det[['customer_id', 'sa_count', 'loan_count', 'cc_count']],
    on='customer_id',
    how='left'
)
feat_G = feat_G.merge(
    demographics[['customer_id', 'nri_flag']],
    on='customer_id',
    how='left'
)
feat_G = feat_G.merge(
    accounts_add[['account_id', 'scheme_code']],
    on='account_id',
    how='left'
)
acc_per_cust = (
    cust_link.groupby('customer_id')['account_id']
    .nunique()
    .rename('acc_per_customer')
)
feat_G = feat_G.merge(acc_per_cust, on='customer_id', how='left')
feat_G['account_age_days'] = (
    REF_TS - feat_G['account_opening_date']
).dt.days.clip(lower=0)
feat_G['days_since_mobile_update'] = (
    REF_TS - feat_G['last_mobile_update_date']
).dt.days
feat_G['no_mobile_update'] = feat_G['last_mobile_update_date'].isna().astype(int)
feat_G['pin_mismatch'] = (
    feat_G['customer_pin'].astype(str).str[:3] !=
    feat_G['branch_pin'].fillna(0).astype(int).astype(str).str[:3]
).astype(int)
if feat_G['nri_flag'].dtype == 'O':
    feat_G['nri_flag'] = feat_G['nri_flag'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
else:
    feat_G['nri_flag'] = feat_G['nri_flag'].fillna(0).astype(int)
if feat_G['kyc_compliant'].dtype == 'O':
    feat_G['kyc_compliant'] = feat_G['kyc_compliant'].map({'Y': 1, 'N': 0}).fillna(0)
else:
    feat_G['kyc_compliant'] = feat_G['kyc_compliant'].fillna(0)
if feat_G['rural_branch'].dtype == 'O':
    feat_G['rural_branch'] = feat_G['rural_branch'].map({'Y': 1, 'N': 0}).fillna(0)
else:
    feat_G['rural_branch'] = feat_G['rural_branch'].fillna(0)
scheme_risk = {
    'PMJJBY': 1.21,
    'SCSS'  : 1.14,
    'REGULAR': 1.00,
    'PMSBY' : 0.99,
    'PMJDY' : 0.99,
    'SSA'   : 0.95,
    'APY'   : 0.81,
}
feat_G['scheme_risk'] = feat_G['scheme_code'].map(scheme_risk).fillna(1.0)
feat_G['has_freeze_date'] = feat_G['freeze_date'].notna().astype(int)
feat_G = feat_G[
    ['account_id', 'account_age_days', 'days_since_mobile_update',
     'no_mobile_update', 'pin_mismatch', 'acc_per_customer',
     'sa_count', 'loan_count', 'cc_count', 'nri_flag',
     'scheme_risk', 'kyc_compliant', 'rural_branch', 'has_freeze_date']
].copy()
feat_G.fillna({
    'sa_count': 0,
    'loan_count': 0,
    'cc_count': 0,
    'acc_per_customer': 1,
}, inplace=True)
print(f"  Group G features: {feat_G.shape}  ({time.time()-t0:.1f}s)")
print(f"\n{'='*60}")
print("9.2  TRAIN-RANGE CHECKS")
print(f"{'='*60}")
check_G = feat_G.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')
for col in [
    'account_age_days',
    'days_since_mobile_update',
    'no_mobile_update',
    'pin_mismatch',
    'acc_per_customer',
    'sa_count',
    'loan_count',
    'cc_count',
    'nri_flag',
    'scheme_risk',
    'kyc_compliant',
    'rural_branch',
    'has_freeze_date',
]:
    mule_med  = check_G.loc[check_G['is_mule'] == 1, col].median()
    legit_med = check_G.loc[check_G['is_mule'] == 0, col].median()
    print(f"  {col:<24}  mule={mule_med:.4f}  legit={legit_med:.4f}")
print("\n  ⚠  has_freeze_date is intentionally preserved here to match")
print("     the original pipeline stage. It is removed later in final cleanup.")
feat_G.to_parquet(f'{OUT_DIR}/feat_G.parquet', index=False)
print(f"\n  Saved → {OUT_DIR}/feat_G.parquet  shape={feat_G.shape}")
print(f"\n{'='*60}")
print("9.4  SANITY CHECKS")
print(f"{'='*60}")
null_pct = feat_G.isnull().mean().max() * 100
n_accs   = feat_G['account_id'].nunique()
print(f"  feat_G: {n_accs:,} accounts  max_null={null_pct:.2f}%")
print(f"  cols   : {list(feat_G.columns)}")
assert n_accs == 160_153, "feat_G account count mismatch"
print("\n  Account count assertion passed ✓")
print("\n" + "="*65)
print("BLOCK 9 COMPLETE ✓")
print("="*65)
print("\nOutputs:")
print(f"  feat_G.parquet   ({feat_G.shape[1]} cols)  → {OUT_DIR}/feat_G.parquet")
import time
import numpy as np
import pandas as pd
OUT_DIR = '/content'
print(f"\n{'='*60}")
print("10.1  GROUP H — BRANCH & NETWORK CONTEXT")
print(f"{'='*60}")
t0 = time.time()
feat_H = accounts[['account_id', 'branch_code']].copy()
feat_H = feat_H.merge(
    branch[['branch_code', 'branch_employee_count', 'branch_type',
            'branch_turnover', 'branch_asset_size']],
    on='branch_code',
    how='left'
)
branch_acc_count = (
    accounts.groupby('branch_code')['account_id']
    .count()
    .rename('branch_acc_count')
)
feat_H = feat_H.merge(branch_acc_count, on='branch_code', how='left')
feat_H['accounts_per_employee'] = (
    feat_H['branch_acc_count'] /
    feat_H['branch_employee_count'].clip(lower=1)
)
# Branch type one-hot encoding
feat_H['branch_type_urban'] = (
    feat_H['branch_type'] == 'urban'
).astype(int)
feat_H['branch_type_semiurban'] = (
    feat_H['branch_type'] == 'semi-urban'
).astype(int)
feat_H['branch_type_rural'] = (
    feat_H['branch_type'] == 'rural'
).astype(int)
feat_H['log_branch_turnover'] = np.log1p(
    feat_H['branch_turnover'].clip(lower=0)
)
feat_H['log_branch_asset_size'] = np.log1p(
    feat_H['branch_asset_size'].clip(lower=0)
)
feat_H = feat_H[
    [
        'account_id',
        'accounts_per_employee',
        'branch_type_urban',
        'branch_type_semiurban',
        'branch_type_rural',
        'log_branch_turnover',
        'log_branch_asset_size',
    ]
].copy()
print(f"  Group H features: {feat_H.shape}  ({time.time()-t0:.1f}s)")
print(f"\n{'='*60}")
print("10.2  TRAIN-RANGE CHECKS")
print(f"{'='*60}")
check_H = feat_H.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')
for col in [
    'accounts_per_employee',
    'branch_type_urban',
    'branch_type_semiurban',
    'branch_type_rural',
    'log_branch_turnover',
    'log_branch_asset_size',
]:
    mule_med  = check_H.loc[check_H['is_mule'] == 1, col].median()
    legit_med = check_H.loc[check_H['is_mule'] == 0, col].median()
    print(f"  {col:<24}  mule={mule_med:.4f}  legit={legit_med:.4f}")
print("\n  NOTE: target-derived branch risk is NOT built here.")
print("        Any branch risk encoding must be added fold-safely at model time.")
feat_H.to_parquet(f'{OUT_DIR}/feat_H.parquet', index=False)
print(f"\n  Saved → {OUT_DIR}/feat_H.parquet  shape={feat_H.shape}")
print(f"\n{'='*60}")
print("10.4  SANITY CHECKS")
print(f"{'='*60}")
null_pct = feat_H.isnull().mean().max() * 100
n_accs   = feat_H['account_id'].nunique()
print(f"  feat_H: {n_accs:,} accounts  max_null={null_pct:.2f}%")
print(f"  cols   : {list(feat_H.columns)}")
assert n_accs == 160_153, "feat_H account count mismatch"
print("\n  Account count assertion passed ✓")
print("\n" + "="*65)
print("BLOCK 10 COMPLETE ✓")
print("="*65)
print("\nOutputs:")
print(f"  feat_H.parquet   ({feat_H.shape[1]} cols)  → {OUT_DIR}/feat_H.parquet")
import os
import time
import numpy as np
import pandas as pd
OUT_DIR = '/content'
print(f"\n{'='*60}")
print("11.1  ASSEMBLE BASE FEATURE MATRIX")
print(f"{'='*60}")
t0 = time.time()
if 'feat_A' not in globals():
    feat_A = pd.read_parquet(f'{OUT_DIR}/feat_A.parquet')
if 'feat_B' not in globals():
    feat_B = pd.read_parquet(f'{OUT_DIR}/feat_B.parquet')
if 'feat_C' not in globals():
    feat_C = pd.read_parquet(f'{OUT_DIR}/feat_C.parquet')
if 'feat_D' not in globals():
    feat_D = pd.read_parquet(f'{OUT_DIR}/feat_D.parquet')
if 'feat_E' not in globals():
    feat_E = pd.read_parquet(f'{OUT_DIR}/feat_E.parquet')
if 'feat_F' not in globals():
    feat_F = pd.read_parquet(f'{OUT_DIR}/feat_F.parquet')
if 'feat_G' not in globals():
    feat_G = pd.read_parquet(f'{OUT_DIR}/feat_G.parquet')
if 'feat_H' not in globals():
    feat_H = pd.read_parquet(f'{OUT_DIR}/feat_H.parquet')
if 'risky_mcc_share_raw' in feat_B.columns:
    feat_B = feat_B.rename(columns={'risky_mcc_share_raw': 'risky_mcc_share'})
feat = all_ids.copy()
for grp_name, grp_df in [
    ('A', feat_A),
    ('B', feat_B),
    ('C', feat_C),
    ('D', feat_D),
    ('E', feat_E),
    ('F', feat_F),
    ('G', feat_G),
    ('H', feat_H),
]:
    feat = feat.merge(grp_df, on='account_id', how='left')
    print(f"  After Group {grp_name}: {feat.shape}")
drop_cols = [
    'first_txn_ts',
    'last_txn_ts',
    'calendar_days',
    'peak_90d_count',
    'avg_90d_count',
    'total_credit_amt',
    'total_debit_amt',
    'avg_running_balance',
    'std_running_balance',
    'total_txns',
]
feat.drop(columns=[c for c in drop_cols if c in feat.columns], inplace=True)
print(f"\nBase feature matrix: {feat.shape}")
print(f"Feature columns ({feat.shape[1]-1}):")
print([c for c in feat.columns if c != 'account_id'])
miss = feat.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss) > 0:
    print(f"\nMissing values before fill:")
    print(miss.to_string())
else:
    print("\nNo missing values before fill ✓")
print(f"\n{'='*60}")
print("11.2  FILL MISSING VALUES")
print(f"{'='*60}")
static_cols = [
    'account_id',
    'account_age_days',
    'days_since_mobile_update',
    'no_mobile_update',
    'pin_mismatch',
    'acc_per_customer',
    'sa_count',
    'loan_count',
    'cc_count',
    'nri_flag',
    'scheme_risk',
    'kyc_compliant',
    'rural_branch',
    'has_freeze_date',
    'accounts_per_employee',
    'branch_type_urban',
    'branch_type_semiurban',
    'branch_type_rural',
    'log_branch_turnover',
    'log_branch_asset_size',
]
txn_based = [c for c in feat.columns if c not in static_cols]
for col in txn_based:
    n_null = feat[col].isnull().sum()
    if n_null > 0:
        fill_val = feat[col].median()
        feat[col].fillna(fill_val, inplace=True)
        print(f"  {col:<24} filled {n_null:>6,} nulls with median={fill_val:.4f}")
static_fills = {
    'days_since_mobile_update': feat['days_since_mobile_update'].median()
                                if 'days_since_mobile_update' in feat.columns else 0,
    'no_mobile_update': 1,
    'acc_per_customer': 1,
    'sa_count': 0,
    'loan_count': 0,
    'cc_count': 0,
    'nri_flag': 0,
    'scheme_risk': 1.0,
    'kyc_compliant': 0,
    'rural_branch': 0,
    'has_freeze_date': 0,
    'accounts_per_employee': feat['accounts_per_employee'].median()
                             if 'accounts_per_employee' in feat.columns else 0,
    'branch_type_urban': 0,
    'branch_type_semiurban': 0,
    'branch_type_rural': 0,
    'log_branch_turnover': 0,
    'log_branch_asset_size': 0,
}
for col, val in static_fills.items():
    if col in feat.columns:
        n_null = feat[col].isnull().sum()
        if n_null > 0:
            feat[col].fillna(val, inplace=True)
            print(f"  {col:<24} filled {n_null:>6,} nulls with value={val}")
null_remaining = int(feat.isnull().sum().sum())
if null_remaining > 0:
    feat.fillna(0, inplace=True)
    print(f"  Remaining {null_remaining:,} nulls filled with 0")
print(f"\nMissing after fill: {feat.isnull().sum().sum()}")
print(f"\n{'='*60}")
print("11.3  SPLIT TRAIN / TEST")
print(f"{'='*60}")
train_ids = set(labels['account_id'])
test_ids  = set(test_acc['account_id'])
feat_train = feat[feat['account_id'].isin(train_ids)].copy()
feat_test  = feat[feat['account_id'].isin(test_ids)].copy()
feat_train = feat_train.merge(
    labels[['account_id', 'is_mule', 'mule_flag_date', 'alert_reason']],
    on='account_id',
    how='left'
)
print(f"  Train features: {feat_train.shape}")
print(f"  Test  features: {feat_test.shape}")
print(f"  Mule rate      : {feat_train['is_mule'].mean():.4f}")
feat_train.to_parquet(f'{OUT_DIR}/features_train.parquet', index=False)
feat_test.to_parquet(f'{OUT_DIR}/features_test.parquet', index=False)
print(f"\nSaved:")
print(f"  {OUT_DIR}/features_train.parquet")
print(f"  {OUT_DIR}/features_test.parquet")
print(f"\n{'='*60}")
print("11.4  SANITY CHECKS")
print(f"{'='*60}")
assert feat_train['account_id'].nunique() == len(labels), "Train account count mismatch"
assert feat_test['account_id'].nunique() == len(test_acc), "Test account count mismatch"
train_feature_cols = [
    c for c in feat_train.columns
    if c not in ['is_mule', 'mule_flag_date', 'alert_reason']
]
assert feat_train[train_feature_cols].isnull().sum().sum() == 0, (
    "Nulls remain in train feature columns"
)
assert feat_test.isnull().sum().sum() == 0, (
    "Nulls remain in test feature columns"
)
print("  Account count assertions passed ✓")
print("  Feature no-null assertions passed ✓")
print("\nAllowed nulls in train metadata:")
for col in ['mule_flag_date', 'alert_reason']:
    if col in feat_train.columns:
        print(f"  {col:<15} {feat_train[col].isnull().sum():>8,}")
print("\n" + "="*65)
print("BLOCK 11 COMPLETE ✓")
print("="*65)
print("\nOutputs:")
print(f"  features_train.parquet  shape={feat_train.shape}  → {OUT_DIR}/features_train.parquet")
print(f"  features_test.parquet   shape={feat_test.shape}   → {OUT_DIR}/features_test.parquet")
import time
import warnings
import numpy as np
import pandas as pd
import duckdb
warnings.filterwarnings('ignore')
OUT_DIR = '/content'
TMP_DIR = '/tmp'
t_start = time.time()
print(f"\n{'='*60}")
print("12.0  FEATURE PACK v2")
print(f"{'='*60}")
if 'con' not in globals():
    con = duckdb.connect()
    con.execute("""
        CREATE OR REPLACE VIEW transactions AS
        SELECT * FROM read_parquet(
            '/content/nfpc_data/transactions/batch-*/part_*.parquet')
    """)
    con.execute("""
        CREATE OR REPLACE VIEW txn_add AS
        SELECT * FROM read_parquet(
            '/content/nfpc_data/transactions_additional/batch-*/part_*.parquet')
    """)
    print("DuckDB views rebuilt ✓")
if 'feat_train' not in globals():
    feat_train = pd.read_parquet(f'{OUT_DIR}/features_train.parquet')
if 'feat_test' not in globals():
    feat_test = pd.read_parquet(f'{OUT_DIR}/features_test.parquet')
all_ids = pd.concat([
    feat_train[['account_id']],
    feat_test[['account_id']]
], ignore_index=True).drop_duplicates()
all_ids.to_parquet(f'{TMP_DIR}/all_ids.parquet', index=False)
print(f"Total accounts: {len(all_ids):,}")
print("\n" + "="*55)
print("12.1  PACK 1 — RECENT-WINDOW RATIOS")
print("="*55)
pack1 = con.execute("""
WITH ids AS (
    SELECT account_id FROM read_parquet('/tmp/all_ids.parquet')
),
txns AS (
    SELECT
        t.account_id,
        CAST(t.transaction_timestamp AS TIMESTAMP) AS ts,
        t.txn_type,
        ABS(t.amount) AS amt,
        ta.balance_after_transaction AS bal
    FROM transactions t
    JOIN txn_add ta ON t.transaction_id = ta.transaction_id
    JOIN ids i      ON t.account_id = i.account_id
),
anchors AS (
    SELECT account_id, MAX(ts) AS anchor_ts
    FROM txns
    GROUP BY account_id
),
lifetime AS (
    SELECT
        t.account_id,
        COUNT(*)                                      AS life_txn_count,
        SUM(CASE WHEN txn_type='C' THEN amt END)      AS life_credit_vol,
        SUM(CASE WHEN txn_type='D' THEN amt END)      AS life_debit_vol,
        COUNT(DISTINCT CAST(ts AS DATE))              AS life_active_days,
        SUM(CASE WHEN ABS(bal) < 500 AND txn_type='D'
                 THEN 1 ELSE 0 END)                   AS life_near_zero,
        AVG(CASE WHEN txn_type='C' THEN amt END)      AS life_avg_credit,
        STDDEV(CASE WHEN txn_type='C' THEN amt END)   AS life_std_credit
    FROM txns t
    GROUP BY t.account_id
),
w30 AS (
    SELECT
        t.account_id,
        COUNT(*)                                      AS w30_txn_count,
        SUM(CASE WHEN txn_type='C' THEN amt END)      AS w30_credit_vol,
        SUM(CASE WHEN txn_type='D' THEN amt END)      AS w30_debit_vol,
        COUNT(DISTINCT CAST(ts AS DATE))              AS w30_active_days,
        SUM(CASE WHEN ABS(bal) < 500 AND txn_type='D'
                 THEN 1 ELSE 0 END)                   AS w30_near_zero,
        COUNT(DISTINCT DATE_TRUNC('week', ts))        AS w30_active_weeks
    FROM txns t
    JOIN anchors a ON t.account_id = a.account_id
    WHERE t.ts >= a.anchor_ts - INTERVAL 30 DAY
    GROUP BY t.account_id
),
w90 AS (
    SELECT
        t.account_id,
        COUNT(*)                                      AS w90_txn_count,
        SUM(CASE WHEN txn_type='C' THEN amt END)      AS w90_credit_vol,
        SUM(CASE WHEN txn_type='D' THEN amt END)      AS w90_debit_vol,
        COUNT(DISTINCT CAST(ts AS DATE))              AS w90_active_days,
        SUM(CASE WHEN ABS(bal) < 500 AND txn_type='D'
                 THEN 1 ELSE 0 END)                   AS w90_near_zero,
        AVG(CASE WHEN txn_type='C' THEN amt END)      AS w90_avg_credit,
        SUM(CASE WHEN amt > 50000 AND txn_type='C'
                 THEN 1 ELSE 0 END)                   AS w90_large_credits
    FROM txns t
    JOIN anchors a ON t.account_id = a.account_id
    WHERE t.ts >= a.anchor_ts - INTERVAL 90 DAY
    GROUP BY t.account_id
)
SELECT
    ids.account_id,
    COALESCE(w30.w30_txn_count, 0)            AS w30_txn_count,
    COALESCE(w90.w90_txn_count, 0)            AS w90_txn_count,
    COALESCE(w30.w30_near_zero, 0)            AS w30_near_zero_count,
    COALESCE(w90.w90_near_zero, 0)            AS w90_near_zero_count,
    COALESCE(w90.w90_large_credits, 0)        AS w90_large_credit_count,
    CASE WHEN l.life_txn_count > 0
         THEN COALESCE(w90.w90_txn_count, 0) * 1.0 / l.life_txn_count
         ELSE 0 END                           AS w90_txn_concentration,
    CASE WHEN l.life_txn_count > 0
         THEN COALESCE(w30.w30_txn_count, 0) * 1.0 / l.life_txn_count
         ELSE 0 END                           AS w30_txn_concentration,
    CASE WHEN l.life_credit_vol > 0
         THEN COALESCE(w90.w90_credit_vol, 0) / l.life_credit_vol
         ELSE 0 END                           AS w90_credit_concentration,
    CASE WHEN COALESCE(w90.w90_txn_count, 0) > 0
         THEN COALESCE(w90.w90_near_zero, 0) * 1.0 / w90.w90_txn_count
         ELSE 0 END                           AS w90_near_zero_rate,
    CASE WHEN COALESCE(w30.w30_txn_count, 0) > 0
         THEN COALESCE(w30.w30_near_zero, 0) * 1.0 / w30.w30_txn_count
         ELSE 0 END                           AS w30_near_zero_rate,
    CASE WHEN l.life_txn_count > 0
              AND (l.life_near_zero * 1.0 / l.life_txn_count) > 0
         THEN (COALESCE(w90.w90_near_zero, 0) * 1.0 /
               NULLIF(w90.w90_txn_count, 0)) /
              (l.life_near_zero * 1.0 / l.life_txn_count)
         ELSE 0 END                           AS w90_near_zero_uplift,
    CASE WHEN l.life_active_days > 0
         THEN COALESCE(w90.w90_active_days, 0) * 1.0 / l.life_active_days
         ELSE 0 END                           AS w90_activity_concentration,
    CASE WHEN l.life_avg_credit > 0 AND w90.w90_avg_credit IS NOT NULL
         THEN w90.w90_avg_credit / l.life_avg_credit
         ELSE 1 END                           AS w90_avg_credit_ratio,
    CASE WHEN l.life_std_credit > 0 AND w90.w90_avg_credit IS NOT NULL
         THEN (w90.w90_avg_credit - l.life_avg_credit) / l.life_std_credit
         ELSE 0 END                           AS w90_credit_zscore,
    CASE WHEN COALESCE(w90.w90_txn_count, 0) > 0
         THEN COALESCE(w30.w30_txn_count, 0) * 1.0 / w90.w90_txn_count
         ELSE 0 END                           AS w30_w90_acceleration
FROM ids
LEFT JOIN lifetime l ON ids.account_id = l.account_id
LEFT JOIN w30        ON ids.account_id = w30.account_id
LEFT JOIN w90        ON ids.account_id = w90.account_id
""").df()
print(f"Pack 1: {pack1.shape}  ({time.time()-t_start:.0f}s)")
print(pack1.describe().round(3).to_string())
print("\n" + "="*55)
print("12.2  PACK 2 — FUND RESIDENCE TIME")
print("="*55)
t2 = time.time()
pack2 = con.execute("""
WITH ids AS (
    SELECT account_id FROM read_parquet('/tmp/all_ids.parquet')
),
txns AS (
    SELECT
        t.account_id,
        CAST(t.transaction_timestamp AS TIMESTAMP) AS ts,
        t.txn_type,
        ABS(t.amount) AS amt,
        ta.balance_after_transaction AS bal,
        ROW_NUMBER() OVER (
            PARTITION BY t.account_id
            ORDER BY CAST(t.transaction_timestamp AS TIMESTAMP)
        ) AS rn
    FROM transactions t
    JOIN txn_add ta ON t.transaction_id = ta.transaction_id
    JOIN ids i      ON t.account_id = i.account_id
),
with_next AS (
    SELECT
        account_id, ts, txn_type, amt, bal, rn,
        LEAD(ts)       OVER (PARTITION BY account_id ORDER BY rn) AS next_ts,
        LEAD(txn_type) OVER (PARTITION BY account_id ORDER BY rn) AS next_type,
        LEAD(amt)      OVER (PARTITION BY account_id ORDER BY rn) AS next_amt,
        LEAD(bal)      OVER (PARTITION BY account_id ORDER BY rn) AS next_bal
    FROM txns
),
credit_to_debit AS (
    SELECT
        account_id,
        EPOCH(next_ts - ts) / 3600.0 AS lag_hours
    FROM with_next
    WHERE txn_type = 'C'
      AND next_type = 'D'
      AND next_ts IS NOT NULL
),
same_day AS (
    SELECT
        account_id,
        COUNT(*) AS same_day_drain_count
    FROM with_next
    WHERE txn_type = 'C'
      AND next_type = 'D'
      AND CAST(ts AS DATE) = CAST(next_ts AS DATE)
      AND next_ts IS NOT NULL
    GROUP BY account_id
),
within_1h AS (
    SELECT account_id, COUNT(*) AS drain_within_1h
    FROM credit_to_debit
    WHERE lag_hours <= 1
    GROUP BY account_id
),
within_6h AS (
    SELECT account_id, COUNT(*) AS drain_within_6h
    FROM credit_to_debit
    WHERE lag_hours <= 6
    GROUP BY account_id
),
within_24h AS (
    SELECT account_id, COUNT(*) AS drain_within_24h
    FROM credit_to_debit
    WHERE lag_hours <= 24
    GROUP BY account_id
),
residence_stats AS (
    SELECT
        account_id,
        COUNT(*) AS n_credit_debit_pairs,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lag_hours)
            AS median_residence_hours,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY lag_hours)
            AS p25_residence_hours,
        AVG(lag_hours) AS avg_residence_hours,
        MIN(lag_hours) AS min_residence_hours
    FROM credit_to_debit
    GROUP BY account_id
),
post_credit_drawdown AS (
    SELECT
        account_id,
        AVG(CASE WHEN txn_type='C' AND next_type='D'
                 THEN (amt - next_amt) / NULLIF(amt, 0)
                 ELSE NULL END)               AS avg_post_credit_drawdown,
        SUM(CASE WHEN txn_type='C' AND next_type='D'
                      AND ABS(next_bal) < 500
                 THEN 1 ELSE 0 END) * 1.0 /
        NULLIF(SUM(CASE WHEN txn_type='C' THEN 1 ELSE 0 END), 0)
                                              AS large_credit_to_zero_rate
    FROM with_next
    GROUP BY account_id
),
credit_counts AS (
    SELECT account_id, COUNT(*) AS n_credits
    FROM txns
    WHERE txn_type = 'C'
    GROUP BY account_id
)
SELECT
    ids.account_id,
    COALESCE(rs.n_credit_debit_pairs, 0)      AS n_credit_debit_pairs,
    COALESCE(rs.median_residence_hours, -1)   AS median_residence_hours,
    COALESCE(rs.p25_residence_hours, -1)      AS p25_residence_hours,
    COALESCE(rs.avg_residence_hours, -1)      AS avg_residence_hours,
    COALESCE(rs.min_residence_hours, -1)      AS min_residence_hours,
    COALESCE(sd.same_day_drain_count, 0)      AS same_day_drain_count,
    CASE WHEN cc.n_credits > 0
         THEN COALESCE(sd.same_day_drain_count, 0) * 1.0 / cc.n_credits
         ELSE 0 END                           AS same_day_drain_rate,
    CASE WHEN cc.n_credits > 0
         THEN COALESCE(w1h.drain_within_1h, 0) * 1.0 / cc.n_credits
         ELSE 0 END                           AS drain_rate_1h,
    CASE WHEN cc.n_credits > 0
         THEN COALESCE(w6h.drain_within_6h, 0) * 1.0 / cc.n_credits
         ELSE 0 END                           AS drain_rate_6h,
    CASE WHEN cc.n_credits > 0
         THEN COALESCE(w24h.drain_within_24h, 0) * 1.0 / cc.n_credits
         ELSE 0 END                           AS drain_rate_24h,
    COALESCE(pcd.avg_post_credit_drawdown, 0) AS avg_post_credit_drawdown,
    COALESCE(pcd.large_credit_to_zero_rate,0) AS large_credit_to_zero_rate
FROM ids
LEFT JOIN residence_stats      rs   ON ids.account_id = rs.account_id
LEFT JOIN same_day             sd   ON ids.account_id = sd.account_id
LEFT JOIN within_1h            w1h  ON ids.account_id = w1h.account_id
LEFT JOIN within_6h            w6h  ON ids.account_id = w6h.account_id
LEFT JOIN within_24h           w24h ON ids.account_id = w24h.account_id
LEFT JOIN post_credit_drawdown pcd  ON ids.account_id = pcd.account_id
LEFT JOIN credit_counts        cc   ON ids.account_id = cc.account_id
""").df()
print(f"Pack 2: {pack2.shape}  ({time.time()-t2:.0f}s)")
print(pack2.describe().round(3).to_string())
print("\n" + "="*55)
print("12.3  PACK 3 — COUNTERPARTY NOVELTY + BRANCH DIVERSITY")
print("="*55)
t3 = time.time()
pack3 = con.execute("""
WITH ids AS (
    SELECT account_id FROM read_parquet('/tmp/all_ids.parquet')
),
txns AS (
    SELECT
        t.account_id,
        CAST(t.transaction_timestamp AS TIMESTAMP) AS ts,
        t.txn_type,
        ABS(t.amount) AS amt,
        t.counterparty_id
    FROM transactions t
    JOIN ids i ON t.account_id = i.account_id
    WHERE t.counterparty_id IS NOT NULL
),
anchors AS (
    SELECT account_id, MAX(ts) AS anchor_ts
    FROM txns
    GROUP BY account_id
),
life_cp AS (
    SELECT
        account_id,
        COUNT(DISTINCT counterparty_id) AS life_unique_cp,
        COUNT(*) AS life_cp_txns
    FROM txns
    GROUP BY account_id
),
w30_cp AS (
    SELECT
        t.account_id,
        COUNT(DISTINCT t.counterparty_id) AS w30_unique_cp
    FROM txns t
    JOIN anchors a ON t.account_id = a.account_id
    WHERE t.ts >= a.anchor_ts - INTERVAL 30 DAY
    GROUP BY t.account_id
),
w90_cp AS (
    SELECT
        t.account_id,
        COUNT(DISTINCT t.counterparty_id) AS w90_unique_cp
    FROM txns t
    JOIN anchors a ON t.account_id = a.account_id
    WHERE t.ts >= a.anchor_ts - INTERVAL 90 DAY
    GROUP BY t.account_id
),
old_cp AS (
    SELECT DISTINCT t.account_id, t.counterparty_id
    FROM txns t
    JOIN anchors a ON t.account_id = a.account_id
    WHERE t.ts < a.anchor_ts - INTERVAL 30 DAY
),
new_cp_30d AS (
    SELECT
        t.account_id,
        COUNT(DISTINCT t.counterparty_id) AS new_cp_30d
    FROM txns t
    JOIN anchors a ON t.account_id = a.account_id
    WHERE t.ts >= a.anchor_ts - INTERVAL 30 DAY
      AND NOT EXISTS (
          SELECT 1
          FROM old_cp o
          WHERE o.account_id = t.account_id
            AND o.counterparty_id = t.counterparty_id
      )
    GROUP BY t.account_id
),
cp_counts AS (
    SELECT account_id, counterparty_id, COUNT(*) AS cp_freq
    FROM txns
    GROUP BY account_id, counterparty_id
),
top1_cp AS (
    SELECT account_id, MAX(cp_freq) AS top1_freq
    FROM cp_counts
    GROUP BY account_id
),
cp_branches AS (
    SELECT
        account_id,
        SUBSTRING(counterparty_id, 1, 8) AS branch_prefix,
        COUNT(*) AS branch_freq
    FROM txns
    GROUP BY account_id, SUBSTRING(counterparty_id, 1, 8)
),
branch_diversity AS (
    SELECT
        account_id,
        COUNT(DISTINCT branch_prefix) AS unique_cp_branches,
        MAX(branch_freq) * 1.0 / NULLIF(SUM(branch_freq), 0) AS top_branch_share
    FROM cp_branches
    GROUP BY account_id
),
outgoing_cp AS (
    SELECT DISTINCT account_id, counterparty_id AS cp
    FROM txns
    WHERE txn_type = 'D'
),
incoming_cp AS (
    SELECT DISTINCT account_id, counterparty_id AS cp
    FROM txns
    WHERE txn_type = 'C'
),
cp_overlap AS (
    SELECT
        o.account_id,
        COUNT(*) AS bidirectional_cp_count
    FROM outgoing_cp o
    JOIN incoming_cp i
      ON o.account_id = i.account_id
     AND o.cp = i.cp
    GROUP BY o.account_id
),
outgoing_unique AS (
    SELECT account_id, COUNT(DISTINCT counterparty_id) AS outgoing_unique_cp
    FROM txns
    WHERE txn_type = 'D'
    GROUP BY account_id
),
incoming_unique AS (
    SELECT account_id, COUNT(DISTINCT counterparty_id) AS incoming_unique_cp
    FROM txns
    WHERE txn_type = 'C'
    GROUP BY account_id
)
SELECT
    ids.account_id,
    COALESCE(nc.new_cp_30d, 0)                AS new_cp_30d,
    CASE WHEN lc.life_unique_cp > 0
         THEN COALESCE(nc.new_cp_30d, 0) * 1.0 / lc.life_unique_cp
         ELSE 0 END                           AS new_cp_30d_rate,
    CASE WHEN lc.life_unique_cp > 0
         THEN COALESCE(w30.w30_unique_cp, 0) * 1.0 / lc.life_unique_cp
         ELSE 0 END                           AS w30_cp_concentration,
    CASE WHEN lc.life_unique_cp > 0
         THEN COALESCE(w90.w90_unique_cp, 0) * 1.0 / lc.life_unique_cp
         ELSE 0 END                           AS w90_cp_concentration,
    CASE WHEN lc.life_cp_txns > 0
         THEN COALESCE(t1.top1_freq, 0) * 1.0 / lc.life_cp_txns
         ELSE 0 END                           AS top1_cp_share,
    COALESCE(bd.unique_cp_branches, 0)        AS unique_cp_branches,
    COALESCE(bd.top_branch_share, 0)          AS top_branch_share,
    COALESCE(co.bidirectional_cp_count, 0)    AS bidirectional_cp_count,
    CASE WHEN lc.life_unique_cp > 0
         THEN COALESCE(co.bidirectional_cp_count, 0) * 1.0 / lc.life_unique_cp
         ELSE 0 END                           AS bidirectional_cp_rate,
    COALESCE(ou.outgoing_unique_cp, 0)        AS outgoing_unique_cp,
    COALESCE(iu.incoming_unique_cp, 0)        AS incoming_unique_cp,
    CASE WHEN COALESCE(iu.incoming_unique_cp, 0) > 0
         THEN COALESCE(ou.outgoing_unique_cp, 0) * 1.0 / iu.incoming_unique_cp
         ELSE 0 END                           AS outgoing_incoming_cp_ratio
FROM ids
LEFT JOIN life_cp          lc  ON ids.account_id = lc.account_id
LEFT JOIN w30_cp           w30 ON ids.account_id = w30.account_id
LEFT JOIN w90_cp           w90 ON ids.account_id = w90.account_id
LEFT JOIN new_cp_30d       nc  ON ids.account_id = nc.account_id
LEFT JOIN top1_cp          t1  ON ids.account_id = t1.account_id
LEFT JOIN branch_diversity bd  ON ids.account_id = bd.account_id
LEFT JOIN cp_overlap       co  ON ids.account_id = co.account_id
LEFT JOIN outgoing_unique  ou  ON ids.account_id = ou.account_id
LEFT JOIN incoming_unique  iu  ON ids.account_id = iu.account_id
""").df()
print(f"Pack 3: {pack3.shape}  ({time.time()-t3:.0f}s)")
print(pack3.describe().round(3).to_string())
print("\n" + "="*55)
print("12.4  MERGING ALL FEATURE PACKS")
print("="*55)
new_features = (
    pack1
    .merge(pack2, on='account_id', how='outer')
    .merge(pack3, on='account_id', how='outer')
)
print(f"New features shape: {new_features.shape}")
print(f"New feature count : {new_features.shape[1] - 1}")
nan_counts = new_features.isnull().sum()
inf_counts = new_features.apply(
    lambda c: np.isinf(c).sum() if c.dtype in [np.float64, np.float32] else 0
)
print(f"\nNaN total: {nan_counts.sum()}")
print(f"Inf total: {inf_counts.sum()}")
new_features = new_features.fillna(0)
for col in new_features.select_dtypes(include=[np.number]).columns:
    new_features[col] = new_features[col].replace(
        [np.inf, -np.inf],
        [new_features[col].quantile(0.99), new_features[col].quantile(0.01)]
    )
feat_train_v2 = feat_train.merge(new_features, on='account_id', how='left')
feat_test_v2  = feat_test.merge(new_features,  on='account_id', how='left')
print(f"\nfeat_train_v2: {feat_train_v2.shape}  (was {feat_train.shape})")
print(f"feat_test_v2 : {feat_test_v2.shape}   (was {feat_test.shape})")
new_cols = [c for c in new_features.columns if c != 'account_id']
print(f"\nTop new features by mule vs legit separation (mean diff):")
mule_mask = feat_train_v2['is_mule'] == 1
rows = []
for col in new_cols:
    mule_mean  = feat_train_v2.loc[mule_mask,  col].mean()
    legit_mean = feat_train_v2.loc[~mule_mask, col].mean()
    mule_std   = feat_train_v2.loc[mule_mask,  col].std()
    legit_std  = feat_train_v2.loc[~mule_mask, col].std()
    pooled_std = np.sqrt((mule_std**2 + legit_std**2) / 2 + 1e-9)
    effect     = abs(mule_mean - legit_mean) / pooled_std
    rows.append({
        'feature': col,
        'mule_mean': mule_mean,
        'legit_mean': legit_mean,
        'effect_size': effect
    })
sig_df = pd.DataFrame(rows).sort_values('effect_size', ascending=False)
print(sig_df.head(20).to_string(index=False))
feat_train_v2.to_parquet(f'{OUT_DIR}/features_train_v2.parquet', index=False)
feat_test_v2.to_parquet(f'{OUT_DIR}/features_test_v2.parquet', index=False)
print(f"\n✓ features_train_v2.parquet — {feat_train_v2.shape}")
print(f"✓ features_test_v2.parquet  — {feat_test_v2.shape}")
print(f"✓ Total time: {time.time()-t_start:.0f}s")
print(f"\nNew features added ({len(new_cols)}):")
for i, c in enumerate(new_cols, 1):
    print(f"  {i:2d}. {c}")
import time
import warnings
import numpy as np
import pandas as pd
import duckdb
warnings.filterwarnings('ignore')
NFPC_DATA = '/content/nfpc_data'
OUT_DIR   = '/content'
TMP_DIR   = '/tmp'
t_start   = time.time()
print(f"\n{'='*60}")
print("13.0  FEATURE PACK v3")
print(f"{'='*60}")
con = duckdb.connect()
con.execute(f"""
    CREATE OR REPLACE VIEW transactions AS
    SELECT * FROM read_parquet(
        '{NFPC_DATA}/transactions/batch-*/part_*.parquet')
""")
con.execute(f"""
    CREATE OR REPLACE VIEW txn_add AS
    SELECT * FROM read_parquet(
        '{NFPC_DATA}/transactions_additional/batch-*/part_*.parquet')
""")
feat_train = pd.read_parquet(f'{OUT_DIR}/features_train_v2.parquet')
feat_test  = pd.read_parquet(f'{OUT_DIR}/features_test_v2.parquet')
all_ids = pd.concat([
    feat_train[['account_id']],
    feat_test[['account_id']]
], ignore_index=True).drop_duplicates()
all_ids.to_parquet(f'{TMP_DIR}/all_ids.parquet', index=False)
labels = pd.read_parquet(f'{NFPC_DATA}/train_labels.parquet')
print(f"Base matrix: train={feat_train.shape}  test={feat_test.shape}")
print(f"Total accounts: {len(all_ids):,}")
print(f"\n{'='*55}")
print("13.1  PACK 4 — DORMANCY & REACTIVATION")
print(f"{'='*55}")
t4 = time.time()
pack4 = con.execute(f"""
WITH ids AS (
    SELECT account_id FROM read_parquet('{TMP_DIR}/all_ids.parquet')
),
daily AS (
    SELECT
        t.account_id,
        CAST(t.transaction_timestamp AS DATE) AS txn_date,
        COUNT(*) AS daily_cnt,
        SUM(CASE WHEN t.txn_type = 'C' THEN ABS(t.amount) ELSE 0 END) AS daily_credit_vol
    FROM transactions t
    JOIN ids i
      ON t.account_id = i.account_id
    GROUP BY t.account_id, CAST(t.transaction_timestamp AS DATE)
),
with_gaps AS (
    SELECT
        account_id,
        txn_date,
        daily_cnt,
        daily_credit_vol,
        LEAD(txn_date) OVER (
            PARTITION BY account_id ORDER BY txn_date
        ) AS next_active_date,
        DATEDIFF(
            'day',
            txn_date,
            LEAD(txn_date) OVER (
                PARTITION BY account_id ORDER BY txn_date
            )
        ) - 1 AS gap_days
    FROM daily
),
gap_stats AS (
    SELECT
        account_id,
        MAX(gap_days)                                    AS max_inactivity_gap,
        SUM(CASE WHEN gap_days >= 30 THEN 1 ELSE 0 END) AS gaps_over_30d,
        SUM(CASE WHEN gap_days >= 60 THEN 1 ELSE 0 END) AS gaps_over_60d,
        SUM(CASE WHEN gap_days >= 90 THEN 1 ELSE 0 END) AS gaps_over_90d
    FROM with_gaps
    WHERE gap_days IS NOT NULL
    GROUP BY account_id
),
longest_gap AS (
    SELECT
        account_id,
        txn_date AS longest_gap_start_date,
        gap_days,
        ROW_NUMBER() OVER (
            PARTITION BY account_id
            ORDER BY gap_days DESC, txn_date DESC
        ) AS rn
    FROM with_gaps
    WHERE gap_days IS NOT NULL
),
best_gap AS (
    SELECT
        account_id,
        longest_gap_start_date,
        gap_days AS longest_gap_days
    FROM longest_gap
    WHERE rn = 1
),
lifetime_intensity AS (
    SELECT
        account_id,
        SUM(daily_cnt) AS total_txns,
        COUNT(*)       AS active_days,
        AVG(daily_cnt) AS avg_daily_intensity
    FROM daily
    GROUP BY account_id
),
post_gap AS (
    SELECT
        d.account_id,
        SUM(d.daily_cnt)        AS post_gap_txns,
        COUNT(*)                AS post_gap_activity_days,
        SUM(d.daily_credit_vol) AS post_gap_credit_vol
    FROM daily d
    JOIN best_gap bg
      ON d.account_id = bg.account_id
    WHERE d.txn_date > bg.longest_gap_start_date
      AND d.txn_date <= bg.longest_gap_start_date + INTERVAL 30 DAY
    GROUP BY d.account_id
),
pre_gap AS (
    SELECT
        d.account_id,
        COUNT(*) AS pre_gap_activity_days
    FROM daily d
    JOIN best_gap bg
      ON d.account_id = bg.account_id
    WHERE d.txn_date < bg.longest_gap_start_date
    GROUP BY d.account_id
)
SELECT
    ids.account_id,
    COALESCE(gs.max_inactivity_gap,  0)    AS max_inactivity_gap,
    COALESCE(gs.gaps_over_30d,       0)    AS gaps_over_30d,
    COALESCE(gs.gaps_over_60d,       0)    AS gaps_over_60d,
    COALESCE(gs.gaps_over_90d,       0)    AS gaps_over_90d,
    COALESCE(pg2.pre_gap_activity_days, 0) AS pre_gap_activity_days,
    COALESCE(pg.post_gap_activity_days, 0) AS post_gap_activity_days,
    COALESCE(pg.post_gap_credit_vol,  0)   AS reactivation_volume,
    CASE
        WHEN li.avg_daily_intensity > 0 AND COALESCE(pg.post_gap_activity_days, 0) > 0
        THEN (pg.post_gap_txns * 1.0 / NULLIF(pg.post_gap_activity_days, 0))
             / li.avg_daily_intensity
        ELSE 0
    END AS burst_ratio_after_gap
FROM ids
LEFT JOIN gap_stats           gs  ON ids.account_id = gs.account_id
LEFT JOIN lifetime_intensity  li  ON ids.account_id = li.account_id
LEFT JOIN post_gap            pg  ON ids.account_id = pg.account_id
LEFT JOIN pre_gap             pg2 ON ids.account_id = pg2.account_id
""").df()
print(f"Pack 4 (dormancy): {pack4.shape}  ({time.time()-t4:.0f}s)")
check4 = pack4.merge(labels[['account_id','is_mule']], on='account_id', how='inner')
print(f"\n  Feature medians (mule vs legit):")
for col in ['max_inactivity_gap', 'gaps_over_30d', 'gaps_over_90d',
            'burst_ratio_after_gap', 'reactivation_volume',
            'pre_gap_activity_days', 'post_gap_activity_days']:
    m = check4.loc[check4['is_mule']==1, col].median()
    l = check4.loc[check4['is_mule']==0, col].median()
    arrow = "↑" if m > l else "↓"
    print(f"    {col:<30}  mule={m:.2f}  legit={l:.2f}  {arrow}")
print(f"\n{'='*55}")
print("13.2  PACK 5 — TRANSACTION VELOCITY")
print(f"{'='*55}")
t5 = time.time()
pack5 = con.execute(f"""
WITH ids AS (
    SELECT account_id FROM read_parquet('{TMP_DIR}/all_ids.parquet')
),
hourly AS (
    SELECT
        t.account_id,
        CAST(t.transaction_timestamp AS DATE) AS txn_date,
        HOUR(CAST(t.transaction_timestamp AS TIMESTAMP)) AS txn_hour,
        COUNT(*) AS cnt
    FROM transactions t
    JOIN ids i
      ON t.account_id = i.account_id
    GROUP BY t.account_id, CAST(t.transaction_timestamp AS DATE),
             HOUR(CAST(t.transaction_timestamp AS TIMESTAMP))
),
daily_agg AS (
    SELECT
        account_id,
        txn_date,
        SUM(cnt) AS daily_cnt
    FROM hourly
    GROUP BY account_id, txn_date
),
daily_with_avg AS (
    SELECT
        account_id,
        txn_date,
        daily_cnt,
        AVG(daily_cnt) OVER (PARTITION BY account_id) AS avg_daily_txns
    FROM daily_agg
),
daily_stats AS (
    SELECT
        account_id,
        MAX(daily_cnt) AS max_daily_txns,
        MAX(avg_daily_txns) AS avg_daily_txns,
        SUM(CASE WHEN daily_cnt > 3 * avg_daily_txns THEN 1 ELSE 0 END) AS burst_count
    FROM daily_with_avg
    GROUP BY account_id
),
hourly_stats AS (
    SELECT
        account_id,
        MAX(cnt) AS max_hourly_txns
    FROM hourly
    GROUP BY account_id
),
inter_txn_gaps AS (
    SELECT
        t.account_id,
        DATEDIFF(
            'second',
            LAG(CAST(t.transaction_timestamp AS TIMESTAMP)) OVER (
                PARTITION BY t.account_id ORDER BY t.transaction_timestamp
            ),
            CAST(t.transaction_timestamp AS TIMESTAMP)
        ) / 3600.0 AS gap_hours
    FROM transactions t
    JOIN ids i
      ON t.account_id = i.account_id
),
gap_stats AS (
    SELECT
        account_id,
        PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY gap_hours) AS inter_txn_gap_p10,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gap_hours) AS inter_txn_gap_p50,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY gap_hours) AS inter_txn_gap_p90,
        AVG(gap_hours)    AS avg_gap_hours,
        STDDEV(gap_hours) AS std_gap_hours
    FROM inter_txn_gaps
    WHERE gap_hours IS NOT NULL
      AND gap_hours >= 0
    GROUP BY account_id
)
SELECT
    ids.account_id,
    COALESCE(ds.max_daily_txns,    0) AS max_daily_txns,
    COALESCE(ds.burst_count,       0) AS burst_count,
    COALESCE(gs.inter_txn_gap_p10, 0) AS inter_txn_gap_p10,
    COALESCE(gs.inter_txn_gap_p50, 0) AS inter_txn_gap_p50,
    COALESCE(gs.inter_txn_gap_p90, 0) AS inter_txn_gap_p90,
    CASE
        WHEN gs.avg_gap_hours > 0
        THEN gs.std_gap_hours / gs.avg_gap_hours
        ELSE 0
    END AS gap_cv,
    COALESCE(hs.max_hourly_txns,   0) AS max_hourly_txns
FROM ids
LEFT JOIN daily_stats  ds ON ids.account_id = ds.account_id
LEFT JOIN gap_stats    gs ON ids.account_id = gs.account_id
LEFT JOIN hourly_stats hs ON ids.account_id = hs.account_id
""").df()
print(f"Pack 5 (velocity): {pack5.shape}  ({time.time()-t5:.0f}s)")
check5 = pack5.merge(labels[['account_id','is_mule']], on='account_id', how='inner')
print(f"\n  Feature medians (mule vs legit):")
for col in ['max_daily_txns', 'burst_count', 'inter_txn_gap_p10',
            'inter_txn_gap_p50', 'gap_cv', 'max_hourly_txns']:
    m = check5.loc[check5['is_mule']==1, col].median()
    l = check5.loc[check5['is_mule']==0, col].median()
    arrow = "↑" if m > l else "↓"
    print(f"    {col:<26}  mule={m:.2f}  legit={l:.2f}  {arrow}")
print(f"\n{'='*55}")
print("13.3  MERGE AND SAVE")
print(f"{'='*55}")
new_features = pack4.merge(pack5, on='account_id', how='outer')
new_features = new_features.fillna(0)
for col in new_features.select_dtypes(include=[np.number]).columns:
    new_features[col] = new_features[col].replace(
        [np.inf, -np.inf],
        [new_features[col].quantile(0.99), 0]
    )
new_cols = [c for c in new_features.columns if c != 'account_id']
print(f"New features added: {len(new_cols)}")
print(f"  {new_cols}")
feat_train_v3 = feat_train.merge(new_features, on='account_id', how='left')
feat_test_v3  = feat_test.merge(new_features,  on='account_id', how='left')
feat_train_v3[new_cols] = feat_train_v3[new_cols].fillna(0)
feat_test_v3[new_cols]  = feat_test_v3[new_cols].fillna(0)
print(f"\nfeat_train_v3: {feat_train_v3.shape}  (was {feat_train.shape})")
print(f"feat_test_v3 : {feat_test_v3.shape}   (was {feat_test.shape})")
mule_mask = feat_train_v3['is_mule'] == 1
rows = []
for col in new_cols:
    m_mean = feat_train_v3.loc[mule_mask,  col].mean()
    l_mean = feat_train_v3.loc[~mule_mask, col].mean()
    m_std  = feat_train_v3.loc[mule_mask,  col].std()
    l_std  = feat_train_v3.loc[~mule_mask, col].std()
    pooled = np.sqrt((m_std**2 + l_std**2) / 2 + 1e-9)
    rows.append({
        'feature': col,
        'mule_mean': m_mean,
        'legit_mean': l_mean,
        'cohen_d': abs(m_mean - l_mean) / pooled
    })
sig_df = pd.DataFrame(rows).sort_values('cohen_d', ascending=False)
print(f"\nTop new features by Cohen's d:")
print(sig_df.head(10).to_string(index=False))
feat_train_v3.to_parquet(f'{OUT_DIR}/features_train_v3.parquet', index=False)
feat_test_v3.to_parquet(f'{OUT_DIR}/features_test_v3.parquet',  index=False)
print(f"\n✓ features_train_v3.parquet — {feat_train_v3.shape}")
print(f"✓ features_test_v3.parquet  — {feat_test_v3.shape}")
print(f"✓ Total time: {time.time()-t_start:.0f}s")
print("\n" + "="*65)
print("BLOCK 13 COMPLETE ✓")
print("="*65)
feat_train = pd.read_parquet('/content/features_train_v3.parquet')
feat_test  = pd.read_parquet('/content/features_test_v3.parquet')
print("Loaded v3 base:", feat_train.shape, feat_test.shape)
import time
import warnings
import numpy as np
import pandas as pd
import duckdb
warnings.filterwarnings('ignore')
NFPC_DATA = '/content/nfpc_data'
OUT_DIR   = '/content'
TMP_DIR   = '/tmp'
REF_TS    = pd.Timestamp('2025-06-30')
t_start   = time.time()
print(f"\n{'='*60}")
print("14.0  FEATURE PACK v4")
print(f"{'='*60}")
if 'con' not in globals():
    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE VIEW transactions AS
        SELECT * FROM read_parquet(
            '{NFPC_DATA}/transactions/batch-*/part_*.parquet')
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW txn_add AS
        SELECT * FROM read_parquet(
            '{NFPC_DATA}/transactions_additional/batch-*/part_*.parquet')
    """)
    print("DuckDB views rebuilt ✓")
feat_train = pd.read_parquet('/content/features_train_v3.parquet')
feat_test  = pd.read_parquet('/content/features_test_v3.parquet')
print("Loaded v3 base:", feat_train.shape, feat_test.shape)
all_ids = pd.concat([
    feat_train[['account_id']],
    feat_test[['account_id']]
], ignore_index=True).drop_duplicates()
all_ids.to_parquet(f'{TMP_DIR}/all_ids.parquet', index=False)
labels = pd.read_parquet(f'{NFPC_DATA}/train_labels.parquet')
print(f"Base matrix: train={feat_train.shape}  test={feat_test.shape}")
accounts     = pd.read_parquet(f'{NFPC_DATA}/accounts.parquet')
customers    = pd.read_parquet(f'{NFPC_DATA}/customers.parquet')
cust_link    = pd.read_parquet(f'{NFPC_DATA}/customer_account_linkage.parquet')
demographics = pd.read_parquet(f'{NFPC_DATA}/demographics.parquet')
product_det  = pd.read_parquet(f'{NFPC_DATA}/product_details.parquet')
def yn_to_int(df, cols):
    for col in cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = (df[col].astype(str).str.upper().str.strip() == 'Y').astype(int)
    return df
customers    = yn_to_int(customers,    ['mobile_banking_flag','internet_banking_flag',
                                        'atm_card_flag','demat_flag',
                                        'credit_card_flag','fastag_flag'])
demographics = yn_to_int(demographics, ['joint_account_flag','nri_flag'])
for col in ['address_last_update_date','passbook_last_update_date']:
    if col in demographics.columns:
        demographics[col] = pd.to_datetime(demographics[col], errors='coerce')
print(f"\n{'='*55}")
print("14.1  PACK 6 — DEMOGRAPHICS & DIGITAL BANKING")
print(f"{'='*55}")
t6 = time.time()
cust_feats = (
    cust_link[['account_id','customer_id']]
    .merge(customers[['customer_id',
                       'mobile_banking_flag','internet_banking_flag',
                       'atm_card_flag','demat_flag',
                       'credit_card_flag','fastag_flag']], on='customer_id', how='left')
    .merge(demographics[['customer_id','joint_account_flag',
                          'address_last_update_date',
                          'passbook_last_update_date']], on='customer_id', how='left')
    .merge(product_det[['customer_id','loan_count','cc_count',
                         'od_count','ka_count','sa_count']], on='customer_id', how='left')
)
for col in ['mobile_banking_flag','internet_banking_flag','atm_card_flag',
            'demat_flag','credit_card_flag','fastag_flag','joint_account_flag',
            'loan_count','cc_count','od_count','ka_count','sa_count']:
    if col in cust_feats.columns:
        cust_feats[col] = cust_feats[col].fillna(0)
cust_feats['digital_banking_score'] = (
    cust_feats['mobile_banking_flag']   +
    cust_feats['internet_banking_flag'] +
    cust_feats['atm_card_flag']         +
    cust_feats['demat_flag']
)
cust_feats['has_credit_card']  = cust_feats['credit_card_flag'].astype(int)
cust_feats['has_fastag']       = cust_feats['fastag_flag'].astype(int)
prod_cols = ['loan_count','cc_count','od_count','ka_count','sa_count']
cust_feats['product_diversity'] = (
    cust_feats[prod_cols].gt(0).sum(axis=1)
)
cust_feats['has_multiple_products'] = (cust_feats['product_diversity'] > 1).astype(int)
cust_feats['address_update_age_days'] = (
    (REF_TS - cust_feats['address_last_update_date']).dt.days
    .clip(lower=0)
    .fillna(9999)     # never updated → large sentinel value
)
cust_feats['passbook_update_age_days'] = (
    (REF_TS - cust_feats['passbook_last_update_date']).dt.days
    .clip(lower=0)
    .fillna(9999)
) if 'passbook_last_update_date' in cust_feats.columns else 9999
pack6_cols = [
    'account_id',
    'digital_banking_score',
    'has_credit_card',
    'has_fastag',
    'product_diversity',
    'has_multiple_products',
    'address_update_age_days',
    'passbook_update_age_days',
    'joint_account_flag',
]
pack6 = cust_feats[[c for c in pack6_cols if c in cust_feats.columns]].copy()
pack6 = pack6.groupby('account_id').first().reset_index()
print(f"Pack 6 (demographics): {pack6.shape}  ({time.time()-t6:.1f}s)")
check6 = pack6.merge(labels[['account_id','is_mule']], on='account_id', how='inner')
print(f"\n  Feature medians (mule vs legit):")
for col in [c for c in pack6.columns if c != 'account_id']:
    m = check6.loc[check6['is_mule']==1, col].median()
    l = check6.loc[check6['is_mule']==0, col].median()
    arrow = "↑" if m > l else "↓"
    print(f"    {col:<30}  mule={m:.2f}  legit={l:.2f}  {arrow}")
print(f"\n{'='*55}")
print("14.2  PACK 7 — GEOGRAPHIC VELOCITY")
print(f"{'='*55}")
t7 = time.time()
pack7 = con.execute("""
WITH ids AS (
    SELECT account_id FROM read_parquet('/tmp/all_ids.parquet')
),
-- Bin lat/lon to ~10km resolution (0.1 degree cells)
geo AS (
    SELECT
        t.account_id,
        CAST(t.transaction_timestamp AS TIMESTAMP)              AS ts,
        ta.latitude,
        ta.longitude,
        ROUND(ta.latitude,  1)                                  AS lat_bin,
        ROUND(ta.longitude, 1)                                  AS lon_bin,
        ta.atm_deposit_channel_code,
        CASE WHEN ta.latitude IS NOT NULL
              AND ta.longitude IS NOT NULL
              AND ta.latitude != 0
              AND ta.longitude != 0
        THEN 1 ELSE 0 END                                       AS has_location
    FROM transactions t
    JOIN txn_add ta ON t.transaction_id = ta.transaction_id
    JOIN ids i ON t.account_id = i.account_id
),
location_stats AS (
    SELECT
        account_id,
        COUNT(DISTINCT (lat_bin, lon_bin))          AS unique_location_count,
        AVG(has_location)                           AS txns_with_location,
        MAX(latitude)  - MIN(latitude)              AS lat_spread,
        MAX(longitude) - MIN(longitude)             AS lon_spread,
        AVG(CASE WHEN atm_deposit_channel_code IS NOT NULL
                      AND atm_deposit_channel_code != ''
                 THEN 1.0 ELSE 0.0 END)             AS atm_deposit_channel_flag
    FROM geo
    GROUP BY account_id
),
loc_entropy AS (
    SELECT
        account_id,
        -SUM(p * LN(p + 1e-9)) AS location_entropy
    FROM (
        SELECT
            account_id,
            lat_bin,
            lon_bin,
            COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY account_id) AS p
        FROM geo
        WHERE has_location = 1
        GROUP BY account_id, lat_bin, lon_bin
    )
    GROUP BY account_id
),
-- Impossible travel: two txns > 5 degrees apart (~555km) within 2 hours
travel_pairs AS (
    SELECT
        g1.account_id,
        MAX(
            CASE
                WHEN ABS(g1.lat_bin - g2.lat_bin) > 5
                  OR ABS(g1.lon_bin - g2.lon_bin) > 5
                THEN 1 ELSE 0
            END
        ) AS impossible_travel_flag
    FROM geo g1
    JOIN geo g2
      ON g1.account_id = g2.account_id
     AND g2.ts > g1.ts
     AND g2.ts <= g1.ts + INTERVAL 2 HOUR
     AND g1.has_location = 1
     AND g2.has_location = 1
    GROUP BY g1.account_id
)
SELECT
    ids.account_id,
    COALESCE(ls.unique_location_count,    1)  AS unique_location_count,
    COALESCE(ls.txns_with_location,       0)  AS txns_with_location,
    COALESCE(SQRT(
        POWER(COALESCE(ls.lat_spread, 0), 2) +
        POWER(COALESCE(ls.lon_spread, 0), 2)
    ), 0)                                     AS max_distance_km_approx,
    COALESCE(le.location_entropy,          0)  AS location_entropy,
    COALESCE(tp.impossible_travel_flag,    0)  AS impossible_travel_flag,
    COALESCE(ls.atm_deposit_channel_flag,  0)  AS atm_deposit_channel_flag
FROM ids
LEFT JOIN location_stats ls ON ids.account_id = ls.account_id
LEFT JOIN loc_entropy    le ON ids.account_id = le.account_id
LEFT JOIN travel_pairs   tp ON ids.account_id = tp.account_id
""").df()
print(f"Pack 7 (geo velocity): {pack7.shape}  ({time.time()-t7:.0f}s)")
check7 = pack7.merge(labels[['account_id','is_mule']], on='account_id', how='inner')
print(f"\n  Feature medians (mule vs legit):")
for col in [c for c in pack7.columns if c != 'account_id']:
    m = check7.loc[check7['is_mule']==1, col].median()
    l = check7.loc[check7['is_mule']==0, col].median()
    arrow = "↑" if m > l else "↓"
    print(f"    {col:<32}  mule={m:.4f}  legit={l:.4f}  {arrow}")
print(f"\n{'='*55}")
print("14.3  MERGE AND SAVE")
print(f"{'='*55}")
new_features = pack6.merge(pack7, on='account_id', how='outer').fillna(0)
for col in new_features.select_dtypes(include=[np.number]).columns:
    new_features[col] = new_features[col].replace([np.inf, -np.inf], 0)
new_cols = [c for c in new_features.columns if c != 'account_id']
print(f"New features added: {len(new_cols)}")
print(f"  {new_cols}")
feat_train_v4 = feat_train.merge(new_features, on='account_id', how='left')
feat_test_v4  = feat_test.merge(new_features,  on='account_id', how='left')
feat_train_v4[new_cols] = feat_train_v4[new_cols].fillna(0)
feat_test_v4[new_cols]  = feat_test_v4[new_cols].fillna(0)
print(f"\nfeat_train_v4: {feat_train_v4.shape}  (was {feat_train.shape})")
print(f"feat_test_v4 : {feat_test_v4.shape}   (was {feat_test.shape})")
mule_mask = feat_train_v4['is_mule'] == 1
rows = []
for col in new_cols:
    m_mean = feat_train_v4.loc[mule_mask,  col].mean()
    l_mean = feat_train_v4.loc[~mule_mask, col].mean()
    m_std  = feat_train_v4.loc[mule_mask,  col].std()
    l_std  = feat_train_v4.loc[~mule_mask, col].std()
    pooled = np.sqrt((m_std**2 + l_std**2) / 2 + 1e-9)
    rows.append({'feature': col, 'cohen_d': abs(m_mean - l_mean) / pooled})
sig_df = pd.DataFrame(rows).sort_values('cohen_d', ascending=False)
print(f"\nTop new features by Cohen's d:")
print(sig_df.head(10).to_string(index=False))
feat_train_v4.to_parquet(f'{OUT_DIR}/features_train_v4.parquet', index=False)
feat_test_v4.to_parquet(f'{OUT_DIR}/features_test_v4.parquet',  index=False)
print(f"\n✓ features_train_v4.parquet — {feat_train_v4.shape}")
print(f"✓ features_test_v4.parquet  — {feat_test_v4.shape}")
print(f"✓ Total time: {time.time()-t_start:.0f}s")
print("\n" + "="*65)
print("BLOCK 14 COMPLETE ✓")
print("="*65)
print("\nNext: Block 15 — leakage cleanup + final v14 matrix assembly")
import pandas as pd
train_v4 = pd.read_parquet('/content/features_train_v4.parquet')
test_v4  = pd.read_parquet('/content/features_test_v4.parquet')
print(train_v4.shape, test_v4.shape)
required_v3_cols = [
    'max_inactivity_gap',
    'gaps_over_30d',
    'burst_ratio_after_gap',
    'gap_cv',
    'max_daily_txns',
    'inter_txn_gap_p50'
]
missing = [c for c in required_v3_cols if c not in train_v4.columns]
print("Missing v3 cols:", missing)
import os
import shutil
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
NFPC_DATA = '/content/nfpc_data'
OUT_DIR   = '/content'
REF_DATE  = pd.Timestamp('2025-06-30')
print(f"\n{'='*60}")
print("15.0  FREEZE PATCH + SAVE v13")
print(f"{'='*60}")
print("\nLoading features_train_v4 / features_test_v4 ...")
feat_train_v4 = pd.read_parquet(f'{OUT_DIR}/features_train_v4.parquet')
feat_test_v4  = pd.read_parquet(f'{OUT_DIR}/features_test_v4.parquet')
required_v3_cols = [
    'max_inactivity_gap',
    'gap_cv',
]
missing_v3 = [c for c in required_v3_cols if c not in feat_train_v4.columns]
if missing_v3:
    raise ValueError(
        "features_train_v4.parquet does not appear to be built on top of v3. "
        f"Missing columns: {missing_v3}. Re-run the fixed Block 14 first."
    )
print(f"  v4 train shape: {feat_train_v4.shape}")
print(f"  v4 test  shape: {feat_test_v4.shape}")
if 'account_id' not in feat_train_v4.columns:
    train_ids = pd.read_parquet(f'{NFPC_DATA}/train_labels.parquet')[['account_id']]
    feat_train_v4.insert(0, 'account_id', train_ids['account_id'].values)
if 'account_id' not in feat_test_v4.columns:
    test_ids = pd.read_parquet(f'{NFPC_DATA}/test_accounts.parquet')[['account_id']]
    feat_test_v4.insert(0, 'account_id', test_ids['account_id'].values)
labels   = pd.read_parquet(f'{NFPC_DATA}/train_labels.parquet')
accounts = pd.read_parquet(f'{NFPC_DATA}/accounts.parquet')
accounts['freeze_date']   = pd.to_datetime(accounts['freeze_date'], errors='coerce')
accounts['unfreeze_date'] = pd.to_datetime(accounts['unfreeze_date'], errors='coerce')
labels['mule_flag_date']  = pd.to_datetime(labels['mule_flag_date'], errors='coerce')
print(f"\n{'='*60}")
print("15.1  CLEAN FREEZE PATCH")
print(f"{'='*60}")
train_freeze = labels[['account_id', 'mule_flag_date']].merge(
    accounts[['account_id', 'freeze_date', 'unfreeze_date']],
    on='account_id', how='left'
)
train_freeze['is_frozen_clean'] = (
    train_freeze['freeze_date'].notna() &
    (train_freeze['freeze_date'] < train_freeze['mule_flag_date'])
).astype(int)
train_freeze['freeze_duration_clean'] = np.where(
    train_freeze['is_frozen_clean'] == 1,
    (REF_DATE - train_freeze['freeze_date']).dt.days.clip(lower=0),
    0
)
train_freeze['was_unfrozen_before_flag'] = (
    train_freeze['unfreeze_date'].notna() &
    (train_freeze['unfreeze_date'] < train_freeze['mule_flag_date'])
).astype(int)
train_patch = train_freeze[
    ['account_id', 'is_frozen_clean', 'freeze_duration_clean', 'was_unfrozen_before_flag']
].copy()
test_freeze = feat_test_v4[['account_id']].merge(
    accounts[['account_id', 'freeze_date', 'unfreeze_date']],
    on='account_id', how='left'
)
test_freeze['is_frozen_clean'] = test_freeze['freeze_date'].notna().astype(int)
test_freeze['freeze_duration_clean'] = np.where(
    test_freeze['is_frozen_clean'] == 1,
    (REF_DATE - test_freeze['freeze_date']).dt.days.clip(lower=0),
    0
)
test_freeze['was_unfrozen_before_flag'] = (
    test_freeze['unfreeze_date'].notna()
).astype(int)
test_patch = test_freeze[
    ['account_id', 'is_frozen_clean', 'freeze_duration_clean', 'was_unfrozen_before_flag']
].copy()
merged = labels[['account_id', 'is_mule']].merge(train_patch, on='account_id', how='left')
mules  = merged[merged['is_mule'] == 1]
legit  = merged[merged['is_mule'] == 0]
print("\nClean freeze feature signal:")
for col in ['is_frozen_clean', 'freeze_duration_clean', 'was_unfrozen_before_flag']:
    mm = mules[col].mean()
    lm = legit[col].mean()
    ps = np.sqrt((mules[col].std()**2 + legit[col].std()**2) / 2 + 1e-9)
    eff = abs(mm - lm) / ps
    print(f"  {col:<30} mule={mm:.4f}  legit={lm:.4f}  effect={eff:.4f}")
DROP_COLS = ['is_frozen', 'freeze_duration_days']
feat_train_v4 = feat_train_v4.drop(columns=DROP_COLS, errors='ignore')
feat_test_v4  = feat_test_v4.drop(columns=DROP_COLS, errors='ignore')
feat_train_v4 = feat_train_v4.merge(train_patch, on='account_id', how='left')
feat_test_v4  = feat_test_v4.merge(test_patch,  on='account_id', how='left')
fill_cols = ['is_frozen_clean', 'freeze_duration_clean', 'was_unfrozen_before_flag']
feat_train_v4[fill_cols] = feat_train_v4[fill_cols].fillna(0)
feat_test_v4[fill_cols]  = feat_test_v4[fill_cols].fillna(0)
print(f"\nPatched v4 train : {feat_train_v4.shape}")
print(f"Patched v4 test  : {feat_test_v4.shape}")
print(f"NaN in train     : {feat_train_v4.isnull().sum().sum()}")
print(f"NaN in test      : {feat_test_v4.isnull().sum().sum()}")
feat_train_v4.to_parquet(f'{OUT_DIR}/features_train_v4.parquet', index=False)
feat_test_v4.to_parquet(f'{OUT_DIR}/features_test_v4.parquet', index=False)
print(f"\n{'='*60}")
print("15.2  BUILD EXACT v13 BASE FILES")
print(f"{'='*60}")
LEAKY_COLS = [
    'is_frozen', 'is_frozen_clean',
    'freeze_duration_days', 'freeze_duration_clean',
    'was_unfrozen_before_flag',
    'avg_balance', 'monthly_avg_balance',
    'quarterly_avg_balance', 'daily_avg_balance',
    'risky_mcc_share',
]
LABEL_COLS = [
    'is_mule', 'mule_flag_date',
    'alert_reason', 'flagged_by_branch',
    'account_opening_date',
]
feat_train_v13 = feat_train_v4.drop(columns=LEAKY_COLS + LABEL_COLS, errors='ignore').copy()
feat_test_v13  = feat_test_v4.drop(columns=LEAKY_COLS + LABEL_COLS,  errors='ignore').copy()
print(f"features_train_v13 : {feat_train_v13.shape}  account_id={'account_id' in feat_train_v13.columns}")
print(f"features_test_v13  : {feat_test_v13.shape}  account_id={'account_id' in feat_test_v13.columns}")
print(f"NaN in train       : {feat_train_v13.isnull().sum().sum()}")
print(f"NaN in test        : {feat_test_v13.isnull().sum().sum()}")
feat_train_v13.to_parquet(f'{OUT_DIR}/features_train_v13.parquet', index=False)
feat_test_v13.to_parquet(f'{OUT_DIR}/features_test_v13.parquet', index=False)
feat_with_label = feat_train_v13.merge(
    labels[['account_id', 'is_mule']],
    on='account_id',
    how='left'
)
base_cols = [c for c in feat_train_v13.columns if c != 'account_id']
mule_mask = feat_with_label['is_mule'] == 1
rows = []
for col in base_cols:
    mm = feat_with_label.loc[mule_mask,  col]
    lm = feat_with_label.loc[~mule_mask, col]
    ps = np.sqrt((mm.std()**2 + lm.std()**2) / 2 + 1e-9)
    rows.append({
        'feature': col,
        'mule_mean': round(mm.mean(), 4),
        'legit_mean': round(lm.mean(), 4),
        'effect_size': round(abs(mm.mean() - lm.mean()) / ps, 4),
    })
manifest = pd.DataFrame(rows).sort_values('effect_size', ascending=False)
manifest.to_csv(f'{OUT_DIR}/features_v13_manifest.csv', index=False)
print(f"\n[OK] features_v13_manifest.csv saved")
print(f"\nTop 20 v13 features:")
print(manifest.head(20).to_string(index=False))
print("\n" + "="*65)
print("BLOCK 15 COMPLETE ✓")
print("="*65)
print("\nOutputs:")
print(f"  features_train_v4.parquet   (patched) → {OUT_DIR}/features_train_v4.parquet")
print(f"  features_test_v4.parquet    (patched) → {OUT_DIR}/features_test_v4.parquet")
print(f"  features_train_v13.parquet            → {OUT_DIR}/features_train_v13.parquet")
print(f"  features_test_v13.parquet             → {OUT_DIR}/features_test_v13.parquet")
print(f"  features_v13_manifest.csv             → {OUT_DIR}/features_v13_manifest.csv")
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
NFPC_DATA = '/content/nfpc_data'
OUT_DIR   = '/content'
print(f"\n{'='*60}")
print("16.0  FINAL v14 FEATURE FILES")
print(f"{'='*60}")
print("Loading v13 parquets...")
feat_train_v13 = pd.read_parquet(f'{OUT_DIR}/features_train_v13.parquet')
feat_test_v13  = pd.read_parquet(f'{OUT_DIR}/features_test_v13.parquet')
labels = pd.read_parquet(f'{NFPC_DATA}/train_labels.parquet')[
    ['account_id', 'is_mule', 'mule_flag_date', 'alert_reason', 'flagged_by_branch']
]
feat_train_v13 = feat_train_v13.merge(labels, on='account_id', how='left')
V14_DROP = [
    'has_freeze_date',
    'account_age_days_y',
    'rural_branch_y',
    'kyc_compliant_y',
    'nri_flag_y',
    'risky_mcc_share',
    'is_mule',
    'mule_flag_date',
    'alert_reason',
    'flagged_by_branch',
]
feat_train_v14 = feat_train_v13.drop(columns=V14_DROP, errors='ignore').copy()
feat_test_v14  = feat_test_v13.drop(columns=V14_DROP,  errors='ignore').copy()
print(f"features_train_v14 : {feat_train_v14.shape}  account_id={'account_id' in feat_train_v14.columns}")
print(f"features_test_v14  : {feat_test_v14.shape}  account_id={'account_id' in feat_test_v14.columns}")
print(f"NaN in train       : {feat_train_v14.isnull().sum().sum()}")
print(f"NaN in test        : {feat_test_v14.isnull().sum().sum()}")
assert 'has_freeze_date' not in feat_train_v14.columns
assert 'account_age_days_y' not in feat_train_v14.columns
assert 'rural_branch_y' not in feat_train_v14.columns
assert 'kyc_compliant_y' not in feat_train_v14.columns
assert 'nri_flag_y' not in feat_train_v14.columns
assert 'is_mule' not in feat_train_v14.columns
assert 'mule_flag_date' not in feat_train_v14.columns
assert 'alert_reason' not in feat_train_v14.columns
assert 'flagged_by_branch' not in feat_train_v14.columns
print("[OK] all checks passed")
feat_train_v14.to_parquet(f'{OUT_DIR}/features_train_v14.parquet', index=False)
feat_test_v14.to_parquet(f'{OUT_DIR}/features_test_v14.parquet', index=False)
print(f"\nSaved:")
print(f"  {OUT_DIR}/features_train_v14.parquet")
print(f"  {OUT_DIR}/features_test_v14.parquet")
print(f"\nFinal feature count (excluding account_id): {feat_train_v14.shape[1] - 1}")
print(f"Sample feature columns: {[c for c in feat_train_v14.columns if c != 'account_id'][:10]}")
print("\n" + "="*65)
print("BLOCK 16 COMPLETE ✓")
