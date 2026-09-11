#!/usr/bin/env python3
"""
Local Zero-Credit Runner for Cymbal Financial Fraud Detection Pipeline
====================================================================
Executes the full 4-stage pipeline locally over the actual codelab dataset
without requiring any Google Cloud credits or active billing:
  Stage 1: Ingestion (data/logs.json -> raw_transactions, dim_payers, dim_payees)
  Stage 2: dbt Transformations (Deduplication, Quarantine, Marts, Quality Tests)
  Stage 3: Distributed ML Training (Random Forest, Feature Scaling, AUC Evaluation)
  Stage 4: Batch Inference & Cloud Spanner Queue Sink (Probability Scoring >= 50%)
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Color helpers for terminal output
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

DB_DUCKDB_PATH = "data/cymbal_fraud.duckdb"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")


def get_db_connection():
    """Connect to DuckDB or fallback to SQLite."""
    try:
        import duckdb
        return duckdb.connect(DB_DUCKDB_PATH)
    except Exception:
        import sqlite3
        return sqlite3.connect("data/cymbal_fraud.db")


# =============================================================================
# STAGE 1: INGESTION
# =============================================================================
def run_ingestion():
    print(f"\n{BLUE}{BOLD}=== [STAGE 1] Ingest Raw Clearinghouse Logs ==={RESET}")
    start_time = time.time()
    
    logs_file = "data/logs.json"
    payers_file = "data/payers.csv"
    payees_file = "data/payees.csv"
    
    if not os.path.exists(logs_file):
        print(f"{RED}Error: {logs_file} not found!{RESET}")
        return False
        
    print(f"Reading raw JSON streaming logs from {logs_file}...")
    records = []
    with open(logs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
                    
    raw_df = pd.DataFrame(records)
    print(f"{GREEN}[OK]{RESET} Ingested {len(raw_df):,} raw transaction records.")
    
    # Load dimension files
    payers_df = pd.read_csv(payers_file) if os.path.exists(payers_file) else pd.DataFrame()
    payees_df = pd.read_csv(payees_file) if os.path.exists(payees_file) else pd.DataFrame()
    
    print(f"{GREEN}[OK]{RESET} Loaded {len(payers_df):,} payers and {len(payees_df):,} payees.")
    
    # Store into database
    con = get_db_connection()
    try:
        con.execute("CREATE OR REPLACE TABLE raw_transactions AS SELECT * FROM raw_df")
        con.execute("CREATE OR REPLACE TABLE dim_payers AS SELECT * FROM payers_df")
        con.execute("CREATE OR REPLACE TABLE dim_payees AS SELECT * FROM payees_df")
    except Exception:
        # SQLite fallback
        raw_df.to_sql("raw_transactions", con, if_exists="replace", index=False)
        payers_df.to_sql("dim_payers", con, if_exists="replace", index=False)
        payees_df.to_sql("dim_payees", con, if_exists="replace", index=False)
        
    con.close()
    
    elapsed = time.time() - start_time
    print(f"{GREEN}{BOLD}Stage 1 Ingestion Complete in {elapsed:.2f}s!{RESET}\n")
    return True


# =============================================================================
# STAGE 2: DBT TRANSFORMATIONS & DATA QUALITY TESTS
# =============================================================================
def run_dbt():
    print(f"\n{BLUE}{BOLD}=== [STAGE 2] dbt Deduplication, Quarantine & Enriched Marts ==={RESET}")
    start_time = time.time()
    
    con = get_db_connection()
    
    # 1. Staging: Deduplicate by transaction_id, take latest timestamp
    print("Executing dbt model: stg_transactions (Deduplicating by transaction_id)...")
    stg_query = """
    CREATE OR REPLACE TABLE stg_transactions AS
    WITH ranked AS (
        SELECT
            transaction_id,
            timestamp,
            payor_id,
            payee_id,
            CAST(amount AS DOUBLE) AS amount,
            currency,
            payment_method,
            status,
            CAST(merchant_mcc AS BIGINT) AS merchant_mcc,
            device_id,
            ip_address,
            CAST(is_fraud AS BIGINT) AS is_fraud,
            ROW_NUMBER() OVER (
                PARTITION BY transaction_id 
                ORDER BY timestamp DESC
            ) AS row_num
        FROM raw_transactions
        WHERE transaction_id IS NOT NULL 
          AND TRIM(transaction_id) != ''
    )
    SELECT
        transaction_id,
        timestamp,
        payor_id,
        payee_id,
        amount,
        currency,
        payment_method,
        status,
        merchant_mcc,
        device_id,
        ip_address,
        is_fraud
    FROM ranked
    WHERE row_num = 1;
    """
    try:
        con.execute(stg_query)
    except Exception:
        # fallback for standard sqlite
        con.execute("""
        CREATE TABLE IF NOT EXISTS stg_transactions AS
        SELECT * FROM raw_transactions WHERE transaction_id IS NOT NULL;
        """)

    # 2. Staging: Quarantine invalid / null transaction IDs
    print("Executing dbt model: invalid_transactions (Quarantining corrupted logs)...")
    inv_query = """
    CREATE OR REPLACE TABLE invalid_transactions AS
    SELECT
        transaction_id,
        timestamp,
        payor_id,
        payee_id,
        amount,
        currency,
        payment_method,
        status,
        merchant_mcc,
        device_id,
        ip_address,
        is_fraud,
        CURRENT_TIMESTAMP AS quarantined_at,
        CASE 
            WHEN transaction_id IS NULL OR TRIM(transaction_id) = '' THEN 'MISSING_TRANSACTION_ID'
            WHEN amount <= 0 OR amount IS NULL THEN 'INVALID_AMOUNT'
            ELSE 'OTHER_ANOMALY'
        END AS quarantine_reason
    FROM raw_transactions
    WHERE transaction_id IS NULL 
       OR TRIM(transaction_id) = ''
       OR amount <= 0
       OR amount IS NULL;
    """
    try:
        con.execute(inv_query)
    except Exception:
        pass

    # 3. Marts: Enriched Transactions joined with Payers & Payees
    print("Executing dbt model: enriched_transactions (Joining payer and payee dimensions)...")
    enriched_query = """
    CREATE OR REPLACE TABLE enriched_transactions AS
    SELECT
        t.transaction_id,
        t.timestamp,
        t.amount,
        t.currency,
        t.payment_method,
        t.status,
        t.merchant_mcc,
        t.device_id,
        t.ip_address,
        
        t.payor_id,
        p.payor_name,
        COALESCE(p.payor_country, 'UNKNOWN') AS payor_country,
        COALESCE(CAST(p.payor_risk_score AS DOUBLE), 0.0) AS payor_risk_score,
        
        t.payee_id,
        e.payee_name,
        COALESCE(e.payee_country, 'UNKNOWN') AS payee_country,
        COALESCE(CAST(e.payee_risk_score AS DOUBLE), 0.0) AS payee_risk_score,
        COALESCE(e.payee_category, 'General') AS payee_category,
        
        t.is_fraud
    FROM stg_transactions t
    LEFT JOIN dim_payers p ON t.payor_id = p.payor_id
    LEFT JOIN dim_payees e ON t.payee_id = e.payee_id;
    """
    con.execute(enriched_query)

    # 4. Data Quality Tests (assertions)
    print("\nRunning dbt Data Quality Tests:")
    # Test 1: Unique transaction_id in stg_transactions
    dup_count = con.execute("SELECT COUNT(*) - COUNT(DISTINCT transaction_id) FROM stg_transactions").fetchone()[0]
    if dup_count == 0:
        print(f"  {GREEN}[PASS]{RESET} test_unique_stg_transactions_transaction_id")
    else:
        print(f"  {RED}[FAIL]{RESET} test_unique_stg_transactions_transaction_id ({dup_count} duplicates found)")

    # Test 2: Not null transaction_id in stg_transactions
    null_count = con.execute("SELECT COUNT(*) FROM stg_transactions WHERE transaction_id IS NULL").fetchone()[0]
    if null_count == 0:
        print(f"  {GREEN}[PASS]{RESET} test_not_null_stg_transactions_transaction_id")
    else:
        print(f"  {RED}[FAIL]{RESET} test_not_null_stg_transactions_transaction_id")

    # Test 3: Unique transaction_id in enriched_transactions
    dup_enriched = con.execute("SELECT COUNT(*) - COUNT(DISTINCT transaction_id) FROM enriched_transactions").fetchone()[0]
    if dup_enriched == 0:
        print(f"  {GREEN}[PASS]{RESET} test_unique_enriched_transactions_transaction_id")
    else:
        print(f"  {RED}[FAIL]{RESET} test_unique_enriched_transactions_transaction_id")

    total_enriched = con.execute("SELECT COUNT(*) FROM enriched_transactions").fetchone()[0]
    quarantined = con.execute("SELECT COUNT(*) FROM invalid_transactions").fetchone()[0]
    print(f"\n{GREEN}[OK]{RESET} Materialized {total_enriched:,} enriched records ({quarantined:,} quarantined).")

    con.close()
    elapsed = time.time() - start_time
    print(f"{GREEN}{BOLD}Stage 2 dbt Transformations Complete in {elapsed:.2f}s!{RESET}\n")
    return True


# =============================================================================
# STAGE 3: ML MODEL TRAINING (RANDOM FOREST)
# =============================================================================
def run_training():
    print(f"\n{BLUE}{BOLD}=== [STAGE 3] Train Distributed Random Forest Classifier ==={RESET}")
    start_time = time.time()
    
    con = get_db_connection()
    df = con.execute("SELECT * FROM enriched_transactions WHERE is_fraud IS NOT NULL").df()
    con.close()
    
    print(f"Loaded {len(df):,} historically labeled transactions for training.")
    fraud_counts = df['is_fraud'].value_counts().to_dict()
    print(f"Class distribution: Legitimate (0): {fraud_counts.get(0, 0):,}, Fraud (1): {fraud_counts.get(1, 0):,}")
    
    # Feature Engineering
    categorical_cols = ["payment_method", "currency", "payor_country", "payee_country", "payee_category"]
    numerical_cols = ["amount", "payor_risk_score", "payee_risk_score", "merchant_mcc"]
    
    # Preprocessing Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report
    import joblib
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1))
    ])
    
    X = df[numerical_cols + categorical_cols]
    y = df['is_fraud'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training on {len(X_train):,} rows, Testing on {len(X_test):,} rows...")
    
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred_proba = rf_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.50).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"  {BOLD}MODEL EVALUATION METRICS{RESET}")
    print("="*40)
    print(f"  Area Under ROC (AUC)  : {GREEN}{auc:.4f}{RESET}")
    print(f"  Area Under PR (AUPRC) : {GREEN}{auprc:.4f}{RESET}")
    print(f"  Accuracy              : {GREEN}{acc*100:.2f}%{RESET}")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    
    # Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf_pipeline, MODEL_PATH)
    print(f"{GREEN}[OK]{RESET} Saved trained model to {MODEL_PATH}")
    
    elapsed = time.time() - start_time
    print(f"{GREEN}{BOLD}Stage 3 ML Training Complete in {elapsed:.2f}s!{RESET}\n")
    return True


# =============================================================================
# STAGE 4: BATCH INFERENCE & SPANNER REVIEW QUEUE SINK
# =============================================================================
def run_inference():
    print(f"\n{BLUE}{BOLD}=== [STAGE 4] Batch Inference & Cloud Spanner Review Queue Sink ==={RESET}")
    start_time = time.time()
    
    import joblib
    if not os.path.exists(MODEL_PATH):
        print(f"{RED}Model file {MODEL_PATH} not found. Running training first...{RESET}")
        run_training()
        
    model = joblib.load(MODEL_PATH)
    print(f"{GREEN}[OK]{RESET} Loaded trained model from {MODEL_PATH}")
    
    con = get_db_connection()
    unlabeled_df = con.execute("SELECT * FROM enriched_transactions WHERE is_fraud IS NULL").df()
    
    if len(unlabeled_df) == 0:
        print(f"{YELLOW}[WARN] No unlabeled transactions with is_fraud IS NULL. Taking simulated holdout stream for scoring.{RESET}")
        unlabeled_df = con.execute("SELECT * FROM enriched_transactions LIMIT 2500").df()
        
    print(f"Scoring {len(unlabeled_df):,} incoming unlabeled transactions...")
    
    categorical_cols = ["payment_method", "currency", "payor_country", "payee_country", "payee_category"]
    numerical_cols = ["amount", "payor_risk_score", "payee_risk_score", "merchant_mcc"]
    
    X_score = unlabeled_df[numerical_cols + categorical_cols]
    fraud_probs = model.predict_proba(X_score)[:, 1]
    
    unlabeled_df["fraud_probability"] = fraud_probs
    unlabeled_df["prediction"] = (fraud_probs >= 0.50).astype(float)
    
    # Filter high risk (>= 50% fraud probability)
    high_risk_df = unlabeled_df[unlabeled_df["fraud_probability"] >= 0.50].copy()
    print(f"{GREEN}[OK]{RESET} Flagged {len(high_risk_df):,} High-Risk Transactions (Probability >= 50%) for Cloud Spanner audit queue.")
    
    # Match Cloud Spanner SparkEvalFraudReviewQueue schema
    spanner_cols = [
        "transaction_id", "amount", "currency", "device_id", "ip_address", "merchant_mcc",
        "payee_id", "payment_method", "payor_id", "status", "timestamp", "payor_name",
        "payor_country", "payor_risk_score", "payee_name", "payee_country", "payee_risk_score",
        "payee_category", "is_fraud", "prediction"
    ]
    
    spanner_export_df = high_risk_df[[col for col in spanner_cols if col in high_risk_df.columns]].copy()
    spanner_export_df["is_fraud"] = 1
    
    # Write to local Spanner table
    con.execute("CREATE OR REPLACE TABLE SparkEvalFraudReviewQueue AS SELECT * FROM spanner_export_df")
    
    sample_alerts = con.execute("SELECT transaction_id, payor_name, payee_name, amount, prediction FROM SparkEvalFraudReviewQueue LIMIT 5").df()
    print("\nSample High-Risk Cloud Spanner Review Queue Entries:")
    print(sample_alerts.to_string(index=False))
    
    con.close()
    elapsed = time.time() - start_time
    print(f"\n{GREEN}{BOLD}Stage 4 Batch Inference Complete in {elapsed:.2f}s!{RESET}\n")
    return True


# =============================================================================
# MAIN CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Zero-Credit Cymbal Fraud Detection Pipeline Runner")
    parser.add_argument("--stage", choices=["all", "ingestion", "dbt", "training", "inference"], default="all",
                        help="Pipeline stage to execute (default: all)")
    args = parser.parse_args()
    
    print(f"\n{GREEN}===================================================================={RESET}")
    print(f"{GREEN}   Cymbal Pay Fraud Pipeline: Local Zero-Credit Execution Engine    {RESET}")
    print(f"{GREEN}===================================================================={RESET}\n")
    
    if args.stage in ["all", "ingestion"]:
        if not run_ingestion():
            sys.exit(1)
    if args.stage in ["all", "dbt"]:
        if not run_dbt():
            sys.exit(1)
    if args.stage in ["all", "training"]:
        if not run_training():
            sys.exit(1)
    if args.stage in ["all", "inference"]:
        if not run_inference():
            sys.exit(1)
            
    print(f"\n{GREEN}{BOLD}===================================================================={RESET}")
    print(f"{GREEN}{BOLD}*** ALL PIPELINE STAGES COMPLETED SUCCESSFULLY WITH ZERO GCP CHARGES! ***{RESET}")
    print(f"{GREEN}{BOLD}===================================================================={RESET}\n")


if __name__ == "__main__":
    main()
