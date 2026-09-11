{{ config(materialized='table') }}

WITH valid_transactions AS (
    SELECT * FROM {{ ref('stg_transactions') }}
),

payers AS (
    SELECT
        payor_id,
        payor_name,
        payor_country,
        CAST(payor_risk_score AS FLOAT64) AS payor_risk_score
    FROM {{ source('transactions_dataset_evals', 'dim_payers') }}
),

payees AS (
    SELECT
        payee_id,
        payee_name,
        payee_country,
        CAST(payee_risk_score AS FLOAT64) AS payee_risk_score,
        payee_category
    FROM {{ source('transactions_dataset_evals', 'dim_payees') }}
)

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
    
    -- Payer dimensional features
    t.payor_id,
    p.payor_name,
    COALESCE(p.payor_country, 'UNKNOWN') AS payor_country,
    COALESCE(p.payor_risk_score, 0.0) AS payor_risk_score,
    
    -- Payee dimensional features
    t.payee_id,
    e.payee_name,
    COALESCE(e.payee_country, 'UNKNOWN') AS payee_country,
    COALESCE(e.payee_risk_score, 0.0) AS payee_risk_score,
    COALESCE(e.payee_category, 'General') AS payee_category,
    
    -- Target classification label (NULL for new / unlabeled records)
    t.is_fraud

FROM valid_transactions t
LEFT JOIN payers p ON t.payor_id = p.payor_id
LEFT JOIN payees e ON t.payee_id = e.payee_id
