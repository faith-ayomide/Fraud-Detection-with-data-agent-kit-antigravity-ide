{{ config(materialized='view') }}

-- Quarantine invalid records missing valid transaction IDs or corrupted metadata
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
    CURRENT_TIMESTAMP() AS quarantined_at,
    CASE 
        WHEN transaction_id IS NULL OR TRIM(transaction_id) = '' THEN 'MISSING_TRANSACTION_ID'
        WHEN amount <= 0 OR amount IS NULL THEN 'INVALID_AMOUNT'
        ELSE 'OTHER_ANOMALY'
    END AS quarantine_reason
FROM {{ source('transactions_dataset_evals', 'raw_transactions') }}
WHERE transaction_id IS NULL 
   OR TRIM(transaction_id) = ''
   OR amount <= 0
   OR amount IS NULL
