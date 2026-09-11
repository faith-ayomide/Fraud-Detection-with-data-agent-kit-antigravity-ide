{{ config(materialized='view') }}

WITH ranked_transactions AS (
    SELECT
        transaction_id,
        timestamp,
        payor_id,
        payee_id,
        CAST(amount AS FLOAT64) AS amount,
        currency,
        payment_method,
        status,
        CAST(merchant_mcc AS INT64) AS merchant_mcc,
        device_id,
        ip_address,
        CAST(is_fraud AS INT64) AS is_fraud,
        ROW_NUMBER() OVER (
            PARTITION BY transaction_id 
            ORDER BY timestamp DESC
        ) AS row_num
    FROM {{ source('transactions_dataset_evals', 'raw_transactions') }}
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
FROM ranked_transactions
WHERE row_num = 1
