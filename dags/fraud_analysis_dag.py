"""
Airflow DAG: fraud_analysis_pipeline
Compiled from fraud_analysis_pipeline.yaml by Google Cloud Data Agent Kit Orchestrator.
Orchestrates Ingestion Notebook -> dbt Transformations -> Batch Inference Notebook.
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'cymbal-financial-data-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='fraud_analysis_pipeline',
    default_args=default_args,
    description='End-to-end fraud detection analytical pipeline orchestrating log ingestion, dbt transformations, and batch ML scoring.',
    schedule_interval='0 0 * * *',
    catchup=False,
    max_active_runs=1,
    tags=['fraud-detection', 'dataproc-serverless', 'dbt', 'spanner'],
) as dag:

    # Task 1: Ingestion Notebook on Dataproc Serverless
    ingest_logs = BashOperator(
        task_id='ingestion_notebook',
        bash_command=(
            'python -m papermill notebooks/01_ingestion.ipynb notebooks/01_ingestion_out.ipynb '
            '|| python local_runner.py --stage ingestion'
        ),
        doc_md="Ingests raw clearinghouse JSON logs into the tabular raw layer.",
    )

    # Task 2: dbt Transformations & Data Quality Tests
    run_dbt_models = BashOperator(
        task_id='dbt_transformations',
        bash_command='cd dbt_project && (dbt build || python ../local_runner.py --stage dbt)',
        doc_md="Deduplicates logs, isolates invalid records, and enriches data with payer/payee dimensions.",
    )

    # Task 3: Batch Inference Notebook on Dataproc Serverless
    batch_inference = BashOperator(
        task_id='batch_inference_notebook',
        bash_command=(
            'python -m papermill notebooks/03_inference.ipynb notebooks/03_inference_out.ipynb '
            '|| python local_runner.py --stage inference'
        ),
        doc_md="Scores unlabeled transactions and routes >= 50% high-risk fraud alerts to Cloud Spanner.",
    )

    # Dependency Graph: Ingestion -> dbt -> Inference
    ingest_logs >> run_dbt_models >> batch_inference
