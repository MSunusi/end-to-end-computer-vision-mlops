from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


# -------------------------
# Default DAG Arguments
# -------------------------
default_args = {
    "owner": "sunusi",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# -------------------------
# Define DAG
# -------------------------
with DAG(
    dag_id="computer_vision_training_pipeline",
    description="YOLO training + MLflow tracking pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # manual trigger for now
    catchup=False,
    tags=["mlops", "computer-vision"],
) as dag:

    train_model = BashOperator(
        task_id="train_and_track_yolo",
        bash_command="""
        cd ~/end-to-end-computer-vision-mlops &&
        python 03-orchestration/scripts/train_and_track.py
        """
    )

    train_model

