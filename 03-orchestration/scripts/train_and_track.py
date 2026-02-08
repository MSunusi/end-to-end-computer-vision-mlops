# 03-orchestration/scripts/train_and_track.py

import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from pathlib import Path


def train_and_log(
    model_file: str,
    data_yaml: str,
    experiment_name: str,
    epochs: int = 1,
    batch: int = 4,
    lr0: float = 0.01,
    device: str = "cpu",
):
    # MLflow setup
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_file):
        mlflow.set_tag("stage", "stage-03-orchestration")
        mlflow.set_tag("developer", "sunusi")

        # Log parameters
        mlflow.log_param("model", model_file)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch", batch)
        mlflow.log_param("lr0", lr0)
        mlflow.log_param("device", device)

        # Train model
        model = YOLO(model_file)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            lr0=lr0,
            device=device,
        )

        # Log metrics
        mlflow.log_metric("mAP50", results.box.map50)
        mlflow.log_metric("mAP50-95", results.box.map)
        mlflow.log_metric("precision", results.box.mp)
        mlflow.log_metric("recall", results.box.mr)

        # Save and log model
        output_model = Path("artifacts") / f"{model_file.replace('.pt','')}_model.pt"
        output_model.parent.mkdir(exist_ok=True)

        model.save(str(output_model))
        mlflow.log_artifact(str(output_model))

        # Log raw PyTorch model
        mlflow.pytorch.log_model(model.model, "torch-model")


if __name__ == "__main__":
    train_and_log(
        model_file="yolov8n.pt",
        data_yaml="data/sesame-plant-detection/data.yaml",
        experiment_name="Stage-03-YOLO-Training",
        epochs=1,
        batch=4,
        lr0=0.01,
        device="cpu",
    )

