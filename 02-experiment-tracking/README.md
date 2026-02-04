# Stage 02: Experiment Tracking with MLflow

After validating the dataset and establishing a functional baseline in Stage 01, the next engineering priority is systematic capture of training runs. Without tracking, hyperparameter choices, metric evolution, and model artifacts become irreproducible and hard to compare — especially when iterating over multiple architectures (YOLOv8n, YOLO11n, YOLO26n).

Stage 02 introduces MLflow to log:

- Hyperparameters
- Training/validation metrics (mAP@50, mAP@50:95, precision, recall, losses)
- Artifacts (model weights, dataset summary CSV, training plots)
- Run metadata (source code version, timestamps, tags)

This creates a searchable history of experiments, enabling side-by-side comparison in the MLflow UI and providing a foundation for future orchestration and model selection.

## Why Experiment Tracking Matters After Validation

Dataset quality is necessary but insufficient. Training introduces variability: different models, learning rates, batch sizes, and random seeds. Tracking ensures every decision is recorded, allowing teams to:

- Reproduce exact runs
- Identify configurations that improve metrics
- Avoid repeating failed experiments
- Maintain organization across long iteration cycles

## MLflow Overview (Practical Context)

MLflow is an open-source Python library (`pip install mlflow`) that covers the ML lifecycle. For this stage we use only the **Tracking** module, which organizes runs inside experiments and captures:

- Parameters (`log_param`)
- Metrics (`log_metric`)
- Artifacts (`log_artifact`, `log_artifacts`)
- Models (`mlflow.pytorch.log_model` or manual artifact logging)
- Automatic metadata (git commit, start/end time, source file)

Ultralytics YOLO provides partial callback-based integration, but we use explicit logging for full control and clarity.

## Installation (Ubuntu Local Machine)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python essentials
sudo apt install python3 python3-pip python3-venv -y

# Create dedicated directory and environment
mkdir -p 02-experiment-tracking && cd 02-experiment-tracking
python3 -m venv mlflow-env
source mlflow-env/bin/activate

# Install MLflow and dependencies#Local Tracking Server
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Open http://localhost:5000 in your browser.

## Colab and Local Tracking (Using ngrok for Remote Access)

  ```bash
#nstall ngrok (Ubuntu):
sudo snap install ngrok
# Or download from https://ngrok.com/download
#Authenticate (sign up at ngrok.com for free authtoken)
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE

#Expose local MLflow UI:
ngrok http 5000
# Copy the forwarding URL (e.g. https://abc123.ngrok-free.app)
```

## Read article for this: https://medium.com/@sunusimuhammada/2c4bd254fa0f
# Readmore about MLflow: https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/02-experiment-tracking
