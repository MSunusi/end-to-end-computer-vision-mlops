# End-to-End Computer Vision MLOps

This repository documents the design and implementation of a **production-oriented computer vision system**, built incrementally using modern MLOps principles.

The project is structured as a **multi-stage journey**, starting from dataset validation and baseline training, and progressing toward experiment tracking, orchestration, deployment, and monitoring.

This is not a collection of isolated notebooks.
It is a system built step by step.

---

## 🎯 Project Objective

The goal is to demonstrate how a **real-world object detection project** can be taken from raw images to a production-ready pipeline using industry-aligned MLOps practices.

The use case is agricultural computer vision (sesame plant detection), but the architecture and lessons generalize to any detection problem.

---

## 🧱 Project Structure

The repository is organized by **stages**, not tools.

Each stage introduces new capabilities while building on artifacts from the previous one.

```text
end-to-end-computer-vision-mlops/
│
├── data/                          # Dataset (kept outside Git history)
│
├── 01-validation-training/        # Dataset validation + baseline model
│   ├── validation.ipynb
│   ├── training.ipynb
│   ├── dataset_summary.csv
│   └── README.md
│
├── 02-experiment-tracking/        # MLflow-based experiment tracking (completed)
│   ├── experiment_track_on_cpu.ipynb
│   ├── experiment_tracking_on_colab.ipynb
│   └── README.md
│
├── 03-orchestration/              # Airflow (planned)
├── 04-serving/                    # Docker + API (planned)
├── 05-ci-cd-monitoring/           # CI/CD + monitoring (planned)
│
└── README.md

````

## 🖥️ Development Environment

Operating System: Ubuntu 24.04

Python Version: Python 3.12 

## Note
this repo contain only the technical implemenation of mlops. to learn deeper for mlops follow this: https://github.com/DataTalksClub/mlops-zoomcamp.git
