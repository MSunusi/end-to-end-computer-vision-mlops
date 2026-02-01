# 🚀 End-to-End Computer Vision MLOps  
## Stage 01 — Data Validation & Model Training

This project is a **hands-on, experiment-driven journey** into building a **production-ready Computer Vision system with MLOps best practices**.

Instead of starting with heavy theory, we begin with **real experiments**, clean structure, and reproducible workflows — the way it’s done in industry.

---

## 🔍 What This Stage Covers (Stage 01)

In this first stage, we focus on **building a solid foundation**:

### ✅ 1. Data Validation
Before training any model, we ensure:
- Dataset paths are correct
- Images and labels are consistent
- Classes are properly defined
- Dataset structure follows expected standards

📌 *Why this matters:*  
Bad data = bad models. Validation saves time and compute.

---

### 🧠 2. Model Training
Once data is validated, we:
- Load the dataset
- Configure training parameters
- Train a baseline Computer Vision model
- Track metrics for future comparison

Short example (simplified):

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50)
