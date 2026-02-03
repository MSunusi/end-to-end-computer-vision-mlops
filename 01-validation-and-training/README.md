# Stage 01: Dataset Validation & Baseline Training

This stage establishes the minimum reliable foundation for any production-grade computer vision pipeline: a verified dataset and a reproducible baseline model.

In production environments, training on unvalidated data wastes compute resources and masks downstream problems. This stage enforces dataset integrity before any modeling effort begins and provides a simple, repeatable baseline that confirms the full data-to-prediction path functions correctly.

## What This Stage Covers

- Systematic validation of YOLO-format dataset (images + labels)
- Detection of structural defects: missing label files, unreadable/corrupted images
- Basic dataset statistics: image counts per split, class distribution
- Baseline training and inference using Ultralytics YOLO 
 



## Folder Structure
01-validation-training/
├── validation.ipynb           # Dataset integrity checks and reporting
├── training.ipynb             # Baseline model training and sample inference
├── requirements.txt           # Exact dependencies pinned for this stage
├── dataset_summary.csv        # Generated validation report (CSV)
└── README.md                  


Training outputs (`runs/`) are git-ignored.

## Validation Logic

The validation notebook performs three checks:

1. **Missing labels**  
   For every `.jpg` in the images directory, verify existence of corresponding `.txt` file with identical stem.

2. **Corrupted / unreadable images**  
   Attempt to load each image with OpenCV (`cv2.imread`). Flag files that return `None` or zero-sized arrays.

3. **Class distribution**  
   Parse every label file, extract class IDs (first token per line), and tally occurrences per split.

Results are printed in a structured report and saved as `dataset_summary.csv` containing:

- Image counts (train / valid)
- Number of missing labels per split
- Number of corrupted images per split
- Class distribution (train / valid)
- Total unique classes detected

These metrics serve as a gate: proceed to training only when validation issues are resolved or explicitly accepted.

## Baseline Training 

The training notebook trains a single Ultralytics YOLO model from a pretrained checkpoint on the provided `data.yaml`.

**Purpose of the baseline**  
- Confirm the dataset is loadable by the training framework
- Verify label format compatibility
- Produce initial predictions to visually sanity-check detections

 


clone and run on your enviroment:

```bash
cd 01-validation-training
pip install -r requirements.txt
# Run validation.ipynb first
# Then training.ipynb
