# 🚗 Driver Drowsiness Detection

A deep learning project using **MobileNetV2 Transfer Learning** to detect driver drowsiness from facial images.

## 📁 Dataset
- **Source:** [Driver Drowsiness Dataset (DDD) — Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- **Total images:** 41,793
  - Drowsy: 22,348 images
  - Non Drowsy: 19,445 images

## 🧠 Model Architecture
- **Base:** MobileNetV2 (pre-trained on ImageNet)
- **Approach:** Transfer Learning + Fine-tuning
- **Input size:** 224 x 224 pixels
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy

## 📊 Training Results

| Phase | Description | Val Accuracy |
|-------|-------------|-------------|
| Phase 1 | Frozen base + custom head | 65% |
| Phase 2 | Fine-tune top layers | 72% |
| Phase 3 | Fine-tune + stronger augmentation | **76%** ✅ |

## 🔍 Error Analysis
- Performed basic error analysis (False Positives & False Negatives)
- Applied **Grad-CAM** visualizations to interpret model decisions
- Found that correct predictions focus on the **eye region**
- Wrong predictions were distracted by **chin, neck, or hands**

## 📂 Project Files

| File | Description |
|------|-------------|
| `notebook.ipynb` | Full Colab notebook |
| `best_model_v2.keras` | Trained model weights |
| `confusion_matrix.png` | Confusion matrix |
| `training_history.png` | Accuracy & loss curves |
| `sample_predictions.png` | Sample predictions |
| `error_analysis.png` | Wrong predictions analysis |
| `gradcam_correct.png` | Grad-CAM correct predictions |
| `gradcam_wrong.png` | Grad-CAM wrong predictions |

## 👥 Tools Used
- Python, TensorFlow, Keras
- Google Colab (GPU)
- Kaggle Dataset
- Grad-CAM for model interpretability
