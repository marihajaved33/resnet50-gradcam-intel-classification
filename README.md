# ResNet50 Fine-Tuning with Grad-CAM Explainability

A deep learning project implementing two-phase transfer learning using **ResNet50** on the **Intel Image Classification** dataset, integrated with **Grad-CAM (Gradient-weighted Class Activation Mapping)** for visual explainability.

---

## 📌 Project Overview

This repository demonstrates an end-to-end Computer Vision pipeline designed to classify natural scene images into six distinct categories: `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`.

To maintain both high classification performance and model interpretability, this project implements:
1. **Two-Phase Transfer Learning:** Sequential feature extraction and gradual fine-tuning of deep convolutional layers.
2. **Explainable AI (XAI):** Grad-CAM heatmaps highlighting regions of interest that drive the model's predictions.

---

## 🛠️ Key Features

- **Dynamic Dataset Management:** Automatic streaming/download using `kagglehub`.
- **Data Augmentation:** Robust training augmentations including zoom, rotation, spatial shifts, vertical/horizontal flips, and brightness jitter.
- **Two-Phase Training Strategy:**
  - **Phase 1 (Head Warm-up):** Base model layers frozen; training custom classification head with Batch Normalization and Dropout.
  - **Phase 2 (Fine-Tuning):** Unfreezing the top 30 layers of ResNet50 with a lower learning rate (`1e-5`).
- **Grad-CAM Visualizations:**
  - Dynamic gradient extraction on layer `conv5_block3_out`.
  - Single-image and batch overlay heatmaps colorized with OpenCV's `JET` colormap.
- **Comprehensive Evaluation:** Detailed classification metrics, confusion matrix, loss/accuracy curves, and visual sample predictions.

---

## 📁 Repository Structure

```text
.
├── GradCAM_ResNet50_Improved.ipynb    # Main Google Colab Notebook
├── README.md                          # Project Documentation
├── .gitignore                         # Git Ignore Rules
└── best_phase2.keras                  # Saved Checkpoint (Generated locally)
```

---

## ⚙️ Quick Start

### Running in Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `GradCAM_ResNet50_Improved.ipynb`.
3. Set runtime environment to **GPU**: `Runtime > Change runtime type > T4 GPU`.
4. Run all cells sequentially.

---

## 📊 Dataset

- **Dataset Name:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Classes (6):** `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`
- **Image Resolution:** $224 \times 224$ pixels

---

## 🔬 Model Architecture & Training Details

| Phase | Target Layers | Optimizer | Base Learning Rate | Callbacks |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Classification Head | Adam | `1e-3` | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| **Phase 2** | Last 30 Layers + Head | Adam | `1e-5` | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## 📈 Explainable AI (Grad-CAM)

Grad-CAM extracts feature activation maps from the final convolutional layer (`conv5_block3_out`) and computes positive gradients relative to predicted target classes.

```python
# Core Grad-CAM extraction logic
with tf.GradientTape() as tape:
    conv_outputs, preds = grad_model(img_array)
    class_channel = preds[:, pred_index]

grads = tape.gradient(class_channel, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
```

---

## 🧰 Dependencies

- `tensorflow >= 2.10.0`
- `kagglehub`
- `opencv-python`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `numpy`

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
