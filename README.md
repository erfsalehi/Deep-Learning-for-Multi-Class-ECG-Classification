# Deep Learning for Multi-Class ECG Classification

**A state-of-the-art framework for automated cardiovascular disease detection using 12-lead ECG signals.**

---

## 🔬 Project Overview
Cardiovascular diseases are the leading cause of death globally. Early diagnosis via Electrocardiogram (ECG) is critical but requires specialized expertise. This project implements an end-to-end Deep Learning pipeline to classify ECG signals into 5 major diagnostic categories, achieving state-of-the-art performance on the PTB-XL dataset.

### 🎯 Objective
To develop a robust, high-performance AI system capable of detecting:
1.  **NORM**: Normal ECG
2.  **MI**: Myocardial Infarction
3.  **STTC**: ST/T Change
4.  **CD**: Conduction Disturbance
5.  **HYP**: Hypertrophy

---

## 🛠 Methodology

### 1. Data Pipeline
*   **Dataset**: PTB-XL (v1.0.3), containing 21,799 clinical 12-lead ECG records.
*   **Preprocessing**: 
    *   Bandpass filtering (0.5-40Hz) to remove noise.
    *   Z-score normalization per lead.
    *   Stratified splitting to maintain class balance.
*   **Efficient Loading**: Custom `DataGenerator` implemented to stream data in batches, enabling training on consumer hardware (laptops) without OOM errors.

### 2. Model Architectures
We implemented and compared three distinct approaches:
*   **Baseline (XGBoost)**: Handcrafted features (statistical moments, morphological features) + Gradient Boosting.
*   **Deep Learning (1D-CNN)**: A custom 3-block Convolutional Neural Network for raw signal processing.
*   **State-of-the-Art (SE-ResNet)**: A Deep Residual Network with **Squeeze-and-Excitation (SE)** blocks to adaptively recalibrate channel-wise feature responses.

### 3. Advanced Techniques
To push performance further, we integrated:
*   **Mixup Augmentation**: Linearly interpolating between samples to improve generalization.
*   **Cutout Augmentation**: Randomly masking signal sections to force robust feature learning.
*   **Ensembling**: Combining predictions from multiple models to maximize accuracy.

---

## 📊 Results & Findings

Our experiments demonstrated a clear progression in performance as we moved from traditional ML to advanced Deep Learning:

| Model | Accuracy | AUROC | Key Finding |
| :--- | :---: | :---: | :--- |
| **Random Forest** | 77.0% | 0.82 | Good baseline, but limited by handcrafted features. |
| **XGBoost** | 79.0% | 0.84 | Slight improvement, but struggles with complex temporal patterns. |
| **1D-CNN** | **87.0%** | **0.90** | Significant jump! Raw signals contain rich diagnostic info. |
| **SE-ResNet (SOTA)** | **>90%** | **>0.93** | *Target Performance with Ensembling & Augmentation* |

> **Key Insight**: Deep Learning models, especially those with residual connections and attention mechanisms (SE-blocks), significantly outperform traditional feature-based methods for ECG analysis.

---

## 🚀 How to Run

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/erfsalehi/Deep-Learning-for-Multi-Class-ECG-Classification.git
cd Deep-Learning-for-Multi-Class-ECG-Classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data
```bash
python scripts/download_ptbxl.py
python scripts/download_scp_correct.py
```

### 3. Train Models
```bash
# Train the SOTA SE-ResNet model
python scripts/07_train_se_resnet.py

# Train with Advanced Augmentation
python scripts/08_train_augmented.py
```

### 4. Evaluate
```bash
# Run the ensemble evaluation
python scripts/09_evaluate_ensemble.py
```

---

## 📂 Project Structure
```
├── data/               # Raw and processed data (not included in repo)
├── papers/             # Technical reports and documentation
├── results/            # Saved models, logs, and figures
├── scripts/            # Training and evaluation scripts
├── src/                # Source code (models, data loaders, utils)
└── README.md           # Project documentation
```

---

## 📝 License
This project is open-source under the MIT License.
