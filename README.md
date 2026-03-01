# Deep Learning for Multi-Class ECG Classification

**A state-of-the-art framework for automated cardiovascular disease diagnostic classification using 12-lead ECG signals.**

---

## 🔬 Project Overview
Cardiovascular diseases are the leading cause of death globally. Early diagnosis via Electrocardiogram (ECG) is critical but requires specialized expertise. This project implements an end-to-end Deep Learning pipeline to classify ECG signals into 5 major diagnostic superclasses, achieving state-of-the-art (SOTA) multi-label performance on the PTB-XL dataset and demonstrating strong external generalization.

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
*   **Datasets**: Trained and internally validated on **PTB-XL** (v1.0.3, 21,799 records). Externally validated on the **Chapman-Shaoxing** dataset (10,646 records).
*   **Preprocessing**: 
    *   Signals resampled to 100 Hz to optimize computational efficiency.
    *   Bandpass filtering (0.5-40Hz) to remove baseline wander and noise.
    *   Z-score normalization applied per lead.
    *   Stratified splitting (using `strat_fold`) to maintain multi-label class balance.
*   **Efficient Loading**: Custom `NpyECGDataGenerator` implemented to stream data in batches, enabling training on consumer hardware without OOM errors.

### 2. Model Architecture: SE-ResNet
The core architecture is a **Squeeze-and-Excitation Residual Network (SE-ResNet)**. 
*   **ResNet Backbone**: Captures complex temporal dependencies via skip connections.
*   **SE Blocks**: Adaptively recalibrate channel-wise feature responses, allowing the network to explicitly model interdependencies between the 12 ECG leads and focus on the most diagnostically relevant channels.

### 3. Advanced Techniques
To address heavy class imbalance and improve generalization, we integrated:
*   **Mixup Augmentation**: Creating synthetic training samples by linearly interpolating between two signals to encourage robust, smooth decision boundaries.
*   **Focal Loss**: A specialized formulation of binary cross-entropy that down-weights easily classified examples (like NORM) and heavily penalizes errors on rare, hard-to-classify conditions (like HYP).
*   **Ensembling**: Combining probability outputs from standard SE-ResNet and Focal Loss SE-ResNet models.

---

## 📊 Results & Findings

Our final SOTA Ensemble model was evaluated on the standardized PTB-XL **Fold 10** test set:

| Model | Macro-AUC | Macro-F1 | F1-NORM | F1-MI | F1-STTC | F1-CD | F1-HYP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline SE-ResNet** | 0.917 | 0.594 | 0.909 | 0.569 | 0.596 | 0.574 | 0.320 |
| **Focal Loss Variant** | 0.914 | 0.601 | 0.908 | 0.577 | 0.612 | 0.585 | 0.324 |
| **Ensemble (SOTA)** | **0.932** | **0.642** | **0.915** | **0.612** | **0.645** | **0.621** | **0.418** |

**External Validation:**
When tested on the demographic shift of the **Chapman-Shaoxing dataset**, the ensemble maintained strong generalization, achieving an **AUROC of 0.844**.

> **Key Insight**: The combination of Squeeze-and-Excitation blocks for lead-wise attention, Focal Loss for imbalance, and Mixup augmentation yields a highly robust multi-label diagnostic tool.

---

## 🚀 How to Run

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/erfsalehi/Deep-Learning-for-Multi-Class-ECG-Classification.git
cd Deep-Learning-for-Multi-Class-ECG-Classification

# Install dependencies (Python 3.8+ recommended)
pip install -r requirements.txt
```

### 2. Download and Preprocess Data
```bash
python scripts/download_ptbxl.py
python scripts/download_scp_correct.py
python preprocess_ptbxl.py
```

### 3. Train Models
```bash
# Train the Baseline SE-ResNet
python training/train_se_resnet.py

# Train the Focal Loss SE-ResNet Variant
python training/train_focal_loss.py

# Train Augmentation Ablations (Mixup, Noise, etc.)
python training/train_ablations.py
```

### 4. Evaluate & Generate Figures
```bash
# Run Fold 10 Evaluation to generate metrics CSV
python evaluation/final_eval.py

# Generate publication-quality figures (saved to results/figures/publication/)
python figures/generate_publication_figures.py
```

---

## 📂 Project Structure
```
├── data/               # Raw and processed datasets
├── evaluation/         # Evaluation, calibration, and Grad-CAM scripts
├── figures/            # Scripts for generating publication figures
├── papers/             # Technical reports, manuscript drafts, and cover letters
├── results/            # Saved models, metrics CSVs, and output figures
├── scripts/            # Data download utilities
├── src/                # Source code (models, data loaders, augmentation)
├── training/           # Model training execution scripts
└── README.md           # Project documentation
```

---

## 📝 License
This project is open-source under the MIT License.
