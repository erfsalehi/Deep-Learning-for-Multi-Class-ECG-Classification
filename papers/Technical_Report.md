# Automated Detection of Cardiovascular Abnormalities using Deep Learning on 12-Lead ECG Signals

**Date:** February 2026  
**Project:** Cardiovascular AI Portfolio  
**Repository:** [GitHub Repository Link]

---

## Abstract

Cardiovascular diseases (CVDs) remain the leading cause of death globally. Early and accurate diagnosis using Electrocardiograms (ECG) is critical but requires specialized medical expertise. This study presents a comparative analysis of machine learning approaches for the automated classification of Normal vs. Abnormal ECG signals using the large-scale PTB-XL dataset. We implemented a robust pipeline comparing traditional machine learning (XGBoost with handcrafted features) against deep learning architectures (1D-CNN and ResNet). Our results demonstrate that deep learning models significantly outperform feature-based baselines, achieving an accuracy of **87%** and an Area Under the Receiver Operating Characteristic (AUROC) of **0.90** with the 1D-CNN model, with further improvements expected from the ResNet architecture. These findings suggest that deep learning can serve as a powerful assistive tool for rapid cardiac screening.

---

## 1. Introduction

The 12-lead Electrocardiogram (ECG) is the gold standard for non-invasive cardiac monitoring. However, manual interpretation is time-consuming and subject to inter-observer variability. With the advent of large public datasets like **PTB-XL**, data-driven approaches have become feasible.

This project aims to:
1.  Develop an end-to-end pipeline for ECG signal processing.
2.  Establish a strong baseline using traditional machine learning and handcrafted features.
3.  Implement and evaluate state-of-the-art deep learning models (1D-CNN and ResNet) for direct signal classification.
4.  Demonstrate the feasibility of training high-performance models on consumer-grade hardware through efficient data engineering.

---

## 2. Methodology

### 2.1 Dataset
We utilized the **PTB-XL** dataset (Version 1.0.3), comprising **21,799** clinical 12-lead ECG records from 18,885 patients. Each record is 10 seconds long, sampled at 100 Hz or 500 Hz. For this study, we used the 100 Hz version to optimize computational efficiency without sacrificing significant diagnostic information.

### 2.2 Preprocessing Pipeline
To ensure data quality, we implemented a uniform preprocessing pipeline:
*   **Bandpass Filtering**: A 4th-order Butterworth filter (0.5–40 Hz) was applied to remove baseline wander and high-frequency noise.
*   **Normalization**: Z-score normalization was applied to each lead independently to standardize signal amplitude ($\mu=0, \sigma=1$).
*   **Data Splitting**: The dataset was split into Training (80%) and Validation/Test (20%) sets, stratified by the "Normal" vs. "Abnormal" label to maintain class distribution.

### 2.3 Baseline Model: Feature Engineering + XGBoost
We extracted **96 handcrafted features** per record (8 features $\times$ 12 leads):
*   Statistical moments: Mean, Standard Deviation, Skewness, Kurtosis.
*   Morphological/Signal features: Minimum, Maximum, Peak-to-Peak Range, Zero-Crossing Rate.

These features were fed into an **XGBoost Classifier** (200 estimators, max_depth=6), a gradient boosting algorithm known for its performance on tabular data.

### 2.4 Deep Learning Models
We moved beyond handcrafted features to learn representations directly from the raw waveform (Shape: $1000 \times 12$).

#### A. 1D-Convolutional Neural Network (1D-CNN)
We designed a custom 3-block CNN architecture:
*   **Layers**: 3 Convolutional blocks with increasing filters (32, 64, 128).
*   **Mechanisms**: Batch Normalization and ReLU activation for stability; Max Pooling for dimensionality reduction.
*   **Head**: Global Average Pooling followed by a Dense layer (64 units) and a Softmax output.

#### B. Residual Network (ResNet)
To capture deeper temporal dependencies, we implemented a ResNet-based architecture:
*   **Structure**: A series of **Residual Blocks** containing skip connections.
*   **Function**: Skip connections allow gradients to flow through deeper networks, preventing the vanishing gradient problem and enabling the model to learn complex signal patterns.
*   **Optimization**: Trained with the Adam optimizer, Learning Rate Reduce-on-Plateau, and Early Stopping.

---

## 3. Results

### 3.1 Quantitative Performance
The models were evaluated on a held-out test set. Key metrics include Accuracy, Precision, Recall, and F1-Score.

| Model | Accuracy | Precision (Abnormal) | Recall (Abnormal) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | 77.0% | 0.76 | 0.75 | 0.76 |
| **XGBoost (Optimized Baseline)** | 79.0% | 0.78 | 0.77 | 0.78 |
| **1D-CNN (Deep Learning)** | **87.0%** | **0.90** | **0.85** | **0.87** |
| **ResNet** | *Training* | *Training* | *Training* | *Training* |

*> Note: The ResNet model is currently training on the full dataset, with early epochs showing >80% accuracy and >0.90 AUROC.*

### 3.2 Visual Analysis
*   **Confusion Matrices**: The 1D-CNN significantly reduced False Negatives compared to the baseline, meaning fewer abnormal cases were missed.
*   **ROC Curves**: The 1D-CNN achieved an **AUROC of 0.90**, indicating excellent discrimination capability across different decision thresholds.

---

## 4. Discussion

### Feature Engineering vs. Representation Learning
The jump in performance from XGBoost (79%) to 1D-CNN (87%) highlights the limitation of handcrafted features. While statistical moments capture basic signal properties, they fail to encode the complex temporal morphology (e.g., ST-segment elevation, T-wave inversion) that deep learning models can learn automatically.

### Efficiency and Scalability
A key achievement of this project was the implementation of a **Custom Data Generator**. This allowed us to train memory-intensive deep learning models on standard hardware (laptop) by streaming data in batches rather than loading the entire 21,000-record dataset into RAM.

---

## 5. Conclusion and Future Work

We successfully developed a high-performance ECG analysis system capable of detecting cardiac abnormalities with **87% accuracy**. The transition from traditional ML to Deep Learning yielded a **+8% absolute improvement** in accuracy.

**Future directions include:**
1.  **Multi-Class Classification**: Extending the binary model to classify specific subclasses (MI, Hypertrophy, Arrhythmia).
2.  **Transformer Architectures**: Experimenting with 1D-Vision Transformers or LSTM-based networks.
3.  **Model Explainability**: Implementing Grad-CAM to visualize which parts of the ECG signal the CNN is focusing on, increasing clinical trust.

---

## Appendix: Technical Stack
*   **Language**: Python 3.8+
*   **Deep Learning**: TensorFlow/Keras
*   **Data Processing**: Pandas, NumPy, WFDB, NeuroKit2
*   **Visualization**: Matplotlib, Seaborn
