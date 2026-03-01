# Automated Multi-Label Classification of 12-Lead ECG Signals using Squeeze-and-Excitation Residual Networks

**Date:** March 2026  
**Project:** Cardiovascular AI Portfolio - SOTA Phase  

---

## Abstract

Cardiovascular diseases (CVDs) remain the leading cause of death globally. Early diagnosis via 12-lead Electrocardiogram (ECG) is critical but requires specialized medical expertise. This study presents a state-of-the-art (SOTA) deep learning framework for the automated multi-label classification of ECG signals into five major diagnostic superclasses: Normal (NORM), Myocardial Infarction (MI), ST/T Change (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP). By leveraging **Squeeze-and-Excitation Residual Networks (SE-ResNet)** combined with **Focal Loss** and **Mixup Augmentation**, our ensemble model achieved a **Macro-AUC of 0.932** and a **Macro-F1 of 0.642** on the PTB-XL dataset. Furthermore, we demonstrated robust generalization through external validation on the Chapman-Shaoxing dataset, achieving an AUROC of 0.844. These results suggest that advanced deep learning architectures can significantly enhance the accuracy and reliability of automated cardiac diagnostics.

---

## 1. Introduction

The 12-lead ECG is the non-invasive gold standard for heart monitoring. However, the increasing volume of data and the shortage of specialized cardiologists demand automated solutions. Traditional machine learning relies on handcrafted features that often miss subtle temporal abnormalities. 

This project transitions from simple binary screening to a comprehensive **multi-label diagnostic system**. Our approach addresses three core challenges in ECG AI:
1.  **Class Imbalance**: Utilizing Focal Loss to prioritize learning from rare but critical conditions like Hypertrophy.
2.  **Generalization**: Implementing Mixup augmentation and SE-blocks to reduce overfitting and improve lead-wise feature selection.
3.  **Trustworthiness**: Combining high-performance ensembling with Grad-CAM interpretability and Expected Calibration Error (ECE) analysis.

---

## 2. Methodology

### 2.1 Dataset and Preprocessing
We utilized the **PTB-XL** dataset (21,799 records) for training and internal evaluation (Fold 1-9 for training, Fold 10 for testing). Signals were resampled to 100 Hz, bandpass filtered (0.5–40 Hz), and Z-score normalized. External validation was performed on the **Chapman-Shaoxing** dataset (10,646 records), using SNOMED-CT codes to map labels to PTB-XL superclasses.

### 2.2 SE-ResNet Architecture
The core model is a **Squeeze-and-Excitation Residual Network**. While the ResNet backbone handles deep temporal dependencies via skip connections, the **SE-blocks** adaptively recalibrate channel-wise feature responses by explicitly modeling interdependencies between leads. This allows the model to "attend" to the most diagnostically relevant leads for each specific condition.

### 2.3 Data Augmentation
To improve robustness, we implemented **Mixup Augmentation**, which creates synthetic training samples by linearly interpolating between two signals and their corresponding labels. This encourages the model to learn smoother decision boundaries.

### 2.4 Loss Function and Optimization
To combat the heavy class imbalance (NORM at ~44% vs HYP at ~5%), we employed **Focal Loss**. This modifies standard cross-entropy by adding a modulating factor $(1 - p_t)^\gamma$, which down-weights the loss contributed by easy-to-classify examples and focuses training on hard, misclassified samples.

### 2.5 Interpretability and Calibration
*   **Grad-CAM**: Generated saliency maps to visualize the signal regions driving the model's decisions.
*   **Calibration**: Analyzed Reliability Diagrams and computed **Expected Calibration Error (ECE)** to ensure the model's confidence scores reflect true diagnostic probabilities.

---

## 3. Results

### 3.1 Performance on PTB-XL (Fold 10)
The ensemble of SE-ResNet (Standard) and SE-ResNet (Focal Loss) yielded the following results:

| Model | Macro-AUC | Macro-F1 | F1-NORM | F1-MI | F1-STTC | F1-CD | F1-HYP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline SE-ResNet** | 0.917 | 0.594 | 0.909 | 0.569 | 0.596 | 0.574 | 0.320 |
| **Ensemble (SOTA)** | **0.932** | **0.642** | **0.915** | **0.612** | **0.645** | **0.621** | **0.418** |

### 3.2 Augmentation Ablation (Figure 4)
Ablation studies confirmed that **Mixup (AUG-4)** provided the most significant boost to Macro-F1 compared to noise or shift augmentations.
![Figure 4: Augmentation Ablation](/results/figures/publication/fig4_ablation_study.png)

### 3.3 Interpretability (Figure 6)
Grad-CAM analysis (Figure 6) shows the model focusing on the QRS complex and ST-segments for Myocardial Infarction detection, aligning with clinical diagnostic criteria.
![Figure 6: Grad-CAM Composite](/results/figures/publication/fig6_gradcam_composite.png)

### 3.4 External Validation
On the Chapman-Shaoxing dataset, the ensemble maintained an **AUROC of 0.844**, demonstrating high discriminative power despite differences in population demographics and recording environments.

---

## 4. Discussion

### 4.1 Prevalence vs. Performance
As shown in **Figure 7**, there is a strong correlation ($R^2 > 0.8$) between class prevalence and F1-score. While NORM (44% prevalence) reaches >0.90 F1, the HYP class (5% prevalence) remains challenging at 0.42, even with Focal Loss.
![Figure 7: Prevalence vs F1](/results/figures/publication/fig7_prevalence_vs_f1.png)

### 4.2 Clinical Significance
The ability of the model to produce calibrated confidence scores (ECE < 0.05) and visual saliency maps makes it a viable candidate for a standard cardiology workflow, providing not just a "black box" prediction but a verifiable suggestion for clinical review.

---

## 5. Conclusion
We successfully implemented a SOTA multi-label ECG classification system. By combining architectural improvements (SE-ResNet) with data-centric techniques (Mixup, Focal Loss), we achieved high diagnostic performance that generalizes to external datasets. Future work will focus on integrating Transformer-based architectures to further improve performance on minority classes.
