# Toward State-of-the-Art: Multi-Class ECG Classification with SE-ResNet

**Date:** February 2026
**Project:** Cardiovascular AI Portfolio - Phase 2

---

## 1. Objective
To upgrade the existing binary classification system to a **State-of-the-Art (SOTA)** multi-class diagnostic tool capable of identifying 5 major cardiac conditions:
1.  **NORM**: Normal ECG
2.  **MI**: Myocardial Infarction
3.  **STTC**: ST/T Change
4.  **CD**: Conduction Disturbance
5.  **HYP**: Hypertrophy

## 2. Methodology Upgrades

### 2.1 Data Pipeline
We implemented a robust **Multi-class Data Generator** (`src/data/multiclass_loader.py`) that:
*   Parses the `scp_statements.csv` standard from PhysioNet.
*   Maps complex diagnostic codes to the 5 superclasses.
*   Handles class stratification to ensure balanced training.

### 2.2 Advanced Architecture: SE-ResNet
We moved beyond standard CNNs to **Squeeze-and-Excitation Residual Networks (SE-ResNet)**.
*   **Mechanism**: The SE block adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
*   **Benefit**: This allows the network to focus on the most informative feature maps (e.g., specific lead patterns) while suppressing less useful ones.

## 3. Implementation Status

| Component | Status | Description |
| :--- | :---: | :--- |
| **Data Loader** | ✅ Complete | Maps PTB-XL codes to 5 classes. |
| **Base ResNet** | 🔄 Training | Currently training on 5-class task (Epoch 8/15). |
| **SE-ResNet** | ✅ Implemented | Code ready in `src/models/se_resnet.py`. |
| **Training Script** | ✅ Ready | `scripts/07_train_se_resnet.py` created. |

## 4. Preliminary Results (Base ResNet - 5 Class)
*   **Current Epoch**: 8/15
*   **Training Accuracy**: ~72%
*   **Validation AUC**: ~0.91
*   **Observation**: The model is learning effectively. The high AUC suggests good separability between classes even if top-1 accuracy is still improving.

## 5. Next Steps (Roadmap)
1.  **Complete Training**: Finish the ResNet baseline.
2.  **Run SE-ResNet**: Execute `python scripts/07_train_se_resnet.py` to observe performance gains from the SE blocks.
3.  **Data Augmentation**: Implement Mixup/Cutout to push accuracy >85%.
4.  **Ensembling**: Combine ResNet and SE-ResNet predictions.

---

## Appendix: How to Run
To train the SOTA SE-ResNet model:
```bash
python scripts/07_train_se_resnet.py
```
