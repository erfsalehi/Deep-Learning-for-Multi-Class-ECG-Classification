# SOTA Improvement Progress

Based on "ECG SOTA Improvement Guide.pdf", we are upgrading the project to achieve State-of-the-Art performance.

## Phase 1: Multi-class Classification (In Progress)
- **Goal**: Move from Binary (Normal/Abnormal) to 5-class classification (NORM, MI, STTC, CD, HYP).
- **Status**: 
    - [x] Downloaded `scp_statements.csv` for label mapping.
    - [x] Implemented `src/data/multiclass_loader.py` for 5-class data generation.
    - [x] Created `scripts/06_train_multiclass.py` to train a ResNet on 5 classes.
    - [ ] Evaluate performance (Target: >85% Accuracy).

## Phase 2: Data Augmentation (Planned)
- **Goal**: Implement Mixup and Cutout to improve generalization.
- **Status**: Pending.

## Phase 3: Advanced Architectures (Partially Done)
- **Goal**: Implement SE-ResNet and Attention mechanisms.
- **Status**:
    - [x] Implemented `src/models/se_resnet.py` (Squeeze-and-Excitation ResNet).
    - [ ] Train SE-ResNet.

## Phase 4: Ensembling (Planned)
- **Goal**: Combine models to boost performance.
- **Status**: Pending.

## Phase 5: Hyperparameter Tuning (Planned)
- **Goal**: Use Optuna to find optimal hyperparameters.
- **Status**: Pending.
