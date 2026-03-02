import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import f1_score
import ast
import sys

# Add project root to path
sys.path.append(os.getcwd())

# Paths
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
MODEL_PATH = 'results/models/se_resnet_focal_best.keras' # Using focal model as it's strongest on HYP

def optimize_thresholds():
    print("Starting Threshold Optimization for HYP class...")
    
    # 1. Load Validation Data (Fold 9)
    X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy'))
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv'))
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    val_idx = df[df['strat_fold'] == 9].index.values
    X_val = X[val_idx]
    
    y_val = np.zeros((len(val_idx), len(classes)))
    for i, idx in enumerate(val_idx):
        labels = df.loc[idx, 'diagnostic_superclass']
        for c_idx, cls_name in enumerate(classes):
            if cls_name in labels:
                y_val[i, c_idx] = 1
                
    # 2. Load Model
    from training.train_focal_loss import FocalLoss
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'FocalLoss': FocalLoss})
    
    # 3. Predict on Validation
    y_prob = model.predict(X_val)
    
    # 4. Optimize threshold for each class, especially HYP
    best_thresholds = np.ones(len(classes)) * 0.5
    best_f1s = np.zeros(len(classes))
    
    for c_idx, cls_name in enumerate(classes):
        prob = y_prob[:, c_idx]
        true = y_val[:, c_idx]
        
        thresholds = np.linspace(0.1, 0.9, 81)
        for t in thresholds:
            pred = (prob > t).astype(int)
            f1 = f1_score(true, pred)
            if f1 > best_f1s[c_idx]:
                best_f1s[c_idx] = f1
                best_thresholds[c_idx] = t
        
        print(f"Class {cls_name}: Best Threshold = {best_thresholds[c_idx]:.3f}, Best F1 = {best_f1s[c_idx]:.4f}")
        
    # 5. Save optimized thresholds
    thresholds_df = pd.DataFrame({
        'class': classes,
        'threshold': best_thresholds,
        'val_f1': best_f1s
    })
    os.makedirs('results/optimization', exist_ok=True)
    thresholds_df.to_csv('results/optimization/optimized_thresholds.csv', index=False)
    print("\nOptimized thresholds saved to results/optimization/optimized_thresholds.csv")

if __name__ == "__main__":
    optimize_thresholds()
