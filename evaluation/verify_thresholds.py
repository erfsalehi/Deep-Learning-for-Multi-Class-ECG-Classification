import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import f1_score, classification_report
import ast
import sys

# Add project root to path
sys.path.append(os.getcwd())

# Paths
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
MODEL_PATH = 'results/models/se_resnet_focal_best.keras'
THRESHOLDS_PATH = 'results/optimization/optimized_thresholds.csv'

def verify_thresholds():
    print("Verifying Optimized Thresholds on Fold 10 (Test Set)...")
    
    # 1. Load Test Data
    X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy'))
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv'))
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    test_idx = df[df['strat_fold'] == 10].index.values
    X_test = X[test_idx]
    
    y_test = np.zeros((len(test_idx), len(classes)))
    for i, idx in enumerate(test_idx):
        labels = df.loc[idx, 'diagnostic_superclass']
        for c_idx, cls_name in enumerate(classes):
            if cls_name in labels:
                y_test[i, c_idx] = 1
                
    # 2. Load Model
    from training.train_focal_loss import FocalLoss
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'FocalLoss': FocalLoss})
    
    # 3. Predict
    y_prob = model.predict(X_test)
    
    # 4. Load Optimized Thresholds
    thresh_df = pd.read_csv(THRESHOLDS_PATH)
    thresholds = thresh_df.set_index('class')['threshold'].to_dict()
    
    # 5. Apply Thresholds and Calculate Metrics
    y_pred = np.zeros_like(y_prob)
    for i, cls_name in enumerate(classes):
        y_pred[:, i] = (y_prob[:, i] > thresholds[cls_name]).astype(int)
        
    print("\nClassification Report with Optimized Thresholds (Fold 10):")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    hyp_f1 = f1_score(y_test[:, 4], y_pred[:, 4])
    print(f"Final HYP F1 on Test Set: {hyp_f1:.4f}")
    
    if hyp_f1 >= 0.500:
        print("Success: REQ-03 fulfilled (HYP F1 >= 0.500)")
    else:
        print("Note: HYP F1 is below 0.500 on test set. Further improvement may be needed.")

if __name__ == "__main__":
    verify_thresholds()
