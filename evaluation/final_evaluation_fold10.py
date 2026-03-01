import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
import ast

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.se_resnet import create_se_resnet
from training.10_train_focal_loss import FocalLoss

# Paths
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
MODELS_DIR = 'results/models/'
ABLATIONS_DIR = 'results/models/ablations/'
OUTPUT_PATH = 'results/fold10_metrics.csv'

def evaluate_model(model_path, X_test, y_test, classes, is_focal=False):
    print(f"Evaluating {os.path.basename(model_path)}...")
    custom_objects = {}
    if is_focal:
        custom_objects['FocalLoss'] = FocalLoss
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None
        
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Macro Metrics
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc_macro = roc_auc_score(y_test, y_pred_proba, average='macro', multi_label='ovr')
    
    # Per-class F1
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    results = {
        'model': os.path.basename(model_path),
        'f1_macro': f1_macro,
        'auc_macro': auc_macro
    }
    
    for i, cls in enumerate(classes):
        results[f'f1_{cls}'] = per_class_f1[i]
        
    return results

def main():
    # 1. Load Data
    X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy'))
    database_path = os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv')
    df = pd.read_csv(database_path)
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = np.zeros((len(df), len(classes)))
    for i, row in df.iterrows():
        for c_idx, cl in enumerate(classes):
            if cl in row['diagnostic_superclass']:
                y[i, c_idx] = 1
                
    # 2. Filter Fold 10
    test_idx = df[df['strat_fold'] == 10].index
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    print(f"Test Set Size (Fold 10): {len(X_test)}")
    
    all_results = []
    
    # 3. List models
    models_to_eval = [
        (os.path.join(MODELS_DIR, 'se_resnet_best.keras'), False),
        (os.path.join(MODELS_DIR, 'se_resnet_focal_best.keras'), True)
    ]
    
    if os.path.exists(ABLATIONS_DIR):
        for filename in sorted(os.listdir(ABLATIONS_DIR)):
            if filename.endswith('.keras'):
                models_to_eval.append((os.path.join(ABLATIONS_DIR, filename), False))
    
    # 4. Evaluate
    for path, is_focal in models_to_eval:
        if os.path.exists(path):
            res = evaluate_model(path, X_test, y_test, classes, is_focal)
            if res:
                all_results.append(res)
        else:
            print(f"Warning: model not found at {path}")
            
    # 5. Save
    results_df = pd.DataFrame(all_results)
    os.makedirs('results/', exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
