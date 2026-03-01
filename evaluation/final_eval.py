import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
import ast
import traceback

sys.path.append(os.getcwd())
try:
    from training.train_focal_loss import FocalLoss
except ImportError:
    # Fallback if renamed
    from training.train_focal_loss import FocalLoss

PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
MODELS_DIR = 'results/models/'
ABLATIONS_DIR = 'results/models/ablations/'
OUTPUT_PATH = 'results/fold10_metrics.csv'

def evaluate_model(model_path, X_test, y_test, classes, is_focal=False):
    print(f"Evaluating {os.path.basename(model_path)}...")
    custom_objects = {'FocalLoss': FocalLoss} if is_focal else {}
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc_macro = roc_auc_score(y_test, y_pred_proba, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        res = {
            'model': os.path.basename(model_path),
            'f1_macro': f1_macro,
            'auc_macro': auc_macro
        }
        
        f1s = f1_score(y_test, y_pred, average=None, zero_division=0)
        for i, cls in enumerate(classes):
            res[f'f1_{cls}'] = f1s[i]
        return res
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")
        traceback.print_exc()
        return None

def main():
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Data path not found: {PROCESSED_DATA_PATH}")
        return

    X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy'))
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv'))
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = np.zeros((len(df), len(classes)))
    for i, row in df.iterrows():
        for c_idx, cl in enumerate(classes):
            if cl in row['diagnostic_superclass']:
                y[i, c_idx] = 1
                
    test_idx = df[df['strat_fold'] == 10].index
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Test Set Size (Fold 10): {len(X_test)}")
    
    all_res = []
    models = []
    base_models = [
        ('se_resnet_best.keras', False),
        ('se_resnet_focal_best.keras', True)
    ]
    for m_name, is_focal in base_models:
        m_path = os.path.join(MODELS_DIR, m_name)
        if os.path.exists(m_path):
            models.append((m_path, is_focal))
            
    if os.path.exists(ABLATIONS_DIR):
        for f in sorted(os.listdir(ABLATIONS_DIR)):
            if f.endswith('.keras'):
                models.append((os.path.join(ABLATIONS_DIR, f), False))
    
    print(f"Found {len(models)} models.")
    for path, is_focal in models:
        res = evaluate_model(path, X_test, y_test, classes, is_focal)
        if res:
            all_res.append(res)
            
    # 5. Ensemble (Baseline + Focal Loss)
    print("Computing Ensemble (Baseline + Focal Loss)...")
    base_path = os.path.join(MODELS_DIR, 'se_resnet_best.keras')
    focal_path = os.path.join(MODELS_DIR, 'se_resnet_focal_best.keras')
    
    if os.path.exists(base_path) and os.path.exists(focal_path):
        m1 = tf.keras.models.load_model(base_path)
        m2 = tf.keras.models.load_model(focal_path, custom_objects={'FocalLoss': FocalLoss})
        
        p1 = m1.predict(X_test)
        p2 = m2.predict(X_test)
        p_ens = (p1 + p2) / 2
        
        y_pred = (p_ens > 0.5).astype(int)
        res_ens = {
            'model': 'Ensemble (Base+Focal)',
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'auc_macro': roc_auc_score(y_test, p_ens, average='macro')
        }
        f1s = f1_score(y_test, y_pred, average=None, zero_division=0)
        for i, cls in enumerate(classes): res_ens[f'f1_{cls}'] = f1s[i]
        all_res.append(res_ens)
    
    # 6. Save
    if all_res:
        results_df = pd.DataFrame(all_res)
        os.makedirs('results/', exist_ok=True)
        results_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Done. Saved to {OUTPUT_PATH}")
    else:
        print("ERROR: No results collected!")

if __name__ == "__main__":
    main()
