import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
import ast
import traceback

# Add project root to path BEFORE importing from submodules
sys.path.append(os.getcwd())

from evaluation.statistical_tests import compute_bootstrap_ci, mcnemar_test, delong_auc_test
from training.train_focal_loss import FocalLoss
from src.models.transformer_ecg import PatchEmbedding

PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
MODELS_DIR = 'results/models/'
BASELINES_DIR = 'results/models/baselines/'
OUTPUT_PATH = 'results/sota_comparison_table.csv'

def get_test_data():
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
    return X[test_idx], y[test_idx], classes

def evaluate_single_model(model_path, X_test, y_test, is_focal=False):
    print(f"Evaluating {os.path.basename(model_path)}...")
    custom_objects = {'PatchEmbedding': PatchEmbedding}
    if is_focal:
        custom_objects['FocalLoss'] = FocalLoss
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    y_proba = model.predict(X_test)
    tf.keras.backend.clear_session()
    return y_proba

def main():
    X_test, y_test, classes = get_test_data()
    
    model_configs = [
        ('Ensemble (SOTA)', os.path.join(MODELS_DIR, 'se_resnet_best.keras'), False), # Will handle ensemble separately
        ('Ribeiro et al. (2020)', os.path.join(BASELINES_DIR, 'ribeiro_resnet_best.keras'), False),
        ('Transformer Baseline', os.path.join(BASELINES_DIR, 'transformer_best.keras'), False),
        ('CinC 2020 Baseline', os.path.join(BASELINES_DIR, 'cinc2020_best.keras'), False)
    ]
    
    results = []
    prob_map = {}
    
    # 1. Evaluate Individual Baselines
    for name, path, is_focal in model_configs:
        if name == 'Ensemble (SOTA)':
            continue
        if os.path.exists(path):
            y_proba = evaluate_single_model(path, X_test, y_test, is_focal)
            prob_map[name] = y_proba
        else:
            print(f"Skipping {name}: model not found at {path}")

    # 2. Evaluate Ensemble (Needs both models)
    base_path = os.path.join(MODELS_DIR, 'se_resnet_best.keras')
    focal_path = os.path.join(MODELS_DIR, 'se_resnet_focal_best.keras')
    if os.path.exists(base_path) and os.path.exists(focal_path):
        print("Evaluating Ensemble...")
        p1 = evaluate_single_model(base_path, X_test, y_test, False)
        p2 = evaluate_single_model(focal_path, X_test, y_test, True)
        p_ens = (p1 + p2) / 2
        prob_map['Ensemble (SOTA)'] = p_ens
    
    # 3. Statistical Analysis
    comparison_data = []
    ensemble_proba = prob_map.get('Ensemble (SOTA)')
    
    for name, y_proba in prob_map.items():
        print(f"Running statistics (n=1000) for {name}...")
        stats = compute_bootstrap_ci(y_test, y_proba, n_bootstraps=1000)
        
        row = {
            'Model': name,
            'AUC': f"{stats['auc_mean']:.3f} ({stats['auc_ci'][0]:.3f}-{stats['auc_ci'][1]:.3f})",
            'F1': f"{stats['f1_mean']:.3f} ({stats['f1_ci'][0]:.3f}-{stats['f1_ci'][1]:.3f})"
        }
        
        # Add per-class F1 with CIs for our Ensemble (and others if space permits)
        for i, cls in enumerate(classes):
            row[f'F1-{cls}'] = f"{stats['per_class_f1_means'][i]:.3f} ({stats['per_class_f1_cis'][i][0]:.3f}-{stats['per_class_f1_cis'][i][1]:.3f})"
        
        # p-values vs Ensemble
        if ensemble_proba is not None and name != 'Ensemble (SOTA)':
            row['p-value (AUC)'] = f"{delong_auc_test(y_test, ensemble_proba, y_proba):.4f}"
            y_pred_ens = (ensemble_proba > 0.5).astype(int)
            y_pred_model = (y_proba > 0.5).astype(int)
            row['p-value (F1)'] = f"{mcnemar_test(y_test, y_pred_ens, y_pred_model):.4f}"
        else:
            row['p-value (AUC)'] = '-'
            row['p-value (F1)'] = '-'
            
        comparison_data.append(row)
        
    df_res = pd.DataFrame(comparison_data)
    print("\nDetailed SOTA Comparison Table (with Per-Class CIs):")
    print(df_res.to_string())
    df_res.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
