import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import ast

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.multiclass_loader import NpyECGDataGenerator

# Set paths
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
MODEL_PATH = 'results/models/se_resnet_best.keras'
FIGURES_DIR = 'results/figures/'
RESULTS_DIR = 'results/'

os.makedirs(FIGURES_DIR, exist_ok=True)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def compute_ece_ci(y_true, y_prob, n_bins=10, n_bootstraps=1000, alpha=0.95):
    bootstrapped_ece = []
    indices = np.arange(len(y_true))
    for _ in range(n_bootstraps):
        resampled_indices = np.random.choice(indices, size=len(indices), replace=True)
        ece = expected_calibration_error(y_true[resampled_indices], y_prob[resampled_indices], n_bins=n_bins)
        bootstrapped_ece.append(ece)
    
    mean_ece = np.mean(bootstrapped_ece)
    lower = np.percentile(bootstrapped_ece, (1-alpha)/2 * 100)
    upper = np.percentile(bootstrapped_ece, (1+alpha)/2 * 100)
    return mean_ece, lower, upper

def evaluate_ptbxl_calibration(model, X_test, y_true, classes):
    print("Starting PTB-XL Calibration Analysis...")
    y_prob = model.predict(X_test, batch_size=32)
    
    ece_results = {}
    plt.figure(figsize=(15, 10))
    for i, cls_name in enumerate(classes):
        prob = y_prob[:, i]
        true = y_true[:, i]
        ece, lower, upper = compute_ece_ci(true, prob)
        ece_results[cls_name] = f"{ece:.4f} ({lower:.4f}-{upper:.4f})"
        
        plt.subplot(2, 3, i + 1)
        prob_true, prob_pred = calibration_curve(true, prob, n_bins=10)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.plot(prob_pred, prob_true, marker='s', label=f'{cls_name} (ECE={ece:.4f})')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'PTB-XL Reliability: {cls_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'calibration_ptbxl.png'), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'calibration_ptbxl.pdf'))
    return ece_results

def evaluate_chapman_calibration(model, classes):
    print("Starting Chapman Calibration Analysis...")
    X_path = 'data/processed/chapman/X_chapman.npy'
    meta_path = 'data/processed/chapman/chapman_database.csv'
    
    if not os.path.exists(X_path):
        print("Chapman data not found.")
        return None

    X = np.load(X_path)
    df = pd.read_csv(meta_path)
    
    mapping = {
        '426177001': 'NORM', '426783006': 'NORM', '427084000': 'NORM', '164884004': 'NORM',
        '164861001': 'MI', '164865005': 'MI', '164909002': 'MI', '22298006': 'MI',
        '55930002': 'STTC', '164931005': 'STTC', '164930006': 'STTC', '164873001': 'STTC',
        '39732003': 'STTC', '427172004': 'STTC', '164917005': 'STTC', '426761007': 'STTC',
        '713427006': 'CD', '713426002': 'CD', '445118002': 'CD', '445211001': 'CD',
        '164903001': 'CD', '164951009': 'CD', '425868008': 'CD', '426112009': 'CD',
        '426660007': 'CD', '63593006': 'CD', '6374002': 'CD',
        '164890007': 'HYP', '164871004': 'HYP', '164872006': 'HYP', '89792004': 'HYP',
        '164934002': 'HYP', '429622005': 'HYP', '428750005': 'HYP'
    }
    
    y_true = np.zeros((len(df), len(classes)))
    df['labels'] = df['labels'].apply(ast.literal_eval)
    for i, labels in enumerate(df['labels']):
        mapped = [mapping.get(str(l)) for l in labels]
        for c_idx, cls_name in enumerate(classes):
            if cls_name in mapped:
                y_true[i, c_idx] = 1

    y_prob = model.predict(X, batch_size=32)
    
    ece_results = {}
    plt.figure(figsize=(15, 10))
    for i, cls_name in enumerate(classes):
        prob = y_prob[:, i]
        true = y_true[:, i]
        ece, lower, upper = compute_ece_ci(true, prob)
        ece_results[cls_name] = f"{ece:.4f} ({lower:.4f}-{upper:.4f})"
        
        plt.subplot(2, 3, i + 1)
        prob_true, prob_pred = calibration_curve(true, prob, n_bins=10)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.plot(prob_pred, prob_true, marker='s', label=f'{cls_name} (ECE={ece:.4f})')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Chapman Reliability: {cls_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'calibration_chapman.png'), dpi=300)
    return ece_results

def evaluate_calibration():
    # 1. Load Data
    X_path = os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy')
    meta_path = os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv')
    
    if not os.path.exists(X_path):
        print(f"Error: Processed data not found at {X_path}. Run preprocess_ptbxl.py first.")
        return

    X = np.load(X_path)
    df = pd.read_csv(meta_path)
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)

    # Use test split (fold 10 is standard for PTB-XL)
    test_idx = df[df.strat_fold == 10].index.values
    X_test = X[test_idx]
    test_df = df.iloc[test_idx].copy()
    
    # Get ground truth
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y_true = np.zeros((len(test_df), len(classes)))
    for i, (_, row) in enumerate(test_df.iterrows()):
        diagnoses = row['diagnostic_superclass']
        for c_idx, cls_name in enumerate(classes):
            if cls_name in diagnoses:
                y_true[i, c_idx] = 1

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        # Try finding any model in the directory if the baseline isn't there yet
        model_files = [f for f in os.listdir('results/models/') if f.endswith('.keras')]
        if model_files:
            MODEL_PATH_ACTUAL = os.path.join('results/models/', model_files[0])
            print(f"Primary model not found, using {MODEL_PATH_ACTUAL} instead.")
        else:
            print(f"Error: No model found in results/models/. Please train a model first.")
            return
    else:
        MODEL_PATH_ACTUAL = MODEL_PATH
    
    model = tf.keras.models.load_model(MODEL_PATH_ACTUAL)
    
    ece_ptbxl = evaluate_ptbxl_calibration(model, X_test, y_true, classes)
    ece_chapman = evaluate_chapman_calibration(model, classes)
    
    # Save text results
    with open(os.path.join(RESULTS_DIR, 'calibration_ece.txt'), 'w') as f:
        f.write("Expected Calibration Error (ECE):\n\nPTB-XL:\n")
        for k, v in ece_ptbxl.items(): f.write(f"{k}: {v}\n")
        if ece_chapman:
            f.write("\nChapman:\n")
            for k, v in ece_chapman.items(): f.write(f"{k}: {v}\n")
    print("Calibration analysis complete.")

if __name__ == "__main__":
    evaluate_calibration()
