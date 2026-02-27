import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.multiclass_loader import MultiClassECGDataGenerator, load_ptbxl_data

# Set paths
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
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
        # Determine points in this bin
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def evaluate_calibration():
    print("Starting Calibration Analysis...")
    
    # 1. Load Data
    try:
        df = load_ptbxl_data(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return

    # Use test split (fold 10 is standard for PTB-XL)
    test_df = df[df.strat_fold == 10].copy()
    test_df['primary_class'] = test_df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    test_df = test_df[test_df['primary_class'] != 'Unknown']
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 3. Predict
    test_gen = MultiClassECGDataGenerator(test_df, DATA_PATH, batch_size=32, shuffle=False)
    y_prob = model.predict(test_gen)
    
    # Get ground truth
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y_true = np.zeros((len(test_df), len(classes)))
    for i, (_, row) in enumerate(test_df.iterrows()):
        diagnoses = row['diagnostic_superclass']
        for c_idx, cls_name in enumerate(classes):
            if cls_name in diagnoses:
                y_true[i, c_idx] = 1

    # 4. Compute ECE and Plot Reliability Diagrams
    ece_results = {}
    
    plt.figure(figsize=(15, 5))
    
    for i, cls_name in enumerate(classes):
        prob = y_prob[:, i]
        true = y_true[:, i]
        
        ece = expected_calibration_error(true, prob)
        ece_results[cls_name] = ece
        
        # Only plot NORM and HYP in the main output figure per PRD
        if cls_name in ['NORM', 'HYP']:
            idx = 1 if cls_name == 'NORM' else 2
            plt.subplot(1, 2, idx)
            
            prob_true, prob_pred = calibration_curve(true, prob, n_bins=10)
            
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
            plt.plot(prob_pred, prob_true, marker='s', label=f'{cls_name} (ECE={ece:.4f})')
            
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Reliability Diagram: {cls_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'calibration_reliability_diagrams.png'), dpi=300)
    plt.savefig(os.path.join(FIGURES_DIR, 'calibration_reliability_diagrams.pdf'))
    
    # Save text results
    with open(os.path.join(RESULTS_DIR, 'calibration_ece.txt'), 'w') as f:
        f.write("Expected Calibration Error (ECE) per class:\n")
        for cls_name, ece in ece_results.items():
            f.write(f"{cls_name}: {ece:.4f}\n")
            
    print("Calibration analysis complete. Figures and results saved.")

if __name__ == "__main__":
    evaluate_calibration()
