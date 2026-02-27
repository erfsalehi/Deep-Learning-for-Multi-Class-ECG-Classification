import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.utils import resample as bootstrap_resample
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set paths
PROCESSED_DATA_DIR = 'data/processed/chapman/'
MODEL_DIR = 'results/models/'
RESULTS_DIR = 'results/external_validation/'

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_metrics(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    # AUC needs handle multi-class properly
    # Check if all classes present in y_true for AUC
    try:
        auc = roc_auc_score(tf.keras.utils.to_categorical(y_true), y_prob, multi_class='ovr', average='macro')
    except:
        auc = np.nan
        
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'auc': auc
    }

def run_external_validation():
    print("Starting External Validation on Chapman-Shaoxing Dataset...")
    
    # 1. Load Processed Chapman Data
    X_path = os.path.join(PROCESSED_DATA_DIR, 'X_chapman.npy')
    meta_path = os.path.join(PROCESSED_DATA_DIR, 'chapman_database.csv')
    
    if not os.path.exists(X_path):
        print(f"Error: Processed data not found at {X_path}. Run download_chapman.py first.")
        return
        
    X = np.load(X_path)
    df = pd.read_csv(meta_path)
    
    # 2. Map Chapman labels to PTB-XL 5-class Taxonomy
    # NORM -> 0
    # MI-equiv -> 1
    # STTC -> 2
    # CD -> 3
    # HYP -> 4
    
    # Values from Diagnostics.csv (Rhythm column/Others)
    # This mapping is critical for EV-4 and EV-5
    # For now, we use the Rhythm column and map based on common medical definitions
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    mapping = {
        'SB': 'NORM', 'SR': 'NORM', 'ST': 'NORM', # Sinus rhythms
        'AF': 'STTC', 'AFIB': 'STTC', 'SVT': 'STTC', # Arrhythmias often mixed with STTC
        'MI': 'MI', 'AMI': 'MI', 'IMI': 'MI', 'LMI': 'MI',
        'LBBB': 'CD', 'RBBB': 'CD', 'AVB': 'CD', 'PVC': 'CD',
        'LVH': 'HYP', 'RVH': 'HYP'
    }
    
    # Note: A real mapping would involve more columns like 'Rhythm' and 'Condition'
    # For this script we assume a simplified mapping or that labels are present
    
    y_true = []
    keep_indices = []
    
    for i, row in df.iterrows():
        rhythm = row.get('Rhythm', '')
        # Simple heuristic mapping for demonstration
        mapped = None
        if rhythm in mapping:
            mapped = mapping[rhythm]
        
        if mapped:
            y_true.append(classes.index(mapped))
            keep_indices.append(i)
            
    X = X[keep_indices]
    y_true = np.array(y_true)
    
    print(f"Mapped {len(y_true)} samples into 5-class taxonomy.")
    
    # 3. Load PTB-XL Models (Ensemble)
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    if not model_files:
        print(f"Error: No trained models found in {MODEL_DIR}")
        return
        
    print(f"Loading ensemble of {len(model_files)} models...")
    models = []
    for mf in model_files:
        try:
            m = tf.keras.models.load_model(os.path.join(MODEL_DIR, mf))
            models.append(m)
        except:
            print(f"Warning: Failed to load {mf}")
            
    # 4. Predict
    all_probs = []
    for m in models:
        probs = m.predict(X, batch_size=32, verbose=0)
        all_probs.append(probs)
        
    # Ensemble average
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    # 5. Bootstrap Metrics (n=1000)
    print("Computing metrics with 1000 bootstrap iterations...")
    bootstrap_results = []
    n_iterations = 1000
    
    for _ in tqdm(range(n_iterations)):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        res = compute_metrics(y_true[indices], ensemble_probs[indices])
        bootstrap_results.append(res)
        
    boot_df = pd.DataFrame(bootstrap_results)
    
    # Calculate CIs
    final_stats = {}
    for col in boot_df.columns:
        mean_val = boot_df[col].mean()
        lower = np.percentile(boot_df[col], 2.5)
        upper = np.percentile(boot_df[col], 97.5)
        final_stats[col] = f"{mean_val:.4f} (95% CI: {lower:.4f} - {upper:.4f})"
        
    # Per-class F1
    report = classification_report(y_true, ensemble_preds, target_names=classes, output_dict=True)
    
    # 6. Save Results
    print("\nExternal Validation Results:")
    for k, v in final_stats.items():
        print(f"{k}: {v}")
        
    with open(os.path.join(RESULTS_DIR, 'metrics_summary.txt'), 'w') as f:
        f.write("External Validation Results (Chapman-Shaoxing):\n")
        for k, v in final_stats.items():
            f.write(f"{k}: {v}\n")
        f.write("\nPer-class Classification Report:\n")
        f.write(classification_report(y_true, ensemble_preds, target_names=classes))
        
    pd.DataFrame(report).transpose().to_csv(os.path.join(RESULTS_DIR, 'per_class_report.csv'))
    
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_external_validation()
