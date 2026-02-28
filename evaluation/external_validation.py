import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.utils import resample as bootstrap_resample
from tqdm import tqdm
import ast

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

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        pos_loss = -y_true * tf.math.pow(1.0 - y_pred, self.gamma) * tf.math.log(y_pred)
        neg_loss = -(1.0 - y_true) * tf.math.pow(y_pred, self.gamma) * tf.math.log(1.0 - y_pred)
        loss = pos_loss + neg_loss
        if self.alpha is not None:
            alpha = tf.cast(self.alpha, tf.float32)
            loss = loss * alpha
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

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
    # Classes: ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # Refined mapping based on SNOMED-CT acronyms in Chapman headers
    mapping = {
        # NORM
        'SB': 'NORM', 'SR': 'NORM', 'ST': 'NORM', 'SA': 'NORM',
        # MI
        'MI': 'MI', 'AMI': 'MI', 'IMI': 'MI', 'LMI': 'MI', 'MIBW': 'MI', 
        'MIFW': 'MI', 'MILW': 'MI', 'MISW': 'MI', 'AQW': 'MI',
        # STTC
        'STDD': 'STTC', 'STE': 'STTC', 'STTC': 'STTC', 'STTU': 'STTC', 
        'TWC': 'STTC', 'TWO': 'STTC', 'AFIB': 'STTC', 'AF': 'STTC', 
        'SVT': 'STTC', 'AT': 'STTC', 'AVNRT': 'STTC', 'AVRT': 'STTC',
        # CD
        'LBBB': 'CD', 'RBBB': 'CD', '1AVB': 'CD', '2AVB': 'CD', '2AVB1': 'CD', 
        '2AVB2': 'CD', '3AVB': 'CD', 'AVB': 'CD', 'VPB': 'CD', 'APB': 'CD', 
        'IVB': 'CD', 'IDC': 'CD', 'JEB': 'CD', 'JPT': 'CD',
        # HYP
        'LVH': 'HYP', 'RVH': 'HYP', 'RAH': 'HYP'
    }
    
    y_true = []
    keep_indices = []
    
    for i, row in df.iterrows():
        # labels column contains a string representation of a list: "['SB', 'LVH']"
        labels_str = row.get('labels', '[]')
        try:
            labels = ast.literal_eval(labels_str)
        except:
            labels = []
            
        mapped_classes = set()
        for l in labels:
            if l in mapping:
                mapped_classes.add(mapping[l])
        
        # If multiple superclasses, we pick one or skip for simplicity in a 5-class multi-class setup (argmax)
        if mapped_classes:
            chosen = None
            for c in classes:
                if c in mapped_classes:
                    chosen = c
                    break
            
            if chosen:
                y_true.append(classes.index(chosen))
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
    custom_objects = {'FocalLoss': FocalLoss}
    for mf in model_files:
        try:
            m = tf.keras.models.load_model(os.path.join(MODEL_DIR, mf), custom_objects=custom_objects)
            models.append(m)
        except Exception as e:
            print(f"Warning: Failed to load {mf} - {e}")
            
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
