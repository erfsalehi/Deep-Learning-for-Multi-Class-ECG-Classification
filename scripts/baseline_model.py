import pandas as pd
import numpy as np
import os
import sys
import joblib
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
from tqdm import tqdm
import xgboost as xgb
import scipy.stats as stats

# Add project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom preprocessor
try:
    from src.data.preprocessing import FeatureExtractor
except ImportError:
    print("Could not import FeatureExtractor from src.data.preprocessing. Defining locally.")
    class FeatureExtractor:
        def __init__(self, fs=100):
            self.fs = fs
        def extract_features(self, ecg_signal):
            features = []
            for lead_idx in range(ecg_signal.shape[1]):
                lead_data = ecg_signal[:, lead_idx]
                features.extend([
                    np.mean(lead_data),
                    np.std(lead_data),
                    np.min(lead_data),
                    np.max(lead_data),
                    np.ptp(lead_data),
                    stats.skew(lead_data),
                    stats.kurtosis(lead_data),
                    ((lead_data[:-1] * lead_data[1:]) < 0).sum()
                ])
            return np.array(features)

# Set paths
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
OUTPUT_DIR = 'results/models/'
FIGURES_DIR = 'results/figures/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data(limit=None):
    print("Loading PTB-XL database...")
    df = pd.read_csv(os.path.join(DATA_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    
    # Parse scp_codes
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    
    # Create binary target: Normal vs Abnormal
    # 'NORM' is the code for Normal ECG
    df['is_normal'] = df['scp_codes'].apply(lambda x: 1 if 'NORM' in x else 0)
    
    if limit:
        print(f"Limiting to first {limit} records for quick testing...")
        df = df.head(limit)
        
    return df

def extract_features_dataset(df):
    X = []
    y = []
    
    print(f"Extracting features for {len(df)} records...")
    extractor = FeatureExtractor()
    
    valid_indices = []
    
    for i, (idx, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
        try:
            record_path = os.path.join(DATA_PATH, row['filename_lr'])
            
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal
            
            feats = extractor.extract_features(signal)
            
            X.append(feats)
            y.append(row['is_normal'])
            valid_indices.append(idx)
            
        except Exception as e:
            # print(f"Error loading record {idx}: {e}")
            continue
            
    return np.array(X), np.array(y)

def train_baseline_model():
    # 1. Load Data
    # Increased limit to 5000 for better performance
    df = load_data(limit=5000) 
    
    # 2. Extract Features
    X, y = extract_features_dataset(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: Normal={sum(y)}, Abnormal={len(y)-sum(y)}")
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Train Model (XGBoost)
    print("Training XGBoost Classifier...")
    # XGBoost is generally faster and more accurate than RF for tabular data
    clf = xgb.XGBClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Abnormal', 'Normal'])
    print(report)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Abnormal', 'Normal'], 
                yticklabels=['Abnormal', 'Normal'])
    plt.title('Confusion Matrix - Baseline XGBoost')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(FIGURES_DIR, 'baseline_xgboost_confusion_matrix.png'))
    print(f"Confusion matrix saved to {FIGURES_DIR}")
    
    # Save Model
    model_path = os.path.join(OUTPUT_DIR, 'baseline_xgboost.pkl')
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_baseline_model()
