import pandas as pd
import numpy as np
import os
import sys
import ast
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import ECGDataGenerator

# Set paths
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
MODEL_PATH = 'results/models/cnn_best.keras'
FIGURES_DIR = 'results/figures/'

def load_data(limit=None):
    print("Loading PTB-XL database...")
    df = pd.read_csv(os.path.join(DATA_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    df['is_normal'] = df['scp_codes'].apply(lambda x: 1 if 'NORM' in x else 0)
    
    if limit:
        df = df.head(limit)
    return df

def evaluate_model():
    # 1. Load Data (same subset as training for validation split consistency)
    LIMIT = 5000 
    df = load_data(limit=LIMIT)
    
    # 2. Split (recreate the split to get the test set)
    # Note: In a real pipeline, you'd save the test indices to a file.
    # Here we rely on the same random_state=42
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_normal'])
    
    print(f"Evaluating on {len(test_df)} samples...")
    
    # 3. Generator
    # Shuffle=False is CRITICAL for evaluation to match predictions with labels
    test_gen = ECGDataGenerator(test_df, DATA_PATH, batch_size=32, shuffle=False)
    
    # 4. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 5. Predict
    print("Generating predictions...")
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Get true labels
    # The generator might drop the last batch if it doesn't fit perfectly
    # So we need to trim predictions or labels to match
    y_true = test_df['is_normal'].values[:len(y_pred)]
    
    # 6. Metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Abnormal', 'Normal']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Abnormal', 'Normal'], 
                yticklabels=['Abnormal', 'Normal'])
    plt.title('Confusion Matrix - 1D-CNN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(FIGURES_DIR, 'cnn_confusion_matrix.png'))
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - 1D-CNN')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'cnn_roc_curve.png'))
    plt.close()
    
    print(f"Plots saved to {FIGURES_DIR}")

if __name__ == "__main__":
    evaluate_model()
