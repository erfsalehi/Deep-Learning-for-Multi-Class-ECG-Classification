import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.multiclass_loader import MultiClassECGDataGenerator, load_ptbxl_data

# Set paths
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
MODELS_DIR = 'results/models/'
FIGURES_DIR = 'results/figures/'

def evaluate_ensemble():
    # 1. Load Data
    print("Loading PTB-XL data for Ensemble Evaluation...")
    try:
        df = load_ptbxl_data(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return

    # Use validation/test set logic (same as training scripts to get same split)
    from sklearn.model_selection import train_test_split
    df['primary_class'] = df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    df = df[df['primary_class'] != 'Unknown']
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['primary_class'])
    
    print(f"Evaluating on {len(test_df)} samples")
    
    # 2. Generator
    test_gen = MultiClassECGDataGenerator(test_df, DATA_PATH, batch_size=32, shuffle=False)
    
    # 3. Load Models
    models_to_ensemble = [
        'resnet_multiclass_best.keras',
        'se_resnet_best.keras',
        'se_resnet_aug_best.keras'
    ]
    
    predictions = []
    valid_models = []
    
    print("Generating predictions from individual models...")
    for model_name in models_to_ensemble:
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            print(f"Loading {model_name}...")
            model = tf.keras.models.load_model(model_path)
            pred = model.predict(test_gen, verbose=1)
            predictions.append(pred)
            valid_models.append(model_name)
        else:
            print(f"Warning: Model {model_name} not found. Skipping.")
            
    if not predictions:
        print("No models found to ensemble!")
        return

    # 4. Ensemble Strategy: Average Probability
    # Truncate to min length just in case of generator drop_last batch mismatch (rare with exact batch_size)
    min_len = min([len(p) for p in predictions])
    predictions = [p[:min_len] for p in predictions]
    
    ensemble_pred_prob = np.mean(predictions, axis=0)
    ensemble_pred_class = np.argmax(ensemble_pred_prob, axis=1)
    
    # Get True Labels
    # Re-generate true labels from generator (it's deterministic with shuffle=False)
    y_true_batches = []
    for i in range(len(test_gen)):
        _, y = test_gen[i]
        y_true_batches.append(y)
    y_true = np.concatenate(y_true_batches)[:min_len]
    y_true_class = np.argmax(y_true, axis=1)
    
    # 5. Metrics
    acc = accuracy_score(y_true_class, ensemble_pred_class)
    print(f"\nEnsemble Accuracy: {acc:.4f}")
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    print("\nClassification Report:")
    print(classification_report(y_true_class, ensemble_pred_class, target_names=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_class, ensemble_pred_class)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Ensemble Confusion Matrix (Acc: {acc:.2%})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(FIGURES_DIR, 'ensemble_confusion_matrix.png'))
    print(f"Confusion matrix saved to {FIGURES_DIR}")

if __name__ == "__main__":
    evaluate_ensemble()
