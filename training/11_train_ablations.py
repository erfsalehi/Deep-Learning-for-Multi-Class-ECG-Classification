import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ast

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.se_resnet import create_se_resnet
from src.data.multiclass_loader import NpyECGDataGenerator
from src.data.augmentation import GaussianNoiseGenerator, ScalingGenerator, ShiftGenerator, MixupGenerator

# Set paths
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
OUTPUT_DIR = 'results/models/ablations/'
RESULTS_DIR = 'results/ablations/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_ablation_experiment(aug_name, train_gen_wrapper, val_gen, epochs=10):
    print(f"\n--- Starting Ablation Experiment: {aug_name} ---")
    
    model = create_se_resnet(num_classes=5)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])
    
    output_path = os.path.join(OUTPUT_DIR, f'se_resnet_{aug_name}.keras')
    
    history = model.fit(
        train_gen_wrapper,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(output_path, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    # Save history to CSV
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(RESULTS_DIR, f'{aug_name}_history.csv'), index=False)
    
    return history

def train_ablations():
    # 1. Config
    BATCH_SIZE = 32
    EPOCHS = 5
    
    # 2. Load Data
    print("Loading Preprocessed PTB-XL data for Augmentation Ablation Study...")
    X_path = os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy')
    meta_path = os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv')
    
    if not os.path.exists(X_path):
        print(f"Error: Processed data not found at {X_path}. Run preprocess_ptbxl.py first.")
        return

    X = np.load(X_path)
    df = pd.read_csv(meta_path)
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    
    # 3. Prepare Multi-label targets
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = np.zeros((len(df), len(classes)))
    for i, row in df.iterrows():
        for c_idx, cl in enumerate(classes):
            if cl in row['diagnostic_superclass']:
                y[i, c_idx] = 1
    
    # 4. Split
    df['primary_class'] = df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.15, random_state=42, stratify=df['primary_class'])
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 5. Base Generators
    base_train_gen = NpyECGDataGenerator(X_train, y_train, batch_size=BATCH_SIZE)
    val_gen = NpyECGDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. Define Augmentation Experiments
    experiments = [
        ("AUG-1_Noise", lambda gen: GaussianNoiseGenerator(gen, sigma=0.01)),
        ("AUG-2_Scaling", lambda gen: ScalingGenerator(gen, scale_range=(0.9, 1.1))),
        ("AUG-3_Shift", lambda gen: ShiftGenerator(gen, max_shift=20)),
        ("AUG-4_Mixup", lambda gen: MixupGenerator(gen, alpha=0.2))
    ]
    
    # 7. Run Experiments
    for name, wrapper in experiments:
        aug_train_gen = wrapper(base_train_gen)
        run_ablation_experiment(name, aug_train_gen, val_gen, epochs=EPOCHS)
        
    print("\nAll ablation experiments completed successfully.")

if __name__ == "__main__":
    train_ablations()
