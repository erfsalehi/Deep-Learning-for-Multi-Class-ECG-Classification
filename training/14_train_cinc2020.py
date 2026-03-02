import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import ast
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cinc2020_resnet import build_cinc2020_resnet
from src.data.multiclass_loader import NpyECGDataGenerator

PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
OUTPUT_DIR = 'results/models/baselines/'
LOG_DIR = 'results/logs/baselines/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train_cinc2020():
    BATCH_SIZE = 32
    EPOCHS = 10
    
    print("Loading PTB-XL data for CinC 2020 Baseline...")
    X_path = os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy')
    meta_path = os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv')
    
    if not os.path.exists(X_path):
        print("Data not found.")
        return

    X = np.load(X_path)
    df = pd.read_csv(meta_path)
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = np.zeros((len(df), len(classes)))
    for i, row in df.iterrows():
        for c_idx, cl in enumerate(classes):
            if cl in row['diagnostic_superclass']:
                y[i, c_idx] = 1
                
    train_idx = df[df['strat_fold'] <= 8].index
    val_idx = df[df['strat_fold'] == 9].index
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    train_gen = NpyECGDataGenerator(X_train, y_train, batch_size=BATCH_SIZE)
    val_gen = NpyECGDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    
    model = build_cinc2020_resnet(input_shape=(1000, 12), num_classes=5)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])
    
    print("Starting CinC 2020 training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'cinc2020_best.keras'), save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6)
        ]
    )
    
    # Save training plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('CinC 2020 Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.legend(); plt.title('CinC 2020 AUC')
    
    os.makedirs('results/figures/baselines/', exist_ok=True)
    plt.savefig('results/figures/baselines/cinc2020_training_history.png')
    print("Completed. Saved to", os.path.join(OUTPUT_DIR, 'cinc2020_best.keras'))

if __name__ == "__main__":
    train_cinc2020()
