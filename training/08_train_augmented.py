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
from src.data.augmentation import MixupGenerator, CutoutGenerator

# Set paths
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
OUTPUT_DIR = 'results/models/'
LOG_DIR = 'results/logs/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train_augmented_se_resnet():
    # 1. Config
    BATCH_SIZE = 32
    EPOCHS = 8
    
    # 2. Load Data
    print("Loading Preprocessed PTB-XL data for Augmented SE-ResNet Training...")
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
    
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    
    # 5. Generators
    base_train_gen = NpyECGDataGenerator(X_train, y_train, batch_size=BATCH_SIZE)
    val_gen = NpyECGDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    
    # Apply Augmentations (Mixup + Cutout)
    cutout_gen = CutoutGenerator(base_train_gen, mask_size=100, probability=0.5)
    train_gen = MixupGenerator(cutout_gen, alpha=0.2)
    
    # 6. Model
    model = create_se_resnet(num_classes=5)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])
    
    # 7. Train
    print("Starting Augmented SE-ResNet training (Mixup + Cutout)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'se_resnet_aug_best.keras'), save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True), 
            tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6)
        ]
    )
    
    # 8. Plot History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title('Accuracy')
    
    os.makedirs('results/figures/', exist_ok=True)
    plt.savefig('results/figures/se_resnet_aug_history.png')
    print("Training complete! Model saved to results/models/se_resnet_aug_best.keras")

if __name__ == "__main__":
    train_augmented_se_resnet()
