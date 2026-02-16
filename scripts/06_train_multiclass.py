import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnet_models import create_resnet
from src.data.multiclass_loader import MultiClassECGDataGenerator, load_ptbxl_data

# Set paths
# Adjust this path based on your actual structure
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
OUTPUT_DIR = 'results/models/'
LOG_DIR = 'results/logs/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train_multiclass():
    # 1. Config
    BATCH_SIZE = 32
    EPOCHS = 15
    # Use full dataset for SOTA attempt
    LIMIT = None 
    
    # 2. Load Data
    print("Loading PTB-XL data for Multi-class Classification...")
    try:
        df = load_ptbxl_data(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return

    if LIMIT:
        df = df.head(LIMIT)
    
    # 3. Split
    # We stratify based on the primary superclass for balance
    # Since 'diagnostic_superclass' is a list, we pick the first one for stratification
    df['primary_class'] = df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    # Filter out Unknown if any (shouldn't be based on load_ptbxl_data filter)
    df = df[df['primary_class'] != 'Unknown']
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['primary_class'])
    
    print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples")
    print(f"Classes: {df['primary_class'].unique()}")
    
    # 4. Generators
    train_gen = MultiClassECGDataGenerator(train_df, DATA_PATH, batch_size=BATCH_SIZE)
    val_gen = MultiClassECGDataGenerator(val_df, DATA_PATH, batch_size=BATCH_SIZE)
    
    # 5. Model
    # 5 classes: NORM, MI, STTC, CD, HYP
    model = create_resnet(num_classes=5)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    # 6. Train
    print("Starting Multi-class ResNet training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'resnet_multiclass_best.keras'), save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
        ]
    )
    
    # 7. Plot History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.savefig('results/figures/resnet_multiclass_history.png')
    print("Training complete! Model saved to results/models/resnet_multiclass_best.keras")

if __name__ == "__main__":
    train_multiclass()
