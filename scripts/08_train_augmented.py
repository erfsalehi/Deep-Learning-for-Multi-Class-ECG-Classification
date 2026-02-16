import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.se_resnet import create_se_resnet
from src.data.multiclass_loader import MultiClassECGDataGenerator, load_ptbxl_data
from src.data.augmentation import MixupGenerator, CutoutGenerator

# Set paths
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
OUTPUT_DIR = 'results/models/'
LOG_DIR = 'results/logs/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train_augmented_se_resnet():
    # 1. Config
    BATCH_SIZE = 32
    EPOCHS = 20 # More epochs needed for Mixup
    LIMIT = None 
    
    # 2. Load Data
    print("Loading PTB-XL data for Augmented SE-ResNet Training...")
    try:
        df = load_ptbxl_data(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return

    if LIMIT:
        df = df.head(LIMIT)
    
    # 3. Split
    df['primary_class'] = df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    df = df[df['primary_class'] != 'Unknown']
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['primary_class'])
    
    print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples")
    
    # 4. Generators
    base_train_gen = MultiClassECGDataGenerator(train_df, DATA_PATH, batch_size=BATCH_SIZE)
    val_gen = MultiClassECGDataGenerator(val_df, DATA_PATH, batch_size=BATCH_SIZE)
    
    # Apply Augmentations (Mixup + Cutout)
    # First apply Cutout
    cutout_gen = CutoutGenerator(base_train_gen, mask_size=100, probability=0.5)
    # Then apply Mixup
    train_gen = MixupGenerator(cutout_gen, alpha=0.2)
    
    # 5. Model
    model = create_se_resnet(num_classes=5)
    
    # Mixup uses soft labels, so we still use categorical_crossentropy
    # We might need to adjust learning rate schedule slightly
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    # 6. Train
    print("Starting Augmented SE-ResNet training (Mixup + Cutout)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'se_resnet_aug_best.keras'), save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True), # Increased patience for augmentation
            tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6)
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
    
    plt.savefig('results/figures/se_resnet_aug_history.png')
    print("Training complete! Model saved to results/models/se_resnet_aug_best.keras")

if __name__ == "__main__":
    train_augmented_se_resnet()
