import pandas as pd
import numpy as np
import os
import sys
import ast
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnet_models import create_resnet
from src.data.data_loader import ECGDataGenerator

# Set paths
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
OUTPUT_DIR = 'results/models/'
LOG_DIR = 'results/logs/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def load_data(limit=None):
    print("Loading PTB-XL database...")
    df = pd.read_csv(os.path.join(DATA_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    df['is_normal'] = df['scp_codes'].apply(lambda x: 1 if 'NORM' in x else 0)
    
    if limit:
        print(f"Limiting to {limit} records.")
        df = df.head(limit)
    return df

def train_resnet():
    # 1. Config for Full Training
    BATCH_SIZE = 32
    EPOCHS = 15
    # Using None to load ALL data (21,799 records)
    # This might take 1-2 hours on a CPU, but it's feasible overnight
    LIMIT = None 
    
    # 2. Load Data
    df = load_data(limit=LIMIT)
    
    # 3. Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_normal'])
    
    print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples")
    
    # 4. Generators
    train_gen = ECGDataGenerator(train_df, DATA_PATH, batch_size=BATCH_SIZE)
    val_gen = ECGDataGenerator(val_df, DATA_PATH, batch_size=BATCH_SIZE)
    
    # 5. Model
    model = create_resnet(num_classes=2)
    
    # Use a slightly lower learning rate for ResNet stability
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auroc')])
    model.summary()
    
    # 6. Train
    print("Starting full ResNet training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'resnet_best.keras'), save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6)
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
    
    plt.savefig('results/figures/resnet_training_history.png')
    print("Training complete! Model and plot saved.")

if __name__ == "__main__":
    train_resnet()
