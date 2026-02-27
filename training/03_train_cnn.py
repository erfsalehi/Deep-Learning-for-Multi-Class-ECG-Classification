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

from src.models.cnn_models import create_1d_cnn
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
    
    # Parse scp_codes
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    
    # Binary target
    df['is_normal'] = df['scp_codes'].apply(lambda x: 1 if 'NORM' in x else 0)
    
    if limit:
        print(f"Limiting to first {limit} records...")
        df = df.head(limit)
        
    return df

def train_cnn():
    # 1. Config
    BATCH_SIZE = 32
    EPOCHS = 10 # Keep it small for laptop
    LIMIT = 5000 # Subset for laptop training
    
    # 2. Load Data
    df = load_data(limit=LIMIT)
    
    # 3. Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_normal'])
    
    print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples")
    
    # 4. Generators
    train_gen = ECGDataGenerator(train_df, DATA_PATH, batch_size=BATCH_SIZE)
    val_gen = ECGDataGenerator(val_df, DATA_PATH, batch_size=BATCH_SIZE)
    
    # 5. Model
    model = create_1d_cnn(num_classes=2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # 6. Train
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'cnn_best.keras'), save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=3)
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
    
    plt.savefig('results/figures/cnn_training_history.png')
    print("Training complete! Model and plot saved.")

if __name__ == "__main__":
    train_cnn()
