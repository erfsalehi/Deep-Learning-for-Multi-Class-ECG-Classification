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
from src.data.multiclass_loader import NpyECGDataGenerator, load_ptbxl_data

# Set paths
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
RAW_DATA_PATH = 'data/raw/ptbxl/' # For metadata
OUTPUT_DIR = 'results/models/'
LOG_DIR = 'results/logs/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary Cross Entropy terms
        pos_loss = -y_true * tf.math.pow(1.0 - y_pred, self.gamma) * tf.math.log(y_pred)
        neg_loss = -(1.0 - y_true) * tf.math.pow(y_pred, self.gamma) * tf.math.log(1.0 - y_pred)
        
        loss = pos_loss + neg_loss
        
        if self.alpha is not None:
            alpha = tf.cast(self.alpha, tf.float32)
            loss = loss * alpha
            
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

def train_focal_loss():
    # 1. Config
    BATCH_SIZE = 32
    EPOCHS = 8
    GAMMA = 2.0
    
    # 2. Load Data
    print("Loading Preprocessed PTB-XL data for Focal Loss Experiment...")
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
    # For stratification in multi-label, we can use a primary class or just random split
    # Let's use the first superclass as primary for stratified split
    df['primary_class'] = df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.15, random_state=42, stratify=df['primary_class'])
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Calculate Alpha (inverse class frequency)
    counts = np.sum(y_train, axis=0)
    weights = len(y_train) / (len(classes) * counts)
    weights = weights / np.sum(weights) * len(classes)
    print(f"Computed class weights (alpha): {dict(zip(classes, weights))}")
    
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    
    # 5. Generators
    train_gen = NpyECGDataGenerator(X_train, y_train, batch_size=BATCH_SIZE)
    val_gen = NpyECGDataGenerator(X_val, y_val, batch_size=BATCH_SIZE)
    
    # 6. Model
    model = create_se_resnet(num_classes=len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=FocalLoss(gamma=GAMMA, alpha=weights), 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])
    
    # 7. Train
    print(f"Starting Focal Loss training (gamma={GAMMA})...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'se_resnet_focal_best.keras'), save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
        ]
    )
    
    # 8. Plot History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Focal Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title('Accuracy')
    
    os.makedirs('results/figures/', exist_ok=True)
    plt.savefig('results/figures/focal_loss_training_history.png')
    print("Training complete! Model saved to results/models/se_resnet_focal_best.keras")

import ast
if __name__ == "__main__":
    train_focal_loss()
