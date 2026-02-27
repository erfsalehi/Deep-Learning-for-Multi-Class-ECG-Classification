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

# Set paths
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
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
        # Clip to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal loss
        loss = cross_entropy * tf.math.pow(1.0 - y_pred, self.gamma)
        
        if self.alpha is not None:
            # alpha should be a vector of weights per class
            alpha = tf.cast(self.alpha, tf.float32)
            loss = loss * alpha
            
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

def train_focal_loss():
    # 1. Config
    BATCH_SIZE = 32
    EPOCHS = 15
    GAMMA = 2.0
    
    # 2. Load Data
    print("Loading PTB-XL data for Focal Loss Experiment...")
    try:
        df = load_ptbxl_data(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return

    # 3. Split
    df['primary_class'] = df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    df = df[df['primary_class'] != 'Unknown']
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['primary_class'])
    
    # Calculate Alpha (inverse class frequency)
    counts = train_df['primary_class'].value_counts()
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    weights = []
    total = len(train_df)
    for c in classes:
        # Inverse proportional weights
        w = total / counts.get(c, 1)
        weights.append(w)
    
    # Normalize weights to sum to num_classes
    weights = np.array(weights)
    weights = weights / weights.sum() * len(classes)
    print(f"Computed class weights (alpha): {dict(zip(classes, weights))}")
    
    print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples")
    
    # 4. Generators
    train_gen = MultiClassECGDataGenerator(train_df, DATA_PATH, batch_size=BATCH_SIZE)
    val_gen = MultiClassECGDataGenerator(val_df, DATA_PATH, batch_size=BATCH_SIZE)
    
    # 5. Model
    model = create_se_resnet(num_classes=len(classes))
    
    # Compile with Focal Loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=FocalLoss(gamma=GAMMA, alpha=weights), 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    # 6. Train
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
    
    # 7. Plot History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Focal Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.savefig('results/figures/focal_loss_training_history.png')
    print("Training complete! Model saved to results/models/se_resnet_focal_best.keras")

if __name__ == "__main__":
    train_focal_loss()
