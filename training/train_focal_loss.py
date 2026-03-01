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

# Set paths
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
OUTPUT_DIR = 'results/models/'

@tf.keras.utils.register_keras_serializable(package='Custom')
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        pos_loss = -y_true * tf.math.pow(1.0 - y_pred, self.gamma) * tf.math.log(y_pred)
        neg_loss = -(1.0 - y_true) * tf.math.pow(y_pred, self.gamma) * tf.math.log(1.0 - y_pred)
        loss = pos_loss + neg_loss
        if self.alpha is not None:
            loss = loss * tf.cast(self.alpha, tf.float32)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

def train_focal_loss():
    BATCH_SIZE, EPOCHS, GAMMA = 32, 8, 2.0
    X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy'))
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv'))
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = np.zeros((len(df), len(classes)))
    for i, row in df.iterrows():
        for c_idx, cl in enumerate(classes):
            if cl in row['diagnostic_superclass']: y[i, c_idx] = 1
    df['primary_class'] = df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.15, random_state=42, stratify=df['primary_class'])
    X_train, y_train = X[train_idx], y[train_idx]
    counts = np.sum(y_train, axis=0)
    weights = len(y_train) / (len(classes) * counts)
    weights = weights / np.sum(weights) * len(classes)
    model = create_se_resnet(num_classes=len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=FocalLoss(gamma=GAMMA, alpha=weights), 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])
    model.fit(NpyECGDataGenerator(X_train, y_train, batch_size=BATCH_SIZE),
              validation_data=NpyECGDataGenerator(X[val_idx], y[val_idx], batch_size=BATCH_SIZE),
              epochs=EPOCHS,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'se_resnet_focal_best.keras'), save_best_only=True)])

if __name__ == "__main__":
    train_focal_loss()
