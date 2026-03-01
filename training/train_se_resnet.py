import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import ast

sys.path.append(os.getcwd())
from src.models.se_resnet import create_se_resnet
from src.data.multiclass_loader import NpyECGDataGenerator

PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
OUTPUT_DIR = 'results/models/'

def train_se_resnet():
    BATCH_SIZE, EPOCHS = 32, 8
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
    model = create_se_resnet(num_classes=len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])
    model.fit(NpyECGDataGenerator(X[train_idx], y[train_idx], batch_size=BATCH_SIZE),
              validation_data=NpyECGDataGenerator(X[val_idx], y[val_idx], batch_size=BATCH_SIZE, shuffle=False),
              epochs=EPOCHS,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'se_resnet_best.keras'), save_best_only=True)])

if __name__ == "__main__":
    train_se_resnet()
