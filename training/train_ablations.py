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
from src.data.augmentation import GaussianNoiseGenerator, ScalingGenerator, ShiftGenerator, MixupGenerator

PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
OUTPUT_DIR = 'results/models/ablations/'
RESULTS_DIR = 'results/ablations/'
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

def run_ablation(name, train_gen, val_gen, epochs=5):
    model = create_se_resnet(num_classes=5)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, f'se_resnet_{name}.keras'), save_best_only=True)])
    pd.DataFrame(history.history).to_csv(os.path.join(RESULTS_DIR, f'{name}_history.csv'), index=False)

def train_ablations():
    X, df = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy')), pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv'))
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = np.zeros((len(df), len(classes)))
    for i, row in df.iterrows():
        for c_idx, cl in enumerate(classes):
            if cl in row['diagnostic_superclass']: y[i, c_idx] = 1
    df['primary_class'] = df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.15, random_state=42, stratify=df['primary_class'])
    base_gen = NpyECGDataGenerator(X[train_idx], y[train_idx], batch_size=32)
    val_gen = NpyECGDataGenerator(X[val_idx], y[val_idx], batch_size=32, shuffle=False)
    exps = [("AUG-1_Noise", lambda g: GaussianNoiseGenerator(g, sigma=0.01)), ("AUG-2_Scaling", lambda g: ScalingGenerator(g)), ("AUG-3_Shift", lambda g: ShiftGenerator(g)), ("AUG-4_Mixup", lambda g: MixupGenerator(g))]
    for name, wrap in exps: run_ablation(name, wrap(base_gen), val_gen)

if __name__ == "__main__":
    train_ablations()
