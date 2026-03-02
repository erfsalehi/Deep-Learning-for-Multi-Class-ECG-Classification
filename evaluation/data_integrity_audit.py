import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
import ast

# Add project root to path
sys.path.append(os.getcwd())

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

# Paths
PROCESSED_PTBXL = 'data/processed/ptbxl/'
PROCESSED_CHAPMAN = 'data/processed/chapman/'
MODELS_DIR = 'results/models/'
THRESHOLDS_PATH = 'results/optimization/optimized_thresholds.csv'

def load_ptbxl_test():
    X = np.load(os.path.join(PROCESSED_PTBXL, 'X_ptbxl.npy'))
    df = pd.read_csv(os.path.join(PROCESSED_PTBXL, 'ptbxl_database_processed.csv'))
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = np.zeros((len(df), len(classes)))
    for i, row in df.iterrows():
        for c_idx, cl in enumerate(classes):
            if cl in row['diagnostic_superclass']: y[i, c_idx] = 1
    test_idx = df[df['strat_fold'] == 10].index
    return X[test_idx], y[test_idx]

def load_chapman_test():
    X = np.load(os.path.join(PROCESSED_CHAPMAN, 'X_chapman.npy'))
    df = pd.read_csv(os.path.join(PROCESSED_CHAPMAN, 'chapman_database.csv'))
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    mapping = {
        '426177001': 'NORM', '426783006': 'NORM', '427084000': 'NORM', '164884004': 'NORM',
        '164861001': 'MI',   '164865005': 'MI',   '164909002': 'MI',   '22298006': 'MI',
        '55930002': 'STTC',  '164931005': 'STTC', '164930006': 'STTC', '59118001': 'STTC',
        '39732003': 'STTC',  '427172004': 'STTC', '164917005': 'STTC', '426761007': 'STTC',
        '164889003': 'STTC', 
        '713427006': 'CD', '713426002': 'CD', '445118002': 'CD', '445211001': 'CD',
        '164903001': 'CD', '164951009': 'CD', '425868008': 'CD', '426112009': 'CD',
        '426660007': 'CD', '63593006': 'CD', '6374002': 'CD', '270492004': 'CD',
        '164890007': 'HYP', '164871004': 'HYP', '164872006': 'HYP', '89792004': 'HYP',
        '164934002': 'HYP', '429622005': 'HYP', '428750005': 'HYP', '164873001': 'HYP',
        '164510008': 'HYP'
    }
    y_true = []
    keep_indices = []
    for i, row in df.iterrows():
        labels = ast.literal_eval(row['labels'])
        m_cls = set(mapping[l] for l in labels if l in mapping)
        if m_cls:
            y_r = np.zeros(len(classes))
            for c in m_cls: y_r[classes.index(c)] = 1
            y_true.append(y_r)
            keep_indices.append(i)
    return X[keep_indices], np.array(y_true)

def main():
    print("--- DATA AUDIT START ---")
    custom_objects = {'FocalLoss': FocalLoss}
    m1 = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'se_resnet_best.keras'), custom_objects=custom_objects)
    m2 = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'se_resnet_focal_best.keras'), custom_objects=custom_objects)
    
    thresh_df = pd.read_csv(THRESHOLDS_PATH)
    thresholds = thresh_df['threshold'].values
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    # 1. PTB-XL Fold 10 Audit
    X_ptb, y_ptb = load_ptbxl_test()
    p1 = m1.predict(X_ptb)
    p2 = m2.predict(X_ptb)
    y_prob = (p1 + p2) / 2
    
    auc_ptb = roc_auc_score(y_ptb, y_prob, average='macro')
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_opt = (y_prob >= thresholds).astype(int)
    
    f1_05 = f1_score(y_ptb, y_pred_05, average='macro')
    f1_opt = f1_score(y_ptb, y_pred_opt, average='macro')
    
    print("\n[PTB-XL Fold 10]")
    print(f"Macro-AUC: {auc_ptb:.4f}")
    print(f"Macro-F1 (0.5): {f1_05:.4f}")
    print(f"Macro-F1 (Opt): {f1_opt:.4f}")
    
    per_cls_f1_opt = f1_score(y_ptb, y_pred_opt, average=None)
    for i, cls in enumerate(classes):
        print(f"F1-{cls} (Opt): {per_cls_f1_opt[i]:.4f}")

    # 2. Chapman Audit
    X_ch, y_ch = load_chapman_test()
    p1_ch = m1.predict(X_ch)
    p2_ch = m2.predict(X_ch)
    y_prob_ch = (p1_ch + p2_ch) / 2
    
    auc_ch = roc_auc_score(y_ch, y_prob_ch, average='macro')
    f1_ch_05 = f1_score(y_ch, (y_prob_ch >= 0.5).astype(int), average='macro')
    
    print("\n[Chapman-Shaoxing]")
    print(f"Macro-AUROC: {auc_ch:.4f}")
    print(f"Macro-F1 (0.5): {f1_ch_05:.4f}")
    
    # Per-class cleanup
    y_pred_ch = (y_prob_ch >= 0.5).astype(int)
    per_cls_f1_ch = f1_score(y_ch, y_pred_ch, average=None)
    for i, cls in enumerate(classes):
        print(f"F1-{cls}: {per_cls_f1_ch[i]:.4f}")

    print("\n--- DATA AUDIT END ---")

if __name__ == "__main__":
    main()
