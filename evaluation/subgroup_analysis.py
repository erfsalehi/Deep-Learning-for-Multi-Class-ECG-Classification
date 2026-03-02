import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
import ast
from tqdm import tqdm

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

PROCESSED_PTBXL = 'data/processed/ptbxl/'
PROCESSED_CHAPMAN = 'data/processed/chapman/'
MODELS_DIR = 'results/models/'
THRESHOLDS_PATH = 'results/optimization/optimized_thresholds.csv'
OUTPUT_PATH = 'results/subgroup_analysis_results.csv'

def load_ptbxl_test():
    X = np.load(os.path.join(PROCESSED_PTBXL, 'X_ptbxl.npy'))
    df = pd.read_csv(os.path.join(PROCESSED_PTBXL, 'ptbxl_database_processed.csv'))
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    y = np.zeros((len(df), len(classes)))
    for i, row in df.iterrows():
        for c_idx, cl in enumerate(classes):
            if cl in row['diagnostic_superclass']:
                y[i, c_idx] = 1
                
    test_idx = df[df['strat_fold'] == 10].index
    return X[test_idx], y[test_idx], df.iloc[test_idx].reset_index(drop=True)

def load_chapman_test():
    X_path = os.path.join(PROCESSED_CHAPMAN, 'X_chapman.npy')
    meta_path = os.path.join(PROCESSED_CHAPMAN, 'chapman_database.csv')
    X = np.load(X_path)
    df = pd.read_csv(meta_path)
    
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
        labels_str = row.get('labels', '[]')
        try: labels = ast.literal_eval(labels_str)
        except: labels = []
        mapped_classes = set()
        for l in labels:
            if l in mapping: mapped_classes.add(mapping[l])
        
        if mapped_classes:
            y_row = np.zeros(len(classes))
            for c in mapped_classes: y_row[classes.index(c)] = 1
            y_true.append(y_row)
            keep_indices.append(i)
            
    return X[keep_indices], np.array(y_true), df.iloc[keep_indices].reset_index(drop=True)

def get_age_group(age):
    if pd.isna(age): return "Unknown"
    if age < 50: return "<50"
    if age <= 70: return "50-70"
    return ">70"

def get_sex_group(sex):
    s = str(sex).strip().lower()
    if s in ['0', '0.0', 'female']: return "Female"
    if s in ['1', '1.0', 'male']: return "Male"
    return "Unknown"

def compute_metrics(y_true, y_prob, thresholds):
    if len(y_true) == 0: return np.nan, np.nan
    y_pred = (y_prob >= thresholds).astype(int)
    
    try: auc = roc_auc_score(y_true, y_prob, average='macro', multi_label='ovr')
    except: auc = np.nan
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return auc, f1

def main():
    thresh_df = pd.read_csv(THRESHOLDS_PATH)
    thresholds = thresh_df['threshold'].values
    
    print("Loading models...")
    custom_objects = {'FocalLoss': FocalLoss}
    m1 = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'se_resnet_best.keras'), custom_objects=custom_objects)
    m2 = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'se_resnet_focal_best.keras'), custom_objects=custom_objects)
    
    datasets = [('PTB-XL', load_ptbxl_test), ('Chapman', load_chapman_test)]
    all_results = []
    
    for ds_name, loader in datasets:
        print(f"\nProcessing {ds_name}...")
        X, y_true, meta_df = loader()
        print(f"Dataset size: {len(X)}")
        
        p1 = m1.predict(X, batch_size=64, verbose=1)
        p2 = m2.predict(X, batch_size=64, verbose=1)
        y_prob = (p1 + p2) / 2
        
        meta_df['AgeGroup'] = meta_df['age'].apply(get_age_group)
        meta_df['SexGroup'] = meta_df['sex'].apply(get_sex_group)
        
        # Breakdown
        for sub_type in ['Overall', 'AgeGroup', 'SexGroup']:
            groups = ['All'] if sub_type == 'Overall' else meta_df[sub_type].unique()
            for g in sorted(groups):
                idx = meta_df.index if sub_type == 'Overall' else meta_df[meta_df[sub_type] == g].index
                if len(idx) < 5: continue # Skip tiny groups
                auc, f1 = compute_metrics(y_true[idx], y_prob[idx], thresholds)
                all_results.append({
                    'Dataset': ds_name, 
                    'Category': sub_type.replace('Group', ''), 
                    'Group': g, 
                    'AUC': auc, 
                    'F1': f1, 
                    'N': len(idx)
                })
                
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(res_df.to_string())

if __name__ == "__main__":
    main()
