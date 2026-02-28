import os
import ast
import pandas as pd
import numpy as np
import wfdb
from tqdm import tqdm
from scipy.signal import resample

# Paths
RAW_DATA_PATH = 'data/raw/ptbxl/'
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'

def load_ptbxl_metadata(data_path):
    print("Loading PTB-XL metadata...")
    df = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    
    # Load scp_statements.csv for superclass mapping
    mapped_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
    mapped_df = mapped_df[mapped_df.diagnostic == 1]
    
    def aggregate_diagnostic(y_dict):
        tmp = []
        for key in y_dict.keys():
            if key in mapped_df.index:
                tmp.append(mapped_df.loc[key].diagnostic_class)
        return list(set(tmp))
        
    df['diagnostic_superclass'] = df['scp_codes'].apply(aggregate_diagnostic)
    return df

def preprocess_ptbxl():
    print(f"Preprocessing PTB-XL and saving to {PROCESSED_DATA_PATH}...")
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    df = load_ptbxl_metadata(RAW_DATA_PATH)
    
    X = []
    y_meta = []
    
    # Process only records with 100Hz (filename_lr)
    print("Processing ECG signals...")
    for ecg_id, row in tqdm(df.iterrows(), total=len(df)):
        record_path = os.path.join(RAW_DATA_PATH, row['filename_lr'])
        
        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal
            
            # Ensure shape is (1000, 12)
            if sig.shape[0] != 1000:
                sig = resample(sig, 1000)
            
            # Z-score normalization per lead
            for lead in range(12):
                mean_val = np.mean(sig[:, lead])
                std_val = np.std(sig[:, lead])
                if std_val > 1e-6:
                    sig[:, lead] = (sig[:, lead] - mean_val) / std_val
                else:
                    sig[:, lead] = sig[:, lead] - mean_val
            
            X.append(sig)
            y_meta.append({
                'ecg_id': ecg_id,
                'diagnostic_superclass': row['diagnostic_superclass'],
                'strat_fold': row['strat_fold']
            })
            
        except Exception as e:
            # print(f"Error processing {ecg_id}: {e}")
            pass

    X = np.array(X)
    print(f"Final X shape: {X.shape}")
    
    if len(X) > 0:
        np.save(os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy'), X)
        out_df = pd.DataFrame(y_meta)
        out_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv'), index=False)
        print(f"Successfully processed {len(X)} records.")
    else:
        print("No records processed successfully.")

if __name__ == "__main__":
    preprocess_ptbxl()
