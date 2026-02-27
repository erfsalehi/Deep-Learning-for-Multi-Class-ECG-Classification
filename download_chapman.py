import wfdb
import os
import zipfile
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample
import ast
from tqdm import tqdm

ZIP_PATH = os.path.join("data", "raw", "chapman.zip")
TARGET_DIR = os.path.join("data", "raw", "chapman")
PROCESSED_DIR = os.path.join("data", "processed", "chapman")

def extract_chapman():
    print(f"Extracting Chapman-Shaoxing dataset...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # Assuming the zip contains a main folder, we extract everything
            zip_ref.extractall(TARGET_DIR)
        print("Extraction complete!")
    else:
        print(f"Zip file {ZIP_PATH} not found.")

def apply_filter(signal_1d, fs):
    # Butterworth bandpass 0.5 - 40 Hz
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 40.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal_1d)
    
    # Notch filter for 50Hz (powerline)
    w0 = 50.0 / nyq
    b_notch, a_notch = iirnotch(w0, 30.0)
    filtered = filtfilt(b_notch, a_notch, filtered)
    
    return filtered

def find_extracted_dir(base_dir):
    # The zip might extract to a subfolder like 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0'
    for root, dirs, files in os.walk(base_dir):
        if 'Diagnostics.csv' in files:
            return root
    return base_dir

def preprocess_and_save():
    print(f"Preprocessing dataset and saving to {PROCESSED_DIR}...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    data_root = find_extracted_dir(TARGET_DIR)
    csv_path = os.path.join(data_root, 'Diagnostics.csv')
    
    if not os.path.exists(csv_path):
        print(f"Could not find Diagnostics.csv in {data_root}.")
        return

    records_df = pd.read_csv(csv_path)
    
    X = []
    y_meta = []
    
    for _, row in tqdm(records_df.iterrows(), total=len(records_df)):
        filename = row['FileName']
        
        # Determine paths
        # Given PhysioNet structure, it usually is under WFDBRecords/01/01001 etc.
        # However, filename usually includes subdir or we can just find it
        if '_' in filename:
            subdir = filename.split('_')[0]
            record_path = os.path.join(data_root, 'WFDBRecords', subdir, filename)
        else:
            record_path = os.path.join(data_root, 'WFDBRecords', filename)
            if not os.path.exists(record_path + '.dat'):
                record_path = os.path.join(data_root, filename)
                
        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal
            fs = record.fs
            
            processed_leads = []
            for lead_idx in range(sig.shape[1]):
                lead_sig = sig[:, lead_idx]
                
                # 1. Bandpass & Notch
                filtered = apply_filter(lead_sig, fs)
                
                # 2. Downsample to 100Hz
                target_len = int(len(filtered) * 100 / fs)
                downsampled = resample(filtered, target_len)
                
                # 3. Z-score normalization
                mean_val = np.mean(downsampled)
                std_val = np.std(downsampled)
                if std_val > 1e-6:
                    normalized = (downsampled - mean_val) / std_val
                else:
                    normalized = downsampled - mean_val
                    
                processed_leads.append(normalized)
            
            # Stack to get (1000, 12)
            processed_sig = np.column_stack(processed_leads)
            
            # Fix shape
            if processed_sig.shape[0] > 1000:
                processed_sig = processed_sig[:1000, :]
            elif processed_sig.shape[0] < 1000:
                pad_width = 1000 - processed_sig.shape[0]
                processed_sig = np.pad(processed_sig, ((0, pad_width), (0, 0)), 'constant')
            
            X.append(processed_sig)
            y_meta.append(row.to_dict())
            
        except Exception as e:
            pass

    X = np.array(X)
    np.save(os.path.join(PROCESSED_DIR, 'X_chapman.npy'), X)
    
    out_df = pd.DataFrame(y_meta)
    out_df.to_csv(os.path.join(PROCESSED_DIR, 'chapman_database.csv'), index=False)
    
    print(f"Processed shape: {X.shape}")

if __name__ == "__main__":
    extract_chapman()
    preprocess_and_save()
