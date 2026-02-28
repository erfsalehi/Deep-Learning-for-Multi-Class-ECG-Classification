import wfdb
import os
import zipfile
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample
import ast
from tqdm import tqdm
import re

# Paths
TARGET_DIR = os.path.join("data", "raw", "chapman")
PROCESSED_DIR = os.path.join("data", "processed", "chapman")
SNOMED_MAPPING_PATH = os.path.join(TARGET_DIR, "ConditionNames_SNOMED-CT.csv")

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

def get_snomed_mapping():
    if not os.path.exists(SNOMED_MAPPING_PATH):
        print(f"Warning: {SNOMED_MAPPING_PATH} not found. Using empty mapping.")
        return {}
    df = pd.read_csv(SNOMED_MAPPING_PATH)
    # Map Snomed_CT to Acronym Name
    mapping = dict(zip(df['Snomed_CT'].astype(str), df['Acronym Name']))
    return mapping

def extract_labels_from_header(header_path, mapping):
    """Extracts diagnostic codes from .hea file and maps them to acronyms."""
    dx_codes = []
    with open(header_path, 'r') as f:
        for line in f:
            if line.startswith('#Dx:'):
                codes = line.split(':')[-1].strip().split(',')
                for code in codes:
                    acronym = mapping.get(code.strip(), code.strip())
                    dx_codes.append(acronym)
    return dx_codes

def preprocess_and_save():
    print(f"Preprocessing dataset and saving to {PROCESSED_DIR}...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    snomed_map = get_snomed_mapping()
    
    # Use RECORDS file to find all records
    records_file = os.path.join(TARGET_DIR, 'RECORDS')
    if not os.path.exists(records_file):
        print(f"Error: {records_file} not found.")
        return

    with open(records_file, 'r') as f:
        record_paths = [line.strip() for line in f if line.strip()]

    X = []
    y_meta = []
    
    for rel_path in tqdm(record_paths):
        # rel_path might be like "WFDBRecords/01/010/"
        # We need to find the .mat and .hea files in that directory
        full_dir = os.path.join(TARGET_DIR, rel_path)
        if not os.path.isdir(full_dir):
            continue

        for filename in os.listdir(full_dir):
            if filename.endswith(".hea"):
                record_id = filename.replace(".hea", "")
                header_path = os.path.join(full_dir, filename)
                record_path = os.path.join(full_dir, record_id)
                
                try:
                    # 1. Extract Labels
                    labels = extract_labels_from_header(header_path, snomed_map)
                    
                    # 2. Load Signal
                    record = wfdb.rdrecord(record_path)
                    sig = record.p_signal
                    fs = record.fs
                    
                    processed_leads = []
                    for lead_idx in range(sig.shape[1]):
                        lead_sig = sig[:, lead_idx]
                        
                        # Apply Filters
                        filtered = apply_filter(lead_sig, fs)
                        
                        # Downsample to 100Hz
                        target_len = 1000 # 10 seconds at 100Hz
                        downsampled = resample(filtered, target_len)
                        
                        # Z-score normalization
                        mean_val = np.mean(downsampled)
                        std_val = np.std(downsampled)
                        if std_val > 1e-6:
                            normalized = (downsampled - mean_val) / std_val
                        else:
                            normalized = downsampled - mean_val
                            
                        processed_leads.append(normalized)
                    
                    # Stack to get (1000, 12)
                    processed_sig = np.column_stack(processed_leads)
                    
                    X.append(processed_sig)
                    y_meta.append({
                        'record_id': record_id,
                        'labels': labels,
                        'age': record.comments[0].split(':')[-1].strip() if len(record.comments) > 0 else '',
                        'sex': record.comments[1].split(':')[-1].strip() if len(record.comments) > 1 else ''
                    })
                    
                except Exception as e:
                    # print(f"Error processing {record_id}: {e}")
                    pass

    X = np.array(X)
    print(f"Final X shape: {X.shape}")
    
    if len(X) > 0:
        np.save(os.path.join(PROCESSED_DIR, 'X_chapman.npy'), X)
        
        out_df = pd.DataFrame(y_meta)
        out_df.to_csv(os.path.join(PROCESSED_DIR, 'chapman_database.csv'), index=False)
        print(f"Successfully processed {len(X)} records.")
    else:
        print("No records processed successfully.")

if __name__ == "__main__":
    preprocess_and_save()
