import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

# Define path (assuming script is run from project root or scripts folder)
# We assume the user runs this from the project root
path = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'

print("Checking for PTB-XL data...")
# Check if data exists
if not os.path.exists(path) or not os.path.exists(os.path.join(path, 'ptbxl_database.csv')):
    print(f"Data directory {path} not found or empty.")
    print("Please download the PTB-XL dataset as described in the Quick Start Guide.")
    # Attempt to create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")
    print("You can download the data using: wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/raw/ptbxl/")
    exit()

try:
    # Load database
    print("Loading database...")
    data = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    print(f"Total records: {len(data)}")
    print(data.head())

    # Load first ECG record
    print("Loading first ECG record...")
    record_path = path + data.iloc[0]['filename_lr']
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal  # Shape: (samples, 12 leads)
    print(f"Signal shape: {signal.shape}")
    print(f"Sampling rate: {record.fs} Hz")
    print(f"Duration: {len(signal)/record.fs:.1f} seconds")

    # Visualize 12-lead ECG
    print("Plotting ECG...")
    fig, axes = plt.subplots(6, 2, figsize=(15, 12))
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for i, (ax, lead_name) in enumerate(zip(axes.flat, lead_names)):
        ax.plot(signal[:1000, i], linewidth=0.5)
        ax.set_title(f'Lead {lead_name}')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'results/figures/my_first_ecg.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    print("SUCCESS! You just loaded and visualized your first ECG!")

except Exception as e:
    print(f"Error: {e}")
