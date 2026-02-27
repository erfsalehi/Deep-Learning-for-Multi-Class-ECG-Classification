import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# Set paths
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
OUTPUT_DIR = 'results/figures/'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print("Loading PTB-XL database...")
    df = pd.read_csv(os.path.join(DATA_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
    
    # Parse scp_codes (stored as string representation of dict)
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    
    return df

def plot_distributions(df):
    print("Generating distribution plots...")
    
    # 1. Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'].dropna(), bins=30, kde=True)
    plt.title('Age Distribution in PTB-XL Dataset')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'age_distribution.png'))
    plt.close()
    
    # 2. Sex Distribution
    plt.figure(figsize=(8, 8))
    sex_counts = df['sex'].value_counts()
    plt.pie(sex_counts, labels=['Male (0)', 'Female (1)'], autopct='%1.1f%%', startangle=90)
    plt.title('Gender Distribution')
    plt.savefig(os.path.join(OUTPUT_DIR, 'gender_distribution.png'))
    plt.close()
    
    # 3. Diagnostic Superclass Distribution
    # Aggregate all diagnostic codes
    all_diagnoses = []
    for codes in df['scp_codes']:
        all_diagnoses.extend(codes.keys())
    
    diag_counts = Counter(all_diagnoses)
    
    # Plot top 20 diagnoses
    top_diagnoses = dict(diag_counts.most_common(20))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(top_diagnoses.values()), y=list(top_diagnoses.keys()))
    plt.title('Top 20 Diagnostic Codes')
    plt.xlabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diagnosis_distribution.png'))
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        data = load_data()
        print(f"Loaded {len(data)} records.")
        plot_distributions(data)
        
        # Print summary statistics
        print("\nDataset Summary:")
        print(f"Total Records: {len(data)}")
        print(f"Age Mean: {data['age'].mean():.1f} ± {data['age'].std():.1f} years")
        print(f"Sex Distribution:\n{data['sex'].value_counts()}")
        
    except FileNotFoundError:
        print(f"Error: Database file not found at {DATA_PATH}")
        print("Please ensure the dataset is downloaded correctly.")
