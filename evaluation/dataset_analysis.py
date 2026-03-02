import pandas as pd
import numpy as np
import ast
import os

def analyze_ptbxl():
    df = pd.read_csv('data/processed/ptbxl/ptbxl_database_processed.csv')
    
    stats = {
        'count': len(df),
        'age_mean': df['age'].mean(),
        'sex_m_pct': (df['sex'] == 0).mean() * 100, # 0=M, 1=F
    }
    
    # Class prevalence
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    for cls in classes:
        stats[f'prevalence_{cls}'] = df['diagnostic_superclass'].apply(lambda l: cls in l).mean() * 100
        
    return stats

def analyze_chapman():
    df = pd.read_csv('data/processed/chapman/chapman_database.csv')
    
    stats = {
        'count': len(df),
        'age_mean': df['age'].mean(),
        'sex_m_pct': (df['sex'] == 'Male').mean() * 100,
    }
    
    # SNOMED mapping from evaluation/external_validation.py
    mapping = {
        '426177001': 'NORM', '426783006': 'NORM', '427084000': 'NORM', '164884004': 'NORM',
        '164861001': 'MI', '164865005': 'MI', '164909002': 'MI', '22298006': 'MI',
        '55930002': 'STTC', '164931005': 'STTC', '164930006': 'STTC', '164873001': 'STTC',
        '39732003': 'STTC', '427172004': 'STTC', '164917005': 'STTC', '426761007': 'STTC',
        '713427006': 'CD', '713426002': 'CD', '445118002': 'CD', '445211001': 'CD',
        '164903001': 'CD', '164951009': 'CD', '425868008': 'CD', '426112009': 'CD',
        '426660007': 'CD', '63593006': 'CD', '6374002': 'CD',
        '164890007': 'HYP', '164871004': 'HYP', '164872006': 'HYP', '89792004': 'HYP',
        '164934002': 'HYP', '429622005': 'HYP', '428750005': 'HYP'
    }
    
    df['labels_list'] = df['labels'].apply(ast.literal_eval)
    
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    for cls in classes:
        def has_class(labels):
            mapped_classes = [mapping.get(str(l)) for l in labels]
            return cls in mapped_classes
        stats[f'prevalence_{cls}'] = df['labels_list'].apply(has_class).mean() * 100
        
    return stats

def main():
    print("Analyzing PTB-XL...")
    ptb_stats = analyze_ptbxl()
    print("\nAnalyzing Chapman...")
    chap_stats = analyze_chapman()
    
    # Create comparison table
    comparison = pd.DataFrame([ptb_stats, chap_stats], index=['PTB-XL', 'Chapman-Shaoxing'])
    print("\nDataset Comparison Table:")
    print(comparison.T)

if __name__ == '__main__':
    main()
