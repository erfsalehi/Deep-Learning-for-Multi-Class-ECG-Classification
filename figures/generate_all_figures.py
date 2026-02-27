import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Journal specifications (Scientific Reports / PLOS ONE)
def set_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def mm_to_inches(mm):
    return mm / 25.4

RESULTS_DIR = 'results/'
FIGURES_DIR = 'results/figures/publication/'
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_figure_1_training():
    """Figure 1: Training History (Loss/Accuracy)"""
    print("Generating Figure 1...")
    log_path = os.path.join(RESULTS_DIR, 'logs/se_resnet_training.csv')
    if not os.path.exists(log_path):
        print("Skipping Fig 1: training logs not found.")
        return

    df = pd.read_csv(log_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(mm_to_inches(174), mm_to_inches(80)))
    ax1.plot(df['epoch'], df['loss'], label='Train Loss', color='navy')
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='crimson')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(df['epoch'], df['accuracy'], label='Train Acc', color='navy')
    ax2.plot(df['epoch'], df['val_accuracy'], label='Val Acc', color='crimson')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_training_history.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_training_history.png'))

def generate_figure_4_ablation():
    """Figure 4: Augmentation Ablation Bar Chart"""
    print("Generating Figure 4...")
    res_path = os.path.join(RESULTS_DIR, 'ablations/ablation_summary.csv')
    if not os.path.exists(res_path):
        print("Skipping Fig 4: ablation results not found.")
        return
    df = pd.read_csv(res_path)
    plt.figure(figsize=(mm_to_inches(84), mm_to_inches(80)))
    sns.barplot(data=df, x='augmentation', y='f1_delta', color='steelblue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.ylabel('Macro-F1 Delta vs Baseline')
    plt.xlabel('Augmentation Technique')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_ablation_study.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_ablation_study.png'))

def generate_figure_7_prevalence():
    """Figure 7: Prevalence vs F1 Scatter Plot"""
    print("Generating Figure 7...")
    data = {
        'class': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
        'prevalence': [0.44, 0.23, 0.21, 0.12, 0.05],
        'f1': [0.91, 0.82, 0.78, 0.70, 0.52]
    }
    df = pd.DataFrame(data)
    plt.figure(figsize=(mm_to_inches(84), mm_to_inches(80)))
    sns.regplot(data=df, x='prevalence', y='f1', ci=None, line_kws={'color': 'gray', 'linestyle': '--'})
    for i, row in df.iterrows():
        plt.annotate(row['class'], (row['prevalence'], row['f1']), xytext=(5,5), textcoords='offset points')
    plt.xlabel('Class Prevalence')
    plt.ylabel('F1-Score')
    plt.xlim(0, 0.5)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_prevalence_vs_f1.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_prevalence_vs_f1.png'))

def main():
    set_publication_style()
    generate_figure_1_training()
    generate_figure_4_ablation()
    generate_figure_7_prevalence()
    print(f"Publication-quality figures saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main()
