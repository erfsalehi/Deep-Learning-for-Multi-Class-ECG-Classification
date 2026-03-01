import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Journal specifications
def set_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
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

FIGURES_DIR = 'results/figures/publication/'
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_figure_4_ablation():
    """Figure 4: Augmentation Ablation Bar Chart (Delta vs Baseline)"""
    print("Generating Figure 4...")
    df = pd.read_csv('results/fold10_metrics.csv')
    
    # Extract baseline
    baseline_f1 = df[df['model'] == 'se_resnet_best.keras']['f1_macro'].values[0]
    
    # Filter ablations
    ablations = df[df['model'].str.contains('AUG')]
    ablations = ablations.copy()
    ablations['delta_f1'] = ablations['f1_macro'] - baseline_f1
    ablations['aug_label'] = ablations['model'].apply(lambda x: x.split('_')[2].replace('.keras', ''))
    
    plt.figure(figsize=(mm_to_inches(84), mm_to_inches(80)))
    sns.barplot(data=ablations, x='aug_label', y='delta_f1', palette='viridis')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.ylabel('Macro-F1 Delta vs Baseline')
    plt.xlabel('Augmentation Technique')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_ablation_study.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_ablation_study.png'))

def generate_figure_6_gradcam_composite():
    """Figure 6: Grad-CAM Saliency 3-Panel Figure"""
    print("Generating Figure 6...")
    files = [
        'results/figures/gradcam_NORM_TP.png',
        'results/figures/gradcam_MI_TP.png',
        'results/figures/gradcam_HYP_FN.png'
    ]
    
    if not all(os.path.exists(f) for f in files):
        print("Skipping Figure 6: Missing individual Grad-CAM files.")
        return

    images = [Image.open(f) for f in files]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(os.path.join(FIGURES_DIR, 'fig6_gradcam_composite.png'))
    # Convert to PDF if possible via pillow or just save png for now

def generate_figure_7_prevalence_vs_f1():
    """Figure 7: Prevalence vs F1 Scatter Plot"""
    print("Generating Figure 7...")
    # Data from Fold 1-9 (prevalence) and Fold 10 (F1)
    # Prevalence: {'NORM': 0.436, 'MI': 0.231, 'STTC': 0.211, 'CD': 0.247, 'HYP': 0.122}
    # F1 (Baseline): [0.909, 0.569, 0.596, 0.573, 0.320]
    
    data = {
        'class': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
        'prevalence': [0.436, 0.231, 0.211, 0.247, 0.122],
        'f1': [0.909, 0.569, 0.596, 0.573, 0.320]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(mm_to_inches(84), mm_to_inches(80)))
    sns.regplot(data=df, x='prevalence', y='f1', ci=None, 
                scatter_kws={'s': 100, 'color': 'navy'}, 
                line_kws={'color': 'gray', 'linestyle': '--', 'linewidth': 1})
    
    for i, row in df.iterrows():
        plt.annotate(row['class'], (row['prevalence'], row['f1']), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Class Prevalence (Training Set)')
    plt.ylabel('F1-Score (Test Set)')
    plt.xlim(0.1, 0.5)
    plt.ylim(0.2, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_prevalence_vs_f1.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_prevalence_vs_f1.png'))

def main():
    set_publication_style()
    generate_figure_4_ablation()
    generate_figure_6_gradcam_composite()
    generate_figure_7_prevalence_vs_f1()
    print(f"All figures saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main()
