import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import ast

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.multiclass_loader import NpyECGDataGenerator

# Set paths
PROCESSED_DATA_PATH = 'data/processed/ptbxl/'
MODEL_PATH = 'results/models/se_resnet_best.keras'
FIGURES_DIR = 'results/figures/'

os.makedirs(FIGURES_DIR, exist_ok=True)

class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

    def compute_heatmap(self, inputs, class_idx):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(inputs)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        guided_grads = tf.cast(conv_outputs > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        heatmap = tf.maximum(cam, 1e-10)
        heatmap = heatmap / tf.reduce_max(heatmap)
        return heatmap.numpy()

def plot_gradcam(signal, heatmap, title, save_path):
    heatmap_resized = cv2.resize(heatmap[0], (1, 1000))
    heatmap_resized = heatmap_resized.flatten()
    
    fig, axes = plt.subplots(12, 1, figsize=(10, 20), sharex=True)
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    time = np.linspace(0, 10, 1000)
    
    for i in range(12):
        axes[i].plot(time, signal[:, i], color='black', alpha=0.7)
        axes[i].imshow(heatmap_resized[np.newaxis, :], aspect='auto', 
                       extent=[time[0], time[-1], np.min(signal[:, i]), np.max(signal[:, i])],
                       cmap='viridis', alpha=0.3)
        axes[i].set_ylabel(leads[i])
        
    plt.xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_gradcam_analysis():
    print("Starting Grad-CAM Saliency Analysis...")
    
    # 1. Load Data
    X_path = os.path.join(PROCESSED_DATA_PATH, 'X_ptbxl.npy')
    meta_path = os.path.join(PROCESSED_DATA_PATH, 'ptbxl_database_processed.csv')
    
    if not os.path.exists(X_path):
        print(f"Error: Processed data not found at {X_path}. Run preprocess_ptbxl.py first.")
        return

    X_all = np.load(X_path)
    df = pd.read_csv(meta_path)
    df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(ast.literal_eval)

    test_idx = df[df.strat_fold == 10].index.values
    X_test = X_all[test_idx]
    test_df = df.iloc[test_idx].copy()
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        model_files = [f for f in os.listdir('results/models/') if f.endswith('.keras')]
        if model_files:
            MODEL_PATH_ACTUAL = os.path.join('results/models/', model_files[0])
            print(f"Primary model not found, using {MODEL_PATH_ACTUAL} instead.")
        else:
            print(f"Error: No model found in results/models/. Please train a model first.")
            return
    else:
        MODEL_PATH_ACTUAL = MODEL_PATH
        
    model = tf.keras.models.load_model(MODEL_PATH_ACTUAL)
    
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer.name
            break
            
    print(f"Using layer: {last_conv_layer}")
    gradcam = GradCAM(model, last_conv_layer)
    
    # 3. Select Representative Cases
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    found_cases = {'NORM_TP': None, 'MI_TP': None, 'HYP_FN': None}
    
    print("Searching for suitable test cases...")
    for i in range(len(X_test)):
        X = X_test[i:i+1]
        y_prob = model.predict(X, verbose=0)
        y_pred = np.argmax(y_prob, axis=-1)[0]
        
        diagnoses = test_df.iloc[i]['diagnostic_superclass']
        
        # Check if diagnoses overlap with classes
        present_classes = [c for c in classes if c in diagnoses]
        if not present_classes: continue
        
        true_label = present_classes[0] # Take first for simplicity
        pred_label = classes[y_pred]
        
        if found_cases['NORM_TP'] is None and true_label == 'NORM' and pred_label == 'NORM':
            found_cases['NORM_TP'] = (X[0], classes.index('NORM'), "Normal Case (TP)")
            
        if found_cases['MI_TP'] is None and true_label == 'MI' and pred_label == 'MI':
            found_cases['MI_TP'] = (X[0], classes.index('MI'), "Myocardial Infarction (TP)")
            
        if found_cases['HYP_FN'] is None and true_label == 'HYP' and pred_label == 'NORM':
            found_cases['HYP_FN'] = (X[0], classes.index('HYP'), "Hypertrophy (FN, misclassified as NORM)")
            
        if all(v is not None for v in found_cases.values()):
            break

    # 4. Generate Visualizations
    for case_id, val in found_cases.items():
        if val is not None:
            signal, class_idx, title = val
            heatmap = gradcam.compute_heatmap(signal[np.newaxis, ...], class_idx)
            save_path = os.path.join(FIGURES_DIR, f'gradcam_{case_id}.png')
            plot_gradcam(signal, heatmap, title, save_path)
            print(f"Generated {case_id} saliency map.")
        else:
            print(f"Warning: Could not find a suitable test case for {case_id}.")

if __name__ == "__main__":
    generate_gradcam_analysis()
