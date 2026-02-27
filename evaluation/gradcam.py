import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.multiclass_loader import MultiClassECGDataGenerator, load_ptbxl_data

# Set paths
DATA_PATH = 'data/raw/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
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

        # Extract gradients with respect to the output of the convolutional layer
        grads = tape.gradient(loss, conv_outputs)

        # Guided gradients
        guided_grads = tf.cast(conv_outputs > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

        # Compute average gradient per feature map
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        # Build Grad-CAM heatmap
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        
        # Apply ReLU
        heatmap = tf.maximum(cam, 1e-10)
        
        # Normalize
        heatmap = heatmap / tf.reduce_max(heatmap)
        return heatmap.numpy()

def plot_gradcam(signal, heatmap, title, save_path):
    # signal shape (1000, 12)
    # heatmap shape (variable size, depending on layer) - need to resize to 1000
    
    heatmap_resized = cv2.resize(heatmap[0], (1, 1000))
    heatmap_resized = heatmap_resized.flatten()
    
    fig, axes = plt.subplots(12, 1, figsize=(10, 20), sharex=True)
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Time axis (scaled to 10 seconds)
    time = np.linspace(0, 10, 1000)
    
    for i in range(12):
        axes[i].plot(time, signal[:, i], color='black', alpha=0.7)
        # Overlay heatmap using background color
        # We can use fill_between to show intensity
        # Use viridis colormap for the intensity
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
    try:
        df = load_ptbxl_data(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return

    test_df = df[df.strat_fold == 10].copy()
    test_df['primary_class'] = test_df['diagnostic_superclass'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    test_df = test_df[test_df['primary_class'] != 'Unknown']
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found. Please train SE-ResNet first.")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Identify last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer.name
            break
            
    print(f"Using layer: {last_conv_layer}")
    gradcam = GradCAM(model, last_conv_layer)
    
    # 3. Select Representative Cases
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # We need a generator to get processed signals
    gen = MultiClassECGDataGenerator(test_df, DATA_PATH, batch_size=1, shuffle=False)
    
    # Representative cases to find:
    # 1. NORM TP
    # 2. MI TP
    # 3. HYP FN (misclassified as NORM)
    
    found_cases = {
        'NORM_TP': None,
        'MI_TP': None,
        'HYP_FN': None
    }
    
    for i in range(len(gen)):
        X, y = gen[i]
        y_prob = model.predict(X, verbose=0)
        y_pred = np.argmax(y_prob, axis=-1)[0]
        y_true_idx = np.argmax(y, axis=-1)[0]
        
        true_label = classes[y_true_idx]
        pred_label = classes[y_pred]
        
        if found_cases['NORM_TP'] is None and true_label == 'NORM' and pred_label == 'NORM':
            found_cases['NORM_TP'] = (X[0], y_true_idx, "Normal Case (TP)")
            
        if found_cases['MI_TP'] is None and true_label == 'MI' and pred_label == 'MI':
            found_cases['MI_TP'] = (X[0], y_true_idx, "Myocardial Infarction (TP)")
            
        if found_cases['HYP_FN'] is None and true_label == 'HYP' and pred_label == 'NORM':
            found_cases['HYP_FN'] = (X[0], y_true_idx, "Hypertrophy (FN, misclassified as NORM)")
            
        if all(v is not None for v in found_cases.values()):
            break

    # 4. Generate Visualizations
    for case_id, (signal, class_idx, title) in found_cases.items():
        if signal is not None:
            heatmap = gradcam.compute_heatmap(signal[np.newaxis, ...], class_idx)
            save_path = os.path.join(FIGURES_DIR, f'gradcam_{case_id}.png')
            plot_gradcam(signal, heatmap, title, save_path)
            print(f"Generated {case_id} saliency map.")
        else:
            print(f"Warning: Could not find a suitable test case for {case_id}.")

if __name__ == "__main__":
    generate_gradcam_analysis()
