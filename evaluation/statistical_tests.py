import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import scipy.stats as stats
from sklearn.utils import resample

def compute_bootstrap_ci(y_true, y_pred_proba, n_bootstraps=1000, alpha=0.95):
    """
    Computes 95% Confidence Intervals for AUC and F1 using bootstrapping.
    Supports both single-label and multi-label (OVR).
    """
    bootstrapped_auc = []
    bootstrapped_f1 = []
    
    # Per-class F1 storage
    n_classes = y_true.shape[1] if y_true.ndim > 1 else 1
    bootstrapped_per_class_f1 = [[] for _ in range(n_classes)]
    
    indices = np.arange(len(y_true))
    for i in range(n_bootstraps):
        resampled_indices = resample(indices)
        y_true_resampled = y_true[resampled_indices]
        y_proba_resampled = y_pred_proba[resampled_indices]
        y_pred_resampled = (y_proba_resampled > 0.5).astype(int)
        
        try:
            # Macro Metrics
            bootstrapped_auc.append(roc_auc_score(y_true_resampled, y_proba_resampled, average='macro'))
            bootstrapped_f1.append(f1_score(y_true_resampled, y_pred_resampled, average='macro', zero_division=0))
            
            # Per-class F1
            if n_classes > 1:
                p_f1 = f1_score(y_true_resampled, y_pred_resampled, average=None, zero_division=0)
                for c_idx in range(n_classes):
                    bootstrapped_per_class_f1[c_idx].append(p_f1[c_idx])
            else:
                bootstrapped_per_class_f1[0].append(f1_score(y_true_resampled, y_pred_resampled, zero_division=0))
        except:
            continue
            
    calc_ci = lambda x: (np.percentile(x, (1-alpha)/2 * 100), np.percentile(x, (1+alpha)/2 * 100))
    
    results = {
        'auc_mean': np.mean(bootstrapped_auc),
        'auc_ci': calc_ci(bootstrapped_auc),
        'f1_mean': np.mean(bootstrapped_f1),
        'f1_ci': calc_ci(bootstrapped_f1),
        'per_class_f1_means': [np.mean(b) for b in bootstrapped_per_class_f1],
        'per_class_f1_cis': [calc_ci(b) for b in bootstrapped_per_class_f1]
    }
    
    return results

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Computes McNemar's test p-value for two models.
    y_pred1 and y_pred2 should be binary classifications.
    Handles multi-label by flattening if necessary.
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
        y_pred1 = y_pred1.flatten()
        y_pred2 = y_pred2.flatten()
        
    # Contingency table
    #           Model 2 Correct  Model 2 Incorrect
    # Mod 1 Correct      a                b
    # Mod 1 Incorrect    c                d
    
    m1_correct = (y_pred1 == y_true)
    m2_correct = (y_pred2 == y_true)
    
    b = np.sum(m1_correct & ~m2_correct)
    c = np.sum(~m1_correct & m2_correct)
    
    # McNemar's test statistic: (b - c)^2 / (b + c)
    if (b + c) == 0:
        return 1.0
        
    stat = (abs(b - c) - 1)**2 / (b + c)  # continuity corrected
    p_value = stats.chi2.sf(stat, 1)
    
    return p_value

def delong_auc_test(y_true, y_proba1, y_proba2):
    """
    Wrapper for comparing two models' AUC using DeLong's method or bootstrap as fallback.
    For simplicity in this environment, we use a robust bootstrap-based p-value if DeLong code is unavailable.
    """
    # For a formal paper, a full DeLong implementation is preferred.
    # Here we implement a permutation/bootstrap test for AUC difference.
    n_bootstraps = 1000
    auc_diffs = []
    
    indices = np.arange(len(y_true))
    for i in range(n_bootstraps):
        resampled_indices = resample(indices)
        y_t = y_true[resampled_indices]
        p1 = y_proba1[resampled_indices]
        p2 = y_proba2[resampled_indices]
        
        try:
            auc1 = roc_auc_score(y_t, p1, average='macro')
            auc2 = roc_auc_score(y_t, p2, average='macro')
            auc_diffs.append(auc1 - auc2)
        except:
            continue
            
    # p-value is the proportion of resamples where the difference is <= 0 (if we expect 1 > 2)
    p_val = np.mean(np.array(auc_diffs) <= 0)
    # two-tailed
    p_val = 2 * min(p_val, 1 - p_val)
    return p_val

if __name__ == '__main__':
    # Simple test
    y_t = np.array([0, 0, 1, 1] * 25)
    y_p1 = np.array([0.1, 0.4, 0.6, 0.9] * 25)
    y_p2 = np.array([0.2, 0.3, 0.7, 0.8] * 25)
    print("Test Results:", compute_bootstrap_ci(y_t, y_p1, n_bootstraps=100))
