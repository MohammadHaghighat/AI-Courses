# === File: evaluation.py ===
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt

def calculate_auc_ap(labels, scores):
    """Calculates Area Under the ROC Curve and Average Precision."""
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap

def calculate_f1(labels, scores):
    """Calculates the F1-score by finding the best threshold."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Exclude the last value which is 1 for precision and 0 for recall
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    return np.max(f1_scores)

def plot_results(results, metric_name, anomaly_ratios):
    """Plots the performance metric against anomaly contamination ratios."""
    plt.figure(figsize=(8, 5))
    plt.plot(anomaly_ratios, results, marker='o', linestyle='-')
    plt.title(f'SRR Performance vs. Anomaly Ratio ({metric_name})')
    plt.xlabel('Anomaly Contamination Ratio')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.xticks(anomaly_ratios)
    plt.savefig(f'srr_performance_{metric_name}.png')
    plt.show()