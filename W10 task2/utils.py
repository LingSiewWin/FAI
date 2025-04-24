# utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from joblib import Parallel, delayed

def calculate_business_metrics(y_true, y_pred, y_prob=None, fn_cost=5, fp_cost=1):
    """
    Calculate business-oriented metrics for model evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    y_prob : array-like, optional
        Predicted probabilities for the positive class
    fn_cost : float, optional
        Cost of a false negative (missing a churner)
    fp_cost : float, optional
        Cost of a false positive (incorrectly predicting churn)
        
    Returns:
    --------
    dict
        Dictionary of business metrics
    """
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate standard metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate business costs
    total_cost = (fn * fn_cost) + (fp * fp_cost)
    cost_per_customer = total_cost / len(y_true)
    
    # Calculate customer retention metrics
    retention_rate = tn / (tn + fn) if (tn + fn) > 0 else 0
    intervention_efficiency = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Calculate profitability metrics (assuming average values)
    avg_customer_value = 1000  # Hypothetical average customer lifetime value
    avg_intervention_cost = 100  # Hypothetical cost of retention intervention
    
    # Potential savings from interventions
    potential_savings = tp * avg_customer_value - (tp + fp) * avg_intervention_cost
    
    # ROI of the churn prevention program
    roi = (potential_savings / ((tp + fp) * avg_intervention_cost)) if (tp + fp) > 0 else 0
    
    # Return all metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_negatives': fn,
        'false_positives': fp,
        'total_business_cost': total_cost,
        'cost_per_customer': cost_per_customer,
        'retention_rate': retention_rate,
        'intervention_efficiency': intervention_efficiency,
        'potential_savings': potential_savings,
        'roi': roi
    }
    
    return metrics

def compute_metrics_at_threshold(y_probs, y_true, threshold, fn_cost, fp_cost):
    """Compute metrics for a single threshold (for parallel processing)."""
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    business_cost = (fn * fn_cost) + (fp * fp_cost)
    avg_customer_value, avg_intervention_cost = 1000, 100
    potential_savings = tp * avg_customer_value - (tp + fp) * avg_intervention_cost
    roi = (potential_savings / ((tp + fp) * avg_intervention_cost)) if (tp + fp) > 0 else 0
    return accuracy, precision, recall, f1, business_cost, roi

def find_optimal_thresholds(model, X, y_true, metric_name='f1', fn_cost=5, fp_cost=1):
    """
    Find the optimal classification threshold based on various metrics.
    
    Parameters:
    -----------
    model : estimator
        Trained classifier with predict_proba method
    X : array-like
        Input features
    y_true : array-like
        True class labels
    metric_name : str, optional
        Metric to optimize ('f1', 'cost', 'precision', 'recall', 'roi')
    fn_cost : float, optional
        Cost of a false negative
    fp_cost : float, optional
        Cost of a false positive
        
    Returns:
    --------
    dict
        Dictionary with optimal thresholds for different metrics
    """
    # Get predicted probabilities
    y_probs = model.predict_proba(X)[:, 1]
    
    # Parallelize threshold computation
    thresholds = np.linspace(0.01, 0.99, 99)
    metrics_list = Parallel(n_jobs=-1)(
        delayed(compute_metrics_at_threshold)(y_probs, y_true, t, fn_cost, fp_cost) for t in thresholds
    )

    metrics = {
        'threshold': list(thresholds),
        'accuracy': [m[0] for m in metrics_list],
        'precision': [m[1] for m in metrics_list],
        'recall': [m[2] for m in metrics_list],
        'f1': [m[3] for m in metrics_list],
        'business_cost': [m[4] for m in metrics_list],
        'roi': [m[5] for m in metrics_list]
    }
    
    # Find optimal thresholds
    results = {
        'accuracy': thresholds[np.argmax(metrics['accuracy'])],
        'precision': thresholds[np.argmax(metrics['precision'])],
        'recall': thresholds[np.argmax(metrics['recall'])],
        'f1': thresholds[np.argmax(metrics['f1'])],
        'business_cost': thresholds[np.argmin(metrics['business_cost'])],
        'roi': thresholds[np.argmax(metrics['roi'])]
    }
    
    # Return threshold based on specified metric
    if metric_name == 'cost':
        optimal_threshold = results['business_cost']
    else:
        optimal_threshold = results[metric_name]
    
    return {
        'optimal_threshold': optimal_threshold,
        'all_thresholds': results,
        'metrics': pd.DataFrame(metrics)
    }