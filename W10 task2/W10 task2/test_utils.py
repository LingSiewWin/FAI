# test_utils.py
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

def calculate_business_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate business metrics based on true labels and predictions.
    
    Parameters:
    - y_true: Array-like of true labels (0 or 1)
    - y_pred: Array-like of predicted labels (0 or 1)
    - y_proba: Array-like of predicted probabilities (optional)
    
    Returns:
    - dict: Dictionary containing business metrics (e.g., cost, revenue)
    """
    # Example: Define costs and revenues
    # - False Positive (FP): Predicting 1 when true label is 0 (e.g., cost of unnecessary action)
    # - False Negative (FN): Predicting 0 when true label is 1 (e.g., missed opportunity cost)
    # - True Positive (TP): Predicting 1 when true label is 1 (e.g., revenue from correct action)
    # - True Negative (TN): Predicting 0 when true label is 0 (no cost/revenue)
    
    # Example cost/revenue values (adjust based on your business case)
    cost_fp = 100  # Cost of a false positive
    cost_fn = 500  # Cost of a false negative
    revenue_tp = 1000  # Revenue from a true positive
    
    # Compute confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate total cost and revenue
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    total_revenue = tp * revenue_tp
    
    return {
        'cost': total_cost,
        'revenue': total_revenue,
        'net_profit': total_revenue - total_cost
    }

def find_optimal_thresholds(model, X, y):
    """
    Find the optimal threshold for a model based on F1 score.
    
    Parameters:
    - model: Trained model with predict_proba method
    - X: Input features (DataFrame or array)
    - y: True labels
    
    Returns:
    - float: Optimal threshold that maximizes F1 score
    """
    y_proba = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]