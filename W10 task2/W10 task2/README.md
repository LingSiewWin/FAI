# Telco Customer Churn Prediction

## Overview
This notebook predicts customer churn using the Telco Customer Churn dataset. It handles class imbalance, evaluates models with business metrics, optimizes classification thresholds, and visualizes performance. The code is modular, production-ready, and includes explainability and testing.

## Requirements
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, shap, joblib
- Dataset: TelcoCustomerChurn.csv

## Usage
1. Ensure `utils.py` is in the same directory as this notebook.
2. Run cells sequentially to preprocess data, train models, and evaluate performance.
3. Models are saved in the `output` directory.
4. Use the `predict_churn` function to make predictions on new data.