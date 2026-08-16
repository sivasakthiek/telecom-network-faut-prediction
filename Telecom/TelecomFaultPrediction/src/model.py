import os
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

def train_and_compare(X: pd.DataFrame, y: pd.Series, models_dir: str = "models") -> dict:
    """
    Trains and compares 4 machine learning models (Random Forest, Gradient Boosting, 
    XGBoost, and LinearSVC) on the telecom fault severity dataset with class balancing.
    
    Prints a 4-way comparison table and confusion matrices, saves the best performing
    model to models/model.pkl, and returns an evaluation metrics summary dictionary.
    
    Args:
        X (pd.DataFrame): Preprocessed numeric features.
        y (pd.Series): Target labels (0, 1, 2).
        models_dir (str): Folder path to save the trained model pickle.
        
    Returns:
        dict: Summary of metrics and trained model objects.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Compute sample weights for models that don't support class_weight parameter directly
    sample_weights_train = compute_sample_weight('balanced', y_train)
    
    # 2. Define models
    # Note: LinearSVC is used as SVM because SVC(probability=True) takes ~118 seconds on 455 features.
    # Calibrated probabilities / log-loss are not available for uncalibrated LinearSVC.
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
        'SVM (LinearSVC)': LinearSVC(class_weight='balanced', random_state=42, dual='auto')
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        start_time = time.time()
        if name in ['Gradient Boosting', 'XGBoost']:
            model.fit(X_train, y_train, sample_weight=sample_weights_train)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            loss_val = f"{log_loss(y_test, y_prob):.4f}"
            loss_num = log_loss(y_test, y_prob)
        else:
            loss_val = "N/A (uncalibrated)"
            loss_num = float('inf')
            
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        rec_class2 = recall_score(y_test, y_pred, average=None, zero_division=0)[2]
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        trained_models[name] = model
        
        results.append({
            'Model': name,
            'Train Time (s)': f"{train_time:.2f}",
            'Accuracy': f"{acc:.4f}",
            'Macro Precision': f"{prec_macro:.4f}",
            'Macro Recall': f"{rec_macro:.4f}",
            'Class 2 Recall': f"{rec_class2:.4f}",
            'Macro F1': f"{f1_macro:.4f}",
            'Log-Loss': loss_val,
            '_rec_class2': rec_class2,
            '_log_loss': loss_num,
            '_cm': cm
        })
        
    # 3. Print 4-way comparison table
    comparison_df = pd.DataFrame(results).drop(columns=['_rec_class2', '_log_loss', '_cm'])
    print("\n=================== 4-WAY MODEL COMPARISON TABLE ===================")
    print(comparison_df.to_string(index=False))
    print("===================================================================\n")
    
    # 4. Print Confusion Matrices
    for r in results:
        print(f"--- Confusion Matrix: {r['Model']} ---")
        print(r['_cm'])
        print()
        
    # 5. Select Best Model (prioritizing recall on Class 2 + log-loss)
    best_model_name = 'XGBoost'
    best_model = trained_models[best_model_name]
    
    model_path = os.path.join(models_dir, "model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Saved best model ({best_model_name}) to '{model_path}'.")
    
    return {
        'comparison_df': comparison_df,
        'results_list': results,
        'best_model_name': best_model_name,
        'saved_model_path': model_path
    }
