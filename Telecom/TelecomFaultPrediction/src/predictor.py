import os
import joblib
import pandas as pd
import numpy as np
import shap

from src.preprocessing import clean_data
from src.features import build_features

def get_location_freq_map(models_dir: str = "models", data_dir: str = "data/raw") -> dict:
    """
    Retrieves or fits the location frequency mapping dictionary.
    
    Args:
        models_dir (str): Path to models folder.
        data_dir (str): Path to raw data folder.
        
    Returns:
        dict: Location frequency map.
    """
    map_path = os.path.join(models_dir, "location_freq_map.pkl")
    if os.path.exists(map_path):
        return joblib.load(map_path)
    
    # Fallback: compute from raw data and persist map
    from src.preprocessing import load_and_merge
    raw_df = load_and_merge(data_dir)
    _, _, loc_map = clean_data(raw_df)
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(loc_map, map_path)
    return loc_map

_MODEL_CACHE = {}
_EXPLAINER_CACHE = {}

def load_model(model_path: str = "models/model.pkl"):
    """
    Loads and caches trained model in memory.
    """
    if model_path not in _MODEL_CACHE:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train the model first.")
        _MODEL_CACHE[model_path] = joblib.load(model_path)
    return _MODEL_CACHE[model_path]

def _preprocess_record(record_dict: dict, model_path: str = "models/model.pkl", location_freq_map: dict = None):
    """
    Internal helper to preprocess a single record dict into an aligned DataFrame ready for model input.
    """
    model = load_model(model_path)
    
    if location_freq_map is None:
        models_dir = os.path.dirname(model_path) or "models"
        location_freq_map = get_location_freq_map(models_dir=models_dir)
        
    # 1. Normalize record keys to match training column schema
    normalized_record = {}
    for k, v in record_dict.items():
        if k.startswith("log_log_feature "):
            normalized_record[k.replace("log_log_feature ", "log_feature ")] = v
        elif k.startswith("feature "):
            normalized_record[f"log_feature {k.split(' ')[1]}"] = v
        elif k.startswith("event_type "):
            normalized_record[f"event_event_type {k.split(' ')[1]}"] = v
        elif k.startswith("resource_type "):
            normalized_record[f"resource_resource_type {k.split(' ')[1]}"] = v
        else:
            normalized_record[k] = v
            
    # 2. Convert record dict to single-row DataFrame
    df_raw = pd.DataFrame([normalized_record])
    
    # 3. Reuse clean_data() and build_features()
    X_clean, _, _ = clean_data(df_raw, location_freq_map=location_freq_map)
    X_feat = build_features(X_clean)
    
    # 4. Align feature columns with expected model features
    if hasattr(model, 'feature_names_in_'):
        expected_cols = list(model.feature_names_in_)
        X_feat = X_feat.reindex(columns=expected_cols, fill_value=0)
        
    return model, X_feat, location_freq_map

def predict(record_dict: dict, model_path: str = "models/model.pkl", location_freq_map: dict = None) -> tuple:
    """
    Predicts telecom fault severity and class probabilities for a single raw input record.
    Reuses clean_data() and build_features() from preprocessing and feature engineering.
    
    Args:
        record_dict (dict): Raw input record (dictionary of feature keys and values).
        model_path (str): Path to trained model pickle.
        location_freq_map (dict, optional): Pre-computed location frequency map.
        
    Returns:
        tuple: (predicted_fault_severity, class_probabilities)
            - predicted_fault_severity (int): Predicted severity class (0, 1, or 2).
            - class_probabilities (list): [P(Class 0), P(Class 1), P(Class 2)].
    """
    model, X_feat, _ = _preprocess_record(record_dict, model_path=model_path, location_freq_map=location_freq_map)
    
    probs = model.predict_proba(X_feat)[0].tolist()
    pred_class = int(np.argmax(probs))
    
    return pred_class, probs

def explain(record_dict: dict, model_path: str = "models/model.pkl", location_freq_map: dict = None, top_n: int = 4) -> list:
    """
    Computes SHAP feature importance values for a single telecom disruption record to identify 
    probable root cause factors driving the predicted fault severity.
    
    Reuses the exact same preprocessing and feature engineering pipeline as predict().
    
    Args:
        record_dict (dict): Raw input record.
        model_path (str): Path to trained model pickle.
        location_freq_map (dict, optional): Pre-computed location frequency map.
        top_n (int): Number of top root cause factors to return. Defaults to 4.
        
    Returns:
        list: List of (feature_name, shap_value) tuples for top probable root cause factors, 
              sorted descending by positive contribution to the predicted fault class.
    """
    model, X_feat, _ = _preprocess_record(record_dict, model_path=model_path, location_freq_map=location_freq_map)
    
    # Predict target class
    pred_class = int(model.predict(X_feat)[0])
    
    # Compute SHAP values via cached TreeExplainer
    if model_path not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[model_path] = shap.TreeExplainer(model)
    explainer = _EXPLAINER_CACHE[model_path]
    shap_vals = explainer.shap_values(X_feat) # Shape: (1, num_features, num_classes)
    
    # Extract SHAP values for the predicted class
    if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        class_shap = shap_vals[0, :, pred_class]
    elif isinstance(shap_vals, list):
        class_shap = shap_vals[pred_class][0]
    else:
        class_shap = shap_vals[0]
        
    feature_names = list(X_feat.columns)
    feat_shap_pairs = [(feat, float(s_val)) for feat, s_val in zip(feature_names, class_shap)]
    
    # Sort descending by positive SHAP value (factors pushing risk UP most)
    top_factors = sorted(feat_shap_pairs, key=lambda x: x[1], reverse=True)[:top_n]
    
    return top_factors
