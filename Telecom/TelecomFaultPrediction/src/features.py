import pandas as pd

def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs high-value summary features on top of existing engineered features:
    - total_volume: sum of all log_feature volume columns per row
      (Justification: Captures overall magnitude of service logging activity per network incident).
    - num_active_log_features: count of non-zero log_feature columns per row
      (Justification: Measures diversity of logging signals active during a disruption).
    - num_event_types: count of non-zero event_type columns per row
      (Justification: Reflects the multiplicity of distinct event triggers associated with the incident).
    - num_resource_types: count of non-zero resource_type columns per row
      (Justification: Quantifies the breadth of infrastructure resource types affected).

    Args:
        X (pd.DataFrame): Cleaned features DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the 4 new summary features appended.
    """
    X_feat = X.copy()

    # Identify feature column subsets by prefix
    log_cols = [c for c in X_feat.columns if c.startswith("log_")]
    event_cols = [c for c in X_feat.columns if c.startswith("event_")]
    resource_cols = [c for c in X_feat.columns if c.startswith("resource_")]

    # 1. total_volume: sum of all log_feature volume columns per row
    # Justification: Captures overall magnitude of service logging activity per network incident.
    X_feat['total_volume'] = X_feat[log_cols].sum(axis=1)

    # 2. num_active_log_features: count of non-zero log_feature columns per row
    # Justification: Measures diversity of logging signals active during a disruption.
    X_feat['num_active_log_features'] = (X_feat[log_cols] > 0).sum(axis=1)

    # 3. num_event_types: count of non-zero event_type columns per row
    # Justification: Reflects the multiplicity of distinct event triggers associated with the incident.
    X_feat['num_event_types'] = (X_feat[event_cols] > 0).sum(axis=1)

    # 4. num_resource_types: count of non-zero resource_type columns per row
    # Justification: Quantifies the breadth of infrastructure resource types affected.
    X_feat['num_resource_types'] = (X_feat[resource_cols] > 0).sum(axis=1)

    return X_feat
