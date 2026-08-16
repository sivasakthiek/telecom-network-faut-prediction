import os
import gc
import pandas as pd

def load_and_merge(data_dir: str) -> pd.DataFrame:
    """
    Loads Telstra Network Disruptions dataset CSVs from data_dir, pivots auxiliary 
    tables (event_type, log_feature, resource_type) into a wide one-row-per-id format, 
    and merges them with train.csv on 'id'.
    
    Args:
        data_dir (str): Directory containing the CSV files.
        
    Returns:
        pd.DataFrame: Merged DataFrame with one row per 'id'.
    """
    # Verify file existence first
    train_path = os.path.join(data_dir, "train.csv")
    event_path = os.path.join(data_dir, "event_type.csv")
    log_path = os.path.join(data_dir, "log_feature.csv")
    resource_path = os.path.join(data_dir, "resource_type.csv")
    severity_path = os.path.join(data_dir, "severity_type.csv")
    
    for path in [train_path, event_path, log_path, resource_path, severity_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load train and severity first and merge
    train = pd.read_csv(train_path)
    severity_type = pd.read_csv(severity_path)
    merged_df = train.merge(severity_type, on='id', how='left')
    
    # Free up memory
    del train
    del severity_type
    gc.collect()
    
    # Load and pivot event_type
    event_type = pd.read_csv(event_path)
    event_pivot = event_type.pivot_table(
        index='id', 
        columns='event_type', 
        aggfunc='size', 
        fill_value=0
    ).reset_index()
    
    # Add prefix to make columns distinct
    event_cols = [col if col == 'id' else f"event_{col}" for col in event_pivot.columns]
    event_pivot.columns = event_cols
    
    # Downcast numeric columns
    for col in event_pivot.columns:
        if col != 'id':
            event_pivot[col] = event_pivot[col].astype('int32')
            
    merged_df = merged_df.merge(event_pivot, on='id', how='left')
    del event_type
    del event_pivot
    gc.collect()
    
    # Load and pivot log_feature
    log_feature = pd.read_csv(log_path)
    log_pivot = log_feature.pivot_table(
        index='id', 
        columns='log_feature', 
        values='volume', 
        aggfunc='sum', 
        fill_value=0
    ).reset_index()
    
    log_cols = [col if col == 'id' else f"log_{col}" for col in log_pivot.columns]
    log_pivot.columns = log_cols
    
    # Downcast numeric columns
    for col in log_pivot.columns:
        if col != 'id':
            log_pivot[col] = log_pivot[col].astype('int32')
            
    merged_df = merged_df.merge(log_pivot, on='id', how='left')
    del log_feature
    del log_pivot
    gc.collect()
    
    # Load and pivot resource_type
    resource_type = pd.read_csv(resource_path)
    resource_pivot = resource_type.pivot_table(
        index='id', 
        columns='resource_type', 
        aggfunc='size', 
        fill_value=0
    ).reset_index()
    
    resource_cols = [col if col == 'id' else f"resource_{col}" for col in resource_pivot.columns]
    resource_pivot.columns = resource_cols
    
    # Downcast numeric columns
    for col in resource_pivot.columns:
        if col != 'id':
            resource_pivot[col] = resource_pivot[col].astype('int32')
            
    merged_df = merged_df.merge(resource_pivot, on='id', how='left')
    del resource_type
    del resource_pivot
    gc.collect()
    
    # Fill NaN values for pivoted features with 0
    feature_cols = [col for col in merged_df.columns if col not in ['id', 'location', 'fault_severity', 'severity_type']]
    merged_df[feature_cols] = merged_df[feature_cols].fillna(0)
    
    return merged_df


def clean_data(df: pd.DataFrame, location_freq_map: dict = None):
    """
    Cleans the merged DataFrame for machine learning:
    1. Frequency-encodes 'location' (uses location_freq_map if provided, else computes from df).
    2. Converts 'severity_type' to numeric format (extracting integer ID).
    3. Drops 'id' column from features set.
    4. Separates target 'fault_severity' from feature set.
    5. Confirms all feature columns are numeric.
    
    Args:
        df (pd.DataFrame): Merged DataFrame from load_and_merge.
        location_freq_map (dict, optional): Pre-fitted mapping of location frequencies. Defaults to None.
        
    Returns:
        tuple: (X_features, y_target, location_freq_map)
            - X_features (pd.DataFrame): Clean numeric features DataFrame.
            - y_target (pd.Series or None): Target fault_severity series if present in input df.
            - location_freq_map (dict): Fitted frequency mapping for location.
    """
    df_clean = df.copy()
    
    # 1. Target separation (if fault_severity column is present)
    y_target = None
    if 'fault_severity' in df_clean.columns:
        y_target = df_clean['fault_severity'].copy()
        df_clean = df_clean.drop(columns=['fault_severity'])
        
    # 2. Drop raw 'id' column from features
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['id'])
        
    # 3. Frequency-encode location
    if 'location' in df_clean.columns:
        if location_freq_map is None:
            location_freq_map = df_clean['location'].value_counts().to_dict()
        df_clean['location'] = df_clean['location'].map(location_freq_map).fillna(0).astype(int)
        
    # 4. Convert severity_type to numeric if present
    if 'severity_type' in df_clean.columns:
        if df_clean['severity_type'].dtype == 'object' or str(df_clean['severity_type'].dtype).startswith('str'):
            df_clean['severity_type'] = (
                df_clean['severity_type']
                .astype(str)
                .str.extract(r'(\d+)')[0]
                .astype(float)
                .fillna(0)
                .astype(int)
            )
            
    # 5. Confirm all remaining feature columns are numeric
    non_numeric_cols = df_clean.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"Leftover non-numeric columns found: {non_numeric_cols}")
        
    return df_clean, y_target, location_freq_map
