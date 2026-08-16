import os
import gc
import pickle
import pandas as pd
import numpy as np
import joblib
import faiss
from sentence_transformers import SentenceTransformer

from src.preprocessing import clean_data
from src.features import build_features
from src.health_score import compute_health_score

def load_and_merge_all(data_dir: str) -> pd.DataFrame:
    """Loads and merges train + test records with auxiliary tables in wide format."""
    print("Loading and merging all datasets...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Combine train and test (test has no fault_severity, fill with NaN)
    all_devs = pd.concat([train, test], ignore_index=True)
    
    severity_type = pd.read_csv(os.path.join(data_dir, "severity_type.csv"))
    merged_df = all_devs.merge(severity_type, on='id', how='left')
    
    # Pivot and merge event_type
    event_type = pd.read_csv(os.path.join(data_dir, "event_type.csv"))
    event_pivot = event_type.pivot_table(index='id', columns='event_type', aggfunc='size', fill_value=0).reset_index()
    event_cols = [col if col == 'id' else f"event_{col}" for col in event_pivot.columns]
    event_pivot.columns = event_cols
    for col in event_pivot.columns:
        if col != 'id':
            event_pivot[col] = event_pivot[col].astype('int32')
    merged_df = merged_df.merge(event_pivot, on='id', how='left')
    
    # Pivot and merge log_feature
    log_feature = pd.read_csv(os.path.join(data_dir, "log_feature.csv"))
    log_pivot = log_feature.pivot_table(index='id', columns='log_feature', values='volume', aggfunc='sum', fill_value=0).reset_index()
    log_cols = [col if col == 'id' else f"log_{col}" for col in log_pivot.columns]
    log_pivot.columns = log_cols
    for col in log_pivot.columns:
        if col != 'id':
            log_pivot[col] = log_pivot[col].astype('int32')
    merged_df = merged_df.merge(log_pivot, on='id', how='left')
    
    # Pivot and merge resource_type
    resource_type = pd.read_csv(os.path.join(data_dir, "resource_type.csv"))
    resource_pivot = resource_type.pivot_table(index='id', columns='resource_type', aggfunc='size', fill_value=0).reset_index()
    resource_cols = [col if col == 'id' else f"resource_{col}" for col in resource_pivot.columns]
    resource_pivot.columns = resource_cols
    for col in resource_pivot.columns:
        if col != 'id':
            resource_pivot[col] = resource_pivot[col].astype('int32')
    merged_df = merged_df.merge(resource_pivot, on='id', how='left')
    
    # Fill NaN values for features with 0
    feature_cols = [col for col in merged_df.columns if col not in ['id', 'location', 'fault_severity', 'severity_type']]
    merged_df[feature_cols] = merged_df[feature_cols].fillna(0)
    
    return merged_df

def build_vector_db():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "raw")
    models_dir = os.path.join(project_root, "models")
    model_path = os.path.join(models_dir, "model.pkl")
    loc_map_path = os.path.join(models_dir, "location_freq_map.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please run or build the app first.")
    
    # 1. Load model and compute health scores for all records
    model = joblib.load(model_path)
    loc_map = joblib.load(loc_map_path)
    
    raw_merged = load_and_merge_all(data_dir)
    
    X_clean, _, _ = clean_data(raw_merged, location_freq_map=loc_map)
    X_feat = build_features(X_clean)
    
    # Align features with model expected inputs
    expected_cols = list(model.feature_names_in_)
    for col in expected_cols:
        if col not in X_feat.columns:
            X_feat[col] = 0
    X_feat = X_feat[expected_cols]
    
    probs = model.predict_proba(X_feat)
    
    health_scores = []
    health_statuses = []
    for p in probs:
        score, status = compute_health_score(p)
        health_scores.append(score)
        health_statuses.append(status)
        
    raw_merged['health_score'] = health_scores
    raw_merged['health_status'] = health_statuses
    
    # 2. Load raw files for detailed aggregates
    print("Aggregating statistics per location...")
    severity_df = pd.read_csv(os.path.join(data_dir, "severity_type.csv"))
    event_df = pd.read_csv(os.path.join(data_dir, "event_type.csv"))
    resource_df = pd.read_csv(os.path.join(data_dir, "resource_type.csv"))
    log_df = pd.read_csv(os.path.join(data_dir, "log_feature.csv"))
    
    # Map raw files by id for quick retrieval
    sev_map = severity_df.groupby('id')['severity_type'].apply(list).to_dict()
    res_map = resource_df.groupby('id')['resource_type'].apply(list).to_dict()
    evt_map = event_df.groupby('id')['event_type'].apply(list).to_dict()
    
    # Map log features
    log_grp = log_df.groupby('id')
    log_map = {}
    for id_val, group in log_grp:
        log_map[id_val] = list(zip(group['log_feature'], group['volume']))
        
    unique_locations = sorted(raw_merged['location'].unique())
    location_summaries = []
    
    for loc in unique_locations:
        loc_rows = raw_merged[raw_merged['location'] == loc]
        ids = loc_rows['id'].tolist()
        
        # Accumulate severities, resources, event types, logs
        severities = set()
        resources = set()
        events = set()
        logs_volume = {}
        
        for id_val in ids:
            if id_val in sev_map:
                severities.update(sev_map[id_val])
            if id_val in res_map:
                resources.update(res_map[id_val])
            if id_val in evt_map:
                events.update(evt_map[id_val])
            if id_val in log_map:
                for feat, vol in log_map[id_val]:
                    logs_volume[feat] = logs_volume.get(feat, 0) + vol
                    
        # Calculate avg health score
        avg_health = float(np.mean(loc_rows['health_score']))
        
        # Calculate status
        if avg_health >= 70.0:
            status = "Healthy"
        elif avg_health >= 40.0:
            status = "Warning"
        else:
            status = "Critical"
            
        # Format strings
        severities_str = ", ".join(sorted(list(severities))) if severities else "None"
        resources_str = ", ".join(sorted(list(resources))) if resources else "None"
        events_str = ", ".join(sorted(list(events))) if events else "None"
        
        # Sort logs by volume descending
        sorted_logs = sorted(logs_volume.items(), key=lambda x: x[1], reverse=True)
        logs_str = ", ".join([f"{feat} (volume: {vol})" for feat, vol in sorted_logs]) if sorted_logs else "None"
        
        summary = (
            f"Location: {loc}\n"
            f"Severity: {severities_str}\n"
            f"Resource Types: {resources_str}\n"
            f"Event Types: {events_str}\n"
            f"Log Features: {logs_str}\n"
            f"Health Score: {avg_health:.1f} ({status})"
        )
        
        location_summaries.append({
            'location': loc,
            'summary': summary
        })
        
    print(f"Generated {len(location_summaries)} location summaries.")
    
    # 2.5 Generate device-level summaries
    device_summaries_dict = {}
    device_summaries_list = []
    
    print("Generating device-level summaries for all records...")
    for idx, row in raw_merged.iterrows():
        dev_id = int(row['id'])
        loc = row['location']
        score = float(row['health_score'])
        status = row['health_status']
        
        # Determine fault severity label
        f_sev = int(row['fault_severity']) if not pd.isna(row['fault_severity']) else 0
        f_sev_label = "No Fault" if f_sev == 0 else ("Minor Fault" if f_sev == 1 else "Major Fault")
        
        # Accumulate items from maps
        severities = sev_map.get(dev_id, [])
        resources = res_map.get(dev_id, [])
        events = evt_map.get(dev_id, [])
        logs = log_map.get(dev_id, [])
        
        severities_str = ", ".join(sorted(severities)) if severities else "None"
        resources_str = ", ".join(sorted(resources)) if resources else "None"
        events_str = ", ".join(sorted(events)) if events else "None"
        
        # Sort logs by volume descending
        sorted_logs = sorted(logs, key=lambda x: x[1], reverse=True)
        logs_str = ", ".join([f"{feat} (volume: {vol})" for feat, vol in sorted_logs]) if sorted_logs else "None"
        
        summary = (
            f"Device ID: {dev_id}\n"
            f"Location: {loc}\n"
            f"Reported Severity: {severities_str}\n"
            f"Resource Type: {resources_str}\n"
            f"Event Types: {events_str}\n"
            f"Log Features: {logs_str}\n"
            f"Fault Severity: {f_sev} ({f_sev_label})\n"
            f"Health Score: {score:.1f} ({status})"
        )
        
        device_summaries_dict[dev_id] = {
            'location': loc,
            'summary': summary
        }
        device_summaries_list.append({
            'device_id': dev_id,
            'summary': summary
        })
        
    print(f"Generated {len(device_summaries_list)} device summaries.")
    
    # 3. Create Embeddings & Save to FAISS
    print("Initializing SentenceTransformer model (all-MiniLM-L6-v2)...")
    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Encode Location Summaries
    texts = [item['summary'] for item in location_summaries]
    print("Computing location embeddings...")
    embeddings = model_st.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    print(f"Location embedding dimension: {dimension}")
    
    # Setup L2 Flat index for locations
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save location index and summaries
    faiss_path = os.path.join(models_dir, "faiss_index.index")
    summaries_path = os.path.join(models_dir, "location_summaries.pkl")
    
    print(f"Saving location FAISS index to {faiss_path}...")
    faiss.write_index(index, faiss_path)
    
    print(f"Saving location summaries mapping to {summaries_path}...")
    with open(summaries_path, "wb") as f:
        pickle.dump(location_summaries, f)
        
    # Encode Device Summaries
    device_texts = [item['summary'] for item in device_summaries_list]
    print("Computing device embeddings...")
    device_embeddings = model_st.encode(device_texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Device embedding dimension: {device_embeddings.shape[1]}")
    
    # Setup L2 Flat index for devices
    device_index = faiss.IndexFlatL2(device_embeddings.shape[1])
    device_index.add(device_embeddings)
    
    # Save device index and summaries
    faiss_devices_path = os.path.join(models_dir, "faiss_devices.index")
    device_summaries_path = os.path.join(models_dir, "device_summaries.pkl")
    
    print(f"Saving device FAISS index to {faiss_devices_path}...")
    faiss.write_index(device_index, faiss_devices_path)
    
    print(f"Saving device summaries mapping to {device_summaries_path}...")
    with open(device_summaries_path, "wb") as f:
        pickle.dump({
            'mapping': device_summaries_dict,
            'list': device_summaries_list
        }, f)
        
    print("Vector Database build complete!")
    
if __name__ == "__main__":
    build_vector_db()
