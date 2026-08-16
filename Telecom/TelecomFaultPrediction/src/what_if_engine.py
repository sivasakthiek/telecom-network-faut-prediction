import os
import re
import pandas as pd
import numpy as np

class WhatIfEngine:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(self.project_root, "models", "model.pkl")
        self._loc_map = None

    def get_loc_map(self):
        if self._loc_map is None:
            from src.predictor import get_location_freq_map
            self._loc_map = get_location_freq_map(models_dir=os.path.join(self.project_root, "models"))
        return self._loc_map

    def _load_raw_record_from_csv(self, device_id: int) -> dict:
        """Lightweight database search to reconstruct raw features for a single device ID."""
        data_dir = os.path.join(self.project_root, "data", "raw")
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")
        
        loc = None
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            row = train_df[train_df['id'] == device_id]
            if not row.empty:
                loc = row.iloc[0]['location']
        if loc is None and os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            row = test_df[test_df['id'] == device_id]
            if not row.empty:
                loc = row.iloc[0]['location']
                
        if loc is None:
            return None
            
        record = {'location': loc}
        
        # Merge severity_type
        sev_path = os.path.join(data_dir, "severity_type.csv")
        if os.path.exists(sev_path):
            sev_df = pd.read_csv(sev_path)
            sev_row = sev_df[sev_df['id'] == device_id]
            if not sev_row.empty:
                record['severity_type'] = sev_row.iloc[0]['severity_type']
                
        # Merge event_type
        evt_path = os.path.join(data_dir, "event_type.csv")
        if os.path.exists(evt_path):
            evt_df = pd.read_csv(evt_path)
            evt_rows = evt_df[evt_df['id'] == device_id]
            for _, r in evt_rows.iterrows():
                record[f"event_{r['event_type']}"] = 1
                
        # Merge resource_type
        res_path = os.path.join(data_dir, "resource_type.csv")
        if os.path.exists(res_path):
            res_df = pd.read_csv(res_path)
            res_rows = res_df[res_df['id'] == device_id]
            for _, r in res_rows.iterrows():
                record[f"resource_{r['resource_type']}"] = 1
                
        # Merge log_feature
        log_path = os.path.join(data_dir, "log_feature.csv")
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            log_rows = log_df[log_df['id'] == device_id]
            for _, r in log_rows.iterrows():
                record[r['log_feature']] = int(r['volume'])
                
        return record

    def analyze(self, device_id: int, df_results: pd.DataFrame = None, raw_records: dict = None) -> dict:
        """
        Executes a deterministic, rule-based what-if scenario impact assessment for the given device ID.
        """
        # 1. Retrieve row stats and raw record
        row_stats = None
        raw_rec = None
        
        if df_results is not None:
            dev_df = df_results[df_results['id'] == device_id]
            if not dev_df.empty:
                row_stats = dev_df.iloc[0].to_dict()
                if raw_records and device_id in raw_records:
                    raw_rec = raw_records[device_id]
                    
        # Fallback to dynamic loading if not loaded/found
        if raw_rec is None:
            raw_rec = self._load_raw_record_from_csv(device_id)
            if raw_rec is None:
                return None
                
        if row_stats is None:
            from src.predictor import predict
            from src.health_score import compute_health_score
            
            pred_class, probs = predict(raw_rec, model_path=self.model_path, location_freq_map=self.get_loc_map())
            health_score, health_status = compute_health_score(probs)
            
            row_stats = {
                'id': device_id,
                'location': raw_rec.get('location', 'Unknown'),
                'severity_type': raw_rec.get('severity_type', 'Unknown'),
                'predicted_class': pred_class,
                'p0': probs[0],
                'p1': probs[1],
                'p2': probs[2],
                'health_score': health_score,
                'status': health_status
            }

        # 2. Get SHAP feature contributions using the predictor module
        from src.predictor import explain
        shap_factors = explain(raw_rec, model_path=self.model_path, location_freq_map=self.get_loc_map())
        
        # Format SHAP features as list of explanations
        shap_explanations = []
        for name, val in shap_factors:
            if val > 0:
                shap_explanations.append(f"{name} (impact weight: +{val:.4f})")
        shap_text = ", ".join(shap_explanations) if shap_explanations else "None detected"

        # 3. Extract telemetry features for description
        active_resources = [k.replace("resource_", "") for k in raw_rec.keys() if k.startswith("resource_")]
        active_events = [k.replace("event_", "") for k in raw_rec.keys() if k.startswith("event_")]
        
        resources_str = ", ".join(active_resources) if active_resources else "none"
        events_str = ", ".join(active_events) if active_events else "none"

        pred_class = int(row_stats['predicted_class'])
        health_score = float(row_stats['health_score'])
        status = row_stats['status']
        loc = row_stats['location']
        
        # 4. Deterministic scenario building using uncertainty-aware wording
        if pred_class == 2:
            current_condition = f"Active major fault disruption (Severity Class 2). Device health score is currently degraded to {health_score:.1f}%."
            risk_level = "HIGH"
            possible_progression = (
                f"If the fault continues to remain unresolved, the underlying hardware triggers may escalate, "
                f"potentially causing cascading node failures, severe network throughput drops, or a complete operational outage at {loc}."
            )
            potential_impact = (
                f"Total service loss could occur at {loc}. Active log trigger events (e.g., {events_str}) indicate severe telemetry volume surge. "
                f"Affected user counts are not recorded in the telemetry dataset. Resource classifications {resources_str} suggest backhaul systems may be impacted."
            )
            recommended_priority = "IMMEDIATE (Priority 1) - Dispatch field technician immediately to replace or repair resource hardware."
            
            confidence_val = float(row_stats['p2']) * 100
            confidence_level = (
                f"High confidence (Model major fault class probability is {confidence_val:.1f}%). "
                f"Top contributing features driving this prediction include: {shap_text}."
            )
        elif pred_class == 1:
            current_condition = f"Active minor fault disruption (Severity Class 1). Device health score is warning status at {health_score:.1f}%."
            risk_level = "MEDIUM"
            possible_progression = (
                f"If left unresolved, this warning condition could escalate to a Class 2 Major Fault if telemetry volumes "
                f"on log features increase. Intermittent packet drops or localized latency might persist."
            )
            potential_impact = (
                f"Performance degradation and latency issues may impact customer connections at {loc}. "
                f"Affected user counts are not recorded in the telemetry dataset. Telemetry points to active events {events_str}."
            )
            recommended_priority = "MEDIUM (Priority 2) - Schedule remote diagnostics review and optimize log daemon settings."
            
            confidence_val = float(row_stats['p1']) * 100
            confidence_level = (
                f"Medium confidence (Model minor fault class probability is {confidence_val:.1f}%). "
                f"Top contributing features driving this prediction include: {shap_text}."
            )
        else:
            current_condition = f"Operating normally (Severity Class 0). Device health score is stable at {health_score:.1f}%."
            risk_level = "LOW"
            possible_progression = (
                f"System is performing within normal bounds. Progression to fault state is unlikely unless "
                f"an anomalous spike in log features or resource signals occurs."
            )
            potential_impact = (
                f"No customer or service impact expected at {loc}. System remains fully operational."
            )
            recommended_priority = "LOW (Priority 3) - Continue routine telemetry monitoring."
            
            confidence_val = float(row_stats['p0']) * 100
            confidence_level = (
                f"High confidence (Model normal class probability is {confidence_val:.1f}%). "
                f"Top contributing features driving this prediction include: {shap_text}."
            )

        return {
            "device_id": device_id,
            "current_condition": current_condition,
            "risk_level": risk_level,
            "possible_progression": possible_progression,
            "potential_impact": potential_impact,
            "recommended_priority": recommended_priority,
            "confidence_level": confidence_level,
            "time_to_impact": "Not available from current data (time-to-failure metrics are not recorded in the telemetry log files)."
        }
