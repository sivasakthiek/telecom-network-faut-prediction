def recommend_action(shap_factors: list, status: str) -> list:
    """
    Generates transparent, rule-based preventive actions based on top SHAP risk factors and network status.
    
    Rules (IF/THEN Logic):
    1. If status == "Healthy" -> "No action needed - network operating normally".
    2. If a "volume" or "log_feature" factor dominates -> "Investigate log/traffic volume spike, check for anomalous logging activity".
    3. If a "location" factor dominates -> "Inspect site-specific infrastructure at this location".
    4. If a "resource_type" or "event_type" factor dominates -> "Inspect the specific resource/event type flagged, check equipment status".
    5. If a "severity_type" factor dominates -> "Review reported severity classification, cross-check with recent incident reports".
    6. If status == "Critical" -> ALWAYS append "Schedule immediate maintenance intervention".
    
    Args:
        shap_factors (list): List of (feature_name, shap_value) tuples from explain().
        status (str): Network status label ("Healthy", "Warning", or "Critical").
        
    Returns:
        list: List of preventive action strings (max 3-4 items).
    """
    # Rule 1: If status == "Healthy", return reassuring default message
    if status == "Healthy":
        return ["No action needed - network operating normally"]
        
    actions = []
    
    # Process top positive SHAP factors (up to top 3)
    pos_factors = [f[0].lower() for f in shap_factors if f[1] > 0][:3]
    
    for feat in pos_factors:
        # Rule 2: Volume / log_feature factor
        if "volume" in feat or "log_feature" in feat or feat.startswith("log_"):
            action = "Investigate log/traffic volume spike, check for anomalous logging activity"
            if action not in actions:
                actions.append(action)
                
        # Rule 3: Location factor
        elif "location" in feat:
            action = "Inspect site-specific infrastructure at this location"
            if action not in actions:
                actions.append(action)
                
        # Rule 4: Resource / Event type factor
        elif "resource" in feat or "event" in feat:
            action = "Inspect the specific resource/event type flagged, check equipment status"
            if action not in actions:
                actions.append(action)
                
        # Rule 5: Severity type factor
        elif "severity" in feat:
            action = "Review reported severity classification, cross-check with recent incident reports"
            if action not in actions:
                actions.append(action)
                
    # Fallback if no specific rule matched despite Warning/Critical status
    if not actions and status != "Healthy":
        actions.append("Monitor network logs closely and perform standard diagnostic check")
        
    # Rule 6: If status is Critical, ALWAYS append immediate maintenance action
    if status == "Critical":
        critical_action = "Schedule immediate maintenance intervention"
        if critical_action not in actions:
            actions.append(critical_action)
            
    return actions[:4]
