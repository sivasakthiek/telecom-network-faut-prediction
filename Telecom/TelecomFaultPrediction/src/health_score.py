import numpy as np

def compute_health_score(probabilities: list) -> tuple:
    """
    Computes a network infrastructure health score (0 to 100) based on predicted fault probabilities.
    
    Formula:
        Health Score = 100 - (P(Class 1)*40 + P(Class 2)*100), clipped to [0, 100].
        
    Status Levels:
        - "Healthy": Score >= 70
        - "Warning": 40 <= Score < 70
        - "Critical": Score < 40
        
    Args:
        probabilities (list): [P(Class 0), P(Class 1), P(Class 2)].
        
    Returns:
        tuple: (health_score: float, status: str)
            - health_score (float): Numeric score from 0.0 to 100.0.
            - status (str): Status label ("Healthy", "Warning", or "Critical").
    """
    p0, p1, p2 = probabilities[0], probabilities[1], probabilities[2]
    
    penalty = (p1 * 40.0) + (p2 * 100.0)
    score = float(np.clip(100.0 - penalty, 0.0, 100.0))
    
    if score >= 70.0:
        status = "Healthy"
    elif score >= 40.0:
        status = "Warning"
    else:
        status = "Critical"
        
    return round(score, 1), status

def format_explanation(shap_factors: list, health_score: float) -> str:
    """
    Formats SHAP root cause factors into a clear, plain-English explanation string for non-technical engineers.
    
    Args:
        shap_factors (list): List of (feature_name, shap_value) tuples from explain().
        health_score (float): Computed health score (0 to 100).
        
    Returns:
        str: Plain-English explanation summary.
    """
    if not shap_factors:
        return "Network operating under normal conditions."
        
    # Take top 2-3 factors with positive SHAP contribution
    pos_factors = [f for f in shap_factors if f[1] > 0][:3]
    
    if not pos_factors:
        return "Network operating under normal conditions with minimal risk factors."
        
    factor_strs = [f"{feat} (+{val:.2f})" for feat, val in pos_factors]
    
    if len(factor_strs) == 1:
        reasons_text = factor_strs[0]
    elif len(factor_strs) == 2:
        reasons_text = f"{factor_strs[0]} and {factor_strs[1]}"
    else:
        reasons_text = f"{', '.join(factor_strs[:-1])}, and {factor_strs[-1]}"
        
    if health_score >= 70:
        return f"Network health is stable. Minor risk indicators: {reasons_text}."
    elif health_score >= 40:
        return f"Score lowered mainly due to {reasons_text}."
    else:
        return f"Score lowered mainly due to {reasons_text}."
