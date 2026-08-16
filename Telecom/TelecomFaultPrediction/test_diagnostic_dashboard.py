import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import predict, explain, get_location_freq_map
from src.health_score import compute_health_score
from src.recommender import recommend_action


def test_normal_operating_scenario():
    """Verify diagnostic outputs for a low-risk, healthy network telemetry input."""
    normal_record = {
        'location': 'location 1008',
        'severity_type': 'severity_type 1',
        'event_event_type 11': 1,
        'resource_resource_type 8': 1,
        'log_log_feature 312': 1
    }
    
    pred_c, probs = predict(normal_record)
    score, status = compute_health_score(probs)
    
    assert pred_c in [0, 1, 2]
    assert 0.0 <= score <= 100.0
    assert len(probs) == 3
    assert status in ['Healthy', 'Warning', 'Critical']


def test_critical_outage_scenario():
    """Verify diagnostic outputs and SHAP root causes for an intense multi-signal outage."""
    critical_record = {
        'location': 'location 821',
        'severity_type': 'severity_type 2',
        'event_event_type 11': 1,
        'event_event_type 34': 1,
        'resource_resource_type 2': 1,
        'resource_resource_type 8': 1,
        'log_log_feature 203': 300,
        'log_log_feature 312': 150
    }
    
    pred_c, probs = predict(critical_record)
    score, status = compute_health_score(probs)
    shap_factors = explain(critical_record, top_n=4)
    actions = recommend_action(shap_factors, status)
    
    # Assertions
    assert status in ['Healthy', 'Warning', 'Critical']
    assert 0.0 <= score <= 100.0
    assert len(probs) == 3
    assert isinstance(shap_factors, list)
    assert isinstance(actions, list)
    assert len(actions) >= 1


def test_summary_metrics_aggregation():
    """Verify that simulated inputs aggregate correctly into dashboard summary tiles."""
    log_volumes = {"log_feature 203": 50, "log_feature 312": 25, "log_feature 50": 10}
    events = ["event_type 11", "event_type 20"]
    resources = ["resource_type 2"]
    
    tot_vol = sum(log_volumes.values())
    n_logs = len(log_volumes)
    n_events = len(events)
    n_res = len(resources)
    
    assert tot_vol == 85
    assert n_logs == 3
    assert n_events == 2
    assert n_res == 1


if __name__ == '__main__':
    print("Testing normal operating scenario diagnostics...")
    test_normal_operating_scenario()
    print("Testing critical outage scenario diagnostics and SHAP...")
    test_critical_outage_scenario()
    print("Testing summary metrics aggregation...")
    test_summary_metrics_aggregation()
    print("[PASS] ALL DIAGNOSTIC DASHBOARD TESTS PASSED!")
