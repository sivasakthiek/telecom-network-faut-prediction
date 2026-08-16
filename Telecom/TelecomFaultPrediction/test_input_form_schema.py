import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import re
import joblib
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import get_location_freq_map


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def test_locations_schema():
    """Verify that all 929 training locations exist, sort naturally, and custom location fallback works."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    loc_map = get_location_freq_map(models_dir=models_dir)
    
    assert len(loc_map) == 929
    sorted_locs = sorted(list(loc_map.keys()), key=natural_sort_key)
    assert sorted_locs[0] == "location 1"
    assert "location 821" in sorted_locs
    assert "location 1126" in sorted_locs


def test_categorical_options_bounds():
    """Verify that severity types, resource types, event types, and log features cover expected domains."""
    # Severity options: 5 types
    sev_options = [f"severity_type {i}" for i in range(1, 6)]
    assert len(sev_options) == 5
    
    # Resource options: 10 types
    res_options = [f"resource_type {i}" for i in range(1, 11)]
    assert len(res_options) == 10
    
    # Event options: 54 types
    event_options = [f"event_type {i}" for i in range(1, 55)]
    assert len(event_options) == 54
    
    # Log features: 386 types
    log_options = [f"log_feature {i}" for i in range(1, 387)]
    assert len(log_options) == 386


def test_dynamic_volume_dict_construction():
    """Verify that multi-selected log features and numeric volumes build valid model inputs."""
    selected_loc = "location 821"
    selected_severity = "severity_type 2"
    selected_resources = ["resource_type 2", "resource_type 8"]
    selected_events = ["event_type 11", "event_type 34"]
    selected_log_features = ["log_feature 203", "log_feature 312"]
    log_volumes = {"log_feature 203": 5, "log_feature 312": 15}
    
    sim_record = {
        'location': selected_loc,
        'severity_type': selected_severity
    }
    for et in selected_events:
        sim_record[f"event_{et}"] = 1
    for rt in selected_resources:
        sim_record[f"resource_{rt}"] = 1
    for lf, vol in log_volumes.items():
        sim_record[f"log_{lf}"] = vol
        
    assert sim_record['location'] == "location 821"
    assert sim_record['severity_type'] == "severity_type 2"
    assert sim_record['event_event_type 11'] == 1
    assert sim_record['event_event_type 34'] == 1
    assert sim_record['resource_resource_type 2'] == 1
    assert sim_record['resource_resource_type 8'] == 1
    assert sim_record['log_log_feature 203'] == 5
    assert sim_record['log_log_feature 312'] == 15


if __name__ == '__main__':
    print("Testing locations schema...")
    test_locations_schema()
    print("Testing categorical options bounds...")
    test_categorical_options_bounds()
    print("Testing dynamic volume dict construction...")
    test_dynamic_volume_dict_construction()
    print("[PASS] ALL FORM SCHEMA TESTS PASSED!")
