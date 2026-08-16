import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import time
import warnings
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import predict, explain, get_location_freq_map, _preprocess_record
from src.health_score import compute_health_score
from src.recommender import recommend_action


def test_custom_record_preprocessing_and_vectorization():
    """Verify that arbitrary raw incident records align cleanly to model feature space without fragmentation warnings."""
    custom_record = {
        'location': 'location 821',
        'severity_type': 'severity_type 2',
        'event_event_type 11': 1,
        'resource_resource_type 2': 1,
        'log_log_feature 203': 5,
        'log_log_feature 312': 12
    }
    
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        model, X_feat, loc_map = _preprocess_record(custom_record)
        
        # Check no performance/fragmentation warnings
        perf_warnings = [w for w in recorded_warnings if issubclass(w.category, (pd_errors := getattr(warnings, 'PerformanceWarning', Warning)))]
        assert len(perf_warnings) == 0, f"Encountered performance warnings: {perf_warnings}"
        
    assert X_feat.shape[0] == 1
    assert hasattr(model, 'feature_names_in_')
    assert X_feat.shape[1] == len(model.feature_names_in_)
    assert list(X_feat.columns) == list(model.feature_names_in_)
    assert X_feat['total_volume'].iloc[0] == 17
    assert X_feat['num_active_log_features'].iloc[0] == 2
    assert X_feat['num_event_types'].iloc[0] == 1
    assert X_feat['num_resource_types'].iloc[0] == 1


def test_end_to_end_fast_inference_and_health_scoring():
    """Verify predict, compute_health_score, explain, and recommend_action outputs."""
    record = {
        'location': 'location 1008',
        'severity_type': 'severity_type 1',
        'event_event_type 11': 1,
        'resource_resource_type 8': 1,
        'log_log_feature 312': 1
    }
    
    # Warm up model in memory
    pred_c, probs = predict(record)
    
    # Measure execution latency
    t0 = time.time()
    pred_c, probs = predict(record)
    score, status = compute_health_score(probs)
    inference_duration_ms = (time.time() - t0) * 1000.0
    
    # Assert valid domain types and ranges
    assert pred_c in [0, 1, 2]
    assert len(probs) == 3
    assert np.isclose(sum(probs), 1.0, atol=1e-3)
    assert 0.0 <= score <= 100.0
    assert status in ['Healthy', 'Warning', 'Critical']
    assert inference_duration_ms < 50.0, f"Inference took too long: {inference_duration_ms:.2f}ms"


def test_unseen_custom_location_fallback():
    """Verify that unlisted / custom location strings execute cleanly using default frequency prior."""
    novel_record = {
        'location': 'location 99999',
        'severity_type': 'severity_type 4',
        'event_event_type 34': 1,
        'resource_resource_type 2': 1,
        'log_log_feature 50': 20
    }
    
    pred_c, probs = predict(novel_record)
    assert pred_c in [0, 1, 2]
    assert len(probs) == 3


def test_explainability_and_actions():
    """Verify SHAP explainability returns ranked anomaly factors and valid recommendations."""
    crit_record = {
        'location': 'location 821',
        'severity_type': 'severity_type 2',
        'event_event_type 11': 1,
        'resource_resource_type 2': 1,
        'log_log_feature 203': 100,
        'log_log_feature 312': 50
    }
    
    pred_c, probs = predict(crit_record)
    score, status = compute_health_score(probs)
    shap_factors = explain(crit_record, top_n=4)
    actions = recommend_action(shap_factors, status)
    
    assert isinstance(shap_factors, list)
    assert len(shap_factors) > 0
    assert isinstance(actions, list)
    assert len(actions) > 0
    for feat_name, val in shap_factors:
        assert isinstance(feat_name, str)
        assert isinstance(val, float)


if __name__ == '__main__':
    print("Running test_custom_record_preprocessing_and_vectorization...")
    test_custom_record_preprocessing_and_vectorization()
    print("Running test_end_to_end_fast_inference_and_health_scoring...")
    test_end_to_end_fast_inference_and_health_scoring()
    print("Running test_unseen_custom_location_fallback...")
    test_unseen_custom_location_fallback()
    print("Running test_explainability_and_actions...")
    test_explainability_and_actions()
    print("[PASS] ALL TESTS PASSED SUCCESSFULLY!")
