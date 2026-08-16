import os
import sys
import dotenv
dotenv.load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm_service import LLMService
from src.what_if_engine import WhatIfEngine

def test_router_regex():
    print("\n--- Test 1: Regex Router Verification ---")
    llm = LLMService(api_key="") # offline mode for fallback routing
    
    # Test cases for what_if intent routing
    tc1 = "What happens if Device 15086 is not fixed?"
    route1 = llm._local_fallback_route(tc1)
    print(f"Query: '{tc1}' -> Route: {route1}")
    assert route1["query_type"] == "what_if", "Failed to route as what_if"
    assert route1["target_device_id"] == 15086, "Failed to extract device ID 15086"
    
    tc2 = "What if TN1045 remains unresolved?"
    route2 = llm._local_fallback_route(tc2)
    print(f"Query: '{tc2}' -> Route: {route2}")
    assert route2["query_type"] == "what_if", "Failed to route as what_if"
    assert route2["target_device_id"] == 1045, "Failed to extract device ID 1045 from TN1045"
    
    tc3 = "What is the impact if this fault continues?"
    route3 = llm._local_fallback_route(tc3)
    print(f"Query: '{tc3}' -> Route: {route3}")
    assert route3["query_type"] == "what_if", "Failed to route as what_if"
    assert route3["refers_to_active_device"] is True, "Failed to detect refers_to_active_device"

    # Verify standard queries are not broken (backward compatibility check)
    tc_norm = "Explain why location 262 is critical"
    route_norm = llm._local_fallback_route(tc_norm)
    print(f"Query: '{tc_norm}' -> Route: {route_norm}")
    assert route_norm["query_type"] == "semantic", "Failed to route standard semantic query"
    
    print("Regex router tests passed successfully!")

def test_what_if_engine():
    print("\n--- Test 2: What-If Engine Verification ---")
    engine = WhatIfEngine()
    
    # Mock data to simulate df_results and raw_records
    import pandas as pd
    mock_df_results = pd.DataFrame([{
        'id': 15086,
        'location': 'location 262',
        'severity_type': 'severity_type 2',
        'fault_severity': 2,
        'predicted_class': 2,
        'p0': 0.05,
        'p1': 0.15,
        'p2': 0.80,
        'health_score': 24.3,
        'status': 'Critical'
    }])
    
    mock_raw_records = {
        15086: {
            'location': 'location 262',
            'severity_type': 'severity_type 2',
            'resource_type 2': 1,
            'event_type 11': 1,
            'log_feature 203': 12
        }
    }
    
    res = engine.analyze(15086, df_results=mock_df_results, raw_records=mock_raw_records)
    print("Structured What-If Output:")
    for k, v in res.items():
        print(f"  {k}: {v}")
        
    assert res["device_id"] == 15086
    assert res["risk_level"] == "HIGH"
    assert "not recorded" in res["potential_impact"].lower() or "not available" in res["potential_impact"].lower()
    assert "not available" in res["time_to_impact"].lower()
    
    # Test for Device ID not found
    res_not_found = engine.analyze(99999, df_results=mock_df_results, raw_records=mock_raw_records)
    assert res_not_found is None, "Engine should return None if device not found"
    
    print("What-If Engine tests passed successfully!")

if __name__ == "__main__":
    test_router_regex()
    test_what_if_engine()
