import os
import dotenv
dotenv.load_dotenv()

from src.rag_service import RAGService
from src.llm_service import LLMService

def run_tests():
    print("=== Running RAG and LLM Pipeline Tests ===")
    
    # 1. Test RAG Service Loading and Retrieval
    print("\n[Test 1] Initializing RAGService...")
    try:
        rag = RAGService()
        print("RAGService initialized successfully.")
        
        print("\nTesting retrieval for 'location 10'...")
        results = rag.retrieve("location 10", k=2)
        print(f"Retrieved {len(results)} results:")
        for idx, res in enumerate(results):
            print(f"  Match {idx+1}: {res['location']}")
            print(f"  Summary:\n{res['summary']}\n")
            
        assert len(results) > 0, "No records retrieved from location FAISS index."
        
        print("\nTesting O(1) device ID lookup for device 15086...")
        dev_res = rag.lookup_device_id(15086)
        if dev_res:
            print(f"Direct Lookup Success: {dev_res['device_id']}")
            print(f"Summary:\n{dev_res['summary']}\n")
        else:
            print("Direct Lookup returned None (Device 15086 not in database?)")

        print("\nTesting retrieval of similar devices for 'resource_type 8'...")
        dev_ret = rag.retrieve_devices("resource_type 8", k=2)
        print(f"Retrieved {len(dev_ret)} device matches:")
        for idx, r in enumerate(dev_ret):
            print(f"  Match {idx+1}: Device #{r['device_id']}")
            print(f"  Summary:\n{r['summary']}\n")
            
        assert len(dev_ret) > 0, "No records retrieved from device FAISS index."
        print("RAG Retrieval tests passed!")
    except Exception as e:
        print(f"RAG Retrieval test failed: {e}")
        return
        
    # 2. Test LLM Router
    print("\n[Test 2] Initializing LLMService...")
    try:
        llm = LLMService()
        print(f"LLMService initialized. OpenRouter Client Active: {llm.client is not None}")
        
        # Test routing for numerical queries
        print("\nTesting intent router for: 'How many total faults are there?'")
        route_res = llm.route_query("How many total faults are there?")
        print(f"Routing classification: {route_res}")
        assert route_res["query_type"] == "numerical", "Should classify 'how many total faults' as numerical."
        
        # Test routing for semantic queries
        print("\nTesting intent router for: 'Explain why location 262 is critical'")
        route_res2 = llm.route_query("Explain why location 262 is critical")
        print(f"Routing classification: {route_res2}")
        assert route_res2["location"] == "location 262", "Should extract location 'location 262'."
        
        # Test routing for out-of-scope queries
        print("\nTesting intent router for unrelated: 'What is the capital of France?'")
        route_res3 = llm.route_query("What is the capital of France?")
        print(f"Routing classification: {route_res3}")
        assert route_res3["is_network_related"] is False, "Should classify capital of France as out of scope."
        
        # Test routing for device ID query
        print("\nTesting intent router for: 'What is the status of device #15086?'")
        route_res4 = llm.route_query("What is the status of device #15086?")
        print(f"Routing classification: {route_res4}")
        assert route_res4["target_device_id"] == 15086, "Should extract target_device_id = 15086."

        # Test routing for active device reference
        print("\nTesting intent router for: 'Why is it critical?'")
        route_res5 = llm.route_query("Why is it critical?")
        print(f"Routing classification: {route_res5}")
        assert route_res5["refers_to_active_device"] is True, "Should flag refers_to_active_device = True."

        print("LLM Routing tests passed!")
        
        # 3. Test LLM Answer Generation Fallback
        print("\n[Test 3] Testing generate_answer with active device context...")
        context = [{"location": "location 10", "summary": "Location: location 10\nSeverity: severity_type 1\nHealth Score: 85.0 (Healthy)"}]
        active_ctx = "Device ID: 15086\nLocation: location_262\nStatus: Critical (Health Score: 24.3)\nTop Anomaly Triggers: log_feature 12 (Impact: +12.3)"
        ans = llm.generate_answer("Why is this device critical?", context, active_ctx)
        print(f"Answer Output:\n{ans}")
        assert "15086" in ans or "critical" in ans or "information" in ans, "Answer output was incorrect."
        print("Answer generation test passed!")
        
    except Exception as e:
        print(f"LLM Service test failed: {e}")
        return

    print("\nAll pipeline tests completed successfully!")

if __name__ == "__main__":
    run_tests()
