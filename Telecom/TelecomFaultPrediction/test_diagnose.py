import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_and_merge
from src.predictor import predict, explain

def test_inference_pipeline():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    merged_df = load_and_merge(data_dir)
    
    # Pick a sample row and convert to dict
    sample_row = merged_df.iloc[0].to_dict()
    print("Testing record:", {k: sample_row[k] for k in list(sample_row.keys())[:5]})
    
    # Try predicting
    pred_class, probs = predict(sample_row)
    print(f"Prediction: class={pred_class}, probs={probs}")
    
    # Try explaining
    factors = explain(sample_row)
    print(f"SHAP Explanations: {factors}")

if __name__ == "__main__":
    test_inference_pipeline()
