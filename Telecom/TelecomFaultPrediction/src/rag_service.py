import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGService:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Default to the models directory in the project structure
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            
        self.faiss_path = os.path.join(models_dir, "faiss_index.index")
        self.summaries_path = os.path.join(models_dir, "location_summaries.pkl")
        self.faiss_devices_path = os.path.join(models_dir, "faiss_devices.index")
        self.device_summaries_path = os.path.join(models_dir, "device_summaries.pkl")
        
        if (not os.path.exists(self.faiss_path) or 
            not os.path.exists(self.summaries_path) or 
            not os.path.exists(self.faiss_devices_path) or 
            not os.path.exists(self.device_summaries_path)):
            raise FileNotFoundError(
                f"Vector database files not found. Please run the build script first.\n"
                f"Missing files: {self.faiss_path}, {self.summaries_path}, {self.faiss_devices_path}, {self.device_summaries_path}"
            )
            
        print("Loading SentenceTransformer model locally...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Loading FAISS indices...")
        self.index = faiss.read_index(self.faiss_path)
        self.device_index = faiss.read_index(self.faiss_devices_path)
        
        print("Loading summaries mappings...")
        with open(self.summaries_path, "rb") as f:
            self.summaries = pickle.load(f)
            
        with open(self.device_summaries_path, "rb") as f:
            dev_data = pickle.load(f)
            self.device_summaries_dict = dev_data['mapping']
            self.device_summaries_list = dev_data['list']
            
    def retrieve(self, query_text: str, k: int = 5) -> list:
        """
        Encodes query_text and performs L2 nearest-neighbor search for locations in FAISS.
        """
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        
        retrieved_records = []
        for idx in indices[0]:
            if 0 <= idx < len(self.summaries):
                retrieved_records.append(self.summaries[idx])
                
        return retrieved_records

    def retrieve_devices(self, query_text: str, k: int = 5) -> list:
        """
        Encodes query_text and performs L2 nearest-neighbor search for devices in FAISS.
        """
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        distances, indices = self.device_index.search(query_embedding, k)
        
        retrieved_records = []
        for idx in indices[0]:
            if 0 <= idx < len(self.device_summaries_list):
                retrieved_records.append(self.device_summaries_list[idx])
                
        return retrieved_records

    def lookup_device_id(self, device_id: int) -> dict:
        """
        O(1) dictionary lookup for a specific device ID.
        """
        try:
            d_id = int(device_id)
        except (ValueError, TypeError):
            return None
            
        if d_id in self.device_summaries_dict:
            res = self.device_summaries_dict[d_id]
            return {
                'device_id': d_id,
                'summary': res['summary'],
                'location': res['location']
            }
        return None
