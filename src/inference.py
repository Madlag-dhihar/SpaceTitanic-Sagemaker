import os
import sys
import types


if 'src' not in sys.modules:
    src_module = types.ModuleType('src')
    src_module.__path__ = [os.path.dirname(__file__)]
    sys.modules['src'] = src_module


import pickle
import json
import pandas as pd

def model_fn(model_dir):
    """Memuat pipeline utuh hasil seleksi dari S3/lokal kontainer"""
    # Mencari pipeline.pkl di dalam folder model/ hasil ekstrak tar
    model_path = os.path.join(model_dir, "model", "pipeline.pkl")
    
    # Fallback jika struktur ekstraksinya langsung di root model_dir
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "pipeline.pkl")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, content_type):
    """Menerima request JSON baik baris tunggal maupun batch banyak data sekaligus"""
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict) and "instances" in data:
            return pd.DataFrame(data["instances"])
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            return pd.DataFrame(data)
    raise ValueError(f"Content-type {content_type} tidak didukung. Gunakan application/json")

def predict_fn(input_data, model):
    """Eksekusi prediksi langsung memanfaatkan pipeline end-to-end"""
    return model.predict(input_data)

def output_fn(prediction, content_type):
    """Mengembalikan respon JSON berisikan array hasil prediksi model"""
    return json.dumps({"prediction": prediction.tolist()}), "application/json"