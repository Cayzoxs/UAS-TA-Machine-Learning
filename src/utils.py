import pickle
import os

def load_components(model_path='models/best_model.pkl', prep_path='models/preprocessing.pkl'):
    """
    Fungsi untuk memuat model XGBoost dan objek Preprocessing (Scaler & Encoder).
    Sangat berguna untuk dipanggil pada file app.py (Streamlit).
    """
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        raise FileNotFoundError("Model atau file preprocessing tidak ditemukan. Pastikan proses training sudah dijalankan.")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(prep_path, 'rb') as f:
        prep = pickle.load(f)
        
    return model, prep
