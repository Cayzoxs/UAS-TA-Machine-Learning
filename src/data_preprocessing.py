import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

def load_and_preprocess(data_url):
    """
    Fungsi untuk memuat data, menghapus kelas 'Enrolled', 
    serta melakukan standarisasi dan encoding target.
    """
    print("Memuat dataset...")
    df = pd.read_csv(data_url, sep=';')
    
    # Memfilter data (fokus pada klasifikasi biner)
    df = df[df['Target'] != 'Enrolled']
    
    # Memisahkan fitur dan target
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Label Encoding untuk target (Dropout/Graduate)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Standard Scaling untuk fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Memastikan folder models ada, lalu menyimpan objek preprocessing
    os.makedirs('models', exist_ok=True)
    with open('models/preprocessing.pkl', 'wb') as f:
        pickle.dump({
            'scaler': scaler, 
            'label_encoder': le, 
            'fitur_kolom': X.columns.tolist()
        }, f)
        
    print("Preprocessing selesai. Pipeline disimpan sebagai 'preprocessing.pkl'")
    
    return X_scaled, y_encoded

if __name__ == "__main__":
    URL = "https://raw.githubusercontent.com/Cayzoxs/UAS-TA-Machine-Learning/main/data/raw/data.csv"
    X, y = load_and_preprocess(URL)
