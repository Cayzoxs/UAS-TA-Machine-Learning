import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

def train_xgboost_model(X_scaled, y_encoded):
    """
    Fungsi untuk melatih model XGBoost dengan SMOTE dan GridSearchCV.
    """
    print("Membagi data menjadi Data Latih (80%) dan Data Uji (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("Menerapkan SMOTE untuk menyeimbangkan data latih...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("Memulai proses Hyperparameter Tuning dengan GridSearchCV (CV=3)...")
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    
    grid_xgb = GridSearchCV(xgb, param_grid, cv=3, scoring='recall', n_jobs=-1)
    grid_xgb.fit(X_train_smote, y_train_smote)
    
    best_xgb = grid_xgb.best_estimator_
    print(f"Training selesai! Parameter Terbaik: {grid_xgb.best_params_}")
    
    # Menyimpan model ke folder models/
    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_xgb, f)
    print("Model XGBoost terbaik berhasil disimpan sebagai 'models/best_model.pkl'")
    
    return best_xgb, X_test, y_test
