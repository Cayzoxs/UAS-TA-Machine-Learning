import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import shap
import warnings
import os

warnings.filterwarnings('ignore')

def evaluate_and_interpret(model, X_test, y_test, target_names, feature_names):
    """
    Fungsi untuk menampilkan metrik klasifikasi dan menyimpan visualisasi.
    """
    print("=== Performa XGBoost (Terbaik) ===")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Memastikan folder reports ada
    os.makedirs('reports', exist_ok=True)
    
    print("Membuat visualisasi Confusion Matrix...")
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - XGBoost (SMOTE)')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    plt.close()
    
    print("Membuat visualisasi SHAP Value...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('Interpretasi Fitur (SHAP Value) - Potensi Dropout')
    plt.tight_layout()
    plt.savefig('reports/shap_summary.png')
    plt.close()
    
    print("Visualisasi evaluasi berhasil disimpan di folder 'reports/'")
