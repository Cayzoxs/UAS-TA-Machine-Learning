# Prediksi Potensi Dropout Mahasiswa Menggunakan XGBoost dan SMOTE

Repositori ini merupakan Capstone Project untuk Ujian Akhir Semester mata kuliah Pembelajaran Mesin. Proyek ini berfokus pada pengembangan model *Machine Learning* *end-to-end* untuk memprediksi potensi *dropout* mahasiswa, mulai dari pra-pemrosesan data hingga *deployment* aplikasi interaktif menggunakan Streamlit.

## 1. Latar Belakang Masalah (Problem Statement)
Institusi pendidikan tinggi sering kali menghadapi tantangan besar dalam mempertahankan mahasiswanya hingga lulus. Tingkat *dropout* (putus kuliah) yang tinggi tidak hanya berdampak buruk pada reputasi institusi, tetapi juga merupakan kerugian besar bagi mahasiswa secara ekonomi dan waktu. Deteksi dini terhadap mahasiswa yang berisiko *dropout* sangat krusial agar pihak kampus dapat memberikan intervensi akademis atau bantuan finansial sebelum terlambat. 

Namun, mendeteksi potensi *dropout* secara manual melalui pendekatan *rule-based* (misalnya hanya melihat IPK rendah) tidaklah efektif. Keputusan seorang mahasiswa untuk berhenti kuliah dipengaruhi oleh kombinasi pola non-linear dari berbagai faktor, seperti latar belakang demografi, kondisi sosial-ekonomi, jalur masuk pendaftaran, hingga kinerja akademik di semester awal. Oleh karena itu, pendekatan *Machine Learning* sangat diperlukan untuk menemukan pola tersembunyi dari interaksi berbagai fitur tersebut secara otomatis dan akurat. 

Tantangan utama dalam pemodelan ini adalah ketidakseimbangan kelas (*imbalanced data*), di mana jumlah mahasiswa yang lulus jauh lebih banyak daripada mahasiswa yang *dropout*. Jika tidak ditangani, model akan cenderung bias ke kelas mayoritas. Proyek ini memecahkan masalah tersebut dengan membandingkan teknik *resampling* SMOTE dan Random Under-Sampling pada algoritma XGBoost.

**Metrik Kesuksesan Proyek:**
Fokus utama dari performa model diukur menggunakan metrik **Recall** pada kelas minoritas (*dropout*). Keberhasilan proyek tercapai jika model mampu meminimalkan *False Negative* (mahasiswa berisiko *dropout* yang gagal terdeteksi oleh sistem), didukung dengan *deployment* aplikasi web yang dapat digunakan oleh staf akademik untuk memprediksi status mahasiswa baru.

## 2. Dataset
Dataset yang digunakan adalah **"Predict Students' Dropout and Academic Success"** yang diperoleh dari repositori publik UCI Machine Learning. Dataset ini bersifat tabular dan berisi fitur demografi, sosial-ekonomi, serta pencapaian akademik mahasiswa.
*   **Sumber Dataset:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
*   **Jumlah Fitur:** 36 Fitur input (kategorikal dan numerikal) dan 1 Fitur target.
*   **Target:** `Dropout`, `Enrolled`, atau `Graduate` (Fokus diubah menjadi klasifikasi biner: `Dropout` vs `Graduate`).

## 3. Struktur Repository
Repository ini ditata mengikuti standar proyek *Machine Learning*:

```text
capstone-project-data-mining/
├── data/
│   ├── raw/                 # Data mentah (.csv)
│   └── processed/           # Data yang sudah di-preprocess
├── notebooks/
│   ├── 01_eda.ipynb         # Eksplorasi data, visualisasi, dan preprocessing
│   └── 02_modeling.ipynb    # Training, resampling SMOTE, evaluasi XGBoost, interpretasi SHAP
├── src/
│   └── utils.py             # Fungsi utilitas tambahan 
├── models/
│   ├── best_model.pkl       # File model XGBoost terbaik yang siap di-deploy
│   └── preprocessing.pkl    # Pipeline preprocessing (scaler/encoder)
├── app/
│   ├── app.py               # Aplikasi Streamlit utama
│   └── requirements.txt     # Daftar dependencies/library
├── reports/
│   └── final_report.pdf     # Laporan teknis UAS
└── README.md                # Dokumentasi proyek ini
