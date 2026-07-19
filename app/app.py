import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="EduML - Prediksi Dropout", page_icon="🎓", layout="wide")

@st.cache_resource
def load_components():
    model_path = os.path.join('models', 'best_model.pkl')
    prep_path = os.path.join('models', 'preprocessing.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(prep_path, 'rb') as f:
        prep = pickle.load(f)
    return model, prep

model, prep = load_components()
scaler = prep['scaler']
le = prep['label_encoder']
fitur_kolom = prep['fitur_kolom']

st.sidebar.title("🎓 EduML Dashboard")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Pilih Menu:", 
                        ["Dashboard Analisis", "Prediksi (Model Demo)", "Evaluasi & Interpretasi", "Dokumentasi & Panduan"])

if menu == "Dashboard Analisis":
    st.title("Dashboard Eksplorasi Data Mahasiswa")
    st.markdown("""
    Selamat datang di aplikasi **EduML**. Aplikasi ini dirancang menggunakan *Machine Learning* (XGBoost + SMOTE) 
    untuk memprediksi mahasiswa yang berisiko putus kuliah (*dropout*).
    
    **Insight Utama dari Data:**
    1. Terdapat ketidakseimbangan kelas (*imbalanced data*), mahasiswa lulus jauh lebih banyak.
    2. Mahasiswa yang memiliki status *Debtor* (tunggakan hutang) memiliki rasio *dropout* yang sangat tinggi.
    3. Nilai IPK di semester 1 dan 2 menjadi indikator terkuat keberhasilan akademik mahasiswa.
    """)
    st.info("Silakan buka menu 'Prediksi (Model Demo)' di sidebar kiri untuk mencoba kecerdasan buatan ini.")

elif menu == "Prediksi (Model Demo)":
    st.title("Simulasi Prediksi Mahasiswa Berisiko")
    st.write("Masukkan parameter akademik dan profil mahasiswa untuk mendeteksi potensi *dropout* sejak dini.")
    
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Akademik")
            grade1 = st.number_input("Rata-rata Nilai Semester 1", min_value=0.0, max_value=20.0, value=12.0)
            grade2 = st.number_input("Rata-rata Nilai Semester 2", min_value=0.0, max_value=20.0, value=12.0)
            age = st.number_input("Umur Saat Mendaftar", min_value=15, max_value=60, value=19)
            
        with col2:
            st.subheader("Data Sosial-Ekonomi")
            debtor = st.selectbox("Apakah memiliki tunggakan/hutang? (Debtor)", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
            tuition = st.selectbox("Apakah uang kuliah sudah lunas?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
            gender = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
            
        submit_button = st.form_submit_button(label="🔍 Deteksi Potensi Dropout")
        
    if submit_button:
        # Mengisi nilai default 0 untuk fitur lain yang tidak diinput manual (agar sesuai bentuk array XGBoost)
        input_data = {col: 0 for col in fitur_kolom}
        
        # Update dengan nilai dari form
        input_data['Curricular units 1st sem (grade)'] = grade1
        input_data['Curricular units 2nd sem (grade)'] = grade2
        input_data['Age at enrollment'] = age
        input_data['Debtor'] = debtor
        input_data['Tuition fees up to date'] = tuition
        input_data['Gender'] = gender
        
        # Konversi ke DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Preprocessing (Scaling)
        df_scaled = scaler.transform(df_input)
        df_scaled = pd.DataFrame(df_scaled, columns=fitur_kolom)
        
        # Prediksi
        prediksi = model.predict(df_scaled)
        hasil_teks = le.inverse_transform(prediksi)[0]
        
        st.markdown("---")
        if hasil_teks == 'Dropout':
            st.error("**HASIL PREDIKSI: RISIKO TINGGI DROPOUT!!!**")
            st.write("Sistem merekomendasikan pihak akademik untuk segera melakukan pendekatan dan pendampingan terhadap mahasiswa ini.")
        else:
            st.success("**HASIL PREDIKSI: AMAN (GRADUATE)**")
            st.write("Mahasiswa ini memiliki profil akademik dan finansial yang stabil untuk melanjutkan perkuliahan hingga lulus.")

elif menu == "Evaluasi & Interpretasi":
    st.title("Evaluasi Kinerja Model")
    st.write("Berdasarkan eksperimen pada tahapan sebelumnya, perpaduan algoritma XGBoost dengan teknik *resampling* SMOTE berhasil memecahkan masalah bias pada *imbalanced data*.")
    
    st.subheader("1. Interpretasi Bisnis (SHAP)")
    st.write("Fitur paling krusial yang menentukan seorang mahasiswa akan *dropout* atau tidak berturut-turut adalah:")
    st.markdown("""
    * **Curricular units 2nd sem (grade)** (Nilai IPK Semester 2)
    * **Curricular units 1st sem (grade)** (Nilai IPK Semester 1)
    * **Tuition fees up to date** (Kelancaran pembayaran uang kuliah)
    * **Debtor** (Kepemilikan hutang)
    """)
    
    st.subheader("2. Mengapa Memilih Metrik Recall?")
    st.write("""
    Dalam kasus klasifikasi *dropout* pendidikan, metrik akurasi (*Accuracy*) bisa sangat menyesatkan karena data didominasi oleh mahasiswa yang lulus. 
    Oleh karena itu, performa dievaluasi menggunakan metrik **Recall**. Model difokuskan untuk meminimalkan *False Negative* 
    (menekan angka mahasiswa yang sebenarnya berisiko *dropout*, namun sistem menebaknya aman).
    """)

elif menu == "Dokumentasi & Panduan":
    st.title("Dokumentasi Sistem")
    
    st.subheader("1. Deskripsi Dataset")
    st.write("Aplikasi ini dibangun menggunakan dataset sekunder *Predict Students' Dropout and Academic Success* yang diunduh dari UCI Machine Learning Repository. Dataset ini mencakup rekam jejak demografi, kondisi sosial-ekonomi, dan performa akademik mahasiswa dari saat mendaftar hingga akhir semester kedua.")
    
    st.subheader("2. Metodologi (Machine Learning Pipeline)")
    st.markdown("""
    * **Prapemrosesan:** Transformasi data kategorikal menjadi numerik (*Label Encoding*) dan penyesuaian skala variabel (*Standard Scaling*).
    * **Penanganan Bias:** Menggunakan teknik **SMOTE** (*Synthetic Minority Over-sampling Technique*) untuk memperbanyak sampel mahasiswa *dropout* secara sintetis agar seimbang dengan jumlah mahasiswa yang lulus.
    * **Pemodelan:** Melatih data menggunakan algoritma **XGBoost** (*Extreme Gradient Boosting*) yang telah dioptimasi hyperparameternya melalui *GridSearchCV*.
    """)
    
    st.subheader("3. Panduan Penggunaan Aplikasi")
    st.markdown("""
    * Buka menu **Dashboard Analisis** untuk melihat eksplorasi wawasan data historis.
    * Buka menu **Prediksi (Model Demo)** dan masukkan profil akademik mahasiswa pada formulir yang disediakan. Klik tombol deteksi untuk melihat prediksi kecerdasan buatan.
    * Buka menu **Evaluasi & Interpretasi** untuk melihat metrik performa model dan transparansi keputusan fitur (*SHAP values*).
    """)
