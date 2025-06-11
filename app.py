
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from pyngrok import ngrok  # Import ngrok

# Set up page config
st.set_page_config(page_title="Prediksi Risiko Penyakit Jantung", page_icon="❤️")
st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.write("Aplikasi ini membantu memprediksi apakah seseorang berisiko terkena penyakit jantung berdasarkan data kesehatan sederhana.")

# Input User
st.header("📋 Masukkan Data Pasien")

# User input
age = st.number_input("Umur", min_value=1, max_value=120, value=50)
sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
cp_type = st.selectbox("Jenis Nyeri Dada", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Tekanan Darah Saat Istirahat (mmHg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Kolesterol (mg/dL)", min_value=100, max_value=600, value=200)
fasting_bs = st.radio("Apakah gula darah puasa > 120 mg/dL?", ["Ya", "Tidak"])
rest_ecg = st.selectbox("Hasil EKG Saat Istirahat", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Detak Jantung Maksimum (BPM)", min_value=60, max_value=250, value=150)
ex_angina = st.radio("Nyeri Dada Saat Olahraga?", ["Ya", "Tidak"])
oldpeak = st.number_input("Oldpeak (Penurunan ST)", min_value=0.0, max_value=6.0, value=1.0)
st_slope = st.selectbox("Kemiringan ST Saat Olahraga", ["Up", "Flat", "Down"])

# Label encoding sesuai data model
def encode_inputs():
    label_map = {
        'Sex': {"F": 0, "M": 1},
        'ChestPainType': {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3},
        'RestingECG': {"LVH": 0, "Normal": 1, "ST": 2},
        'ExerciseAngina': {"N": 0, "Y": 1},
        'ST_Slope': {"Down": 0, "Flat": 1, "Up": 2}
    }

    return pd.DataFrame({
        'Age': [age],
        'Sex': [label_map['Sex']["M" if sex == "Laki-laki" else "F"]],
        'ChestPainType': [label_map['ChestPainType'][cp_type]],
        'RestingBP': [resting_bp],
        'Cholesterol': [chol],
        'FastingBS': [1 if fasting_bs == "Ya" else 0],
        'RestingECG': [label_map['RestingECG'][rest_ecg]],
        'MaxHR': [max_hr],
        'ExerciseAngina': [label_map['ExerciseAngina']["Y" if ex_angina == "Ya" else "N"]],
        'Oldpeak': [oldpeak],
        'ST_Slope': [label_map['ST_Slope'][st_slope]]
    })

# Load model
@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")  # Pastikan model ini sudah kamu simpan

# Load KMeans model (Clustering)
@st.cache_resource
def load_kmeans():
    return joblib.load("kmeans_model.pkl")  # Pastikan model KMeans ini sudah kamu simpan

model = load_model()
kmeans_model = load_kmeans()

# Mapping cluster numbers to human-readable labels
cluster_labels = {
    0: "Cluster Sehat",
    1: "Cluster Rentan",
    2: "Cluster Berisiko Tinggi"
}

# Tombol Prediksi
if st.button("🔍 Prediksi Sekarang"):
    input_df = encode_inputs()
    st.write("🔎 Data yang Diproses:", input_df)

    # Prediksi menggunakan model Naive Bayes
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Prediksi cluster menggunakan KMeans
    cluster_label = kmeans_model.predict(input_df)[0]

    # Menampilkan hasil prediksi
    st.subheader("🧠 Hasil Prediksi")
    if prediction == 1:
        st.error(f"⚠️ Anda kemungkinan **berisiko terkena penyakit jantung**. (Skor Probabilitas: {probability:.2f})")
        st.write("👉 Disarankan untuk konsultasi ke dokter dan mulai gaya hidup sehat.")
    else:
        st.success(f"✅ Anda kemungkinan **tidak berisiko** terkena penyakit jantung. (Skor Probabilitas: {probability:.2f})")
        st.write("👍 Tetap jaga pola makan, olahraga rutin, dan periksa kesehatan secara berkala.")

    # Menampilkan hasil cluster dalam kata-kata
    st.write(f"🎯 Data ini berada pada **{cluster_labels.get(cluster_label, 'Unknown Cluster')}**.")
    st.write("🔍 Cluster ini menunjukkan karakteristik tertentu yang dapat membantu dalam penentuan risiko.")

st.markdown("---")
st.caption("Model ini dibuat hanya untuk tujuan edukasi dan tidak menggantikan diagnosis medis profesional.")
