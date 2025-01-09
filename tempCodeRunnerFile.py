import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from knn import KNN

# Page configuration
st.set_page_config(page_title="Job Recommendation System", layout="wide")

# Load Dataset and Model
@st.cache_resource
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Dataset tidak ditemukan. Pastikan file tersedia.")
        return None

# Load model and transformer objects
knn_model = joblib.load('knn_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
scaler = joblib.load('normalization_params.joblib')

# Load dataset
data_file = "dataset_fiks.csv"  # Ganti dengan nama file dataset Anda
data = load_data(data_file)

# Navigation menu
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Menu", ["Rekomendasi Pekerjaan", "Tentang Aplikasi"])

if menu == "Rekomendasi Pekerjaan":
    st.title("🎯 Sistem Rekomendasi Pekerjaan")
    st.markdown("Dapatkan rekomendasi pekerjaan berdasarkan profil Anda.")
    
    if data is not None:
        # Input User
        gender = st.selectbox("Jenis Kelamin:", ["Male", "Female", "Prefer not to say"])
        gender_map = {"Male": 0, "Female": 1, "Prefer not to say": 2}
        gender_numeric = gender_map[gender]

        major = st.selectbox("Jurusan Sarjana:", data["UG Specialization (Major)"].unique())
        interests = st.text_input("Minat Utama (Pisahkan dengan koma):", placeholder="Contoh: technology, data science, ai")
        skills = st.text_input("Keterampilan (Pisahkan dengan koma):", placeholder="Contoh: python, machine learning, sql")
        cgpa = st.slider("Rata-rata Nilai Akademik:", min_value=2.0, max_value=4.0, step=0.1)
        
        certification = st.selectbox("Apakah Anda Memiliki Sertifikasi?", ["No", "Yes"])
        certification_map = {"No": 0, "Yes": 1}
        certification_numeric = certification_map[certification]

        certification_course_title = st.text_input("Judul Sertifikasi (jika ada):", placeholder="Contoh: Data Science Certification")
        
        status = st.selectbox("Status Kerja Saat Ini:", ["Not Working", "Working"])
        status_map = {"Not Working": 0, "Working": 1}
        status_numeric = status_map[status]

        # Fungsi untuk memfilter pekerjaan
        def filter_jobs(data, gender_numeric, major, interests, skills, cgpa, certification_numeric, status_numeric):
            # Filter data berdasarkan input pengguna
            filtered_data = data[(
                (data["Gender"] == gender_numeric) &
                (data["UG Specialization (Major)"] == major) &
                (data["Average CGPA/Percentage"] >= cgpa) &
                (data["Certification Courses"] == certification_numeric) &
                (data["Working Status"] == status_numeric)
            )].copy()

            # Prioritaskan kecocokan minat dan keterampilan
            filtered_data["Interest Match"] = filtered_data["Interests"].apply(
                lambda x: len(set(x.split(", ")) & set(interests.split(", ")))
            )
            filtered_data["Skill Match"] = filtered_data["Skills"].apply(
                lambda x: len(set(x.split(", ")) & set(skills.split(", ")))
            )
            filtered_data["Total Match"] = filtered_data["Interest Match"] + filtered_data["Skill Match"]

            return filtered_data.sort_values(by="Total Match", ascending=False)

        # Validasi Input
        if st.button("🔍 Cari Pekerjaan"):
            if not interests:
                st.warning("Harap isi minimal satu minat.")
            elif not skills:
                st.warning("Harap isi minimal satu keterampilan.")
            else:
                recommendations = filter_jobs(data, gender_numeric, major, interests, skills, cgpa, certification_numeric, status_numeric)
                st.subheader("Rekomendasi Pekerjaan:")

                # Gabungkan input teks
                combined_text = interests + ' ' + skills + ' ' + certification_course_title + ' ' + major

                # Transformasi teks input menggunakan TF-IDF vectorizer
                input_tfidf = vectorizer.transform([combined_text]).toarray()

                # Periksa jumlah fitur input dan pastikan cocok dengan jumlah fitur yang diharapkan oleh scaler
                input_features = input_tfidf.shape[1]
                expected_features = scaler.feature_names_in_.shape[0]

                if input_features != expected_features:
                    st.warning(f"Jumlah fitur input tidak sesuai. Fitur input memiliki {input_features} fitur, tetapi scaler mengharapkan {expected_features} fitur.")
                else:
                    # Normalisasi input dengan scaler yang dimuat
                    input_normalized = scaler.transform(input_tfidf)

                    # Prediksi menggunakan KNN
                    prediction = knn_model.predict(input_normalized)

                    # Menampilkan hasil prediksi
                    st.write(f"Predicted Career Category: {prediction[0]}")

                if not recommendations.empty:
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"{row['Mapped Category']}")
                else:
                    st.warning("Tidak ada rekomendasi pekerjaan yang sesuai dengan kriteria Anda.")
                
elif menu == "Tentang Aplikasi":
    st.title("Tentang Aplikasi Job Recommendation")
    st.markdown(
        """
        ### Deskripsi
        Aplikasi ini membantu pengguna mendapatkan rekomendasi pekerjaan berdasarkan profil mereka, seperti:
        - Jenis kelamin
        - Jurusan sarjana
        - Minat utama
        - Keterampilan
        - Rata-rata nilai akademik
        - Sertifikat
        - Status kerja saat ini
        """
    )