import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import os

# --- Pustaka Model (Diperlukan untuk memuat model nyata) ---
tf_keras_imported = False
try:
    import tensorflow as tf
    from tensorflow import keras
    tf_keras_imported = True
except ImportError:
    st.error("❌ Pustaka TensorFlow/Keras tidak ditemukan. Klasifikasi TIDAK AKAN BERJALAN.")

yolo_imported = False
try:
    from ultralytics import YOLO
    import torch
    yolo_imported = True
except ImportError:
    st.error("❌ Pustaka Ultralytics/PyTorch tidak ditemukan. Deteksi Objek TIDAK AKAN BERJALAN.")

# --- Ketergantungan CV2 (Diisolasi dari Error libGL) ---
cv2_loaded = False
try:
    import cv2 # Untuk manipulasi gambar (Deteksi Objek)
    cv2_loaded = True
except ImportError as e:
    if "libGL.so.1" in str(e):
        st.warning("⚠️ Peringatan: cv2 gagal dimuat (libGL.so.1 hilang). Deteksi objek NYATA mungkin gagal dalam visualisasi.")
    else:
        st.warning(f"⚠️ Peringatan: cv2 gagal dimuat: {e}. Deteksi objek NYATA mungkin gagal dalam visualisasi.")
# -------------------------------------------------------------

# --- Konfigurasi File Model ---
YOLO_MODEL_PATH = "Siti Naura Khalisa_Laporan 4.pt"
CLASSIFICATION_MODEL_PATH = "SitiNauraKhalisa_Laporan2.h5"
CLASS_LABELS = ['Clean', 'Messy'] # Label untuk klasifikasi

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Analisis Ruangan: Bersih vs. Berantakan",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Gaya Kustom (Custom Styling) ---
st.markdown("""
<style>
    /* Mengubah warna latar belakang sidebar dan elemen utama */
    .css-1d391kg, .st-emotion-cache-1c7v0n3 {
        background-color: #f0f2f6; /* Latar belakang abu-abu terang */
    }
    /* Mengubah warna latar belakang utama */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Style untuk kotak hasil klasifikasi */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-top: 20px;
        font-size: 1.5em;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease-in-out;
    }
    .clean-bg {
        background-color: #4CAF50; /* Hijau untuk Bersih */
        border: 4px solid #388E3C;
    }
    .messy-bg {
        background-color: #F44336; /* Merah untuk Berantakan */
        border: 4px solid #D32F2F;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)


# --- Fungsi Pemuatan Model (Menggunakan Cache Streamlit) ---

@st.cache_resource
def load_yolo_model(path):
    """Memuat model Deteksi Objek (YOLO)."""
    if not yolo_imported:
        return None

    if not os.path.exists(path):
        st.error(f"❌ File model YOLO tidak ditemukan: {path}")
        return None
    try:
        model = YOLO(path)
        st.success("✅ Model Deteksi Objek (YOLO) berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO dari file .pt: {e}")
        return None

@st.cache_resource
def load_classification_model(path):
    """Memuat model Klasifikasi (Keras/TensorFlow)."""
    if not tf_keras_imported:
        return None

    if not os.path.exists(path):
        st.error(f"❌ File model Klasifikasi tidak ditemukan: {path}")
        return None
    try:
        model = keras.models.load_model(path)
        st.success("✅ Model Klasifikasi (Keras) berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model Klasifikasi dari file .h5: {e}")
        return None

# Muat Model di awal
yolo_model = load_yolo_model(YOLO_MODEL_PATH)
classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)


# --- Halaman Deteksi Objek ---

def object_detection_page():
    st.title("Deteksi Objek di Ruangan 🔎")
    st.subheader("Fokus: Identifikasi Item yang Menyebabkan Kekacauan (Menggunakan Model YOLO Nyata)")

    # Periksa ketersediaan model dan dependensi
    if not yolo_model:
        st.warning("⚠️ Deteksi Objek dinonaktifkan. Model YOLO gagal dimuat (lihat pesan error di atas).")
        st.info("Pastikan pustaka Ultralytics/PyTorch terinstal dan file model (.pt) ada.")
        return # Keluar jika model tidak dimuat

    uploaded_file = st.file_uploader(
        "Unggah Gambar Ruangan...",
        type=["jpg", "jpeg", "png", "webp"],
        key="detection_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Gambar Ruangan yang Diunggah.', use_column_width=True)

        st.markdown("---")

        if st.button("Jalankan Deteksi Objek", key="run_detection"):
            with st.spinner('Sedang memproses deteksi objek dengan model nyata...'):

                if not cv2_loaded:
                    st.error("❌ Pustaka visualisasi (`cv2`) atau dependensi grafis (`libGL.so.1`) gagal dimuat. Model YOLO dimuat, tetapi tidak dapat memvisualisasikan kotak deteksi.")
                    st.info("Prediksi tidak dapat ditampilkan. Pastikan `cv2` dan `libGL1` (jika di Linux) terinstal dengan benar.")
                    return

                try:
                    # Logika Prediksi Model YOLO Nyata
                    img_array = np.array(image.resize((640, 640))) # Resize untuk input model

                    # Melakukan prediksi
                    results = yolo_model(img_array)

                    # Mengambil gambar dengan kotak deteksi
                    annotated_img = results[0].plot() # Hasil plot adalah numpy array BGR

                    # Konversi BGR ke RGB untuk Streamlit
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                    st.subheader("Hasil Deteksi Objek (Model Nyata):")
                    st.image(annotated_img, caption="Hasil Deteksi YOLO", use_column_width=True)
                    st.success("Deteksi objek berhasil diselesaikan!")

                except Exception as e:
                    st.error(f"⚠️ Kesalahan saat menjalankan prediksi model YOLO: {e}. Model dan pustaka dimuat, tetapi prediksi gagal. Tidak ada hasil yang ditampilkan.")
                    st.exception(e)


# --- Halaman Klasifikasi Ruangan (Clean/Messy) ---

def classification_page():
    st.title("Klasifikasi Ruangan: Bersih atau Berantakan? 🧹")
    st.subheader("Fokus: Penilaian Keseluruhan Kebersihan Ruangan (Menggunakan Model Keras Nyata)")

    # Periksa ketersediaan model
    if not classification_model:
        st.warning("⚠️ Klasifikasi dinonaktifkan. Model Keras/TensorFlow gagal dimuat (lihat pesan error di atas).")
        st.info("Pastikan pustaka TensorFlow/Keras terinstal dan file model (.h5) ada.")
        return # Keluar jika model tidak dimuat

    uploaded_file = st.file_uploader(
        "Unggah Gambar Ruangan...",
        type=["jpg", "jpeg", "png", "webp"],
        key="classification_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Gambar Ruangan yang Diunggah.', use_column_width=True)

        st.markdown("---")

        if st.button("Jalankan Klasifikasi", key="run_classification"):
            with st.spinner('Sedang melakukan klasifikasi ruangan dengan model nyata...'):
                predicted_label = "ERROR"
                predicted_confidence = 0.0

                try:
                    # Logika Prediksi Model Keras Nyata
                    img = image.resize((224, 224)) # Contoh ukuran input Keras
                    img_array = keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
                    img_array /= 255.0 # Normalisasi (sesuaikan dengan training Anda)

                    # Prediksi
                    predictions = classification_model.predict(img_array)

                    # Ambil label hasil prediksi
                    predicted_index = np.argmax(predictions, axis=1)[0]
                    predicted_label = CLASS_LABELS[predicted_index]
                    predicted_confidence = predictions[0][predicted_index] * 100

                except Exception as e:
                    st.error(f"⚠️ Kesalahan saat menjalankan prediksi model Keras: {e}.")
                    st.info("Prediksi gagal dieksekusi, periksa input gambar dan struktur model Anda.")
                    st.exception(e)
                    return # Keluar jika prediksi gagal

                # --- Menampilkan Hasil Klasifikasi ---
                st.subheader("Hasil Klasifikasi (Model Nyata):")

                if predicted_label == 'Clean':
                    bg_class = "clean-bg"
                    emoji = "✨"
                    message = "Selamat! Ruangan ini diklasifikasikan sebagai **BERSIH**."
                else:
                    bg_class = "messy-bg"
                    emoji = "⚠️"
                    message = "Peringatan! Ruangan ini diklasifikasikan sebagai **BERANTAKAN**."

                # Kotak Hasil Klasifikasi
                st.markdown(
                    f'<div class="result-box {bg_class}">{emoji} {predicted_label.upper()} {emoji}</div>',
                    unsafe_allow_html=True
                )

                st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{predicted_confidence:.2f}%")
                st.markdown(f"***Pesan Analisis:*** {message}")


# --- Sidebar Navigasi ---

st.sidebar.title("Navigasi Aplikasi")
menu = st.sidebar.radio(
    "Pilih Fitur:",
    ("Klasifikasi Ruangan", "Deteksi Objek")
)

if menu == "Klasifikasi Ruangan":
    classification_page()
elif menu == "Deteksi Objek":
    object_detection_page()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Deteksi (YOLO):** `{YOLO_MODEL_PATH}`")
st.sidebar.markdown(f"**Model Klasifikasi (Keras):** `{CLASSIFICATION_MODEL_PATH}`")
st.sidebar.caption("Pastikan kedua file model berada di folder yang sama.")
