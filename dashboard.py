import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2 # Untuk manipulasi gambar (Deteksi Objek)

# --- Pustaka Model (Diperlukan untuk memuat model nyata) ---
# Import ini diperlukan agar fungsi load_model dapat berjalan
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    st.warning("Pustaka TensorFlow/Keras tidak ditemukan. Fungsi klasifikasi akan disimulasikan.")

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    st.warning("Pustaka Ultralytics/PyTorch tidak ditemukan. Fungsi deteksi objek akan disimulasikan.")
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
    if not os.path.exists(path):
        st.error(f"File model YOLO tidak ditemukan: {path}")
        return None
    try:
        model = YOLO(path)
        st.success("✅ Model Deteksi Objek (YOLO) berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO dari file .pt: {e}")
        st.info("Logika deteksi objek akan disimulasikan.")
        return None

@st.cache_resource
def load_classification_model(path):
    """Memuat model Klasifikasi (Keras/TensorFlow)."""
    if not os.path.exists(path):
        st.error(f"File model Klasifikasi tidak ditemukan: {path}")
        return None
    try:
        # Pemuatan menggunakan Keras/TensorFlow
        model = keras.models.load_model(path)
        st.success("✅ Model Klasifikasi (Keras) berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model Klasifikasi dari file .h5: {e}")
        st.info("Logika klasifikasi akan disimulasikan.")
        return None

# Muat Model di awal
yolo_model = load_yolo_model(YOLO_MODEL_PATH)
classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)


# --- Fungsi Prediksi Simulasi (Jika model gagal dimuat) ---
def simulate_detection(image):
    """Simulasi deteksi objek dengan menggambar kotak acak."""
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    # Warna Merah untuk Berantakan
    color = (255, 0, 0) 
    thickness = 2
    
    # Simulasi 3 kotak deteksi
    cv2.rectangle(img_array, (w//5, h//5), (w//3, h//3), color, thickness)
    cv2.putText(img_array, 'Objek Berantakan', (w//5, h//5 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.rectangle(img_array, (2*w//3, h//4), (3*w//4, 2*h//3), color, thickness)
    cv2.putText(img_array, 'Pakaian', (2*w//3, h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.rectangle(img_array, (w//4, 3*h//4), (w//2, 4*h//5), color, thickness)
    cv2.putText(img_array, 'Buku', (w//4, 3*h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return Image.fromarray(img_array)

def simulate_classification():
    """Simulasi hasil klasifikasi acak."""
    # Memberikan hasil 'Messy' lebih sering untuk demonstrasi
    return np.random.choice(CLASS_LABELS, p=[0.4, 0.6]) 

# --- Halaman Deteksi Objek ---

def object_detection_page():
    st.title("Deteksi Objek di Ruangan 🔎")
    st.subheader("Fokus: Identifikasi Item yang Menyebabkan Kekacauan")

    uploaded_file = st.file_uploader(
        "Unggah Gambar Ruangan (untuk dideteksi itemnya)...", 
        type=["jpg", "jpeg", "png", "webp"],
        key="detection_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Gambar Ruangan yang Diunggah.', use_column_width=True)
        
        st.markdown("---")

        if st.button("Jalankan Deteksi Objek", key="run_detection"):
            with st.spinner('Sedang memproses deteksi objek...'):
                if yolo_model:
                    # Logika Prediksi Model YOLO Nyata
                    try:
                        # Resizing gambar agar tidak terlalu besar saat diproses oleh model
                        img_array = np.array(image.resize((640, 640))) 
                        
                        # Melakukan prediksi
                        results = yolo_model(img_array)
                        
                        # Mengambil gambar dengan kotak deteksi
                        annotated_img = results[0].plot() # Hasil plot adalah numpy array BGR
                        
                        # Konversi BGR ke RGB untuk Streamlit
                        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        
                        st.subheader("Hasil Deteksi Objek:")
                        st.image(annotated_img, caption="Hasil Deteksi YOLO", use_column_width=True)
                        st.success("Deteksi objek berhasil diselesaikan!")

                    except Exception as e:
                        st.warning(f"Terjadi kesalahan saat menjalankan prediksi YOLO: {e}. Menggunakan simulasi.")
                        # Fallback ke simulasi
                        result_image = simulate_detection(image.resize((600, 400))) 
                        st.image(result_image, caption="Hasil Deteksi (Simulasi)", use_column_width=True)
                        st.info("Prediksi disimulasikan karena model nyata gagal dieksekusi.")
                else:
                    # Logika Prediksi Simulasi
                    result_image = simulate_detection(image.resize((600, 400)))
                    st.subheader("Hasil Deteksi Objek (Simulasi):")
                    st.image(result_image, caption="Hasil Deteksi (Simulasi)", use_column_width=True)
                    st.info("Model Deteksi Objek tidak dimuat, menggunakan prediksi simulasi.")


# --- Halaman Klasifikasi Ruangan (Clean/Messy) ---

def classification_page():
    st.title("Klasifikasi Ruangan: Bersih atau Berantakan? 🧹")
    st.subheader("Fokus: Penilaian Keseluruhan Kebersihan Ruangan")

    uploaded_file = st.file_uploader(
        "Unggah Gambar Ruangan (untuk diklasifikasi kebersihannya)...", 
        type=["jpg", "jpeg", "png", "webp"],
        key="classification_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Gambar Ruangan yang Diunggah.', use_column_width=True)
        
        st.markdown("---")

        if st.button("Jalankan Klasifikasi", key="run_classification"):
            with st.spinner('Sedang melakukan klasifikasi ruangan...'):
                predicted_label = None

                if classification_model:
                    # Logika Prediksi Model Keras Nyata
                    try:
                        # Preprocessing gambar (sesuaikan dengan input training model Anda)
                        img = image.resize((224, 224)) # Contoh ukuran input Keras
                        img_array = keras.preprocessing.image.img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
                        img_array /= 255.0 # Normalisasi (jika model Anda menggunakan ini)

                        # Prediksi
                        predictions = classification_model.predict(img_array)
                        
                        # Ambil label hasil prediksi
                        predicted_index = np.argmax(predictions, axis=1)[0]
                        predicted_label = CLASS_LABELS[predicted_index]
                        predicted_confidence = predictions[0][predicted_index] * 100

                    except Exception as e:
                        st.warning(f"Terjadi kesalahan saat menjalankan prediksi Keras: {e}. Menggunakan simulasi.")
                        predicted_label = simulate_classification()
                        predicted_confidence = 85.0 # Nilai simulasi

                else:
                    # Logika Prediksi Simulasi
                    predicted_label = simulate_classification()
                    predicted_confidence = 85.0 # Nilai simulasi
                    st.info("Model Klasifikasi tidak dimuat, menggunakan prediksi simulasi.")


                # --- Menampilkan Hasil Klasifikasi ---
                st.subheader("Hasil Klasifikasi:")
                
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
