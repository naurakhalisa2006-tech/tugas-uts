import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# 1. KONFIGURASI HALAMAN UNIK (Untuk estetika & layout)
# ==========================
st.set_page_config(
    page_title="Vision AI Dashboard - Clean/Messy Room",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menambahkan gaya kustom dengan CSS Injector untuk judul unik, keren, dan berwarna
st.markdown(
    """
    <style>
    /* Judul Utama: Keren & Berwarna */
    .big-font {
        font-size: 36px !important;
        font-weight: 800;
        color: #4a41d4; /* Deep Violet/Blue yang modern */
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.15); /* Efek keren */
        padding-top: 10px;
    }
    .main-title {
        text-align: center;
        margin-top: -20px;
        margin-bottom: 20px;
    }
    /* Mengatur nilai metrik agar lebih besar dan mudah dibaca */
    .stMetric > div[data-testid="stMetricValue"] {
        font-size: 36px;
        font-weight: bold;
    }
    /* Class Khusus untuk Hasil Klasifikasi Berwarna */
    .clean-result { color: #1e8449; } /* Hijau tua untuk 'Clean' */
    .messy-result { color: #c0392b; } /* Merah tua untuk 'Messy' */
    
    /* Tombol Upload */
    .stFileUploader {
        border: 2px dashed #9b59b6;
        border-radius: 10px;
        padding: 10px;
        background-color: #f8f8f8;
    }
    </style>
    """, unsafe_allow_html=True
)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Define paths
    yolo_path = "model/Siti Naura Khalisa_Laporan 4.pt"
    classifier_path = "model/SitiNauraKhalisa_Laporan2.h5"

    yolo_model = None
    classifier = None
    yolo_loaded = False
    classifier_loaded = False

    # Load YOLO model
    try:
        yolo_model = YOLO(yolo_path)
        yolo_loaded = True
    except Exception:
        pass # Error handled below

    # Load Keras Classification model
    try:
        classifier = tf.keras.models.load_model(classifier_path)
        classifier_loaded = True
    except Exception:
        pass # Error handled below
        
    return yolo_model, classifier, yolo_loaded, classifier_loaded

# Memanggil fungsi load_models dan mendapatkan status
yolo_model, classifier, yolo_loaded, classifier_loaded = load_models()

# !!! PERUBAHAN UTAMA UNTUK KLASIFIKASI "CLEAN/MESSY" !!!
CLASS_NAMES = ["Clean Room", "Messy Room"]

# ==========================
# UI - TAMPILAN UTAMA UNIK
# ==========================
st.markdown('<div class="main-title"><p class="big-font">🏡 Vision AI Dashboard: Ruangan Rapi atau Berantakan? 🧹</p></div>', unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #666;'>Aplikasi Terstruktur dan Rapi untuk Deteksi Objek dan Klasifikasi Kondisi Ruangan.</h4>", unsafe_allow_html=True)

# Garis pemisah yang lebih elegan (Rapi)
st.divider()


# --------------------------
# SIDEBAR (Rapi & Terstruktur)
# --------------------------
st.sidebar.header("⚙️ Pengaturan & Status")
menu = st.sidebar.selectbox("Pilih Fungsionalitas:", ["Klasifikasi Gambar (Clean/Messy)", "Deteksi Objek (YOLO)"])
st.sidebar.markdown("---")

# Visual feedback model status di sidebar (Mudah Dipahami)
st.sidebar.subheader("Status Model")
if yolo_loaded:
    st.sidebar.success("✅ YOLO Model (Deteksi) Siap")
else:
    st.sidebar.error("❌ YOLO Model GAGAL dimuat")

if classifier_loaded:
    st.sidebar.success("✅ Klasifikasi Model Siap")
else:
    st.sidebar.error("❌ Klasifikasi Model GAGAL dimuat")

st.sidebar.markdown("---")
st.sidebar.caption("Project UTS - Siti Naura Khalisa")
st.sidebar.info("💡 Pastikan file model berada di folder 'model/' dengan penamaan yang benar.")


# --------------------------
# UPLOADER SECTION
# --------------------------
uploaded_file = st.file_uploader("🖼️ Unggah Gambar Ruangan untuk Analisis", type=["jpg", "jpeg", "png"])

st.divider()

# Logika Tampilan
if uploaded_file is None:
    # Tampilan "Empty State" (Mudah Dipahami)
    st.markdown(
        """
        <div style="text-align: center; padding: 50px; border: 2px solid #ddd; border-radius: 10px;">
            <h3>⬆️ Mulai Analisis Anda!</h3>
            <p>Silakan unggah gambar ruangan (Clean atau Messy) di atas.</p>
            <p>Gunakan sidebar untuk beralih antara Klasifikasi dan Deteksi Objek.</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    
else:
    # --------------------------
    # Layout Utama: Gambar vs. Hasil (Terstruktur)
    # --------------------------
    img = Image.open(uploaded_file)
    col_img, col_result = st.columns([1, 1])
    
    with col_img:
        st.subheader("📸 Gambar Sumber Anda")
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    with col_result:
        st.subheader(f"🔍 Hasil: {menu}")
        
        # ==========================
        # KLASIFIKASI GAMBAR LOGIC
        # ==========================
        if menu == "Klasifikasi Gambar (Clean/Messy)":
            if classifier and classifier_loaded:
                st.info("Memulai proses Klasifikasi Kondisi Ruangan...")
                
                # Dynamic TARGET_SIZE calculation
                try:
                    input_shape = classifier.input_shape
                    TARGET_SIZE = (input_shape[1], input_shape[2])
                    if None in TARGET_SIZE or TARGET_SIZE[0] == 0:
                        TARGET_SIZE = (224, 224) 
                except Exception:
                    TARGET_SIZE = (224, 224)

                try:
                    # Preprocessing
                    img_resized = img.resize(TARGET_SIZE)
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) 
                    img_array = img_array / 255.0

                    # Prediksi
                    prediction = classifier.predict(img_array, verbose=0)
                    class_index = np.argmax(prediction)
                    
                    predicted_class = CLASS_NAMES[class_index]
                    confidence = np.max(prediction)
                    
                    st.success("🎉 Analisis Kondisi Ruangan Selesai!")
                    
                    # Logika untuk Tampilan Berwarna (Keren & Berwarna)
                    if "Clean" in predicted_class:
                        color_class = "clean-result"
                        icon = "✨"
                    else:
                        color_class = "messy-result"
                        icon = "⚠️"
                        
                    # Menampilkan Status Utama dengan warna
                    st.markdown(f'<h1 class="{color_class}">{icon} {predicted_class}</h1>', unsafe_allow_html=True)
                    
                    # Tampilan Metrik yang Rapi
                    col_met1, col_met2 = st.columns(2)
                    with col_met1:
                        st.metric(
                            label="🎯 Tingkat Keyakinan (Confidence)", 
                            value=f"{confidence * 100:.2f}%"
                        )
                    with col_met2:
                        st.metric(
                            label="Ukuran Input Model", 
                            value=f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
                        )
                    
                    # Tampilkan detail probabilitas lainnya dalam expander
                    with st.expander("📊 Lihat Probabilitas Penuh"):
                        prob_list = list(zip(CLASS_NAMES, prediction[0]))
                        prob_list.sort(key=lambda x: x[1], reverse=True)
                        st.table(
                            [
                                {"Kelas": name, "Probabilitas": f"{prob * 100:.2f}%"}
                                for name, prob in prob_list
                            ]
                        )
                        

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat Klasifikasi: {e}")
                    st.warning("Pastikan ukuran gambar yang diproses sesuai dengan model Anda.")
            else:
                st.warning("Model Klasifikasi Gambar tidak tersedia. Cek status di sidebar.")


        # ==========================
        # DETEKSI OBJEK LOGIC
        # ==========================
        elif menu == "Deteksi Objek (YOLO)":
            if yolo_model and yolo_loaded:
                st.info("Memulai proses Deteksi Objek dengan YOLO...")
                try:
                    # Deteksi objek
                    results = yolo_model(img, verbose=False)
                    result_img = results[0].plot()
                    
                    st.success("🎉 Deteksi Objek Selesai!")
                    
                    # Menampilkan gambar hasil deteksi di kolom yang sama
                    st.subheader("🖼️ Gambar Hasil Deteksi")
                    st.image(result_img, caption="Gambar dengan Bounding Box", use_container_width=True)
                    
                    # Tampilkan ringkasan deteksi
                    boxes = results[0].boxes
                    
                    st.metric(label="Jumlah Total Objek Ditemukan", value=len(boxes))
                    
                    if len(boxes) > 0:
                        st.info(f"✅ Ditemukan **{len(boxes)}** objek dalam gambar.")
                    else:
                        st.warning("Tidak ada objek yang terdeteksi. Coba gambar lain yang berisi objek.")
                        
                except Exception as e:
                    st.error(f"❌ Gagal menjalankan Deteksi Objek: {e}")
            else:
                st.warning("Model Deteksi Objek (YOLO) tidak tersedia. Cek status di sidebar.")
