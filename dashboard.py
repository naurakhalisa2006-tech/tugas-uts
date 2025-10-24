import streamlit as st
import numpy as np
from PIL import Image

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
@st.cache_resource(show_spinner="Memuat model AI...")
def load_models():
    # Pindahkan impor library berat ke sini untuk menghindari konflik Import saat start up
    # Kita hanya mencoba mengimpor jika kita berada di lingkungan yang mendukungnya
    try:
        # Impor library ML utama
        from ultralytics import YOLO
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image
        # Impor cv2 secara eksplisit karena digunakan untuk konversi warna setelah YOLO
        import cv2 
    except ImportError as e:
        # Jika salah satu library utama gagal, kembalikan status gagal
        return None, None, False, False, f"ImportError: {e}. Pastikan pustaka seperti 'ultralytics' dan 'tensorflow' sudah terinstal."

    # Define paths
    yolo_path = "model/Siti Naura Khalisa_Laporan 4.pt"
    classifier_path = "model/SitiNauraKhalisa_Laporan2.h5"

    yolo_model = None
    classifier = None
    yolo_loaded = False
    classifier_loaded = False
    error_message = None

    # Load YOLO model
    try:
        yolo_model = YOLO(yolo_path)
        yolo_loaded = True
    except FileNotFoundError:
        error_message = "❌ YOLO Model GAGAL dimuat: File '.pt' tidak ditemukan di folder 'model/'."
    except Exception as e:
        error_message = f"❌ YOLO Model GAGAL dimuat: Error saat memuat YOLO: {e}"

    # Load Keras Classification model
    try:
        # Menghilangkan pesan warning TF
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

        classifier = tf.keras.models.load_model(classifier_path)
        classifier_loaded = True
    except FileNotFoundError:
        if not error_message: error_message = "❌ Klasifikasi Model GAGAL dimuat: File '.h5' tidak ditemukan di folder 'model/'."
    except Exception as e:
        if not error_message: error_message = f"❌ Klasifikasi Model GAGAL dimuat: Error saat memuat Keras: {e}"
        
    # Mengembalikan model dan status
    return yolo_model, classifier, yolo_loaded, classifier_loaded, error_message

# Memanggil fungsi load_models dan mendapatkan status
yolo_model, classifier, yolo_loaded, classifier_loaded, model_error = load_models()

# Jika model berhasil dimuat, kita perlu mengimpor modul yang dibutuhkan dari TensorFlow (di luar @st.cache_resource)
# Ini diperlukan agar Streamlit dapat mengakses modul preprocessing
keras_image = None
tf_loaded = False
cv2 = None 
if classifier_loaded or yolo_loaded:
    try:
        from tensorflow.keras.preprocessing import image as keras_image
        import tensorflow as tf
        # Mengimpor cv2 lagi secara global jika berhasil di 'load_models' untuk digunakan di YOLO
        try:
            import cv2 
        except ImportError:
            pass # Lanjutkan tanpa cv2 jika gagal
        tf_loaded = True
    except ImportError:
        keras_image = None
        tf_loaded = False

# !!! KLASIFIKASI "CLEAN/MESSY" !!!
# Urutan: Index 0 = Messy Room, Index 1 = Clean Room.
CLASS_NAMES = ["Messy Room", "Clean Room"]

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
menu = st.sidebar.selectbox("Pilih Fungsionalitas:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar (Clean/Messy)"])
st.sidebar.markdown("---")

# Visual feedback model status di sidebar (Mudah Dipahami)
st.sidebar.subheader("Status Model")
if yolo_loaded:
    st.sidebar.success("✅ YOLO Model (Deteksi) Siap")
else:
    st.sidebar.error("❌ YOLO Model (Deteksi) GAGAL")

if classifier_loaded:
    st.sidebar.success("✅ Klasifikasi Model Siap")
else:
    st.sidebar.error("❌ Klasifikasi Model GAGAL")

if model_error:
    st.sidebar.caption("Detail Error:")
    st.sidebar.exception(Exception(model_error.split(": ")[-1])) # Tampilkan hanya bagian error yang relevan


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
            if classifier and classifier_loaded and keras_image:
                st.info("Memulai proses Klasifikasi Kondisi Ruangan...")
                
                # Dynamic TARGET_SIZE calculation
                try:
                    # Ambil input shape dari model (Bisa jadi (None, H, W, C))
                    input_shape = classifier.input_shape
                    TARGET_SIZE = (input_shape[1], input_shape[2])
                    # Jika shape tidak terdefinisi, gunakan default yang umum (224x224)
                    if None in TARGET_SIZE or TARGET_SIZE[0] == 0:
                        TARGET_SIZE = (224, 224)  
                except Exception:
                    # Default jika gagal mendapatkan shape dari model
                    TARGET_SIZE = (224, 224)

                try:
                    # Preprocessing
                    img_resized = img.resize(TARGET_SIZE)
                    # Menggunakan modul yang sudah diimpor jika berhasil
                    img_array = keras_image.img_to_array(img_resized) 
                    img_array = np.expand_dims(img_array, axis=0) 
                    img_array = img_array / 255.0 # Normalisasi (sesuai asumsi training)

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
                    # --- LOGIKA DEBUG INPUT SHAPE ---
                    expected_flattened_size = 'Unknown'
                    received_flattened_size = 'Unknown'

                    try:
                        # Mencoba menemukan ukuran input yang diharapkan oleh lapisan Dense 
                        for layer in classifier.layers:
                            if 'dense' in layer.name:
                                # Mengambil ukuran input (sebelumnya sudah di-flatten)
                                expected_flattened_size = layer.input_shape[1] 
                                break
                        
                        # Mencoba menghitung ukuran input yang diterima (H * W * C)
                        received_flattened_size = TARGET_SIZE[0] * TARGET_SIZE[1] * 3 
                    except Exception:
                        pass
                        
                    st.error(f"❌ Terjadi kesalahan saat Klasifikasi: {e}")
                    st.warning(f"""
                        **⚠️ Kesalahan Kritis (Input Shape Mismatch):**
                        Model Keras Anda mengharapkan input yang menghasilkan **{expected_flattened_size}** fitur 
                        pada lapisan Dense/Input.
                        
                        Namun, gambar yang di-*resize* ke **{TARGET_SIZE[0]}x{TARGET_SIZE[1]}** mungkin menghasilkan jumlah fitur berbeda (**{received_flattened_size}**).
                        
                        **Solusi:** Silakan ganti nilai `TARGET_SIZE` di kode Anda 
                        (saat ini dihitung sebagai {TARGET_SIZE[0]}x{TARGET_SIZE[1]}) dengan ukuran 
                        yang Anda gunakan saat melatih model `SitiNauraKhalisa_Laporan2.h5`. (Misalnya 224x224 atau 150x150).
                        """)
            else:
                st.warning("Model Klasifikasi Gambar tidak tersedia. Cek status di sidebar. Detail Error: " + (model_error if model_error else "Library TensorFlow/Keras gagal dimuat."))


        # ==========================
        # DETEKSI OBJEK LOGIC
        # ==========================
        elif menu == "Deteksi Objek (YOLO)":
            if yolo_model and yolo_loaded:
                st.info("Memulai proses Deteksi Objek dengan YOLO...")
                try:
                    # Deteksi objek
                    results = yolo_model(img, verbose=False)
                    # result_img adalah numpy array (BGR)
                    result_img = results[0].plot()
                    
                    st.success("🎉 Deteksi Objek Selesai!")
                    
                    # Menampilkan gambar hasil deteksi di kolom yang sama
                    st.subheader("🖼️ Gambar Hasil Deteksi")
                    # Mengkonversi dari BGR (output YOLO plot) ke RGB untuk Streamlit
                    if cv2 is not None:
                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        st.image(result_img_rgb, caption="Gambar dengan Bounding Box", use_container_width=True)
                    else:
                        # Fallback jika cv2 gagal diimpor
                        st.image(result_img, caption="Gambar dengan Bounding Box (Mungkin Terdapat Perubahan Warna)", use_container_width=True)

                    
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
                st.warning("Model Deteksi Objek (YOLO) tidak tersedia. Cek status di sidebar. Detail Error: " + (model_error if model_error else "Library Ultralytics gagal dimuat."))
