import streamlit as st
from PIL import Image
import io
import random
import numpy as np

# --- 0. Penyiapan Pustaka dan Penanganan Kesalahan ---

# Variabel global untuk model yang gagal dimuat
yolo_imported = False
cnn_imported = False

try:
    # Import untuk YOLO
    import torch
    from ultralytics import YOLO
    yolo_imported = True
    
    # Import untuk CNN Keras
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    cnn_imported = True
except ImportError as e:
    st.error(f"⚠️ Pustaka Machine Learning yang diperlukan (torch/ultralytics/tensorflow) tidak ditemukan. Beberapa fungsi akan menggunakan simulasi. Error: {e}")

# --- 1. Konfigurasi Halaman Streamlit (Aesthetic) ---

# Mengatur tampilan halaman dengan tema custom
st.set_page_config(
    page_title="Cute Vision AI: Aesthetic Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS untuk nuansa kawaii/ciwii
st.markdown("""
<style>
    /* Mengubah warna latar belakang dan teks Streamlit */
    .main {
        background: linear-gradient(135deg, #FFC7D4, #D4C7FF); /* Gradien ciwii-pink ke lavender */
        color: #52436D; /* ciwii-text */
    }
    h1, h2, h3, h4, .st-emotion-cache-10trblm { /* Header dan title */
        color: #52436D !important;
        text-shadow: 2px 2px 0px #FFD89C; /* Soft Peach shadow */
    }
    .stButton>button {
        background-color: #D4C7FF; /* ciwii-lavender */
        color: white;
        font-weight: bold;
        border: 2px solid #52436D;
        border-radius: 16px;
        box-shadow: 4px 4px 0px 0px #FFD89C;
        transition: all 0.1s;
    }
    .stButton>button:hover {
        background-color: #D4C7FF;
        box-shadow: 6px 6px 0px 0px #FFD89C;
        transform: translate(-1px, -1px);
    }
    .stButton>button:active {
        transform: translateY(4px);
        box-shadow: 0 0 #52436D;
    }
    .stFileUploader > div:first-child > div:first-child {
        border: 4px dashed #C7FFD1; /* ciwii-mint dashed border */
        border-radius: 20px;
        background-color: rgba(255, 255, 255, 0.7);
        padding: 30px;
    }
    /* Kotak Hasil */
    .ciwii-box {
        background-color: white;
        border: 3px solid #52436D;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 8px 8px 0px 0px #FFD89C;
        margin-top: 20px;
    }
    /* Chips */
    .ciwii-chip-clean {
        background-color: #C7FFD1; /* Soft Mint */
        color: #52436D;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 5px;
        font-size: 0.9em;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
    }
    .ciwii-chip-messy {
        background-color: #FFC7D4; /* Pastel Pink */
        color: #52436D;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 5px;
        font-size: 0.9em;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. Pemuatan Model (Menambahkan Fallback Aman) ---

@st.cache_resource
def load_yolo_model(model_path):
    """Memuat model Deteksi Objek YOLO (.pt)."""
    if not yolo_imported:
        st.warning("Model YOLO tidak dapat diimpor, menggunakan mode simulasi.")
        return None
        
    try:
        model_yolo = YOLO(model_path)
        st.success("✅ Model YOLO (.pt) berhasil dimuat.")
        return model_yolo
    except Exception as e:
        # Menangkap error seperti 'Ran out of input'
        st.error(f"❌ Gagal memuat model YOLO: {e}. Akan menggunakan mode simulasi.")
        return None

@st.cache_resource
def load_cnn_model(model_path):
    """Memuat model Klasifikasi Citra CNN (.h5)."""
    if not cnn_imported:
        st.warning("Model CNN tidak dapat diimpor, menggunakan mode simulasi.")
        return None
        
    try:
        model_cnn = load_model(model_path, compile=False) 
        st.success("✅ Model CNN (.h5) berhasil dimuat.")
        return model_cnn
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: {e}. Akan menggunakan mode simulasi.")
        return None

# Path file model yang telah diunggah
yolo_model_path = "model/Siti Naura Khalisa_Laporan 4.pt" 
cnn_model_path = "model/SitiNauraKhalisa_Laporan2.h5"

# Memuat model saat aplikasi Streamlit dimulai
yolo_model = load_yolo_model(yolo_model_path)
cnn_model = load_cnn_model(cnn_model_path)
    
# --- 3. Data Konstan dan Logika Analisis ---

# Data Simulasi (Gunakan nama kelas dari pelatihan model Anda)
MESSY_CLASSES = ["Baju Kotor", "Kabel Berantakan", "Piring Bekas", "Sampah Kertas", "Kaos Kaki Hilang", "Sprei Kusut", "Buku Tergeletak"]
CLEAN_CLASSES = ["Karpet Rapi", "Meja Bersih", "Tanaman Hias", "Buku Tersusun", "Kaca Mengkilap", "Bantal Tersusun", "Lilin Aromaterapi"]

def run_yolo_detection(image: Image.Image, model: YOLO):
    """Fungsi untuk menjalankan model YOLO (.pt) atau mode simulasi."""
    
    # 1. Fallback / Mode Simulasi jika model gagal dimuat
    if model is None:
        st.warning("⚠️ MODE SIMULASI: YOLO Gagal Dimuat. Hasil deteksi disimulasikan.")
        # Simulasikan deteksi dengan campuran objek
        detected_objects = random.sample(MESSY_CLASSES + CLEAN_CLASSES, k=random.randint(3, 7))
        return list(set(detected_objects))
        
    # 2. Inferensi Model Nyata
    st.info("YOLO: Sedang melakukan Deteksi Objek...")
    try:
        results = model(image, conf=0.4, iou=0.5, verbose=False) 
        
        detected_objects = []
        for result in results:
            class_names = result.names
            for cls_index in result.boxes.cls.tolist():
                name = class_names.get(int(cls_index), f"Unknown Class {int(cls_index)}")
                detected_objects.append(name)

        detected_objects = list(set(detected_objects))
        
        if not detected_objects:
            detected_objects = ["Tidak ada objek spesifik terdeteksi"]
            
        return detected_objects
        
    except Exception as e:
        st.error(f"❌ Kesalahan saat inferensi YOLO: {e}. Menggunakan Simulasi.")
        # Fallback ke mock logic jika inferensi gagal
        detected_objects = random.sample(MESSY_CLASSES + CLEAN_CLASSES, k=random.randint(3, 7))
        return list(set(detected_objects))


def run_cnn_classification(image: Image.Image, model: tf.keras.Model):
    """Fungsi untuk menjalankan model Klasifikasi Citra CNN (.h5) atau mode simulasi."""
    
    # 1. Fallback / Mode Simulasi jika model gagal dimuat
    if model is None:
        st.warning("⚠️ MODE SIMULASI: CNN Gagal Dimuat. Hasil klasifikasi disimulasikan.")
        final_category = random.choice(["Clean", "Messy"])
        confidence_str = f"{random.uniform(70.0, 99.9):.1f}% Confidence (Simulated)"
        return final_category, confidence_str
        
    # 2. Inferensi Model Nyata
    st.info("CNN: Sedang melakukan Klasifikasi Ruangan...")
    try:
        # Asumsi: Model Keras Anda dilatih dengan input 224x224, 3 channel (RGB)
        target_size = (224, 224) 
        
        # Preprocessing gambar
        img = image.resize(target_size) 
        img_array = np.array(img, dtype='float32') 
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0 
        
        # Kelas output yang diasumsikan
        class_labels = ["Clean", "Messy"] 
        
        # Inferensi
        prediction = model.predict(img_array, verbose=0)
        
        # Ambil index kelas dengan probabilitas tertinggi
        predicted_index = np.argmax(prediction[0])
        confidence_value = prediction[0][predicted_index]
        
        final_category = class_labels[predicted_index]
        confidence_str = f"{confidence_value * 100:.1f}% Confidence"
        
        return final_category, confidence_str

    except Exception as e:
        st.error(f"❌ Kesalahan saat inferensi CNN: {e}. Menggunakan Simulasi.")
        # Fallback ke mock logic jika inferensi gagal
        final_category = random.choice(["Clean", "Messy"])
        confidence_str = f"{random.uniform(70.0, 99.9):.1f}% Confidence (Simulated Fallback)"
        return final_category, confidence_str

def combine_results(yolo_results, cnn_category):
    """Menggabungkan hasil YOLO dan CNN untuk rekomendasi akhir."""
    # Hitung berapa banyak objek berantakan yang ditemukan oleh YOLO
    messy_count = sum(1 for obj in yolo_results if obj in MESSY_CLASSES)
    
    # Tentukan kategori rekomendasi dari gabungan kedua model
    if cnn_category == "Clean" and messy_count <= 2:
        # Konsisten: CNN Clean, YOLO sedikit/tidak berantakan
        return "Clean", messy_count
    elif cnn_category == "Messy" and messy_count >= 3:
        # Konsisten: CNN Messy, YOLO banyak berantakan
        return "Messy", messy_count
    elif messy_count >= 3 and cnn_category == "Clean":
        # Konflik: CNN bilang bersih, tapi YOLO menemukan banyak sampah (Prioritas YOLO untuk item spesifik)
        return "Messy", messy_count
    else:
        # Konflik atau Clean sejati (misal: CNN Messy tapi YOLO hanya deteksi 1-2 item)
        # Ambil keputusan berdasarkan CNN jika YOLO tidak jelas
        return cnn_category, messy_count


# --- 4. Tampilan Utama Aplikasi Streamlit ---

def main():
    # Header
    st.markdown(f"""
        <div style="text-align: center;">
            <h1 style="font-size: 3.5em; font-weight: 800; line-height: 1.1;">
                <span style="display: inline-block; transform: rotate(6deg); color: #FFC7D4;">🎀</span> 
                Cute Room Analyzer 
                <span style="display: inline-block; transform: rotate(-6deg); color: #D4C7FF;">✨</span>
            </h1>
            <p style="font-size: 1.2em; margin-top: -10px;">Magic untuk Analisis Ruangan Bersih vs Berantakan!</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Drop foto ruanganmu di sini atau klik untuk upload! (Max 5MB, JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Memuat gambar
        image = Image.open(uploaded_file).convert('RGB') # Pastikan format RGB untuk CNN
        
        # Tampilkan gambar yang diunggah
        st.subheader("📸 Ruanganmu")
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

        st.markdown('<div class="ciwii-box">', unsafe_allow_html=True)
        st.subheader("🧠 AI Sedang Berpikir...")
        
        # Tombol Analisis
        if st.button("✨ Mulai Analisis Ciwii!"):
            st.session_state['analysis_done'] = False
            with st.spinner("Memproses dengan Dual Model (YOLO dan CNN)..."):
                
                # 1. Deteksi Objek (YOLO)
                yolo_results = run_yolo_detection(image, yolo_model)
                
                # 2. Klasifikasi Gambar (CNN)
                cnn_category, confidence_str = run_cnn_classification(image, cnn_model)
                
                # 3. Gabungkan Hasil
                final_category, messy_count = combine_results(yolo_results, cnn_category)

                # Simpan hasil di session state untuk ditampilkan
                st.session_state['yolo_results'] = yolo_results
                st.session_state['cnn_category'] = cnn_category
                st.session_state['confidence'] = confidence_str
                st.session_state['final_category'] = final_category
                st.session_state['messy_count'] = messy_count
                st.session_state['analysis_done'] = True
                
        st.markdown('</div>', unsafe_allow_html=True)


    # Tampilkan Hasil Analisis
    if st.session_state.get('analysis_done', False):
        yolo_results = st.session_state['yolo_results']
        cnn_category = st.session_state['cnn_category']
        confidence_str = st.session_state['confidence']
        final_category = st.session_state['final_category']
        messy_count = st.session_state['messy_count']

        st.markdown("---")
        st.markdown('<div class="ciwii-box">', unsafe_allow_html=True)

        # --- Hasil Deteksi Objek (YOLO) ---
        st.markdown("<h3>🔎 Deteksi Objek YOLO</h3>", unsafe_allow_html=True)
        
        chips_html = ""
        for item in yolo_results:
            is_messy = item in MESSY_CLASSES
            chip_class = "ciwii-chip-messy" if is_messy else "ciwii-chip-clean"
            icon = "⚠️" if is_messy else "✨"
            chips_html += f'<span class="{chip_class}">{icon} {item}</span>'
            
        st.markdown(f'<div style="display: flex; flex-wrap: wrap;">{chips_html}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- Hasil Klasifikasi Ruangan (CNN) & Rekomendasi ---
        
        if final_category == "Clean":
            emoji = "👑"
            category_text = "Ruangan Bersih Sempurna"
            recommendation = f"Pertahankan kebersihannya! Ruanganmu mencerminkan kedamaian. Ditemukan {messy_count} item yang sedikit berantakan. Sekarang, tambahkan satu barang baru yang lucu!"
            box_style = "background-color: #C7FFD1; border: 4px solid #52436D; border-radius: 20px; padding: 20px; box-shadow: 6px 6px 0px 0px #FFC7D4;"
        else:
            emoji = "🚨"
            category_text = "Ruangan Berantakan Parah"
            recommendation = f"Yuk, mari kita mulai beres-beres! Ditemukan {messy_count} item berantakan (seperti yang ditunjukkan di atas). Fokus pada 5 barang teratas yang terdeteksi. Kamu pasti bisa!"
            box_style = "background-color: #FFC7D4; border: 4px solid #52436D; border-radius: 20px; padding: 20px; box-shadow: 6px 6px 0px 0px #C7FFD1;"

        
        st.markdown(f"""
            <div style="{box_style}">
                <h3 style="font-size: 2em; font-weight: 800; color: #52436D; text-shadow: none;">
                    {emoji} Kategori: {category_text}
                </h3>
                <p style="font-size: 1.1em; margin-top: 10px; font-weight: 600;">Confidence: {confidence_str} (dari CNN)</p>
                
                <h4 style="margin-top: 20px; font-size: 1.3em; font-weight: 700; color: #52436D; text-shadow: none;">
                    🌟 Tips Ciwii Beres-Beres!
                </h4>
                <p>{recommendation}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# Jalankan fungsi utama
if __name__ == '__main__':
    # Inisialisasi session state
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False
        
    main()

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding-top: 20px; color: #52436D80; font-size: 0.8em;">
    <p>Dibangun menggunakan Streamlit. Memuat model <code>Siti Naura Khalisa_Laporan 4.pt</code> (YOLO) dan <code>SitiNauraKhalisa_Laporan2.h5</code> (CNN). Jika model gagal dimuat, aplikasi akan menggunakan Mode Simulasi.</p>
</div>
""", unsafe_allow_html=True)
