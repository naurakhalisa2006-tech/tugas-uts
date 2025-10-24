import streamlit as st
import random
import time
import json
import io
import os
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pandas as pd
import numpy as np

# --- 1. Import Pustaka Machine Learning (Hanya akan berfungsi jika diinstal) ---
try:
    from ultralytics import YOLO
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    ML_LIBRARIES_LOADED = True
except ImportError:
    # st.warning("Pustaka 'ultralytics' atau 'tensorflow' tidak ditemukan. Aplikasi akan berjalan dalam mode SIMULASI yang ditingkatkan.") # Dihapus agar tidak konflik
    ML_LIBRARIES_LOADED = False

# --- 2. Konfigurasi dan Styling (Tema Cute Vision AI / Girly Pastel) ---

st.set_page_config(
    page_title="ROOM INSIGHT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Palet Warna Cute Vision AI
BG_LIGHT = "#F0F0FF"            # Very Light Lavender/White
CARD_BG = "#FFFFFF"             # Soft White Card Background
TEXT_DARK = "#333333"           # Main text color (Dark)
ACCENT_PRIMARY_PINK = "#FF88AA" # Soft Coral Pink (Main Accent)
ACCENT_PURPLE = "#AA88FF"       # Pastel Purple/Lilac (Clean Status)
ACCENT_PINK_MESSY = "#FF3366"   # Hot Pink/Fuschia (Messy Status)
BUTTON_COLOR_SOFT = "#FFB3D9"   # Soft Pink Button BG

# TAMBAHAN BARU: Light Blue Glow sesuai permintaan
ACCENT_LIGHT_BLUE = "#88AAFF" # Soft, light sky blue/periwinkle

TEXT_CLEAN_STATUS = ACCENT_PURPLE
TEXT_MESSY_STATUS = ACCENT_PINK_MESSY

custom_css = f"""
<style>
    /* Definisi Keyframe untuk Efek Soft Glow di Tombol */
    @keyframes soft-glow {{
        0% {{
            box-shadow: 0 0 5px {ACCENT_PRIMARY_PINK}, 0 0 10px {ACCENT_PRIMARY_PINK};
        }}
        50% {{
            box-shadow: 0 0 8px {ACCENT_PRIMARY_PINK}, 0 0 15px {ACCENT_PRIMARY_PINK};
        }}
        100% {{
            box-shadow: 0 0 5px {ACCENT_PRIMARY_PINK}, 0 0 10px {ACCENT_PRIMARY_PINK};
        }}
    }}

    /* --- KEYFRAME BARU: SOFT CORAL PINK & LIGHT BLUE GLOW --- */
    /* Menggunakan ACCENT_LIGHT_BLUE sebagai glow sekunder */
    @keyframes soft-coral-glow {{
        0%, 100% {{
            /* Soft Coral Pink shadow + Light Blue glow */
            text-shadow: 0 0 7px {ACCENT_PRIMARY_PINK}, 0 0 15px {ACCENT_PRIMARY_PINK}, 0 0 25px {ACCENT_LIGHT_BLUE};
            color: {TEXT_DARK};
            opacity: 0.9;
        }}
        50% {{
            /* Brighter glow at midpoint, color pulses to the pink accent */
            text-shadow: 0 0 10px {ACCENT_PRIMARY_PINK}, 0 0 25px {ACCENT_PRIMARY_PINK}, 0 0 40px {ACCENT_LIGHT_BLUE};
            color: {ACCENT_PRIMARY_PINK}; 
            opacity: 1;
        }}
    }}

    /* Keyframe Glitch untuk Efek Hover */
    @keyframes glitch {{
        0% {{ transform: translate(0); }}
        20% {{ transform: translate(-2px, 2px); opacity: 0.9; }}
        40% {{ transform: translate(-1px, -1px); opacity: 0.85; }}
        60% {{ transform: translate(3px, 1px); opacity: 0.92; }}
        80% {{ transform: translate(1px, -2px); opacity: 0.88; }}
        100% {{ transform: translate(0); }}
    }}

    /* --- PERUBAHAN UTAMA DI SINI --- */
    .main-title {{
        color: {TEXT_DARK}; 
        font-size: 5rem; /* UKURAN DIPERBESAR DAN DIKOREKSI (50rem terlalu besar) */
        font-weight: 900;
        letter-spacing: 5px;
        text-transform: uppercase;
        margin-bottom: 0px;
        position: relative;
        text-align: center;
        /* Menggunakan animasi SOFT CORAL GLOW BARU */
        animation: soft-coral-glow 2s ease-in-out infinite alternate;
        transition: color 0.3s;
    }}

    .main-title:hover {{
        color: {ACCENT_PRIMARY_PINK}; /* Ganti warna saat hover */
        animation: glitch 0.2s linear infinite; /* Tambah glitch saat hover */
    }}
    
    .subtitle-center {{
        color: {ACCENT_PRIMARY_PINK};
        font-size: 1.2rem; /* UKURAN SUBTITLE TETAP KECIL */
        margin-top: 5px;
        text-align: center;
        font-weight: 600;
        padding-bottom: 30px;
        border-bottom: 3px solid {ACCENT_PRIMARY_PINK};
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }}
    
    /* --- END ANIMASI JUDUL BARU --- */

    .stApp {{
        background-color: {BG_LIGHT};
        color: {TEXT_DARK};
        font-family: 'Inter', sans-serif;
    }}
    /* H1 default Streamlit disembunyikan agar tidak konflik dengan .main-title */
    h1 {{
        display: none; 
    }}
    .modern-card {{
        background-color: {CARD_BG};
        border: 1px solid {ACCENT_PRIMARY_PINK};
        box-shadow: 0 4px 15px rgba(255, 136, 170, 0.4);
        border-radius: 18px; /* Lebih rounded */
        padding: 25px;
        margin-bottom: 25px;
    }}
    h2 {{
        color: {ACCENT_PRIMARY_PINK};
        border-bottom: 1px solid #DDDDDD;
        padding-bottom: 5px;
    }}
    
    /* GAYA UPLOADER CUTE */
    [data-testid="stFileUploader"] {{
        min-height: 200px; 
        padding: 15px 25px 15px 25px !important;
        margin-top: 15px;
        border: 3px dashed {ACCENT_PRIMARY_PINK} !important; 
        border-radius: 18px;
        background-color: {CARD_BG} !important; 
        pointer-events: auto !important;
        box-shadow: 0 2px 10px rgba(255, 136, 170, 0.3);
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] p, [data-testid="stFileUploader"] label {{
        color: {TEXT_DARK} !important;
        font-weight: 500;
    }}
    
    /* GAYA TOMBOL 'BROWSE FILES' DI DALAM UPLOADER */
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {ACCENT_PRIMARY_PINK} !important;
        color: {CARD_BG} !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 10px !important;
        transition: all 0.3s !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2) !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: {ACCENT_PINK_MESSY} !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    }}
    
    /* Tombol Initiate Dual-Model Analysis (Tombol utama non-uploader) */
    .stButton > button:nth-child(1) {{
        background-color: {BUTTON_COLOR_SOFT}; 
        color: {TEXT_DARK} !important;
        font-weight: bold;
        border-radius: 15px;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(255, 179, 217, 0.6);
        height: 55px;
        font-size: 18px;
        animation: soft-glow 3s infinite alternate;
    }}
    .stButton > button:nth-child(1):hover {{
        background-color: {ACCENT_PRIMARY_PINK};
        color: {CARD_BG} !important;
        box-shadow: 0 6px 15px rgba(255, 179, 217, 0.8);
    }}

    /* Tombol Kembali (Return Button) */
    .stButton > button[data-testid="stElement" i]:not(:nth-child(1)) {{
        background-color: {CARD_BG}; 
        color: {ACCENT_PRIMARY_PINK} !important;
        border: 2px solid {ACCENT_PRIMARY_PINK} !important;
        box-shadow: 0 0 5px {ACCENT_PRIMARY_PINK};
        border-radius: 10px;
        margin-top: 20px;
    }}

    .status-metric-card {{
        background-color: {CARD_BG};
        border: 2px solid #EEEEEE;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }}
    
    .clean-status-text {{ color: {TEXT_CLEAN_STATUS}; font-weight: 900; font-size: 32px; text-shadow: 0 0 2px {TEXT_CLEAN_STATUS}; }}
    .messy-status-text {{ color: {TEXT_MESSY_STATUS}; font-weight: 900; font-size: 32px; text-shadow: 0 0 2px {TEXT_MESSY_STATUS}; }}
    
    .clean-border {{ border-color: {ACCENT_PURPLE} !important; border-width: 4px !important; box-shadow: 0 0 15px rgba(170, 136, 255, 0.6) !important; }}
    .messy-border {{ border-color: {ACCENT_PINK_MESSY} !important; border-width: 4px !important; box-shadow: 0 0 15px rgba(255, 51, 102, 0.6) !important; }}
    
    .tips-box {{
        padding: 25px;
        border-radius: 18px;
        margin-top: 30px;
        margin-bottom: 30px;
        border-left: 5px solid;
    }}

    .tips-box h3 {{
        color: inherit !important;
        font-weight: bold;
        padding-bottom: 5px;
        border-bottom: none;
    }}
    .tips-box ul {{
        list-style-type: '💖 '; /* Cute List Style */
        padding-left: 20px;
        color: {TEXT_DARK};
    }}
    p {{ color: {TEXT_DARK}; }}

    /* Menyembunyikan log container karena tidak digunakan lagi */
    .log-container {{ display: none; }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Inisialisasi State Session
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'execution_log_data' not in st.session_state:
    # Cukup set string placeholder karena log tidak ditampilkan
    st.session_state.execution_log_data = "Log display is disabled."
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'UPLOAD' # Mengelola navigasi

# --- 3. Fungsi Pemuatan Model (Menggunakan Cache Resource Streamlit) ---
@st.cache_resource
def load_ml_model():
    """Memuat model YOLO dan CNN ke memori dengan caching."""
    # Hanya mencoba memuat jika pustaka berhasil diimpor
    if not ML_LIBRARIES_LOADED:
        return None, None
    
    try:
        # PENTING: Ganti path ini jika lokasi file Anda berbeda
        YOLO_MODEL_PATH = "model/Siti Naura Khalisa_Laporan 4.pt"
        CNN_MODEL_PATH = "model/SitiNauraKhalisa_Laporan2.h5"

        # Model 1: Deteksi Objek (YOLOv8)
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # Model 2: Klasifikasi Ruangan (Keras/CNN)
        # Suppress TensorFlow warnings/messages
        tf.get_logger().setLevel('ERROR')
        cnn_model = load_model(CNN_MODEL_PATH)
        
        return yolo_model, cnn_model
    except Exception as e:
        # Mengubah ini menjadi error eksplisit
        st.error(f"FATAL ERROR: Gagal memuat file model dari path. Pastikan file berada di folder 'model/' dan namanya benar: {e}")
        return None, None

# --- 4. Fungsi Real Inference dan Pemrosesan Data (TETAP SAMA) ---

def run_yolo_detection(yolo_model, image_path):
    """Menjalankan inferensi YOLOv8 dan memproses hasilnya."""
    
    # Menjalankan prediksi (Mode 'save=False' dan 'conf' dapat disesuaikan)
    results = yolo_model.predict(
        source=image_path, 
        conf=0.25, # Ambang batas kepercayaan minimum
        iou=0.7,   # Ambang batas IOU
        verbose=False,
        save=False
    )
    
    # Ambil hasil dari batch pertama (image_path tunggal)
    result = results[0] 
    
    detections = []
    messy_count = 0
    
    # Mendefinisikan ID Kelas yang dianggap "Messy" (Ganti sesuai nama kelas model Anda)
    MESSY_CLASS_NAMES = ["Scattered_Clothes", "Loose_Cables", "Unsorted_Papers", "Trash_Object", "Empty_Bottles", "Food_Wrapper"]
    class_names = result.names # Peta ID ke nama kelas
    
    
    for box in result.boxes:
        # Koordinat bounding box (normalized, xywh format)
        x_norm, y_norm, w_norm, h_norm = box.xywhn[0].tolist() 
        conf = box.conf.item() 
        class_id = box.cls.item()
        label = class_names[int(class_id)]
        
        is_messy_item = label in MESSY_CLASS_NAMES
        classification_tag = 'UNOPTIMIZED' if is_messy_item else 'STRUCTURED'
        
        if is_messy_item:
            messy_count += 1
            
        detections.append({
            "asset_id": label.upper().replace('_', '-'),
            "confidence_score": round(conf, 4),
            "classification_tag": classification_tag,
            "normalized_coordinates": [
                round(x_norm, 4), 
                round(y_norm, 4),
                round(w_norm, 4),
                round(h_norm, 4)
            ]
        })

    return detections, messy_count

def run_cnn_classification(cnn_model, image_bytes):
    """Menjalankan klasifikasi CNN pada gambar mentah."""
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Perbaikan ukuran gambar untuk menghindari shape mismatch pada model Keras/CNN
    target_size = (128, 128) 
    image_resized = image.resize(target_size)
    
    # Konversi ke array Numpy dan normalisasi (misalnya 0-1)
    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array = img_array / 255.0 # Normalisasi
    
    # Prediksi
    predictions = cnn_model.predict(img_array, verbose=0)[0]
    
    # Asumsikan output adalah (Conf_Clean, Conf_Messy). Periksa indeks ini
    # Ganti urutan indeks ini (0 atau 1) sesuai dengan urutan kelas model CNN Anda
    conf_clean = predictions[0]  # Ganti indeks ini jika CLEAN bukan kelas 0
    conf_messy = predictions[1]  # Ganti indeks ini jika MESSY bukan kelas 1
    
    return conf_clean, conf_messy

# --- 5. Fungsi Utilitas Visualisasi (format_execution_log dihapus) ---

def draw_boxes_on_image(image_bytes, detections):
    """Menggambar Bounding Box Neon pada Gambar dari hasil deteksi."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    
    # Menggunakan warna tema baru
    CLEAN_RGB = ImageColor.getrgb(ACCENT_PURPLE)
    MESSY_RGB = ImageColor.getrgb(ACCENT_PINK_MESSY)
    TEXT_BG_RGB = ImageColor.getrgb(CARD_BG) # Teks hitam di atas kotak putih/terang
    
    try:
        font_size = max(15, min(image_width // 40, 30))
        font = ImageFont.load_default(size=font_size) 
    except IOError:
        font = ImageFont.load_default()
        
    for det in detections:
        # Koordinat Normalized (x_center, y_center, w, h) dari YOLO
        x_norm, y_norm, w_norm, h_norm = det['normalized_coordinates']
        
        # Konversi YOLO xywhn ke PIXEL xyxy
        x_min = int((x_norm - w_norm/2) * image_width)
        y_min = int((y_norm - h_norm/2) * image_height)
        x_max = int((x_norm + w_norm/2) * image_width)
        y_max = int((y_norm + h_norm/2) * image_height)
        
        # Clamp koordinat
        x_max = min(x_max, image_width - 1)
        y_max = min(y_max, image_height - 1)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        
        label = det['asset_id']
        confidence = det['confidence_score']
        
        box_rgb = CLEAN_RGB if det['classification_tag'] == 'STRUCTURED' else MESSY_RGB
        
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_rgb, width=4) 
        
        text_content = f"{label} [{int(confidence * 100)}%]"
        
        try:
            text_bbox = draw.textbbox((0, 0), text_content, font=font) 
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text_content, font=font)
            
        text_x = x_min
        text_y = y_min - text_height - 5 
        
        if text_y < 0:
            text_y = y_max + 5
            
        draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill=box_rgb)
        draw.text((text_x + 2, text_y + 2), text_content, font=font, fill=TEXT_BG_RGB) 
        
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    return buf.getvalue()

# Fungsi format_execution_log dihapus

# --- 6. Fungsi Utama Alur Kerja ML (Disesuaikan untuk navigasi) ---

def run_ml_analysis():
    """Menjalankan analisis ML nyata dan beralih ke halaman report."""
    
    # PERHATIAN: Periksa status pustaka sebelum melanjutkan
    if not ML_LIBRARIES_LOADED:
        st.error("Analisis dibatalkan: Pustaka Machine Learning gagal dimuat saat startup.")
        return

    if st.session_state.uploaded_file is None:
        st.error("Sila muat naik imej ruangan dahulu.")
        return
    
    # 1. Muat Model
    yolo_model, cnn_model = load_ml_model()
    
    # PERHATIAN: Periksa status pemuatan model
    if yolo_model is None or cnn_model is None:
        st.error("Analisis dibatalkan: File Model gagal dimuat dari direktori.")
        return

    # Reset hasil
    st.session_state.analysis_results = None
    st.session_state.processed_image = None
    st.session_state.execution_log_data = "Log display is disabled." # Tetap set string placeholder
    
    
    # Placeholder log dan progress bar
    log_placeholder = st.empty()
    log_placeholder.markdown(f'<p style="color: {TEXT_DARK};">SYSTEM> Initiating inference. Loading models...</p>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0, text="Loading Tensor Core & Running Inference...")
    
    # 2. Persiapan Data
    image_bytes = st.session_state.uploaded_file.getvalue()
    
    # Simpan file secara sementara untuk dibaca oleh YOLO (YOLOv8 sering membaca dari path)
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_bytes)
        
    # --- ALUR KERJA ML NYATA ---
    
    # A. Model 1: Deteksi Objek (YOLOv8)
    progress_bar.progress(30, text="[Step 1/2] Executing YOLOv8 Detection...")
    try:
        detections, messy_count = run_yolo_detection(yolo_model, temp_path)
    except Exception as e:
        st.error(f"Error saat menjalankan Deteksi YOLOv8: {e}")
        progress_bar.empty()
        return

    # B. Model 2: Klasifikasi Akhir (Keras/CNN)
    progress_bar.progress(60, text="[Step 2/2] Executing Keras/CNN Classification...")
    try:
        conf_clean, conf_messy = run_cnn_classification(cnn_model, image_bytes)
    except Exception as e:
        # Menangkap error CNN/Dense layer di sini
        st.error(f"Error saat menjalankan Klasifikasi CNN: {e}")
        progress_bar.empty()
        return
    
    progress_bar.progress(90, text="[Step 2/2] Post-processing results...")
    
    # --- PENENTUAN STATUS AKHIR (Berdasarkan CNN + Hybrid Rule) ---
    
    MESSY_DETECTION_THRESHOLD = 3 
    
    if conf_clean > conf_messy:
        is_clean = True
    else:
        is_clean = False
        
    is_overridden = False
    if is_clean and messy_count >= MESSY_DETECTION_THRESHOLD:
        is_clean = False # Override status
        is_overridden = True
    
    if is_clean:
        final_status = "STATUS: RUANGAN RAPI - OPTIMAL" # Diterjemahkan
        final_message = f"Pemeriksaan Integritas Sistem: HIJAU (LILAC). Kepercayaan kebersihan {round(conf_clean * 100, 2)}%. Organisasi yang luar biasa. (YOLO Messy Count: {messy_count})."
    else:
        final_status = "STATUS: RUANGAN BERANTAKAN - PERINGATAN" # Diterjemahkan
        
        if is_overridden:
            final_message = f"HYBRID OVERRIDE: YOLO mendeteksi {messy_count} aset TIDAK OPTIMAL (Ambang batas: {MESSY_DETECTION_THRESHOLD}). Status Akhir: PERINGATAN. (CNN Conf: {round(conf_clean * 100, 2)}% Rapi)."
        else:
            final_message = f"Pemeriksaan Integritas Sistem: MERAH (HOT PINK). Kepercayaan berantakan {round(conf_messy * 100, 2)}%. Kekacauan terdeteksi. Rekomendasi: Segera Rapikan."
    
    # C. Visualisasi Bounding Box (Menggunakan hasil YOLO)
    processed_image_bytes = draw_boxes_on_image(image_bytes, detections)
    
    results = {
        "final_status": final_status,
        "is_clean": is_clean,
        "conf_clean": round(conf_clean * 100, 2),
        "conf_messy": round(conf_messy * 100, 2),
        "messy_count": messy_count, 
        "detection_model": "Siti Naura Khalisa_Laporan 4.pt",
        "classification_model": "SitiNauraKhalisa_Laporan2.h5",
        "detections": detections,
        "final_message": final_message,
        "is_overridden": is_overridden # Tambahkan flag untuk logging
    }
    
    progress_bar.progress(100, text="Analysis Complete. Generating Report.")
    progress_bar.empty()

    # Hapus file sementara
    try:
        os.remove(temp_path)
    except OSError:
        pass

    st.session_state.processed_image = processed_image_bytes
    st.session_state.analysis_results = results
    
    # Log placeholder tetap di sini, tidak perlu diisi data log riil
    st.session_state.execution_log_data = "Log display is disabled for Cute Vision AI theme."
    
    log_placeholder.empty()
    
    # --- BERALIH KE HALAMAN REPORT ---
    st.session_state.app_state = 'REPORT'
    st.rerun() # Memaksa Streamlit untuk me-render ulang

# --- 7. Fungsi Utility Tips/Apresiasi ---

def get_tips_and_appreciation(is_clean, messy_count, is_overridden):
    """Menghasilkan konten HTML untuk tips atau apresiasi."""
    if is_clean:
        return {
            "title": "✅ STATUS OPTIMAL: APRESIASI KERAPIHAN",
            "icon": "✨",
            "color": ACCENT_PURPLE,
            "content": f"""
                <p>Selamat! Ruangan Anda menunjukkan tingkat kerapihan yang luar biasa. Sistem kami mendeteksi sedikit atau tidak ada <b>ASET TIDAK OPTIMAL</b> (jumlah: {messy_count}).</p>
                <p><b>Tips Maintenance:</b></p>
                <ul>
                    <li>Lanjutkan dengan prinsip 'Less is More': Pastikan setiap barang memiliki tempatnya yang spesifik.</li>
                    <li>Audit Digital: Jika ini ruang kerja, pertimbangkan untuk merapikan file digital secara berkala, seperti yang Anda lakukan pada aset fisik.</li>
                    <li>Sistem 5 Menit: Lakukan audit kerapihan cepat 5 menit setiap hari untuk mencegah penumpukan.</li>
                </ul>
            """
        }
    else: # Messy or Overridden
        if is_overridden:
            override_note = f"<p style='color:{ACCENT_PINK_MESSY}; font-weight:bold;'>CATATAN SISTEM: Meskipun klasifikasi CNN awal mungkin 'Rapi', YOLOv8 mendeteksi {messy_count} item tidak optimal yang signifikan, memicu Aturan Hibrida OVERRIDE ke status PERINGATAN.</p>"
        else:
            override_note = ""

        return {
            "title": "🚨 STATUS PERINGATAN: SARAN OPTIMASI RUANGAN",
            "icon": "🧹",
            "color": ACCENT_PINK_MESSY,
            "content": f"""
                {override_note}
                <p>Ruangan Anda teridentifikasi sebagai <b>TIDAK OPTIMAL / BERANTAKAN</b>. Ini menunjukkan adanya aset-aset yang perlu dikelola ulang. Model Deteksi kami mengidentifikasi <b>{messy_count} item TIDAK OPTIMAL</b>.</p>
                <p><b>Rekomendasi Tindakan (De-Clutter Protocol):</b></p>
                <ul>
                    <li>Fokus pada Aset Berisiko: Prioritaskan merapikan item yang terdeteksi (seperti **Pakaian Berserakan** atau **Kertas Tidak Teratur**).</li>
                    <li>Prinsip 4 Kotak: Gunakan 4 kotak: Sampah, Donasi, Simpan (jauh), dan Simpan (di sini). Segera distribusikan aset berdasarkan kategori ini.</li>
                    <li>Re-Scan: Setelah merapikan, muat ulang gambar ruangan Anda dan jalankan analisis kembali untuk memverifikasi Status Optimal.</li>
                </ul>
            """
        }

# --- 8. Fungsi Render Halaman (Pemisahan UI) ---

def render_upload_page():
    """Halaman 1: Upload Gambar Saja."""
    
    # --- JUDUL BARU TERPUSAT ---
    st.markdown(f"""
        <header>
            <div style="text-align: center;">
                <p class="main-title">ROOM INSIGHT</p>
                <p class="subtitle-center">CUTE VISION AI - Klasifikasikan Kerapihan Ruangan Anda</p>
            </div>
        </header>
        <div style="margin-bottom: 40px;"></div>
        """, unsafe_allow_html=True)
    # --- AKHIR JUDUL BARU ---
    
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_PINK};">1. Data Input Matrix (Upload Payload)</h2>', unsafe_allow_html=True)

    # Menampilkan pesan error jika ML tidak dimuat
    if not ML_LIBRARIES_LOADED:
        st.error("GAGAL PENTING: Pustaka 'ultralytics' atau 'tensorflow' tidak ditemukan. Analisis ML TIDAK DAPAT DILANJUTKAN.")
        
    uploaded_file = st.file_uploader(
        "Upload Image File (JPG/PNG) | Initiating Payload Protocol", 
        type=["jpg", "jpeg", "png"],
        key="uploader_main",
        help="Unggah file gambar ruangan untuk dianalisis."
    )

    st.session_state.uploaded_file = uploaded_file

    st.markdown('</div>', unsafe_allow_html=True) 

    st.markdown('<div style="padding-top: 20px;">', unsafe_allow_html=True)
    
    # Tombol dinonaktifkan jika tidak ada file yang diunggah ATAU jika pustaka ML gagal dimuat
    button_disabled = st.session_state.uploaded_file is None or not ML_LIBRARIES_LOADED
    
    if st.button("💖 INITIATE DUAL-MODEL ANALYSIS", disabled=button_disabled, use_container_width=True):
        # PANGGIL FUNGSI ML NYATA
        with st.spinner('Running Dual-Model Analysis...'):
            run_ml_analysis() 
            
    if st.session_state.uploaded_file and not ML_LIBRARIES_LOADED:
        st.warning("Analisis dinonaktifkan karena pustaka Machine Learning tidak tersedia. Harap instal 'ultralytics' dan 'tensorflow' untuk fungsi penuh.")

    st.markdown('</div>', unsafe_allow_html=True)

def render_report_page():
    """Halaman 2: Tampilan Laporan Analisis."""

    results = st.session_state.analysis_results
    
    # Cek apakah ada hasil
    if not results or not st.session_state.uploaded_file:
        st.error("Sesi analisis tidak valid. Kembali ke halaman utama.")
        st.session_state.app_state = 'UPLOAD'
        st.rerun()
        return

    st.markdown(f"""
        <header>
            <div style="text-align: center;">
                <h2 style="color: {ACCENT_PRIMARY_PINK}; border-bottom: none; padding-bottom: 5px;">LAPORAN ANALISIS</h2>
                <p style="color: {TEXT_DARK}; font-size: 14px;">Laporan lengkap hasil deteksi objek (YOLOv8) dan klasifikasi kerapihan (CNN) untuk: **{st.session_state.uploaded_file.name.upper()}**</p>
            </div>
        </header>
        <div style="margin-bottom: 20px;"></div>
        """, unsafe_allow_html=True)

    # --- 1. GAMBAR YANG DIUNGGAH (ATAS) ---
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_PINK};">1. Visualisasi Deteksi Objek</h2>', unsafe_allow_html=True)

    border_class = 'clean-border' if results['is_clean'] else 'messy-border'
    
    st.markdown(f"""
        <div style="border: 4px solid #34495E; border-radius: 15px; padding: 5px; background-color: {CARD_BG}; margin-bottom: 25px;" class="{border_class}">
        """, unsafe_allow_html=True)

    if st.session_state.processed_image:
        st.image(st.session_state.processed_image, caption='VISUALIZATION: Live Detection Grid (YOLO v8 Output)', use_container_width=True)
    else:
        image_data = st.session_state.uploaded_file.getvalue()
        st.image(image_data, caption='Image Data Stream (Original)', use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f"<hr style='border-top: 1px solid {ACCENT_PRIMARY_PINK}; box-shadow: 0 0 5px {ACCENT_PRIMARY_PINK}; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

    # --- 2. KLASIFIKASI FINAL STATUS (UTAMA) ---
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_PINK};">2. Final Classification Status</h2>', unsafe_allow_html=True)

    col_report, col_clean_conf, col_messy_conf = st.columns([2, 1, 1])

    status_main_text = results['final_status'].split(': ')[1]
    css_class_status = 'clean-status-text' if results['is_clean'] else 'messy-status-text'
    message = results['final_message']
    
    with col_report:
        border_color = ACCENT_PURPLE if results['is_clean'] else ACCENT_PINK_MESSY
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {border_color}; box-shadow: 0 0 10px {border_color};">
                <p style="color: {TEXT_DARK}; font-size: 14px; margin-bottom: 5px; font-weight: bold;">CLASSIFICATION REPORT (Final Status)</p>
                <p class="{css_class_status}" style="font-size: 32px; margin-top: 5px;">{status_main_text}</p>
                <p style="font-size: 12px; color: {TEXT_DARK}; opacity: 0.7;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col_clean_conf:
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {ACCENT_PURPLE}; background-color: {CARD_BG}; box-shadow: 0 0 8px {ACCENT_PURPLE};">
                <p style="color: {ACCENT_PURPLE}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">CONFIDENCE: RAPI</p>
                <p style="color: {TEXT_DARK}; font-size: 28px; font-weight: bold;">{results["conf_clean"]}%</p>
                <p style="color: {TEXT_DARK}; font-size: 10px; margin-top: 5px; opacity: 0.6;">(Dari Model {results['classification_model']})</p>
            </div>
            """, unsafe_allow_html=True)
        
    with col_messy_conf:
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {ACCENT_PINK_MESSY}; background-color: {CARD_BG}; box-shadow: 0 0 8px {ACCENT_PINK_MESSY};">
                <p style="color: {ACCENT_PINK_MESSY}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">CONFIDENCE: BERANTAKAN</p>
                <p style="color: {TEXT_DARK}; font-size: 28px; font-weight: bold;">{results["conf_messy"]}%</p>
                <p style="color: {TEXT_DARK}; font-size: 10px; margin-top: 5px; opacity: 0.6;">(Dari Model {results['classification_model']})</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown(f"<hr style='border-top: 1px solid {ACCENT_PRIMARY_PINK}; box-shadow: 0 0 5px {ACCENT_PRIMARY_PINK}; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)
    
    # --- 3. TIPS / APRESIASI ---
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_PINK};">3. Tindakan Rekomendasi</h2>', unsafe_allow_html=True)
    
    tips = get_tips_and_appreciation(results['is_clean'], results['messy_count'], results.get('is_overridden', False))
    
    st.markdown(f"""
        <div class="tips-box" style="background-color: {CARD_BG}; border-color: {tips['color']}; box-shadow: 0 0 10px rgba({tips['color'][1:]}, 0.5);">
            <h3 style="color: {tips['color']};">{tips['icon']} {tips['title']}</h3>
            {tips['content']}
        </div>
        """, unsafe_allow_html=True)

    # --- 4. LOG DAN DETAIL TABEL (DIHAPUS SESUAI PERMINTAAN) ---
    # Bagian ini dikosongkan/dihapus

    # Tombol untuk kembali
    st.button("↩ KEMBALI KE HALAMAN UPLOAD", on_click=lambda: st.session_state.update(app_state='UPLOAD', analysis_results=None, processed_image=None), use_container_width=False)


# --- 9. FUNGSI UTAMA APP CONTROLLER ---
if st.session_state.app_state == 'UPLOAD':
    render_upload_page()
elif st.session_state.app_state == 'REPORT':
    render_report_page()
else:
    # Fallback, seharusnya tidak terjadi
    st.session_state.app_state = 'UPLOAD'
    st.rerun()
