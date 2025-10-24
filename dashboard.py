import streamlit as st
import random
import time
import json
import io
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pandas as pd
import numpy as np

# --- 1. Import Pustaka Machine Learning (Hanya akan berfungsi jika diinstal) ---
try:
    from ultralytics import YOLO
    # Peringatan: TensorFlow seringkali harus dimuat secara penuh
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    ML_LIBRARIES_LOADED = True
except ImportError:
    st.error("GAGAL PENTING: Pustaka 'ultralytics' atau 'tensorflow' tidak ditemukan. Analisis ML TIDAK DAPAT DILANJUTKAN.")
    ML_LIBRARIES_LOADED = False

# --- 2. Konfigurasi dan Styling (Tema Cyber Pastel / Vaporwave) ---

st.set_page_config(
    page_title="ROOM INSIGHT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Palet Warna
BG_DARK = "#121212"            
CARD_BG = "#1e1e1e"            
TEXT_LIGHT = "#F0F0F0"        
ACCENT_PRIMARY_NEON = "#39FF14"  # Hijau Neon
NEON_CYAN = "#00FFFF"         # Bersih
NEON_MAGENTA = "#FF073A"      # Berantakan
BUTTON_COLOR_NEON = "#39FF14"    # Hijau Neon

custom_css = f"""
<style>
    /* Definisi Keyframe untuk Efek Neon Flicker */
    @keyframes neon-flicker {{
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {{
            text-shadow: 0 0 5px {ACCENT_PRIMARY_NEON}, 0 0 10px {ACCENT_PRIMARY_NEON}, 0 0 20px {ACCENT_PRIMARY_NEON}, 0 0 40px rgba(57, 255, 20, 0.5);
            opacity: 1;
        }}
        20%, 24%, 55% {{
            text-shadow: 0 0 2px {ACCENT_PRIMARY_NEON}, 0 0 5px {ACCENT_PRIMARY_NEON};
            opacity: 0.9;
        }}
    }}

    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_LIGHT};
        font-family: 'Inter', sans-serif;
    }}
    h1 {{
        color: {ACCENT_PRIMARY_NEON};
        border-bottom: 3px solid {ACCENT_PRIMARY_NEON};
        padding-bottom: 10px;
        animation: neon-flicker 1.8s infinite alternate; 
    }}
    .modern-card {{
        background-color: {CARD_BG};
        border: 1px solid {ACCENT_PRIMARY_NEON};
        box-shadow: 0 0 15px rgba(57, 255, 20, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }}
    h2 {{
        color: {ACCENT_PRIMARY_NEON};
        border-bottom: 1px solid #333333;
        padding-bottom: 5px;
    }}
    
    /* GAYA UPLOADER */
    [data-testid="stFileUploader"] {{
        min-height: 150px; 
        padding: 10px 20px 10px 20px !important;
        margin-top: 10px;
        border: 2px dashed {ACCENT_PRIMARY_NEON} !important; 
        border-radius: 12px;
        background-color: {BG_DARK} !important; 
    }}
    
    /* GAYA TOMBOL UTAMA */
    .stButton > button:nth-child(1) {{
        background-color: {BUTTON_COLOR_NEON}; 
        color: {BG_DARK} !important;
        font-weight: 900;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 0 10px {BUTTON_COLOR_NEON}, 0 0 20px {BUTTON_COLOR_NEON};
    }}
    .stButton > button:nth-child(1):hover {{
        background-color: #2ECC71; /* Slightly darker green */
        box-shadow: 0 0 15px {BUTTON_COLOR_NEON}, 0 0 25px {BUTTON_COLOR_NEON};
    }}

    /* STATUS CARD STYLES */
    .status-metric-card {{
        background-color: {CARD_BG};
        border: 2px solid #333333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.3);
    }}
    
    .clean-status-text {{ color: {NEON_CYAN}; font-weight: 900; font-size: 36px; text-shadow: 0 0 8px {NEON_CYAN}; }}
    .messy-status-text {{ color: {NEON_MAGENTA}; font-weight: 900; font-size: 36px; text-shadow: 0 0 8px {NEON_MAGENTA}; }}
    
    .clean-border-visual {{ border: 4px solid {NEON_CYAN} !important; box-shadow: 0 0 20px rgba(0, 255, 255, 0.6) !important; }}
    .messy-border-visual {{ border: 4px solid {NEON_MAGENTA} !important; box-shadow: 0 0 20px rgba(255, 7, 58, 0.6) !important; }}
    
    .log-container {{
        background-color: {CARD_BG}; 
        color: {TEXT_LIGHT}; 
        border: 1px solid {ACCENT_PRIMARY_NEON} !important;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        height: 140px;
        overflow-y: auto;
    }}
    
    .advice-clean {{ background-color: rgba(0, 255, 255, 0.1); border-left: 5px solid {NEON_CYAN}; padding: 10px; border-radius: 5px; }}
    .advice-messy {{ background-color: rgba(255, 7, 58, 0.1); border-left: 5px solid {NEON_MAGENTA}; padding: 10px; border-radius: 5px; }}
    
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
    st.session_state.execution_log_data = []

# --- 3. Fungsi Pemuatan Model (Menggunakan Cache Resource Streamlit) ---
@st.cache_resource
def load_ml_model():
    """Memuat model YOLO dan CNN ke memori dengan caching."""
    if not ML_LIBRARIES_LOADED:
        return None, None
    
    try:
        # PENTING: Ganti path ini jika lokasi file Anda berbeda
        YOLO_MODEL_PATH = "model/Siti Naura Khalisa_Laporan 4.pt" # Menggunakan nama file yang diunggah
        CNN_MODEL_PATH = "model/SitiNauraKhalisa_Laporan2.h5"     # Menggunakan nama file yang diunggah

        # Model 1: Deteksi Objek (YOLOv8)
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # Model 2: Klasifikasi Ruangan (Keras/CNN)
        tf.get_logger().setLevel('ERROR')
        cnn_model = load_model(CNN_MODEL_PATH)
        
        return yolo_model, cnn_model
    except Exception as e:
        st.error(f"FATAL ERROR: Gagal memuat file model dari path. Pastikan file berada di direktori yang sama dan namanya benar: {e}")
        return None, None

# --- 4. Fungsi Real Inference dan Pemrosesan Data ---

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
    
    result = results[0] 
    
    detections = []
    messy_count = 0
    
    # Mendefinisikan ID Kelas yang dianggap "Messy" (SESUAIKAN DENGAN MODEL ANDA)
    MESSY_CLASS_NAMES = ["Scattered_Clothes", "Loose_Cables", "Unsorted_Papers", "Trash_Object", "Empty_Bottles", "Food_Wrapper"]
    class_names = result.names # Peta ID ke nama kelas
    
    
    for box in result.boxes:
        # Koordinat bounding box (normalized, xywh format)
        x_norm, y_norm, w_norm, h_norm = box.xywhn[0].tolist() 
        conf = box.conf.item() 
        class_id = box.cls.item()
        
        # Pengecekan bounds untuk class_id
        if int(class_id) < len(class_names):
            label = class_names[int(class_id)]
        else:
            label = f"UNKNOWN_CLASS_{int(class_id)}"
            
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
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
    img_array = img_array / 255.0 # Normalisasi
    
    # Prediksi
    predictions = cnn_model.predict(img_array, verbose=0)[0]
    
    # Asumsi output adalah (Conf_Clean, Conf_Messy).
    # Pastikan urutan ini benar sesuai dengan model Anda.
    conf_clean = predictions[0]  
    conf_messy = predictions[1]  
    
    return conf_clean, conf_messy

# --- 5. Fungsi Utilitas Visualisasi dan Log ---

def draw_boxes_on_image(image_bytes, detections):
    """Menggambar Bounding Box Neon pada Gambar dari hasil deteksi."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    
    CLEAN_RGB = ImageColor.getrgb(NEON_CYAN)
    MESSY_RGB = ImageColor.getrgb(NEON_MAGENTA)
    TEXT_RGB = ImageColor.getrgb(BG_DARK)
    
    try:
        # Atur ukuran font secara dinamis
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
        draw.text((text_x + 2, text_y + 2), text_content, font=font, fill=TEXT_RGB) 
        
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    return buf.getvalue()

def format_execution_log(results, uploaded_file_name):
    """Membuat format log tekstual yang menyerupai output konsol"""
    log_lines = []
    
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] INFO: System Initialized.")
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 4))}] DATA: Payload Acquired: {uploaded_file_name}.")

    # Log Model 1: Deteksi
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 3))}] MODEL-DETECTION: Loading <b>{results['detection_model']}</b> (YOLO V8).")
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 2))}] INFERENCE-YOLO: Detecting {len(results['detections'])} Assets. Messy Count: {results['messy_count']}.")
    
    # Log Model 2: Klasifikasi
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 1))}] MODEL-CLASSIFICATION: Loading <b>{results['classification_model']}</b> (Keras/CNN).")
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY-CNN: Initial Classification Complete. (Clean Conf: {results['conf_clean']}%, Messy Conf: {results['conf_messy']}%)")
    
    # Log Final Decision (Sekarang sepenuhnya berdasarkan YOLO)
    tag_color = NEON_CYAN if results['is_clean'] else NEON_MAGENTA
    
    final_status = results['final_status'].split(': ')[1]
    
    # Pesan Log yang diperbarui untuk mencerminkan keputusan YOLO-only
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] FINAL-DECISION: Status ditetapkan berdasarkan YOLOv8 Messy Count ({results['messy_count']}). Status: <span style='color:{tag_color};'><b>{final_status}</b></span>.")
    
    return '<br>'.join(log_lines)

def get_advice(is_clean, messy_count, detections):
    """Memberikan saran berdasarkan hasil analisis."""
    if is_clean:
        advice_html = f"""
        <div class="advice-clean">
            <h3 style="color: {NEON_CYAN}; margin-top: 0; font-size: 18px;">✨ Protokol Kebersihan Lulus!</h3>
            <p style="color: {TEXT_LIGHT}; font-size: 14px; margin-bottom: 0;">Ruangan Anda dalam kondisi **OPTIMAL**. Terus pertahankan organisasi yang sangat baik ini. Tidak ada tindakan segera yang diperlukan.</p>
            <p style="color: {TEXT_LIGHT}; font-size: 12px; margin-top: 5px; opacity: 0.8;">*Tips Lanjutan: Coba lakukan pembersihan cepat mingguan untuk menjaga momentum!</p>
        </div>
        """
    else:
        messy_items = [d['asset_id'] for d in detections if d['classification_tag'] == 'UNOPTIMIZED']
        item_summary = ""
        
        # Membuat ringkasan item paling banyak (maks 3 unik)
        if messy_items:
            item_counts = pd.Series(messy_items).value_counts()
            top_items = item_counts.head(3).index.tolist()
            if len(top_items) == 1:
                item_summary = f" Fokuskan pada membersihkan **{top_items[0]}**."
            elif len(top_items) == 2:
                item_summary = f" Fokuskan pada membersihkan **{top_items[0]}** dan **{top_items[1]}**."
            else:
                item_summary = f" Fokuskan pada membersihkan **{top_items[0]}**, **{top_items[1]}**, dan sisa sampah."
        
        advice_html = f"""
        <div class="advice-messy">
            <h3 style="color: {NEON_MAGENTA}; margin-top: 0; font-size: 18px;">🚨 KODE MERAH: Perlu Tindakan Segera!</h3>
            <p style="color: {TEXT_LIGHT}; font-size: 14px; margin-bottom: 0;">Sistem mendeteksi **{messy_count}** aset yang tidak terorganisir. {item_summary}</p>
            <p style="color: {TEXT_LIGHT}; font-size: 12px; margin-top: 5px; opacity: 0.8;">*Tindakan Cepat: Segera kembalikan barang-barang ke tempatnya atau buang sampah yang terdeteksi.</p>
        </div>
        """
    return advice_html

# --- 6. Fungsi Utama Alur Kerja ML ---

def run_ml_analysis():
    """Menjalankan alur kerja ML yang nyata."""
    
    if not ML_LIBRARIES_LOADED:
        st.error("Analisis dibatalkan: Pustaka Machine Learning gagal dimuat saat startup.")
        return

    if st.session_state.uploaded_file is None:
        st.error("Sila muat naik imej ruangan dahulu.")
        return
    
    # 1. Muat Model
    yolo_model, cnn_model = load_ml_model()
    
    if yolo_model is None or cnn_model is None:
        st.error("Analisis dibatalkan: File Model gagal dimuat dari direktori.")
        return

    st.session_state.analysis_results = None
    st.session_state.processed_image = None
    st.session_state.execution_log_data = []

    
    # Placeholder log dan progress bar
    log_placeholder = st.empty()
    log_placeholder.markdown(f'<p style="color: {TEXT_LIGHT};">SYSTEM> Initiating inference. Loading models...</p>', unsafe_allow_html=True)
    
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

    # B. Model 2: Klasifikasi Akhir (Keras/CNN) - Dijalankan untuk Metrik, BUKAN untuk Keputusan Akhir
    progress_bar.progress(60, text="[Step 2/2] Executing Keras/CNN Classification (for metrics only)...")
    try:
        conf_clean, conf_messy = run_cnn_classification(cnn_model, image_bytes)
    except Exception as e:
        st.error(f"Error saat menjalankan Klasifikasi CNN: {e}")
        progress_bar.empty()
        return
    
    progress_bar.progress(90, text="[Step 2/2] Post-processing results...")
    
    # --- PENENTUAN STATUS AKHIR (Berdasarkan DETEKSI YOLO SAJA) ---
    # Sesuai permintaan pengguna: status akhir ditentukan HANYA oleh hitungan YOLO
    MESSY_DETECTION_THRESHOLD = 3
    
    # Hitungan confidence dari CNN (untuk display)
    conf_clean_perc = round(conf_clean * 100, 2)
    conf_messy_perc = round(conf_messy * 100, 2)

    # 1. Penentuan Status Penuh Berdasarkan YOLO (Messy Item Count)
    is_overridden = False # Flag ini sekarang tidak relevan, tapi dipertahankan untuk kompatibilitas data structure
    
    if messy_count < MESSY_DETECTION_THRESHOLD:
        is_clean = True
        # Status BERSIH ditentukan karena jumlah item berantakan di bawah ambang batas
        final_status = "STATUS: RUANGAN BERSIH (OPTIMAL)"
        final_message = f"KEPUTUSAN YOLOV8 (SOLE SOURCE): {messy_count} asset UNOPTIMIZED terdeteksi (di bawah batas {MESSY_DETECTION_THRESHOLD}). CNN Conf: {conf_clean_perc}% Clean."
    else:
        is_clean = False
        # Status BERANTAKAN ditentukan karena jumlah item berantakan mencapai atau melebihi ambang batas
        final_status = "STATUS: RUANGAN BERANTAKAN (PERINGATAN)"
        final_message = f"KEPUTUSAN YOLOV8 (SOLE SOURCE): {messy_count} asset UNOPTIMIZED terdeteksi (MELEBIHI batas {MESSY_DETECTION_THRESHOLD}). CNN Conf: {conf_messy_perc}% Messy."
    
    # C. Visualisasi Bounding Box (Menggunakan hasil YOLO)
    processed_image_bytes = draw_boxes_on_image(image_bytes, detections)
    
    results = {
        "final_status": final_status,
        "is_clean": is_clean,
        "conf_clean": conf_clean_perc, # Simpan dalam persentase
        "conf_messy": conf_messy_perc, # Simpan dalam persentase
        "messy_count": messy_count, 
        "detection_model": "Siti Naura Khalisa_Laporan 4.pt",
        "classification_model": "SitiNauraKhalisa_Laporan2.h5",
        "detections": detections,
        "final_message": final_message,
        "is_overridden": is_overridden # Tambahkan flag untuk logging
    }
    
    progress_bar.progress(100, text="Analysis Complete. Generating Report.")
    progress_bar.empty()

    st.session_state.processed_image = processed_image_bytes
    st.session_state.analysis_results = results
    
    st.session_state.execution_log_data = format_execution_log(results, st.session_state.uploaded_file.name)
    
    log_placeholder.empty()
    st.success("SYSTEM> Analysis Completed. Report Generated and Visualization Rendered.")
    
# --- 7. Tata Letak Streamlit (UI) ---

st.markdown(f"""
    <header>
        <h1>ROOM INSIGHT <span style="font-size: 18px; margin-left: 15px; color: {ACCENT_PRIMARY_NEON};">DUAL-MODEL CLASSIFIER</span></h1>
        <p style="color: {TEXT_LIGHT}; font-size: 14px;">Klasifikasikan kerapihan ruangan menggunakan model Deteksi Objek (YOLOv8) dan Klasifikasi Gambar (CNN).</p>
    </header>
    <div style="margin-bottom: 20px;"></div>
    """, unsafe_allow_html=True)


col_main_input, col_main_results = st.columns([1, 2])

with col_main_input:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">1. Payload Acquisition</h2>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Image File (JPG/PNG) | Initiating Payload Protocol", 
        type=["jpg", "jpeg", "png"],
        key="uploader",
        help="Unggah file gambar ruangan untuk dianalisis."
    )
    st.session_state.uploaded_file = uploaded_file

    st.markdown('</div>', unsafe_allow_html=True) 

    st.markdown('<div style="padding-top: 20px;">', unsafe_allow_html=True)
    
    button_disabled = st.session_state.uploaded_file is None or not ML_LIBRARIES_LOADED
    
    if st.button("⚡ INITIATE DUAL-MODEL ANALYSIS", disabled=button_disabled, use_container_width=True):
        run_ml_analysis() 
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # CONFIDENCE METRICS (Pindahkan ke Input Column)
    results = st.session_state.analysis_results
    if results:
        st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 18px; margin-top: 20px;">Confidence Score (CNN)</h3>', unsafe_allow_html=True)
        col_conf_clean, col_conf_messy = st.columns(2)

        with col_conf_clean:
            st.markdown(f"""
                <div class="status-metric-card" style="border-color: {NEON_CYAN}; background-color: {CARD_BG}; box-shadow: 0 0 8px {NEON_CYAN};">
                    <p style="color: {NEON_CYAN}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">CLEAN</p>
                    <p style="color: {TEXT_LIGHT}; font-size: 24px; font-weight: bold;">{results["conf_clean"]}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_conf_messy:
            st.markdown(f"""
                <div class="status-metric-card" style="border-color: {NEON_MAGENTA}; background-color: {CARD_BG}; box-shadow: 0 0 8px {NEON_MAGENTA};">
                    <p style="color: {NEON_MAGENTA}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">MESSY</p>
                    <p style="color: {TEXT_LIGHT}; font-size: 24px; font-weight: bold;">{results["conf_messy"]}%</p>
                </div>
                """, unsafe_allow_html=True)

with col_main_results:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">2. Visualization & Final Status</h2>', unsafe_allow_html=True)
    
    border_class = ""
    status_text = "Awaiting Image Payload"
    status_color = ACCENT_PRIMARY_NEON
    
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        border_class = 'clean-border-visual' if results['is_clean'] else 'messy-border-visual'
        status_main_text = results['final_status'].split(': ')[1]
        css_class_status = 'clean-status-text' if results['is_clean'] else 'messy-status-text'
        status_text = f'FINAL STATUS: <span class="{css_class_status}">{status_main_text}</span>'
        
        # Display Final Status Box
        # Mengganti pesan ringkas di sini agar lebih mencerminkan keputusan YOLO
        final_message_display = results['final_message'].split(': ')[0] + results['final_message'].split(': ')[1]
        
        st.markdown(f"""
            <div class="status-metric-card" style="margin-top: -10px; border-color: {'#00FFFF' if results['is_clean'] else '#FF073A'};">
                <p style="color: {TEXT_LIGHT}; font-size: 14px; margin-bottom: 5px; font-weight: bold;">{final_message_display}</p>
                <p style="margin: 0;">{status_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
    elif st.session_state.uploaded_file:
        status_text = "Image Loaded. Click Initiate Analysis."
        
    st.markdown(f"""
        <div style="border: 2px solid #34495E; border-radius: 10px; padding: 5px; background-color: {BG_DARK};" class="{border_class}">
        """, unsafe_allow_html=True)

    if st.session_state.uploaded_file:
        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, caption='VISUALIZATION: Live Detection Grid (YOLO v8 Output)', use_container_width=True)
        else:
            image_data = st.session_state.uploaded_file.getvalue()
            st.image(image_data, caption='Image Data Stream (Original) - Click Analysis to Process', use_container_width=True)
    else:
        st.image("https://placehold.co/1200x675/121212/39FF14?text=UPLOAD+IMAGE+TO+ACTIVATE+SCANNER+MODULE", caption="Awaiting Image Payload", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- FITUR BARU: Quick Action Terminal ---
st.markdown(f"<hr style='border-top: 1px solid {ACCENT_PRIMARY_NEON}; box-shadow: 0 0 5px {ACCENT_PRIMARY_NEON}; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
st.markdown('<div class="modern-card">', unsafe_allow_html=True)
st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">3. Quick Action Terminal (Rekomendasi)</h2>', unsafe_allow_html=True)

if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    st.markdown(get_advice(results['is_clean'], results['messy_count'], results['detections']), unsafe_allow_html=True)
else:
    st.info("Saran pembersihan akan ditampilkan di sini setelah analisis selesai.")
    
st.markdown('</div>', unsafe_allow_html=True)
    
# --- TAMPILAN 4: EXECUTION LOG & DETAILED TABLE (Sebagai Detail Lanjutan) ---
st.markdown(f"<hr style='border-top: 1px dashed #333333; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

col_log, col_table = st.columns(2)

with col_log:
    st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; border-bottom: 1px solid #333333; padding-bottom: 5px;">Execution Log</h3>', unsafe_allow_html=True)
    
    log_content = ""
    if st.session_state.analysis_results:
        log_content = st.session_state.execution_log_data
    else:
        log_content = f"""[{time.strftime('%H:%M:%S')}] INFO: System Initialized. Awaiting Input Payload.<br>
[{time.strftime('%H:%M:%S')}] MODEL: Dual-Model System Idle. ({'ML Libraries NOT loaded' if not ML_LIBRARIES_LOADED else 'ML Libraries loaded'})"""

    st.markdown(f"""
        <div class="log-container">
            {log_content}
        </div>
        """, unsafe_allow_html=True)

with col_table:
    st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; border-bottom: 1px solid #333333; padding-bottom: 5px;">Detected Asset Detail (YOLO)</h3>', unsafe_allow_html=True)

    if st.session_state.analysis_results:
        df = pd.DataFrame(st.session_state.analysis_results['detections'])
        df = df.rename(columns={
            'asset_id': 'Asset ID', 
            'confidence_score': 'Conf. Deteksi (%)', 
            'classification_tag': 'Tag Kerapihan',
            'normalized_coordinates': 'Koordinat Norm.'
        })
        df['Conf. Deteksi (%)'] = (df['Conf. Deteksi (%)'] * 100).round(2).astype(str) + '%'
        
        st.dataframe(
            df, 
            use_container_width=True,
            height=140 
        )

    else:
        st.info("Tabel aset akan muncul di sini.")
