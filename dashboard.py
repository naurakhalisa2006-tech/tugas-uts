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
# Menggunakan penanda global untuk membedakan mode simulasi/nyata
try:
    from ultralytics import YOLO
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    ML_LIBRARIES_LOADED = True
    # Menghapus st.warning di sini
except ImportError:
    # Mengubah ke st.error dan memastikan status False
    ML_LIBRARIES_LOADED = False

# --- 2. Konfigurasi dan Styling (Tema Cyber Pastel / Vaporwave) ---

st.set_page_config(
    page_title="ROOM INSIGHT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BG_DARK = "#1A1A2E"             
CARD_BG = "#2C3E50"             
TEXT_LIGHT = "#EAEAEA"          
ACCENT_PRIMARY_NEON = "#4DFFFF" 
NEON_CYAN = "#00FFFF"           
NEON_MAGENTA = "#FF00FF"        
TEXT_CLEAN_LIGHT = NEON_CYAN    
TEXT_MESSY_LIGHT = NEON_MAGENTA 
BUTTON_COLOR_NEON = "#3498DB"

custom_css = f"""
<style>
    /* Definisi Keyframe untuk Efek Neon Flicker */
    @keyframes neon-flicker {{
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {{
            text-shadow: 0 0 5px {ACCENT_PRIMARY_NEON}, 0 0 10px {ACCENT_PRIMARY_NEON}, 0 0 20px {ACCENT_PRIMARY_NEON}, 0 0 40px rgba(77, 255, 255, 0.5);
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
        border-bottom: 2px solid {ACCENT_PRIMARY_NEON};
        padding-bottom: 10px;
        animation: neon-flicker 1.8s infinite alternate; 
    }}
    .modern-card {{
        background-color: {CARD_BG};
        border: 1px solid {ACCENT_PRIMARY_NEON};
        box-shadow: 0 0 15px rgba(77, 255, 255, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }}
    h2 {{
        color: {ACCENT_PRIMARY_NEON};
        border-bottom: 1px solid #34495E;
        padding-bottom: 5px;
    }}
    
    /* GAYA UPLOADER CYBER */
    [data-testid="stFileUploader"] {{
        min-height: 200px; 
        padding: 10px 20px 10px 20px !important;
        margin-top: 10px;
        border: 2px dashed {ACCENT_PRIMARY_NEON} !important; 
        border-radius: 12px;
        background-color: {BG_DARK} !important; 
        pointer-events: auto !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] p, [data-testid="stFileUploader"] label {{
        color: {TEXT_LIGHT} !important;
    }}
    
    /* GAYA TOMBOL 'BROWSE FILES' DI DALAM UPLOADER */
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {CARD_BG} !important;
        color: {ACCENT_PRIMARY_NEON} !important;
        font-weight: bold !important;
        border: 1px solid {ACCENT_PRIMARY_NEON} !important;
        border-radius: 8px !important;
        transition: all 0.3s !important;
        box-shadow: 0 0 5px {ACCENT_PRIMARY_NEON} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: #3C5A6C !important;
        box-shadow: 0 0 10px {ACCENT_PRIMARY_NEON} !important;
    }}
    
    /* Tombol Initiate YOLO V8 Algorithm (Tombol utama non-uploader) */
    .stButton > button:nth-child(1) {{
        background-color: {BUTTON_COLOR_NEON}; 
        color: {TEXT_LIGHT} !important;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 0 10px {NEON_CYAN}, 0 0 20px {NEON_CYAN};
        height: 50px;
        font-size: 18px;
    }}
    .stButton > button:nth-child(1):hover {{
        background-color: #2980B9;
        box-shadow: 0 0 15px {NEON_CYAN}, 0 0 25px {NEON_CYAN};
    }}

    /* Tombol Kembali (Return Button) */
    .stButton > button[data-testid="stElement" i]:not(:nth-child(1)) {{
        background-color: {CARD_BG}; 
        color: {ACCENT_PRIMARY_NEON} !important;
        border: 1px solid {ACCENT_PRIMARY_NEON} !important;
        box-shadow: 0 0 5px {ACCENT_PRIMARY_NEON};
        margin-top: 20px;
    }}


    .status-metric-card {{
        background-color: {CARD_BG};
        border: 2px solid #34495E;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.3);
    }}
    
    .clean-status-text {{ color: {TEXT_CLEAN_LIGHT}; font-weight: 900; font-size: 28px; text-shadow: 0 0 5px {TEXT_CLEAN_LIGHT}; }}
    .messy-status-text {{ color: {TEXT_MESSY_LIGHT}; font-weight: 900; font-size: 28px; text-shadow: 0 0 5px {TEXT_MESSY_LIGHT}; }}
    
    .clean-border {{ border-color: {NEON_CYAN} !important; border-width: 4px !important; box-shadow: 0 0 15px rgba(0, 255, 255, 0.6) !important; }}
    .messy-border {{ border-color: {NEON_MAGENTA} !important; border-width: 4px !important; box-shadow: 0 0 15px rgba(255, 0, 255, 0.6) !important; }}
    
    .info-card-clean {{ background-color: {NEON_CYAN}; color: {BG_DARK}; border-color: {NEON_CYAN}; box-shadow: 0 0 8px {NEON_CYAN}; }}
    .info-card-messy {{ background-color: {NEON_MAGENTA}; color: {BG_DARK}; border-color: {NEON_MAGENTA}; box-shadow: 0 0 8px {NEON_MAGENTA}; }}

    .caption-clean {{ color: {TEXT_CLEAN_LIGHT}; }}
    .caption-messy {{ color: {TEXT_MESSY_LIGHT}; }}

    p {{ color: {TEXT_LIGHT}; }}
    
    .log-container {{
        background-color: {BG_DARK}; 
        color: {TEXT_LIGHT}; 
        border-color: {ACCENT_PRIMARY_NEON} !important; 
        box-shadow: 0 0 8px rgba(77, 255, 255, 0.4);
        padding: 15px;
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        height: 140px;
        overflow-y: auto;
    }}
    
    .tips-box {{
        padding: 20px;
        border-radius: 12px;
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
        list-style-type: square;
        padding-left: 20px;
        color: {TEXT_LIGHT};
    }}
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

# --- 5. Fungsi Utilitas Visualisasi dan Log (TETAP SAMA) ---

def draw_boxes_on_image(image_bytes, detections):
    """Menggambar Bounding Box Neon pada Gambar dari hasil deteksi."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    
    CLEAN_RGB = ImageColor.getrgb(NEON_CYAN)
    MESSY_RGB = ImageColor.getrgb(NEON_MAGENTA)
    TEXT_RGB = ImageColor.getrgb(BG_DARK)
    
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
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY-CNN: Initial Classification Complete.")
    
    # Log Hybrid Rule
    tag_color = NEON_CYAN if results['is_clean'] else NEON_MAGENTA
    final_status_report = "REPORT"
    
    if results.get('is_overridden'):
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] <span style='color:{NEON_MAGENTA};'>WARNING: Hybrid Rule Triggered! Overriding CNN result due to high messy count ({results['messy_count']}).</span>")
        final_status_report = "OVERRIDE-REPORT"
    
    final_status = results['final_status'].split(': ')[1]
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] {final_status_report}: Final Status: <span style='color:{tag_color};'><b>{final_status}</b></span> (Clean Conf: {results['conf_clean']}%, Messy Conf: {results['conf_messy']}%).")
    
    return '<br>'.join(log_lines)

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
        final_status = "STATUS: ROOM CLEAN - OPTIMAL"
        final_message = f"System Integrity Check: GREEN (CYAN NEON). Cleanliness confidence {round(conf_clean * 100, 2)}%. Excellent organization. (YOLO Messy Count: {messy_count})."
    else:
        final_status = "STATUS: ROOM MESSY - ALERT"
        
        if is_overridden:
            final_message = f"HYBRID OVERRIDE: YOLO detected {messy_count} UNOPTIMIZED assets (Threshold: {MESSY_DETECTION_THRESHOLD}). Final Status: ALERT. (CNN Conf: {round(conf_clean * 100, 2)}% Clean)."
        else:
            final_message = f"System Integrity Check: RED (MAGENTA NEON). Messy confidence {round(conf_messy * 100, 2)}%. Clutter detected. Recommendation: De-clutter immediately."
    
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
    
    st.session_state.execution_log_data = format_execution_log(results, st.session_state.uploaded_file.name)
    
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
            "color": NEON_CYAN,
            "content": f"""
                <p>Selamat! Ruangan Anda menunjukkan tingkat kerapihan yang luar biasa. Sistem kami mendeteksi sedikit atau tidak ada <b>UNOPTIMIZED ASSET</b> (jumlah: {messy_count}).</p>
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
             override_note = f"<p style='color:{NEON_MAGENTA}; font-weight:bold;'>CATATAN SISTEM: Meskipun klasifikasi CNN awal mungkin 'Clean', YOLOv8 mendeteksi {messy_count} item unoptimized yang signifikan, memicu Aturan Hibrida OVERRIDE ke status ALERT.</p>"
        else:
             override_note = ""

        return {
            "title": "🚨 STATUS ALERT: SARAN OPTIMASI RUANGAN",
            "icon": "🧹",
            "color": NEON_MAGENTA,
            "content": f"""
                {override_note}
                <p>Ruangan Anda teridentifikasi sebagai <b>UNOPTIMIZED / MESSY</b>. Ini menunjukkan adanya aset-aset yang perlu dikelola ulang. Model Deteksi kami mengidentifikasi <b>{messy_count} item UNOPTIMIZED</b>.</p>
                <p><b>Rekomendasi Tindakan (De-Clutter Protocol):</b></p>
                <ul>
                    <li>Fokus pada Aset Berisiko: Prioritaskan merapikan item yang terdeteksi (lihat tabel di bawah, seperti **Scattered Clothes** atau **Unsorted Papers**).</li>
                    <li>Prinsip 4 Kotak: Gunakan 4 kotak: Sampah, Donasi, Simpan (jauh), dan Simpan (di sini). Segera distribusikan aset berdasarkan kategori ini.</li>
                    <li>Re-Scan: Setelah merapikan, muat ulang gambar ruangan Anda dan jalankan analisis kembali untuk memverifikasi Status Optimal.</li>
                </ul>
            """
        }

# --- 8. Fungsi Render Halaman (Pemisahan UI) ---

def render_upload_page():
    """Halaman 1: Upload Gambar Saja."""
    
    st.markdown(f"""
        <header>
            <h1>ROOM INSIGHT <span style="font-size: 18px; margin-left: 15px; color: {ACCENT_PRIMARY_NEON};">CLEAN OR MESSY?</span></h1>
            <p style="color: {TEXT_LIGHT}; font-size: 16px;">Klasifikasikan kerapihan ruangan Anda menggunakan arsitektur model ganda (Deteksi + Klasifikasi).</p>
        </header>
        <div style="margin-bottom: 40px;"></div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">1. Data Input Matrix (Upload Payload)</h2>', unsafe_allow_html=True)

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
    
    if st.button("⚡ INITIATE DUAL-MODEL ANALYSIS", disabled=button_disabled, use_container_width=True):
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
            <h1>ANALYSIS REPORT: <span style="font-size: 18px; margin-left: 15px; color: {ACCENT_PRIMARY_NEON};">{st.session_state.uploaded_file.name.upper()}</span></h1>
            <p style="color: {TEXT_LIGHT}; font-size: 14px;">Laporan lengkap hasil deteksi objek (YOLOv8) dan klasifikasi kerapihan (CNN).</p>
        </header>
        <div style="margin-bottom: 20px;"></div>
        """, unsafe_allow_html=True)

    # --- 1. GAMBAR YANG DIUNGGAH (ATAS) ---
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">1. Visualisasi Deteksi Objek</h2>', unsafe_allow_html=True)

    border_class = 'clean-border' if results['is_clean'] else 'messy-border'
    
    st.markdown(f"""
        <div style="border: 4px solid #34495E; border-radius: 10px; padding: 5px; background-color: {BG_DARK}; margin-bottom: 20px;" class="{border_class}">
        """, unsafe_allow_html=True)

    if st.session_state.processed_image:
        st.image(st.session_state.processed_image, caption='VISUALIZATION: Live Detection Grid (YOLO v8 Output)', use_container_width=True)
    else:
        image_data = st.session_state.uploaded_file.getvalue()
        st.image(image_data, caption='Image Data Stream (Original)', use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f"<hr style='border-top: 1px solid {ACCENT_PRIMARY_NEON}; box-shadow: 0 0 5px {ACCENT_PRIMARY_NEON}; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)

    # --- 2. KLASIFIKASI FINAL STATUS (UTAMA) ---
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">2. Final Classification Status</h2>', unsafe_allow_html=True)

    col_report, col_clean_conf, col_messy_conf = st.columns([2, 1, 1])

    status_main_text = results['final_status'].split(': ')[1]
    css_class_status = 'clean-status-text' if results['is_clean'] else 'messy-status-text'
    message = results['final_message']
    
    with col_report:
        border_color = NEON_CYAN if results['is_clean'] else NEON_MAGENTA
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {border_color}; box-shadow: 0 0 10px {border_color};">
                <p style="color: {TEXT_LIGHT}; font-size: 14px; margin-bottom: 5px; font-weight: bold;">CLASSIFICATION REPORT (Final Status)</p>
                <p class="{css_class_status}" style="font-size: 32px; margin-top: 5px;">{status_main_text}</p>
                <p style="font-size: 12px; color: {TEXT_LIGHT}; opacity: 0.7;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col_clean_conf:
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {NEON_CYAN}; background-color: {CARD_BG}; box-shadow: 0 0 8px {NEON_CYAN};">
                <p style="color: {NEON_CYAN}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">CONFIDENCE: CLEAN</p>
                <p style="color: {TEXT_LIGHT}; font-size: 28px; font-weight: bold;">{results["conf_clean"]}%</p>
                <p style="color: {TEXT_LIGHT}; font-size: 10px; margin-top: 5px; opacity: 0.6;">(From Model {results['classification_model']})</p>
            </div>
            """, unsafe_allow_html=True)
        
    with col_messy_conf:
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {NEON_MAGENTA}; background-color: {CARD_BG}; box-shadow: 0 0 8px {NEON_MAGENTA};">
                <p style="color: {NEON_MAGENTA}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">CONFIDENCE: MESSY</p>
                <p style="color: {TEXT_LIGHT}; font-size: 28px; font-weight: bold;">{results["conf_messy"]}%</p>
                <p style="color: {TEXT_LIGHT}; font-size: 10px; margin-top: 5px; opacity: 0.6;">(From Model {results['classification_model']})</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown(f"<hr style='border-top: 1px solid {ACCENT_PRIMARY_NEON}; box-shadow: 0 0 5px {ACCENT_PRIMARY_NEON}; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    # --- 3. TIPS / APRESIASI ---
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">3. Tindakan Rekomendasi</h2>', unsafe_allow_html=True)
    
    tips = get_tips_and_appreciation(results['is_clean'], results['messy_count'], results.get('is_overridden', False))
    
    st.markdown(f"""
        <div class="tips-box" style="background-color: {CARD_BG}; border-color: {tips['color']}; box-shadow: 0 0 10px rgba({tips['color'][1:]}, 0.5);">
            <h3 style="color: {tips['color']};">{tips['icon']} {tips['title']}</h3>
            {tips['content']}
        </div>
        """, unsafe_allow_html=True)

    # --- 4. LOG DAN DETAIL TABEL ---
    st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Log Eksekusi Model Ganda</h3>', unsafe_allow_html=True)

    log_content = st.session_state.execution_log_data
    st.markdown(f"""
        <div class="log-container">
            {log_content}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Tabel Detail Aset Terdeteksi ({results["detection_model"]})</h3>', unsafe_allow_html=True)

    df = pd.DataFrame(results['detections'])
    df = df.rename(columns={
        'asset_id': 'Asset ID (Deteksi)', 
        'confidence_score': 'Conf. Deteksi (%)', 
        'classification_tag': 'Tag Kerapihan',
        'normalized_coordinates': 'Koordinat Norm. (x, y, w, h)'
    })
    df['Conf. Deteksi (%)'] = (df['Conf. Deteksi (%)'] * 100).round(2).astype(str) + '%'
    
    st.dataframe(
        df, 
        use_container_width=True,
        height=250 
    )

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
