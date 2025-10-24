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
    # Memuat pustaka ML yang diperlukan
    from ultralytics import YOLO
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    ML_LIBRARIES_LOADED = True
except ImportError:
    st.error("GAGAL PENTING: Pustaka 'ultralytics' atau 'tensorflow' tidak ditemukan. Analisis ML TIDAK DAPAT DILANJUTKAN.")
    ML_LIBRARIES_LOADED = False

# --- 2. Konfigurasi dan Styling (Tema Soft Light / Muted Green YANG LEBIH DINAMIS) ---

st.set_page_config(
    page_title="SPATIAL AUDIT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Warna Baru (Soft Light / Muted Green Theme - Updated) ---
BG_DARK = "#F5F7F9"              # Soft Light Gray / Off-White (Main Background)
CARD_BG = "#FFFFFF"              # Pure White Card Background
TEXT_DARK = "#1F2937"            # Darker Slate Blue / Near-Black (Main Text)
ACCENT_GREEN = "#10B981"         # Muted Emerald Green (Primary Action / Clean Status)
SECONDARY_LIGHT = "#E0F2F1"      # Very light Teal/Mint for subtle section background
ALERT_RED = "#EF4444"            # Standard Red (Used for Messy Status)
GRID_LINE = "#CBD5E1"            # Light Border Gray (Subtle separator)

custom_css = f"""
<style>
    /* Global Styles */
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_DARK};
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, h4 {{
        color: {TEXT_DARK};
        font-weight: 700; /* Lebih tebal untuk hierarki */
    }}
    
    /* Main Header - Lebih Dinamis */
    h1 {{
        color: {ACCENT_GREEN};
        border-bottom: 3px solid {ACCENT_GREEN}; /* Border aksen */
        padding-bottom: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-size: 2.5em;
        text-shadow: 2px 2px 5px rgba(16, 185, 129, 0.2); 
    }}
    
    /* Section Headers - Lebih halus */
    .data-module-card h2 {{
        color: {TEXT_DARK};
        font-weight: 700;
        border-bottom: 2px dashed {SECONDARY_LIGHT}; /* Border yang lebih lembut */
        padding-bottom: 8px;
        margin-bottom: 15px;
    }}
    .data-module-card h2 span {{
        color: {ACCENT_GREEN};
        font-size: 1.1em;
        margin-right: 5px;
    }}
    
    /* Card/Module Style - Lebih Rounded & Angkat */
    .data-module-card {{
        background-color: {CARD_BG};
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); /* Shadow yang lebih dalam dan bersih */
        border-radius: 10px; /* Sudut lebih membulat */
        padding: 25px;
        margin-bottom: 25px;
    }}
    
    /* File Uploader Style - Lebih menonjol */
    [data-testid="stFileUploader"] {{
        min-height: 200px; 
        padding: 20px !important;
        margin-top: 15px;
        border: 3px dashed {ACCENT_GREEN} !important; 
        border-radius: 10px;
        background-color: {SECONDARY_LIGHT} !important; /* Gunakan warna sekunder */
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {CARD_BG} !important;
        color: {ACCENT_GREEN} !important;
        font-weight: bold !important;
        border: 1px solid {ACCENT_GREEN} !important;
        border-radius: 8px !important;
        transition: all 0.3s !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }}

    /* Main Action Button Style - Lebih Bold */
    .stButton > button:nth-child(1) {{
        background-color: {ACCENT_GREEN}; 
        color: {CARD_BG} !important;
        font-weight: 800; /* Extra bold */
        border-radius: 8px;
        border: none;
        padding: 12px 25px;
        transition: all 0.3s;
        box-shadow: 0 5px 15px rgba(16, 185, 129, 0.4); 
        text-transform: uppercase;
    }}
    .stButton > button:nth-child(1):hover {{
        background-color: #0d946d; 
        transform: translateY(-2px); /* Efek hover */
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.6);
    }}

    /* Status and Metric Cards - Lebih Jelas */
    .status-box {{
        padding: 20px 25px;
        border-radius: 8px; /* Sudut lebih membulat */
        font-weight: 600;
        margin-top: 10px;
        text-align: center;
        border: 2px solid;
        transition: all 0.3s;
    }}
    .status-clean {{
        background-color: rgba(16, 185, 129, 0.1); 
        border-color: {ACCENT_GREEN};
        color: {ACCENT_GREEN};
    }}
    .status-messy {{
        background-color: rgba(239, 68, 68, 0.1); 
        border-color: {ALERT_RED};
        color: {ALERT_RED};
    }}
    .status-text-large {{
        font-size: 38px; /* Lebih besar */
        font-weight: 900;
        margin: 5px 0;
        letter-spacing: -1px;
    }}

    /* Log Console - Lebih Elegan */
    .log-container {{
        background-color: {TEXT_DARK}; /* Background gelap untuk log */
        color: #FFFFFF; /* Teks putih untuk kontras */
        border: 1px solid {ACCENT_GREEN}; /* Border aksen hijau */
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2); 
        padding: 15px;
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 12px;
        height: 160px; /* Sedikit lebih tinggi */
        overflow-y: auto;
    }}
    /* Recommendations - Menggunakan aksen yang lebih menonjol */
    .recommendation-list {{
        list-style-type: none;
        padding-left: 0;
    }}
    .recommendation-list li {{
        margin-bottom: 12px;
        padding: 15px;
        border-left: 5px solid {ACCENT_GREEN}; /* Border kiri yang tebal */
        background-color: {SECONDARY_LIGHT}; /* Soft secondary background */
        border-radius: 5px;
        color: {TEXT_DARK};
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    .recommendation-list li strong {{
        color: #0d946d; /* Darker green for bold text inside recs */
    }}
    
    /* Placeholder Status Card */
    .placeholder-status {{
        border: 2px dashed {GRID_LINE}; 
        background-color: {SECONDARY_LIGHT};
        border-radius: 8px;
        padding: 30px;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .placeholder-status p {{
        color:{GRID_LINE}; 
        font-weight: bold; 
        margin: 0; 
        text-align: center;
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
    st.session_state.execution_log_data = "" # Ubah ke string untuk kemudahan


# --- 3. Fungsi Pemuatan Model (Menggunakan Cache Resource Streamlit) ---
@st.cache_resource
def load_ml_model():
    """Memuat model YOLO dan CNN ke memori dengan caching."""
    if not ML_LIBRARIES_LOADED:
        return None, None
    
    try:
        # PENTING: Ganti path ini jika lokasi file Anda berbeda
        YOLO_MODEL_PATH = "model/Siti Naura Khalisa_Laporan 4.pt"
        CNN_MODEL_PATH = "model/SitiNauraKhalisa_Laporan2.h5"

        # Model 1: Deteksi Objek (YOLOv8)
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # Model 2: Klasifikasi Ruangan (Keras/CNN)
        tf.get_logger().setLevel('ERROR')
        # Setting run_eagerly=True is sometimes necessary for Keras models in Streamlit, 
        # but often slows down training/inference. Keep it commented unless needed.
        # cnn_model = load_model(CNN_MODEL_PATH, compile=False, custom_objects={}, options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))
        cnn_model = load_model(CNN_MODEL_PATH, compile=False)
        
        return yolo_model, cnn_model
    except Exception as e:
        st.error(f"FATAL ERROR: Gagal memuat file model dari path. Pastikan file berada di folder 'model/' dan namanya benar: {e}")
        return None, None

# --- 4. Fungsi Real Inference dan Pemrosesan Data ---

def run_yolo_detection(yolo_model, image_path):
    """Menjalankan inferensi YOLOv8 dan memproses hasilnya."""
    
    # Mendefinisikan ID Kelas yang dianggap "Messy" (Ganti sesuai nama kelas model Anda)
    MESSY_CLASS_NAMES = ["Scattered_Clothes", "Loose_Cables", "Unsorted_Papers", "Trash_Object", "Empty_Bottles", "Food_Wrapper"]
    
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
    
    target_size = (128, 128) # Pastikan ini sesuai dengan input model Anda
    image_resized = image.resize(target_size)
    
    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
    img_array = img_array / 255.0 # Normalisasi
    
    predictions = cnn_model.predict(img_array, verbose=0)[0]
    
    # ASUMSI: predictions[0] adalah CLEAN, predictions[1] adalah MESSY
    conf_clean = predictions[0] 
    conf_messy = predictions[1] 
    
    return conf_clean, conf_messy

# --- 5. Fungsi Utilitas Visualisasi dan Log ---

def draw_boxes_on_image(image_bytes, detections):
    """Menggambar Bounding Box pada Gambar dari hasil deteksi."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    
    # Warna Baru
    CLEAN_RGB = ImageColor.getrgb(ACCENT_GREEN)
    MESSY_RGB = ImageColor.getrgb(ALERT_RED)
    TEXT_RGB = ImageColor.getrgb(CARD_BG) # Putih/White untuk kontras pada fill box
    
    try:
        # Menggunakan font default PIL, sesuaikan ukuran
        font_size = max(15, min(image_width // 40, 30))
        font = ImageFont.load_default(size=font_size) 
    except IOError:
        font = ImageFont.load_default()
        
    for det in detections:
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
        
        # Gambar Bounding Box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_rgb, width=3) 
        
        # Gambar Label di atas kotak
        text_content = f"{label} {int(confidence * 100)}%"
        
        try:
            text_bbox = draw.textbbox((0, 0), text_content, font=font) 
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text_content, font=font)
            
        text_x = x_min
        text_y = y_min - text_height - 5 
        
        if text_y < 0:
            text_y = y_max + 5
            
        # Background untuk teks
        draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill=box_rgb)
        draw.text((text_x + 2, text_y + 2), text_content, font=font, fill=TEXT_RGB) 
        
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    return buf.getvalue()

def format_execution_log(results, uploaded_file_name):
    """Membuat format log tekstual yang menyerupai output konsol data."""
    # Ubah warna log sesuai tema baru (lebih kontras)
    LOG_BG_COLOR = TEXT_DARK # Warna background log adalah TEXT_DARK
    LOG_TEXT_COLOR = "#FFFFFF"
    LOG_ACCENT_COLOR = ACCENT_GREEN
    LOG_ALERT_COLOR = ALERT_RED
    
    log_lines = []
    
    # Menggunakan HTML tag span untuk styling log line per line
    def log_line(message, color=LOG_TEXT_COLOR):
        return f"[{time.strftime('%H:%M:%S')}] <span style='color:{color};'>{message}</span>"

    log_lines.append(log_line(f"SYS_INIT: Audit System Initialized."))
    log_lines.append(log_line(f"DATA_LOAD: Input Payload Acquired: {uploaded_file_name}.", "#9CA3AF")) # Grayer color for data load

    # Log Model 1: Deteksi
    log_lines.append(log_line(f"MODEL_A (YOLOv8): Loading Detection Core <b>{results['detection_model']}</b>.", LOG_ACCENT_COLOR))
    log_lines.append(log_line(f"INFERENCE_DET: Detected {len(results['detections'])} assets. UNOPTIMIZED count: {results['messy_count']}."))
    
    # Log Model 2: Klasifikasi
    log_lines.append(log_line(f"MODEL_B (CNN): Loading Classification Core <b>{results['classification_model']}</b>.", LOG_ACCENT_COLOR))
    log_lines.append(log_line(f"INFERENCE_CLASS: Clean Conf: {results['conf_clean']}%, Messy Conf: {results['conf_messy']}%.", "#9CA3AF"))
    
    # Log Hybrid Rule
    tag_color = ACCENT_GREEN if results['is_clean'] else LOG_ALERT_COLOR
    
    if results.get('is_overridden'):
        log_lines.append(log_line(f"AUDIT_RULE: Hybrid Override Triggered! High clutter count ({results['messy_count']}) forces MESSY status.", LOG_ALERT_COLOR))
        
    final_status = results['final_status'].split(': ')[1]
    log_lines.append(log_line(f"AUDIT_REPORT: Final Status: <b>{final_status}</b>.", tag_color))
    
    return '<br>'.join(log_lines)

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
    st.session_state.execution_log_data = ""

    
    # Placeholder log dan progress bar
    log_placeholder = st.empty()
    # Mengubah warna teks pesan menjadi TEXT_DARK agar kontras di light mode
    log_placeholder.markdown(f'<p style="color: {TEXT_DARK};">SYS_AUDIT> Initiating inference. Loading models...</p>', unsafe_allow_html=True) 
    
    progress_bar = st.progress(0, text="[PROCESS_FLOW] Loading Tensor Core & Running Inference...")
    
    # 2. Persiapan Data
    image_bytes = st.session_state.uploaded_file.getvalue()
    
    # Simpan file secara sementara untuk dibaca oleh YOLO
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_bytes)
        
    # --- ALUR KERJA ML NYATA ---
    
    # A. Model 1: Deteksi Objek (YOLOv8)
    progress_bar.progress(30, text="[PROCESS_FLOW] Executing YOLOv8 Detection...")
    try:
        detections, messy_count = run_yolo_detection(yolo_model, temp_path)
    except Exception as e:
        st.error(f"Error saat menjalankan Deteksi YOLOv8: {e}")
        progress_bar.empty()
        return

    # B. Model 2: Klasifikasi Akhir (Keras/CNN)
    progress_bar.progress(60, text="[PROCESS_FLOW] Executing Keras/CNN Classification...")
    try:
        conf_clean, conf_messy = run_cnn_classification(cnn_model, image_bytes)
    except Exception as e:
        st.error(f"Error saat menjalankan Klasifikasi CNN: {e}")
        progress_bar.empty()
        return
    
    progress_bar.progress(90, text="[PROCESS_FLOW] Post-processing results...")
    
    # --- PENENTUAN STATUS AKHIR (Berdasarkan CNN + Hybrid Rule) ---
    MESSY_DETECTION_THRESHOLD = 3 
    
    # 1. Penentuan Awal (Berdasarkan CNN)
    is_clean = conf_clean > conf_messy
        
    # 2. Implementasi Hybrid Rule: Override 'Clean' to 'Messy' jika hitungan YOLO tinggi
    is_overridden = False
    if is_clean and messy_count >= MESSY_DETECTION_THRESHOLD:
        is_clean = False # Override status
        is_overridden = True
    
    # 3. Final Status Assignment dan Pesan
    if is_clean:
        final_status = "STATUS: AUDIT PASS - OPTIMAL SPATIAL INTEGRITY"
        final_message = f"System Integrity: GREEN ({round(conf_clean * 100, 2)}% Clean Confidence). Excellent organization. (YOLO Unoptimized Count: {messy_count})."
    else:
        final_status = "STATUS: AUDIT FAIL - CRITICAL CLUTTER ALERT"
        
        if is_overridden:
            final_message = f"AUDIT OVERRIDE: YOLO detected {messy_count} UNOPTIMIZED assets. Status is ALERT, regardless of CNN score. Recommendation: Immediate action required."
        else:
            final_message = f"System Integrity: RED ({round(conf_messy * 100, 2)}% Messy Confidence). Significant clutter detected. Recommendation: De-clutter and re-audit."
    
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
        "is_overridden": is_overridden
    }
    
    progress_bar.progress(100, text="[PROCESS_FLOW] Analysis Complete. Generating Report.")
    progress_bar.empty()

    st.session_state.processed_image = processed_image_bytes
    st.session_state.analysis_results = results
    
    st.session_state.execution_log_data = format_execution_log(results, st.session_state.uploaded_file.name)
    
    log_placeholder.empty()
    st.success("SYS_AUDIT> Analysis Completed. Report Generated.")

# --- 7. Fitur Baru: Rekomendasi Optimasi Spasial ---

def generate_recommendations(results):
    """Menghasilkan saran berdasarkan hasil audit."""
    if results['is_clean']:
        return ["Ruangan Anda memiliki integritas spasial yang optimal. Pertahankan!"]
    
    # Jika Messy, hasilkan rekomendasi spesifik
    recs = []
    
    # Identifikasi item messy yang paling sering
    messy_items = [d['asset_id'] for d in results['detections'] if d['classification_tag'] == 'UNOPTIMIZED']
    item_counts = pd.Series(messy_items).value_counts()
    
    if item_counts.empty:
             # Ini terjadi jika override rule yang memicu 'Messy' tapi YOLO count masih rendah
             recs.append("Meskipun statusnya ALERT, clutter yang terdeteksi relatif tersebar. Fokus pada penataan ulang barang-barang kecil di permukaan meja.")
    else:
        # Rekomendasi berdasarkan item yang terdeteksi
        for item, count in item_counts.head(3).items():
            if "CLOTHES" in item:
                recs.append(f"Audit mendeteksi **{count} item pakaian** tidak tertata. Rekomendasi: Gunakan *hamper* atau segera lipat dan simpan ke dalam lemari untuk mengoptimalkan ruang vertikal.")
            elif "PAPER" in item:
                recs.append(f"Ditemukan **{count} dokumen/kertas** yang tidak terorganisir. Rekomendasi: Digitalisasi atau gunakan sistem folder berlabel (misalnya, *binder*) untuk meminimalkan *visual noise*.")
            elif "CABLES" in item:
                recs.append(f"Terdapat deteksi **kabel longgar**. Rekomendasi: Gunakan *cable ties* atau *cable management box* untuk menjaga jalur akses data tetap bersih dan aman.")
            elif "TRASH" in item or "BOTTLES" in item or "WRAPPER" in item:
                recs.append(f"Deteksi sisa **sampah atau pembungkus** ({item}). Rekomendasi: Pastikan tempat sampah diletakkan di lokasi yang strategis dan kosongkan setiap hari.")
            else:
                recs.append(f"Beberapa aset teridentifikasi sebagai UNOPTIMIZED. Lakukan audit manual di area sekitarnya untuk memulihkan kerapihan.")
                
    # Rekomendasi Umum
    if results['is_overridden']:
        recs.append("CATATAN: Status ALERT dipicu oleh kuantitas clutter, meskipun klasifikasi gambar terlihat Clean. Fokus pada pembersihan area berulang.")

    return recs

# --- 8. Tata Letak Streamlit (UI) ---

st.markdown(f"""
    <header>
        <h1>SPATIAL AUDIT INTERFACE</h1>
        <p style="color: {TEXT_DARK}; font-size: 18px; font-weight: 400;">— ANALISIS INTEGRITAS SPASIAL A.I. DUAL-MODEL —</p>
    </header>
    <div style="margin-bottom: 30px;"></div>
    """, unsafe_allow_html=True)


col_input, col_detection = st.columns([1, 1]) 

with col_input:
    st.markdown('<div class="data-module-card">', unsafe_allow_html=True)
    st.markdown(f'<h2><span style="color: {ACCENT_GREEN};">01.</span> INPUT MATRIX</h2>', unsafe_allow_html=True)
    
    # Status Payload - Lebih menonjolkan ACCENT_GREEN
    file_display = st.session_state.uploaded_file.name if st.session_state.uploaded_file else "AWAITING IMAGE FILE"
    
    st.markdown(f"""
        <div class="status-box" style="border: 1px dashed {GRID_LINE}; background-color: {SECONDARY_LIGHT};">
            <p style="margin: 0; font-weight: 700; font-size: 14px; color:{TEXT_DARK};">PAYLOAD STATUS:</p>
            <p style="margin: 0; font-size: 16px; font-weight: 600; color:{ACCENT_GREEN if st.session_state.uploaded_file else TEXT_DARK};">{file_display}</p>
        </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Image File (JPG/PNG) | Initiating Payload Protocol", 
        type=["jpg", "jpeg", "png"],
        key="uploader",
        help="Unggah file gambar ruangan (kamar, meja kerja, dll.) untuk dianalisis."
    )

    st.session_state.uploaded_file = uploaded_file

    st.markdown('</div>', unsafe_allow_html=True) 
    
    st.markdown('<div style="padding-top: 10px;">', unsafe_allow_html=True)
    
    # Tombol dinonaktifkan jika tidak ada file yang diunggah ATAU jika pustaka ML gagal dimuat
    button_disabled = st.session_state.uploaded_file is None or not ML_LIBRARIES_LOADED
    
    if st.button("EXECUTE DUAL-MODEL AUDIT", disabled=button_disabled, use_container_width=True):
        # PANGGIL FUNGSI ML NYATA
        run_ml_analysis() 
        
    st.markdown('</div>', unsafe_allow_html=True)


with col_detection:
    st.markdown('<div class="data-module-card">', unsafe_allow_html=True)
    st.markdown(f'<h2><span style="color: {ACCENT_GREEN};">02.</span> VISUAL DETECTOR GRID</h2>', unsafe_allow_html=True)

    # Tambahkan border/styling pada gambar
    
    if st.session_state.uploaded_file:
        # Tentukan border berdasarkan status hasil
        if st.session_state.analysis_results:
            border_color = ACCENT_GREEN if st.session_state.analysis_results['is_clean'] else ALERT_RED
            caption = 'VISUALIZATION: Bounding Box Output (YOLO v8)' if st.session_state.processed_image else 'VISUALIZATION: Original Image Stream'
        else:
            border_color = GRID_LINE
            caption = 'VISUALIZATION: Original Image Stream'
            
        image_style = f"border: 4px solid {border_color}; border-radius: 8px; padding: 5px; background-color: {BG_DARK}; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);"

        st.markdown(f'<div style="{image_style}">', unsafe_allow_html=True)

        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, caption=caption, use_container_width=True)
        else:
            image_data = st.session_state.uploaded_file.getvalue()
            st.image(image_data, caption=caption, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        # URL Placeholder disesuaikan dengan skema warna baru
        placeholder_bg = SECONDARY_LIGHT.replace('#', '')
        placeholder_text = TEXT_DARK.replace('#', '')
        st.image(f"https://placehold.co/1200x675/{placeholder_bg}/{placeholder_text}?text=UPLOAD+IMAGE+TO+ACTIVATE+AUDIT+SCANNER", caption="Awaiting Input Data Stream", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# --- Separator ---
st.markdown(f"<hr style='border-top: 2px solid {SECONDARY_LIGHT}; margin-top: 25px; margin-bottom: 25px;'>", unsafe_allow_html=True) 

# --- TAMPILAN 2: EXECUTION REPORT & METRICS ---
st.markdown('<div class="data-module-card" style="margin-top: 20px;">', unsafe_allow_html=True)
st.markdown(f'<h2><span style="color: {ACCENT_GREEN};">03.</span> AUDIT REPORT & METRICS</h2>', unsafe_allow_html=True)

results = st.session_state.analysis_results

col_report, col_conf_metrics = st.columns([1, 1])

with col_report:
    st.markdown(f'<h4>FINAL AUDIT STATUS</h4>', unsafe_allow_html=True)

    if results:
        status_main_text = results['final_status'].split(': ')[1]
        css_class_status = 'status-clean' if results['is_clean'] else 'status-messy'
        message = results['final_message']
        
        # Mengubah warna teks pesan menjadi TEXT_DARK agar kontras di light mode
        message_color = TEXT_DARK 
        
        st.markdown(f"""
            <div class="status-box {css_class_status}" style="text-align: left; padding: 25px; border-width: 4px;">
                <p style="margin: 0; font-size: 14px; font-weight: 600;">REPORT SUMMARY:</p>
                <p class="status-text-large" style="color: {TEXT_DARK}; margin-bottom: 15px;">{status_main_text}</p>
                <p style="font-size: 14px; color: {message_color}; font-weight: 500; opacity: 0.8;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        # Placeholder yang lebih baik
        st.markdown(f"""
            <div class="placeholder-status" style="height: 180px;">
                <p style="color:#A0AEC0; font-weight: 600; margin: 0; opacity: 0.8;">-- INFERENCE DATA PENDING --</p>
            </div>
            """, unsafe_allow_html=True)

with col_conf_metrics:
    st.markdown(f'<h4>CONFIDENCE SCORES</h4>', unsafe_allow_html=True)
    if results:
        col_clean, col_messy = st.columns(2)
        with col_clean:
            st.markdown(f"""
                <div class="status-box status-clean">
                    <p style="color: {ACCENT_GREEN}; font-size: 14px; margin-bottom: 5px; font-weight: 700;">CLEAN CONFIDENCE</p>
                    <p style="color: {TEXT_DARK}; font-size: 32px; font-weight: 800;">{results["conf_clean"]}%</p>
                    <p style="color: {TEXT_DARK}; font-size: 12px; opacity: 0.6;">(From Model B)</p>
                </div>
                """, unsafe_allow_html=True)
        with col_messy:
            st.markdown(f"""
                <div class="status-box status-messy">
                    <p style="color: {ALERT_RED}; font-size: 14px; margin-bottom: 5px; font-weight: 700;">MESSY CONFIDENCE</p>
                    <p style="color: {TEXT_DARK}; font-size: 32px; font-weight: 800;">{results["conf_messy"]}%</p>
                    <p style="color: {TEXT_DARK}; font-size: 12px; opacity: 0.6;">(From Model B)</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="placeholder-status" style="height: 180px; margin-top: 10px;">
                <p style="color:#A0AEC0; font-weight: 600; margin: 0; opacity: 0.8;">-- NO DATA --</p>
            </div>
            """, unsafe_allow_html=True)

# --- Log and Table Section ---
col_log, col_recommendation = st.columns([1, 1])

with col_log:
    st.markdown(f'<h3 style="color: {TEXT_DARK}; font-size: 24px; margin-top: 30px; border-bottom: 2px solid {SECONDARY_LIGHT}; padding-bottom: 8px;">Execution Log (Model Pipeline)</h3>', unsafe_allow_html=True)
    
    log_content = st.session_state.execution_log_data or f"""[{time.strftime('%H:%M:%S')}] SYS_INIT: Audit System Initialized. Awaiting Input Payload.<br>
[{time.strftime('%H:%M:%S')}] DATA_LOAD: No active payload detected.<br>
[{time.strftime('%H:%M:%S')}] MODEL: Models are idle. Please execute audit."""
    
    st.markdown(f"""
        <div class="log-container">
            {log_content}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f'<h3 style="color: {TEXT_DARK}; font-size: 24px; margin-top: 30px; border-bottom: 2px solid {SECONDARY_LIGHT}; padding-bottom: 8px;">Detected Asset Details (Model A)</h3>', unsafe_allow_html=True)

    if results:
        df = pd.DataFrame(results['detections'])
        df = df.rename(columns={
            'asset_id': 'Asset ID', 
            'confidence_score': 'Conf. (%)', 
            'classification_tag': 'Tag Kerapihan',
            'normalized_coordinates': 'Koordinat Norm. (x, y, w, h)'
        })
        df['Conf. (%)'] = (df['Conf. (%)'] * 100).round(2).astype(str) + '%'
        
        # Highlight Tag Kerapihan
        def highlight_messy(val):
            # Menggunakan warna teks ALERT_RED di background yang sangat tipis untuk light mode
            color = f'background-color: rgba(239, 68, 68, 0.1); color: {ALERT_RED}; font-weight: bold;' if val == 'UNOPTIMIZED' else f'color: {ACCENT_GREEN};'
            return color
            
        st.dataframe(
            df.style.applymap(highlight_messy, subset=['Tag Kerapihan']), 
            use_container_width=True,
            height=200 
        )

    else:
        st.info("Tabel detail aset akan muncul setelah analisis berhasil dijalankan.")


with col_recommendation:
    st.markdown(f'<h3 style="color: {TEXT_DARK}; font-size: 24px; margin-top: 30px; border-bottom: 2px solid {SECONDARY_LIGHT}; padding-bottom: 8px;">Spatial Optimization Recommendations</h3>', unsafe_allow_html=True)
    
    if results:
        recs = generate_recommendations(results)
        list_items = "".join([f"<li>{r}</li>" for r in recs])
        st.markdown(f"""
            <ul class="recommendation-list">
                {list_items}
            </ul>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="placeholder-status" style="height: 300px;">
                <p style="color:#A0AEC0; font-weight: 600; margin: 0; text-align: center; opacity: 0.8;">-- REKOMENDASI HANYA AKAN MUNCUL SETELAH AUDIT DATA SELESAI --</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
