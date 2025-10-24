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

# --- 2. Konfigurasi dan Styling (Tema Soft Light / Muted Green) ---

st.set_page_config(
    page_title="SPATIAL AUDIT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Warna Baru (Soft Light / Muted Green Theme) ---
BG_DARK = "#F5F7F9"              # Soft Light Gray / Off-White (Main Background)
CARD_BG = "#FFFFFF"              # Pure White Card Background
TEXT_LIGHT = "#334155"           # Dark Slate Blue / Main Text
ACCENT_BLUE = "#10B981"          # Muted Emerald Green (Primary Action / Clean Status)
ALERT_RED = "#EF4444"            # Standard Red (Used for Messy Status)
BUTTON_COLOR = "#10B981"
GRID_LINE = "#CBD5E1"            # Light Border Gray

custom_css = f"""
<style>
    /* Global Styles */
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_LIGHT};
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, h4 {{
        color: {TEXT_LIGHT};
        font-weight: 600;
    }}
    h1 {{
        color: {ACCENT_BLUE};
        border-bottom: 3px solid {GRID_LINE};
        padding-bottom: 15px;
        letter-spacing: 2px;
        text-transform: uppercase;
        /* Shadow disesuaikan untuk light mode */
        text-shadow: 0 0 3px rgba(16, 185, 129, 0.2); 
    }}
    
    /* Card/Module Style */
    .data-module-card {{
        background-color: {CARD_BG};
        border: 1px solid {GRID_LINE};
        /* Box shadow disesuaikan untuk light mode */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); 
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 20px;
    }}
    .data-module-card h2 {{
        color: {ACCENT_BLUE};
        border-bottom: 1px dashed {GRID_LINE};
        padding-bottom: 10px;
    }}
    
    /* File Uploader Style */
    [data-testid="stFileUploader"] {{
        min-height: 180px; 
        padding: 10px 20px 10px 20px !important;
        margin-top: 10px;
        border: 2px dashed {ACCENT_BLUE} !important; 
        border-radius: 4px;
        background-color: {BG_DARK} !important; /* Gunakan BG_DARK (soft light gray) */
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {CARD_BG} !important;
        color: {ACCENT_BLUE} !important;
        font-weight: bold !important;
        border: 1px solid {ACCENT_BLUE} !important;
        border-radius: 4px !important;
        transition: all 0.3s !important;
    }}

    /* Main Action Button Style */
    .stButton > button:nth-child(1) {{
        background-color: {BUTTON_COLOR}; 
        color: {CARD_BG} !important; /* Text putih agar kontras dengan hijau */
        font-weight: bold;
        border-radius: 4px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); /* Shadow disesuaikan */
        text-transform: uppercase;
    }}
    .stButton > button:nth-child(1):hover {{
        background-color: #0d946d; /* Darker shade of emerald */
    }}

    /* Status and Metric Cards */
    .status-box {{
        padding: 15px 20px;
        border-radius: 4px;
        font-weight: bold;
        margin-top: 10px;
        text-align: center;
        border: 2px solid;
    }}
    .status-clean {{
        /* Muted but visible on light background */
        background-color: rgba(16, 185, 129, 0.1); 
        border-color: {ACCENT_BLUE};
        color: {ACCENT_BLUE};
    }}
    .status-messy {{
        /* Muted but visible on light background */
        background-color: rgba(239, 68, 68, 0.1); 
        border-color: {ALERT_RED};
        color: {ALERT_RED};
    }}
    .status-text-large {{
        font-size: 32px;
        font-weight: 800;
        margin: 5px 0;
    }}

    /* Log Console */
    .log-container {{
        background-color: {CARD_BG}; /* White background */
        color: {TEXT_LIGHT}; /* Dark text */
        border: 1px solid {GRID_LINE};
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        padding: 15px;
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        height: 140px;
        overflow-y: auto;
    }}
    /* Recommendations */
    .recommendation-list {{
        list-style-type: none;
        padding-left: 0;
    }}
    .recommendation-list li {{
        margin-bottom: 10px;
        padding: 10px;
        border-left: 3px solid {ACCENT_BLUE};
        background-color: {BG_DARK}; /* Soft light gray background for list items */
        border-radius: 2px;
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
    CLEAN_RGB = ImageColor.getrgb(ACCENT_BLUE)
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
    log_lines = []
    
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] SYS_INIT: Audit System Initialized.")
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 4))}] DATA_LOAD: Input Payload Acquired: {uploaded_file_name}.")

    # Log Model 1: Deteksi
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 3))}] MODEL_A (YOLOv8): Loading Detection Core <b>{results['detection_model']}</b>.")
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 2))}] INFERENCE_DET: Detected {len(results['detections'])} assets. UNOPTIMIZED count: {results['messy_count']}.")
    
    # Log Model 2: Klasifikasi
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 1))}] MODEL_B (CNN): Loading Classification Core <b>{results['classification_model']}</b>.")
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] INFERENCE_CLASS: Clean Conf: {results['conf_clean']}%, Messy Conf: {results['conf_messy']}%.")
    
    # Log Hybrid Rule
    tag_color = ACCENT_BLUE if results['is_clean'] else ALERT_RED
    
    if results.get('is_overridden'):
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] <span style='color:{ALERT_RED};'>AUDIT_RULE: Hybrid Override Triggered! High clutter count ({results['messy_count']}) forces MESSY status.</span>")
        
    final_status = results['final_status'].split(': ')[1]
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] AUDIT_REPORT: Final Status: <span style='color:{tag_color};'><b>{final_status}</b></span>.")
    
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
    log_placeholder.markdown(f'<p style="color: {TEXT_LIGHT};">SYS_AUDIT> Initiating inference. Loading models...</p>', unsafe_allow_html=True)
    
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
        <p style="color: {TEXT_LIGHT}; font-size: 16px;">Analisis Integritas Spasial A.I. Dual-Model (Deteksi Objek + Klasifikasi)</p>
    </header>
    <div style="margin-bottom: 25px;"></div>
    """, unsafe_allow_html=True)


col_input, col_detection = st.columns([1, 1]) 

with col_input:
    st.markdown('<div class="data-module-card">', unsafe_allow_html=True)
    st.markdown(f'<h2><span style="color: {ACCENT_BLUE};">01.</span> INPUT MATRIX</h2>', unsafe_allow_html=True)
    
    # Status Payload
    file_display = st.session_state.uploaded_file.name if st.session_state.uploaded_file else "AWAITING IMAGE FILE"
    status_card_class = "status-box status-clean" if st.session_state.uploaded_file else "status-box"
    st.markdown(f"""
        <div class="status-box" style="border: 1px dashed {GRID_LINE};">
            <p style="margin: 0; font-weight: bold; font-size: 14px; color:{TEXT_LIGHT};">PAYLOAD STATUS:</p>
            <p style="margin: 0; font-size: 14px; font-weight: 500; color:{ACCENT_BLUE if st.session_state.uploaded_file else TEXT_LIGHT};">{file_display}</p>
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
    st.markdown(f'<h2><span style="color: {ACCENT_BLUE};">02.</span> VISUAL DETECTOR GRID</h2>', unsafe_allow_html=True)

    # Tambahkan border/styling pada gambar
    image_style = f"border: 2px solid {ACCENT_BLUE if st.session_state.analysis_results and st.session_state.analysis_results['is_clean'] else ALERT_RED if st.session_state.analysis_results else GRID_LINE}; border-radius: 4px; padding: 5px; background-color: {BG_DARK};"

    st.markdown(f'<div style="{image_style}">', unsafe_allow_html=True)

    if st.session_state.uploaded_file:
        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, caption='VISUALIZATION: Bounding Box Output (YOLO v8)', use_container_width=True)
        else:
            image_data = st.session_state.uploaded_file.getvalue()
            st.image(image_data, caption='VISUALIZATION: Original Image Stream', use_container_width=True)
            
    else:
        # URL Placeholder disesuaikan dengan skema warna baru
        placeholder_bg = BG_DARK.replace('#', '')
        placeholder_text = TEXT_LIGHT.replace('#', '')
        st.image(f"https://placehold.co/1200x675/{placeholder_bg}/{placeholder_text}?text=UPLOAD+IMAGE+TO+ACTIVATE+AUDIT+SCANNER", caption="Awaiting Input Data Stream", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- Separator ---
st.markdown(f"<hr style='border-top: 1px dashed {GRID_LINE}; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True) 

# --- TAMPILAN 2: EXECUTION REPORT & METRICS ---
st.markdown('<div class="data-module-card" style="margin-top: 20px;">', unsafe_allow_html=True)
st.markdown(f'<h2><span style="color: {ACCENT_BLUE};">03.</span> AUDIT REPORT & METRICS</h2>', unsafe_allow_html=True)

results = st.session_state.analysis_results

col_report, col_conf_metrics = st.columns([1, 1])

with col_report:
    st.markdown(f'<h4>{f"FINAL AUDIT STATUS" if results else "Status Analysis"}</h4>', unsafe_allow_html=True)

    if results:
        status_main_text = results['final_status'].split(': ')[1]
        css_class_status = 'status-clean' if results['is_clean'] else 'status-messy'
        message = results['final_message']
        
        # Mengubah warna teks pesan menjadi TEXT_LIGHT agar kontras di light mode
        message_color = TEXT_LIGHT 
        
        st.markdown(f"""
            <div class="status-box {css_class_status}" style="text-align: left; padding: 20px; border-width: 3px;">
                <p style="margin: 0; font-size: 14px; font-weight: 500;">REPORT SUMMARY:</p>
                <p class="status-text-large" style="color: inherit; margin-bottom: 15px;">{status_main_text}</p>
                <p style="font-size: 13px; color: {message_color}; font-weight: 500; opacity: 0.8;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown(f"""
            <div class="status-box" style="border: 2px dashed {GRID_LINE}; padding: 30px;">
                <p style="color:{GRID_LINE}; font-weight: bold; margin: 0;">-- INFERENCE DATA PENDING --</p>
            </div>
            """, unsafe_allow_html=True)

with col_conf_metrics:
    st.markdown(f'<h4>{f"CONFIDENCE SCORES" if results else "Confidence Scores"}</h4>', unsafe_allow_html=True)
    if results:
        col_clean, col_messy = st.columns(2)
        with col_clean:
            st.markdown(f"""
                <div class="status-box status-clean">
                    <p style="color: {ACCENT_BLUE}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">CLEAN CONFIDENCE</p>
                    <p style="color: {TEXT_LIGHT}; font-size: 28px; font-weight: bold;">{results["conf_clean"]}%</p>
                    <p style="color: {TEXT_LIGHT}; font-size: 10px; opacity: 0.6;">(From Model B)</p>
                </div>
                """, unsafe_allow_html=True)
        with col_messy:
            st.markdown(f"""
                <div class="status-box status-messy">
                    <p style="color: {ALERT_RED}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">MESSY CONFIDENCE</p>
                    <p style="color: {TEXT_LIGHT}; font-size: 28px; font-weight: bold;">{results["conf_messy"]}%</p>
                    <p style="color: {TEXT_LIGHT}; font-size: 10px; opacity: 0.6;">(From Model B)</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="status-box" style="border: 2px dashed {GRID_LINE}; padding: 30px; margin-top: 10px;">
                <p style="color:{GRID_LINE}; font-weight: bold; margin: 0;">-- NO DATA --</p>
            </div>
            """, unsafe_allow_html=True)

# --- Log and Table Section ---
col_log, col_recommendation = st.columns([1, 1])

with col_log:
    st.markdown(f'<h3 style="color: {TEXT_LIGHT}; font-size: 20px; margin-top: 25px; border-bottom: 1px dashed {GRID_LINE}; padding-bottom: 5px;">Execution Log (Model Pipeline)</h3>', unsafe_allow_html=True)
    
    log_content = st.session_state.execution_log_data or f"""[{time.strftime('%H:%M:%S')}] SYS_INIT: Audit System Initialized. Awaiting Input Payload.<br>
[{time.strftime('%H:%M:%S')}] DATA_LOAD: No active payload detected.<br>
[{time.strftime('%H:%M:%S')}] MODEL: Models are idle. Please execute audit."""
    
    st.markdown(f"""
        <div class="log-container">
            {log_content}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f'<h3 style="color: {TEXT_LIGHT}; font-size: 20px; margin-top: 25px; border-bottom: 1px dashed {GRID_LINE}; padding-bottom: 5px;">Detected Asset Details (Model A)</h3>', unsafe_allow_html=True)

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
            color = f'background-color: rgba(239, 68, 68, 0.1); color: {ALERT_RED}; font-weight: bold;' if val == 'UNOPTIMIZED' else f'color: {ACCENT_BLUE};'
            return color
            
        st.dataframe(
            df.style.applymap(highlight_messy, subset=['Tag Kerapihan']), 
            use_container_width=True,
            height=200 
        )

    else:
        st.info("Tabel detail aset akan muncul setelah analisis berhasil dijalankan.")


with col_recommendation:
    st.markdown(f'<h3 style="color: {TEXT_LIGHT}; font-size: 20px; margin-top: 25px; border-bottom: 1px dashed {GRID_LINE}; padding-bottom: 5px;">Spatial Optimization Recommendations</h3>', unsafe_allow_html=True)
    
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
            <div class="status-box" style="border: 2px dashed {GRID_LINE}; padding: 20px; margin-top: 10px; height: 300px;">
                <p style="color:{GRID_LINE}; font-weight: bold; margin: 0; text-align: center;">-- REKOMENDASI HANYA AKAN MUNCUL SETELAH AUDIT DATA SELESAI --</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
