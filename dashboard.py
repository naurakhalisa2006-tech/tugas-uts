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
    st.warning("Pustaka 'ultralytics' atau 'tensorflow' tidak ditemukan. Aplikasi akan berjalan dalam mode SIMULASI yang ditingkatkan.")
    ML_LIBRARIES_LOADED = False

# --- 2. Konfigurasi dan Styling (Tema Cyber Pastel / Vaporwave) ---
# ... (Bagian Styling dan CSS tetap sama) ...

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
        min-height: 150px; 
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
    }}
    .stButton > button:nth-child(1):hover {{
        background-color: #2980B9;
        box-shadow: 0 0 15px {NEON_CYAN}, 0 0 25px {NEON_CYAN};
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
    
    .clean-status-text {{ color: {TEXT_CLEAN_LIGHT}; font-weight: 900; font-size: 24px; text-shadow: 0 0 5px {TEXT_CLEAN_LIGHT}; }}
    .messy-status-text {{ color: {TEXT_MESSY_LIGHT}; font-weight: 900; font-size: 24px; text-shadow: 0 0 5px {TEXT_MESSY_LIGHT}; }}
    
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
def load_ml_models():
    """Memuat model YOLO dan CNN ke memori dengan caching."""
    if not ML_LIBRARIES_LOADED:
        return None, None # Kembali jika pustaka tidak dimuat
    
    try:
        # PENTING: Ganti path ini jika lokasi file Anda berbeda
        YOLO_MODEL_PATH = "models/Siti_Naura_Khalisa_Laporan_4.pt"
        CNN_MODEL_PATH = "models/SitiNauraKhalisa_Laporan2.h5"

        # Model 1: Deteksi Objek (YOLOv8)
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # Model 2: Klasifikasi Ruangan (Keras/CNN)
        # Suppress TensorFlow warnings/messages
        tf.get_logger().setLevel('ERROR')
        cnn_model = load_model(CNN_MODEL_PATH)
        
        return yolo_model, cnn_model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file berada di folder 'models/' dan pustaka terinstal: {e}")
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
    
    # Ambil hasil dari batch pertama (image_path tunggal)
    result = results[0] 
    
    detections = []
    messy_count = 0
    
    # Mendefinisikan ID Kelas yang dianggap "Messy" (Ganti sesuai nama kelas model Anda)
    # Contoh: 'Scattered_Clothes' mungkin memiliki class ID 1, 'Trash_Object' ID 5.
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
    
    # Keras/CNN memerlukan input dengan ukuran tetap (misalnya 224x224)
    target_size = (224, 224) 
    image_resized = image.resize(target_size)
    
    # Konversi ke array Numpy dan normalisasi (misalnya 0-1)
    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array = img_array / 255.0 # Normalisasi
    
    # Prediksi
    predictions = cnn_model.predict(img_array, verbose=0)[0]
    
    # Asumsikan output adalah (Conf_Clean, Conf_Messy)
    # Ganti urutan indeks ini (0 atau 1) sesuai dengan urutan kelas model CNN Anda
    conf_clean = predictions[0]  # Ganti indeks ini jika CLEAN bukan kelas 0
    conf_messy = predictions[1]  # Ganti indeks ini jika MESSY bukan kelas 1
    
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
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY-CNN: Final Classification Complete. Input: Raw Image + Detection Context.")
    
    # Log Final
    tag_color = NEON_CYAN if results['is_clean'] else NEON_MAGENTA
    final_status = results['final_status'].split(': ')[1]
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] REPORT: Final Status: <span style='color:{tag_color};'><b>{final_status}</b></span> (Clean Conf: {results['conf_clean']}%, Messy Conf: {results['conf_messy']}%).")
    
    return '<br>'.join(log_lines)

# --- 6. Fungsi Utama Alur Kerja ML (Menggantikan Simulasi) ---

def run_ml_analysis():
    """Menggantikan simulate_yolo_analysis dengan alur kerja ML yang nyata."""
    if st.session_state.uploaded_file is None:
        st.error("Sila muat naik imej ruangan dahulu.")
        return
    
    # 1. Muat Model
    yolo_model, cnn_model = load_ml_models()
    
    if yolo_model is None or cnn_model is None:
        # Jika model gagal dimuat, kembali ke simulasi jika perlu, atau hentikan.
        st.error("Gagal menjalankan analisis ML: Model tidak tersedia atau gagal dimuat.")
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
    with open("temp_upload.jpg", "wb") as f:
        f.write(image_bytes)
        
    # --- ALUR KERJA ML NYATA ---
    
    # A. Model 1: Deteksi Objek (YOLOv8)
    progress_bar.progress(30, text="[Step 1/2] Executing YOLOv8 Detection...")
    try:
        detections, messy_count = run_yolo_detection(yolo_model, "temp_upload.jpg")
    except Exception as e:
        st.error(f"Error saat menjalankan Deteksi YOLOv8: {e}")
        progress_bar.empty()
        return

    # B. Model 2: Klasifikasi Akhir (Keras/CNN)
    progress_bar.progress(60, text="[Step 2/2] Executing Keras/CNN Classification...")
    try:
        conf_clean, conf_messy = run_cnn_classification(cnn_model, image_bytes)
    except Exception as e:
        st.error(f"Error saat menjalankan Klasifikasi CNN: {e}")
        progress_bar.empty()
        return
    
    progress_bar.progress(90, text="[Step 2/2] Post-processing results...")
    
    # --- PENENTUAN STATUS AKHIR (Berdasarkan CNN) ---
    if conf_clean > conf_messy:
        final_status = "STATUS: ROOM CLEAN - OPTIMAL"
        is_clean = True
        final_message = f"System Integrity Check: GREEN (CYAN NEON). Cleanliness confidence {round(conf_clean * 100, 2)}%. Excellent organization."
    else:
        final_status = "STATUS: ROOM MESSY - ALERT"
        is_clean = False
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
        "final_message": final_message
    }
    
    progress_bar.progress(100, text="Analysis Complete. Generating Report.")
    progress_bar.empty()

    st.session_state.processed_image = processed_image_bytes
    st.session_state.analysis_results = results
    
    st.session_state.execution_log_data = format_execution_log(results, st.session_state.uploaded_file.name)
    
    log_placeholder.empty()
    st.success("SYSTEM> Analysis Completed. Report Generated and Visualization Rendered.")
    
# --- 7. Tata Letak Streamlit (UI) ---
# ... (Sama seperti file sebelumnya, dengan fungsi simulasi yang diganti) ...

st.markdown(f"""
    <header>
        <h1>ROOM INSIGHT <span style="font-size: 18px; margin-left: 15px; color: {ACCENT_PRIMARY_NEON};">CLEAN OR MESSY?</span></h1>
        <p style="color: {TEXT_LIGHT}; font-size: 14px;">Klasifikasikan kerapihan ruangan Anda menggunakan arsitektur model ganda (Deteksi + Klasifikasi).</p>
        <p style="color: {NEON_MAGENTA}; font-size: 12px; font-weight: bold;">CATATAN: Pastikan Anda telah menginstal 'ultralytics' dan 'tensorflow' dan menempatkan file model di folder 'models/'.</p>
    </header>
    <div style="margin-bottom: 20px;"></div>
    """, unsafe_allow_html=True)


col_input, col_detection = st.columns([1, 1]) 

with col_input:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">1. Data Input Matrix</h2>', unsafe_allow_html=True)
    
    status_card_class = "info-card-clean" if st.session_state.uploaded_file else "info-card-messy"
    status_text = "PAYLOAD ACQUIRED" if st.session_state.uploaded_file else "AWAITING PAYLOAD"
    file_display = st.session_state.uploaded_file.name if st.session_state.uploaded_file else "No file uploaded"
    
    st.markdown(f"""
        <div class="status-metric-card {status_card_class}">
            <p style="margin: 0; font-weight: bold; font-size: 14px; color:{BG_DARK};">{status_text}</p>
            <p style="margin: 0; font-size: 14px; font-weight: 500; color:{BG_DARK};">{file_display}</p>
        </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Image File (JPG/PNG) | Initiating Payload Protocol", 
        type=["jpg", "jpeg", "png"],
        key="uploader",
        help="Unggah file gambar ruangan untuk dianalisis."
    )

    st.session_state.uploaded_file = uploaded_file

    st.markdown('</div>', unsafe_allow_html=True) 
    
    st.markdown('<div style="padding-top: 20px;">', unsafe_allow_html=True)
    
    button_disabled = st.session_state.uploaded_file is None
    
    if st.button("⚡ INITIATE DUAL-MODEL ANALYSIS", disabled=button_disabled, use_container_width=True):
        # PANGGIL FUNGSI ML NYATA
        run_ml_analysis() 
        
    st.markdown('</div>', unsafe_allow_html=True)


with col_detection:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">2. Live Detection Grid </h2>', unsafe_allow_html=True)

    border_class = ""
    caption_text = "Awaiting Image Payload"
    caption_style = ACCENT_PRIMARY_NEON
    
    if st.session_state.analysis_results is not None:
        border_class = 'clean-border' if st.session_state.analysis_results['is_clean'] else 'messy-border'
        caption_text = 'BOUNDING BOXES: (Rendered Status Visualization)'
        caption_style = NEON_CYAN if st.session_state.analysis_results['is_clean'] else NEON_MAGENTA
    elif st.session_state.uploaded_file:
        caption_text = 'BOUNDING BOXES: (Ready for Analysis - Click Initiate)'
        caption_style = ACCENT_PRIMARY_NEON

    st.markdown(f"""
        <div style="border: 4px solid #34495E; border-radius: 10px; padding: 5px; background-color: {BG_DARK};" class="{border_class}">
        """, unsafe_allow_html=True)

    if st.session_state.uploaded_file:
        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, caption='VISUALIZATION: Live Detection Grid (YOLO v8 Output)', use_container_width=True)
        else:
            image_data = st.session_state.uploaded_file.getvalue()
            st.image(image_data, caption='Image Data Stream (Original)', use_container_width=True)
            
        st.markdown(f'<p style="text-align: center; color: {caption_style}; font-weight: bold; margin-top: 10px; text-shadow: 0 0 3px {caption_style};">'
                    f'{caption_text}</p>', unsafe_allow_html=True)
    else:
        st.image("https://placehold.co/1200x675/1A1A2E/4DFFFF?text=UPLOAD+IMAGE+TO+ACTIVATE+SCANNER+MODULE", caption="Awaiting Image Payload", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(f"<hr style='border-top: 1px solid {ACCENT_PRIMARY_NEON}; box-shadow: 0 0 5px {ACCENT_PRIMARY_NEON}; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True) 

# --- TAMPILAN 2: EXECUTION REPORT & METRICS ---
st.markdown('<div class="modern-card" style="margin-top: 30px;">', unsafe_allow_html=True)
st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON}; border-bottom: 2px solid {ACCENT_PRIMARY_NEON}; padding-bottom: 10px;">Execution Report & Metrics</h2>', unsafe_allow_html=True)

results = st.session_state.analysis_results

col_report, col_clean_conf, col_messy_conf = st.columns([2, 1, 1])

if results:
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

else:
    st.markdown(f"""
        <div style="text-align: center; padding: 40px; border: 2px dashed {ACCENT_PRIMARY_NEON}; border-radius: 10px; background-color: {CARD_BG};">
            <h3 style="font-size: 24px; color:{ACCENT_PRIMARY_NEON};">METRICS AWAITING INFERENCE</h3>
            <p style="font-size: 16px; color: {TEXT_LIGHT};">Upload image and click 'INITIATE DUAL-MODEL ANALYSIS' to generate report.</p>
        </div>
    """, unsafe_allow_html=True)


st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Log Eksekusi Model Ganda (Dual-Model Execution Log)</h3>', unsafe_allow_html=True)

log_content = ""
if results:
    log_content = st.session_state.execution_log_data
else:
    log_content = f"""[{time.strftime('%H:%M:%S')}] INFO: System Initialized. Awaiting Input Payload.<br>
[{time.strftime('%H:%M:%S')}] DATA: No active payload detected. <br>
[{time.strftime('%H:%M:%S')}] MODEL: Detection Model (<b>Siti Naura Khalisa_Laporan 4.pt</b>) and Classification Model (<b>SitiNauraKhalisa_Laporan2.h5</b>) are idle."""

st.markdown(f"""
    <div class="log-container">
        {log_content}
    </div>
    """, unsafe_allow_html=True)

st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Tabel Detail Aset Terdeteksi ({results["detection_model"] if results else "..."})</h3>', unsafe_allow_html=True)

if st.session_state.analysis_results:
    df = pd.DataFrame(st.session_state.analysis_results['detections'])
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

else:
    st.info("Tabel detail akan muncul setelah analisis berhasil dijalankan.")

st.markdown('</div>', unsafe_allow_html=True)
