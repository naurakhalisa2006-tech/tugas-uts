import streamlit as st
import random
import time
import json
import io
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageColor

# --- 1. Pustaka untuk Integrasi Model ---
# Catatan: Pustaka ini harus diinstal di lingkungan Anda: pip install ultralytics tensorflow keras
try:
    from ultralytics import YOLO # Untuk model .pt (Deteksi Objek)
    import tensorflow as tf # Untuk model .h5 (Klasifikasi Akhir)
    import numpy as np
    
    # Nonaktifkan pesan peringatan TensorFlow/CUDA yang tidak perlu
    tf.get_logger().setLevel('ERROR') 

except ImportError:
    # Setel flag jika pustaka tidak ditemukan
    st.error("Pustaka ML (ultralytics/tensorflow) tidak ditemukan. Menjalankan dalam mode SIMULASI.")
    YOLO = None
    tf = None

# --- 2. Konfigurasi dan Pemuatan Model Global ---
MODEL_DETECTION_PATH = "Siti Naura Khalisa_Laporan 4.pt"
MODEL_CLASSIFICATION_PATH = "SitiNauraKhalisa_Laporan2.h5"

# Inisialisasi Model (Hanya sekali)
@st.cache_resource
def load_models():
    """Memuat model Deteksi dan Klasifikasi ke dalam cache Streamlit."""
    
    # 1. Model Deteksi Objek (YOLO/PyTorch)
    detection_model = None
    if YOLO:
        try:
            st.info(f"Memuat Model Deteksi: {MODEL_DETECTION_PATH}...")
            # Gunakan jalur lokal. Anda harus meletakkan file .pt di folder yang sama.
            detection_model = YOLO(MODEL_DETECTION_PATH)
            st.success("Model Deteksi (YOLO V8) berhasil dimuat.")
        except Exception as e:
            st.warning(f"Gagal memuat model PyTorch YOLO: {e}. Menggunakan mode simulasi deteksi.")
            detection_model = None

    # 2. Model Klasifikasi Akhir (Keras/TensorFlow)
    classification_model = None
    if tf:
        try:
            st.info(f"Memuat Model Klasifikasi: {MODEL_CLASSIFICATION_PATH}...")
            # Gunakan jalur lokal. Anda harus meletakkan file .h5 di folder yang sama.
            classification_model = tf.keras.models.load_model(MODEL_CLASSIFICATION_PATH)
            st.success("Model Klasifikasi (Keras) berhasil dimuat.")
        except Exception as e:
            st.warning(f"Gagal memuat model Keras: {e}. Menggunakan mode simulasi klasifikasi.")
            classification_model = None
            
    return detection_model, classification_model

# Muat model saat aplikasi Streamlit dimulai
YOLO_MODEL, KERAS_CLASSIFICATION_MODEL = load_models()

# --- 3. Konfigurasi dan Styling (Tema Cyber Pastel / Vaporwave) ---
# (Kode Styling tetap sama agar tampilan tetap bagus)

st.set_page_config(
    page_title="ROOM INSIGHT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Definisi Warna untuk Tema Cyber Pastel (Dark Theme with Neon Accents)
BG_DARK = "#1A1A2E"              # Latar Belakang Biru Indigo Gelap (Futuristic)
CARD_BG = "#2C3E50"              # Latar Belakang Kartu Abu-abu Biru Tua
TEXT_LIGHT = "#EAEAEA"          # Teks Utama Putih Cerah
ACCENT_PRIMARY_NEON = "#4DFFFF"  # Neon Biru Muda (Aksen Utama)

# Warna Status Dinamis (Neon)
NEON_CYAN = "#00FFFF"            # Neon Cyan (CLEAN)
NEON_MAGENTA = "#FF00FF"         # Neon Magenta (MESSY)

# Warna Teks Kontras (Di atas card gelap)
TEXT_CLEAN_LIGHT = NEON_CYAN     # Cyan
TEXT_MESSY_LIGHT = NEON_MAGENTA  # Magenta

# Tombol Neon (Sedikit lebih jenuh untuk kontras)
BUTTON_COLOR_NEON = "#3498DB"

# CSS Kustom untuk menyesuaikan tema Streamlit ke Cyber Pastel Dynamic
custom_css = f"""
<style>
    /* Definisi Keyframe untuk Efek Neon Flicker */
    @keyframes neon-flicker {{
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {{
            /* Puncak Glow */
            text-shadow: 0 0 5px {ACCENT_PRIMARY_NEON}, 0 0 10px {ACCENT_PRIMARY_NEON}, 0 0 20px {ACCENT_PRIMARY_NEON}, 0 0 40px rgba(77, 255, 255, 0.5);
            opacity: 1;
        }}
        20%, 24%, 55% {{
            /* Flicker (Redup) */
            text-shadow: 0 0 2px {ACCENT_PRIMARY_NEON}, 0 0 5px {ACCENT_PRIMARY_NEON};
            opacity: 0.9;
        }}
    }}

    /* Latar Belakang Gelap Vaporwave */
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_LIGHT};
        font-family: 'Inter', sans-serif;
    }}
    h1 {{
        color: {ACCENT_PRIMARY_NEON};
        border-bottom: 2px solid {ACCENT_PRIMARY_NEON};
        padding-bottom: 10px;
        /* Efek Neon Glow DENGAN ANIMASI */
        animation: neon-flicker 1.8s infinite alternate; 
    }}
    /* Gaya Kartu Cyber dengan Bayangan Neon */
    .modern-card {{
        background-color: {CARD_BG};
        border: 1px solid {ACCENT_PRIMARY_NEON}; /* Border Neon Tipis */
        box-shadow: 0 0 15px rgba(77, 255, 255, 0.4); /* Neon Glow! */
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }}
    h2 {{
        color: {ACCENT_PRIMARY_NEON};
        border-bottom: 1px solid #34495E; /* Garis pemisah gelap */
        padding-bottom: 5px;
    }}
    
    /* --- GAYA UPLOADER CYBER --- */
    [data-testid="stFileUploader"] {{
        min-height: 150px; 
        padding: 10px 20px 10px 20px !important;
        margin-top: 10px;
        /* Menggunakan ACCENT_PRIMARY_NEON untuk border */
        border: 2px dashed {ACCENT_PRIMARY_NEON} !important; 
        border-radius: 12px;
        /* Menggunakan BG_DARK agar tampak menyatu dengan latar belakang app */
        background-color: {BG_DARK} !important; 
        pointer-events: auto !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] p, [data-testid="stFileUploader"] label {{
        color: {TEXT_LIGHT} !important;
    }}
    
    /* --- GAYA TOMBOL 'BROWSE FILES' DI DALAM UPLOADER (PENTING!) --- */
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {CARD_BG} !important; /* Latar belakang gelap */
        color: {ACCENT_PRIMARY_NEON} !important; /* Teks Neon */
        font-weight: bold !important;
        border: 1px solid {ACCENT_PRIMARY_NEON} !important; /* Border Neon */
        border-radius: 8px !important;
        transition: all 0.3s !important;
        box-shadow: 0 0 5px {ACCENT_PRIMARY_NEON} !important; /* Subtle neon glow */
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: #3C5A6C !important; /* Hover effect */
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
        /* Neon Glow pada Tombol */
        box-shadow: 0 0 10px {NEON_CYAN}, 0 0 20px {NEON_CYAN};
    }}
    .stButton > button:nth-child(1):hover {{
        background-color: #2980B9;
        box-shadow: 0 0 15px {NEON_CYAN}, 0 0 25px {NEON_CYAN};
    }}

    /* Card Status Kecil */
    .status-metric-card {{
        background-color: {CARD_BG};
        border: 2px solid #34495E; /* Warna default gelap */
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.3);
    }}
    
    /* GAYA UNTUK STATUS MESSY/CLEAN (Teks Neon) */
    .clean-status-text {{ color: {TEXT_CLEAN_LIGHT}; font-weight: 900; font-size: 24px; text-shadow: 0 0 5px {TEXT_CLEAN_LIGHT}; }}
    .messy-status-text {{ color: {TEXT_MESSY_LIGHT}; font-weight: 900; font-size: 24px; text-shadow: 0 0 5px {TEXT_MESSY_LIGHT}; }}
    
    /* Border Dinamis Neon */
    .clean-border {{ border-color: {NEON_CYAN} !important; border-width: 4px !important; box-shadow: 0 0 15px rgba(0, 255, 255, 0.6) !important; }}
    .messy-border {{ border-color: {NEON_MAGENTA} !important; border-width: 4px !important; box-shadow: 0 0 15px rgba(255, 0, 255, 0.6) !important; }}
    
    /* Info Cards Dinamis (untuk Input Matrix) */
    .info-card-clean {{ background-color: {NEON_CYAN}; color: {BG_DARK}; border-color: {NEON_CYAN}; box-shadow: 0 0 8px {NEON_CYAN}; }}
    .info-card-messy {{ background-color: {NEON_MAGENTA}; color: {BG_DARK}; border-color: {NEON_MAGENTA}; box-shadow: 0 0 8px {NEON_MAGENTA}; }}

    .caption-clean {{ color: {TEXT_CLEAN_LIGHT}; }}
    .caption-messy {{ color: {TEXT_MESSY_LIGHT}; }}

    p {{ color: {TEXT_LIGHT}; }}
    
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Inisialisasi State Session (Tetap Sama)
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'execution_log_data' not in st.session_state:
    st.session_state.execution_log_data = []

# --- 4. Fungsi Deteksi dan Klasifikasi ASLI / SIMULASI ---

# Fungsi SIMULASI (Digunakan sebagai fallback jika model gagal dimuat)
def generate_mock_detections(is_clean):
    # Menggunakan logika simulasi yang sama seperti kode asli Anda
    clean_items = ["Desk_Unit", "Chair_Ergonomic", "Bookshelf_Structured", "Monitor_System", "Rug_Area", "Wall_Lamp", "PC_Tower"]
    messy_items = ["Scattered_Clothes", "Loose_Cables", "Unsorted_Papers", "Trash_Object", "Empty_Bottles", "Food_Wrapper"]
    num_items = random.randint(3, 7) + (0 if is_clean else 3)
    detections = []
    for i in range(num_items):
        is_messy_item = False
        if is_clean:
            label = random.choice(clean_items)
            confidence = round(random.uniform(0.75, 0.95), 4)
        else:
            if random.random() < 0.7:
                label = random.choice(messy_items)
                confidence = round(random.uniform(0.65, 0.90), 4)
                is_messy_item = True
            else:
                label = random.choice(clean_items)
                confidence = round(random.uniform(0.55, 0.80), 4)
        x_norm = random.uniform(0.05, 0.7)
        y_norm = random.uniform(0.05, 0.7)
        w_norm = random.uniform(0.1, 0.3)
        h_norm = random.uniform(0.1, 0.3)
        detections.append({
            "asset_id": label.upper().replace('_', '-'),
            "confidence_score": confidence,
            "classification_tag": 'UNOPTIMIZED' if is_messy_item else 'STRUCTURED',
            "normalized_coordinates": [
                round(x_norm, 4), 
                round(y_norm, 4),
                round(w_norm, 4),
                round(h_norm, 4)
            ]
        })
    return sorted(detections, key=lambda x: x['confidence_score'], reverse=True)


def draw_boxes_on_image(image_bytes, detections, class_names={0: 'STRUCTURED', 1: 'UNOPTIMIZED'}):
    # Logika untuk menggambar bounding box pada gambar (Tidak ada perubahan)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    
    # Menggunakan warna Neon
    CLEAN_RGB = ImageColor.getrgb(NEON_CYAN)
    MESSY_RGB = ImageColor.getrgb(NEON_MAGENTA)
    TEXT_RGB = ImageColor.getrgb(BG_DARK) # Teks gelap pada latar belakang kotak neon
    
    try:
        font_size = max(15, min(image_width // 40, 30))
        font = ImageFont.load_default() 
    except IOError:
        font = ImageFont.load_default()
        
    for det in detections:
        # Menangani data deteksi dari YOLO asli (result.boxes) atau Simulasi
        if isinstance(det, dict):
            # Data Simulasi
            x_norm, y_norm, w_norm, h_norm = det['normalized_coordinates']
            x_min = int(x_norm * image_width)
            y_min = int(y_norm * image_height)
            x_max = int((x_norm + w_norm) * image_width)
            y_max = int((y_norm + h_norm) * image_height)
            label = det['asset_id']
            confidence = det['confidence_score']
            tag = det['classification_tag']
        else:
            # Data Asli dari YOLO result (format [x_min, y_min, x_max, y_max])
            x_min, y_min, x_max, y_max = [int(val) for val in det[0]]
            confidence = det[1]
            class_id = det[2]
            label = class_names.get(class_id, f'Class-{class_id}')
            tag = 'UNOPTIMIZED' if 'Messy' in label or 'Unsorted' in label else 'STRUCTURED' 
        
        x_max = min(x_max, image_width - 1)
        y_max = min(y_max, image_height - 1)
            
        # Warna box berdasarkan status
        box_rgb = CLEAN_RGB if tag == 'STRUCTURED' else MESSY_RGB
        
        # Gambar kotak dengan warna neon yang cerah
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
            
        # Background teks dengan warna neon (untuk kontras)
        draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill=box_rgb)
        # Teks dengan warna latar belakang gelap agar kontras dengan neon
        draw.text((text_x + 2, text_y + 2), text_content, font=font, fill=TEXT_RGB) 
        
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    return buf.getvalue()

def format_execution_log(results, uploaded_file_name):
    # Membuat format log tekstual yang menyerupai output konsol (Tidak ada perubahan)
    log_lines = []
    
    # 1. Start Log
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] INFO: System Initialized.")
    
    # 2. Data Payload
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 3))}] DATA: Payload Acquired: {uploaded_file_name}.")

    # 3. Model Loading/Inference
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 2))}] MODEL: Initiating YOLO V8 inference ({results['detection_model']}).")
    if YOLO_MODEL:
        log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 1))}] INFERENCE: Running Real YOLO V8 Inference...")
    
    # 4. Detection Count
    tag = 'STRUCTURED' if results['is_clean'] else 'UNOPTIMIZED'
    tag_color = NEON_CYAN if results['is_clean'] else NEON_MAGENTA
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 1))}] DETECT: {len(results['detections'])} Assets Tagged ({results['messy_count']} Unoptimized).")
    
    # 5. Classification
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY: Keras Model ({results['classification_model']}) classified image features.")
    
    # 6. Final Report
    final_conf = results['conf_clean'] if results['is_clean'] else results['conf_messy']
    final_status = results['final_status'].split(': ')[1]
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] REPORT: Final Status Classification: <span style='color:{tag_color};'><b>{final_status}</b></span> (Confidence: {final_conf}%).")
    
    # Gabungkan dengan <br> untuk markdown
    return '<br>'.join(log_lines)

# --- 5. Fungsi Analisis Utama (REAL/SIMULASI) ---
def simulate_yolo_analysis():
    if st.session_state.uploaded_file is None:
        st.error("Sila muat naik imej ruangan dahulu.")
        return
    st.session_state.analysis_results = None
    st.session_state.processed_image = None
    
    st.session_state.execution_log_data = [
        f"[{time.strftime('%H:%M:%S')}] INFO: System Initialized. Awaiting user action."
    ]
    
    log_placeholder = st.empty()
    log_placeholder.markdown(f'<p style="color: {TEXT_LIGHT};">SYSTEM> Initiating inference. Loading model <b>{MODEL_DETECTION_PATH}</b>...</p>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0, text="Loading Tensor Core & Running Inference...")
    
    image_data = st.session_state.uploaded_file.getvalue()
    input_image = Image.open(io.BytesIO(image_data))
    
    
    # --- A. Tahap 1: Deteksi Objek (Model .pt) ---
    real_detections = False
    
    if YOLO_MODEL:
        try:
            progress_bar.progress(30, text="Running YOLO Detection (Model .pt)...")
            
            # Melakukan inferensi dengan model YOLO yang dimuat
            # Diasumsikan model dilatih dengan kelas berantakan/rapi
            results_list = YOLO_MODEL(input_image) 
            result = results_list[0]
            
            detections_for_drawing = []
            messy_count = 0
            
            # Konversi hasil YOLO ke format yang dapat diolah
            for box in result.boxes:
                # Format box: [x_min, y_min, x_max, y_max]
                coordinates = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                # Menggunakan class_names dari model atau fallback
                class_name = result.names.get(class_id, f'Class-{class_id}')
                
                # Menentukan apakah objek tersebut 'messy' untuk perhitungan
                is_messy_object = 'Messy' in class_name or 'Unsorted' in class_name or 'Cable' in class_name
                if is_messy_object:
                    messy_count += 1
                
                detections_for_drawing.append((coordinates, confidence, class_id))

            mock_detections = [] # Kosongkan simulasi
            real_detections = True
            
        except Exception as e:
            st.warning(f"Error saat menjalankan inferensi YOLO: {e}. Beralih ke data simulasi.")
            real_detections = False
    
    if not real_detections:
        # Fallback ke Simulasi
        is_clean_sim = random.random() < 0.6
        mock_detections = generate_mock_detections(is_clean_sim) 
        messy_count = sum(1 for d in mock_detections if d['classification_tag'] == 'UNOPTIMIZED')
        detections_for_drawing = mock_detections

    
    # --- B. Tahap 2: Klasifikasi Ruangan (Model .h5) ---
    progress_bar.progress(60, text="Running Keras Classification (Model .h5)...")
    
    conf_clean = 0.5
    conf_messy = 0.5
    
    if KERAS_CLASSIFICATION_MODEL:
        try:
            # Pra-pemrosesan gambar untuk model Keras
            target_size = KERAS_CLASSIFICATION_MODEL.input_shape[1:3] # Ambil (height, width) yang dibutuhkan model
            processed_input = input_image.resize(target_size)
            processed_input = np.array(processed_input) / 255.0  # Normalisasi
            processed_input = np.expand_dims(processed_input, axis=0) # Tambahkan dimensi batch
            
            # Prediksi
            prediction = KERAS_CLASSIFICATION_MODEL.predict(processed_input, verbose=0)[0]
            
            # Diasumsikan model Keras menghasilkan 2 kelas (Clean/Messy)
            # Anda perlu menyesuaikan indeks ini berdasarkan output model Anda.
            # Contoh: prediction[0] = Clean, prediction[1] = Messy
            conf_clean = prediction[0].item()
            conf_messy = 1.0 - conf_clean # Atau prediction[1].item() jika ada 2 output
            
            st.info("Prediksi Keras: [Clean/Messy]")
            
        except Exception as e:
            st.warning(f"Error saat menjalankan model Keras: {e}. Beralih ke perhitungan klasifikasi berbasis deteksi.")
            # Tetap gunakan perhitungan berbasis deteksi sebagai fallback

    # --- C. Tahap 3: Klasifikasi Gabungan dan Laporan ---
    progress_bar.progress(90, text="Generating Final Report...")
    
    # Logika Penentuan Status Akhir: Gabungkan hasil Keras dan Messy Count
    
    # 1. Klasifikasi awal (berdasarkan Keras, jika berhasil)
    if KERAS_CLASSIFICATION_MODEL:
        is_clean_final = conf_clean > conf_messy
        
    # 2. Jika Keras gagal, gunakan Messy Count (sama seperti simulasi asli)
    else:
        is_clean_final = messy_count <= 2
        if is_clean_final:
            conf_clean = random.uniform(0.85, 0.98)
            conf_messy = 1.0 - conf_clean
        else:
            conf_messy = random.uniform(0.75, 0.95)
            conf_clean = 1.0 - conf_messy

    # Teks Final
    if is_clean_final:
        final_status = "STATUS: ROOM CLEAN - OPTIMAL"
        final_message = "System Integrity Check: GREEN (CYAN NEON). Minimum clutter detected. Excellent organization."
    else:
        final_status = "STATUS: ROOM MESSY - ALERT"
        final_message = "System Integrity Check: RED (MAGENTA NEON). High probability of unoptimized state. Clutter detected."

    # Gambar kotak pada gambar yang diunggah
    processed_image_bytes = draw_boxes_on_image(image_data, detections_for_drawing, class_names=YOLO_MODEL.names if YOLO_MODEL else None)
    
    results = {
        "final_status": final_status,
        "is_clean": is_clean_final,
        "conf_clean": round(conf_clean * 100, 2),
        "conf_messy": round(conf_messy * 100, 2),
        "messy_count": messy_count, 
        "detection_model": MODEL_DETECTION_PATH, 
        "classification_model": MODEL_CLASSIFICATION_PATH, 
        # Simpan dalam format yang konsisten untuk ditampilkan
        "detections": mock_detections if not real_detections else [
            {'asset_id': result.names.get(det[2], f'Class-{det[2]}'), 
             'confidence_score': det[1], 
             'classification_tag': ('UNOPTIMIZED' if 'Messy' in result.names.get(det[2], '') else 'STRUCTURED'),
             'normalized_coordinates': [
                 det[0][0]/input_image.width, det[0][1]/input_image.height, 
                 (det[0][2]-det[0][0])/input_image.width, (det[0][3]-det[0][1])/input_image.height 
             ]} for det in detections_for_drawing
        ],
        "final_message": final_message
    }
    
    st.session_state.processed_image = processed_image_bytes
    st.session_state.analysis_results = results
    
    # Generate the log data for display
    st.session_state.execution_log_data = format_execution_log(results, st.session_state.uploaded_file.name)
    
    progress_bar.progress(100, text="Analysis Complete!")
    time.sleep(0.5)
    progress_bar.empty()
    log_placeholder.empty()
    st.success("SYSTEM> Analysis Completed. Report Generated and Visualization Rendered.")

# --- 6. Tata Letak Streamlit (UI) ---

# Diperbarui: Mengubah Judul dan Subjudul
st.markdown(f"""
    <header>
        <h1>ROOM INSIGHT <span style="font-size: 18px; margin-left: 15px; color: {ACCENT_PRIMARY_NEON};">CLEAN OR MESSY?</span></h1>
        <p style="color: {TEXT_LIGHT}; font-size: 14px;">Klasfikasikan kerapihan ruangan anada dengan cepat dan akurat.</p>
    </header>
    <div style="margin-bottom: 20px;"></div>
    """, unsafe_allow_html=True)


# --- TAMPILAN 1: TATA LETAK DUA KOLOM UTAMA (ATAS) ---
# Menggunakan [1, 1] agar hampir 50/50
col_input, col_detection = st.columns([1, 1]) 

# Kolom 1: Data Input Matrix
with col_input:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">1. Data Input Matrix</h2>', unsafe_allow_html=True)
    
    # Status Card
    status_card_class = "info-card-clean" if st.session_state.uploaded_file else "info-card-messy"
    status_text = "PAYLOAD ACQUIRED" if st.session_state.uploaded_file else "AWAITING PAYLOAD"
    file_display = st.session_state.uploaded_file.name if st.session_state.uploaded_file else "No file uploaded"
    
    st.markdown(f"""
        <div class="status-metric-card {status_card_class}">
            <p style="margin: 0; font-weight: bold; font-size: 14px; color:{BG_DARK};">{status_text}</p>
            <p style="margin: 0; font-size: 14px; font-weight: 500; color:{BG_DARK};">{file_display}</p>
        </div>
        """, unsafe_allow_html=True)

    # FILE UPLOADER STREAMLIT STANDAR
    uploaded_file = st.file_uploader(
        "Upload Image File (JPG/PNG) | Initiating Payload Protocol", 
        type=["jpg", "jpeg", "png"],
        key="uploader",
        help="Unggah file gambar ruangan untuk dianalisis."
    )

    # Perbarui state session
    st.session_state.uploaded_file = uploaded_file

    st.markdown('</div>', unsafe_allow_html=True) 
    
    # Tombol Analisis
    st.markdown('<div style="padding-top: 20px;">', unsafe_allow_html=True)
    
    button_disabled = st.session_state.uploaded_file is None
    
    if st.button("⚡ START", disabled=button_disabled, use_container_width=True):
        simulate_yolo_analysis()
        
    st.markdown('</div>', unsafe_allow_html=True)


# Kolom 2: Live Detection Grid
with col_detection:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">2. Live Detection Grid </h2>', unsafe_allow_html=True)

    # Tentukan warna border dinamis
    border_class = ""
    if st.session_state.analysis_results is not None:
        border_class = 'clean-border' if st.session_state.analysis_results['is_clean'] else 'messy-border'

    # Container untuk menampung gambar dengan border dinamis
    st.markdown(f"""
        <div style="border: 4px solid #34495E; border-radius: 10px; padding: 5px; background-color: {BG_DARK};" class="{border_class}">
        """, unsafe_allow_html=True)

    if st.session_state.uploaded_file:
        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, caption='VISUALIZATION: Live Detection Grid (Drawn by YOLO Model)', use_container_width=True)
            caption_class = 'caption-clean' if st.session_state.analysis_results['is_clean'] else 'caption-messy'
            st.markdown(f'<p class="{caption_class}" style="text-align: center; font-weight: bold; margin-top: 10px; text-shadow: 0 0 3px {NEON_CYAN if st.session_state.analysis_results["is_clean"] else NEON_MAGENTA};">'
                        'BOUNDING BOXES: (Rendered Status Visualization)</p>', unsafe_allow_html=True)
        else:
            # Tampilkan gambar asli jika belum diproses
            image_data = st.session_state.uploaded_file.getvalue()
            st.image(image_data, caption='Image Data Stream (Original)', use_container_width=True)
            st.markdown(f'<p style="text-align: center; color: {ACCENT_PRIMARY_NEON}; font-weight: bold; margin-top: 10px; text-shadow: 0 0 3px {ACCENT_PRIMARY_NEON};">'
                        'BOUNDING BOXES: (Ready for Analysis - Click Initiate)</p>', unsafe_allow_html=True)
    else:
        st.image("https://placehold.co/1200x675/1A1A2E/4DFFFF?text=UPLOAD+IMAGE+TO+ACTIVATE+SCANNER+MODULE", caption="Awaiting Image Payload", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(f"<hr style='border-top: 1px solid {ACCENT_PRIMARY_NEON}; box-shadow: 0 0 5px {ACCENT_PRIMARY_NEON}; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True) 

# --- TAMPILAN 2: EXECUTION REPORT & METRICS (Satu Kolom Penuh) ---
st.markdown('<div class="modern-card" style="margin-top: 30px;">', unsafe_allow_html=True)
st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON}; border-bottom: 2px solid {ACCENT_PRIMARY_NEON}; padding-bottom: 10px;">Execution Report & Metrics</h2>', unsafe_allow_html=True)

results = st.session_state.analysis_results

# BARIS 1: TIGA KOLOM METRICS (2 Unit, 1 Unit, 1 Unit)
col_report, col_clean_conf, col_messy_conf = st.columns([2, 1, 1])

if results:
    status_main_text = results['final_status'].split(': ')[1]
    css_class_status = 'clean-status-text' if results['is_clean'] else 'messy-status-text'
    message = results['final_message']
    
    # Kolom 1 (2 Unit Lebar): Classification Report
    with col_report:
        border_color = NEON_CYAN if results['is_clean'] else NEON_MAGENTA
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {border_color}; box-shadow: 0 0 10px {border_color};">
                <p style="color: {TEXT_LIGHT}; font-size: 14px; margin-bottom: 5px; font-weight: bold;">CLASSIFICATION REPORT (Final Status)</p>
                <p class="{css_class_status}" style="font-size: 32px; margin-top: 5px;">{status_main_text}</p>
                <p style="font-size: 12px; color: {TEXT_LIGHT}; opacity: 0.7;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
    # Kolom 2 (1 Unit Lebar): Confidence: Clean
    with col_clean_conf:
        # Card Neon Cyan
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {NEON_CYAN}; background-color: {CARD_BG}; box-shadow: 0 0 8px {NEON_CYAN};">
                <p style="color: {NEON_CYAN}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">CONFIDENCE: CLEAN</p>
                <p style="color: {TEXT_LIGHT}; font-size: 28px; font-weight: bold;">{results["conf_clean"]}%</p>
            </div>
            """, unsafe_allow_html=True)
        
    # Kolom 3 (1 Unit Lebar): Confidence: Messy
    with col_messy_conf:
        # Card Neon Magenta
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {NEON_MAGENTA}; background-color: {CARD_BG}; box-shadow: 0 0 8px {NEON_MAGENTA};">
                <p style="color: {NEON_MAGENTA}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">CONFIDENCE: MESSY</p>
                <p style="color: {TEXT_LIGHT}; font-size: 28px; font-weight: bold;">{results["conf_messy"]}%</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Placeholder saat belum ada hasil
    st.markdown(f"""
        <div style="text-align: center; padding: 40px; border: 2px dashed {ACCENT_PRIMARY_NEON}; border-radius: 10px; background-color: {CARD_BG};">
            <h3 style="font-size: 24px; color:{ACCENT_PRIMARY_NEON};">METRICS AWAITING INFERENCE</h3>
            <p style="font-size: 16px; color: {TEXT_LIGHT};">Upload image and click '⚡ START' to generate report.</p>
        </div>
    """, unsafe_allow_html=True)


# BARIS 2: LOG OBJEK TERDETEKSI (Execution Log)
st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Log Objek Terdeteksi (Execution Log)</h3>', unsafe_allow_html=True)

# Container untuk Log yang bergulir
log_content = ""
if results:
    log_content = st.session_state.execution_log_data
else:
    log_content = f"""[{time.strftime('%H:%M:%S')}] INFO: System Initialized. Awaiting Input Payload.<br>
[{time.strftime('%H:%M:%S')}] DATA: No active payload detected. <br>
[{time.strftime('%H:%M:%S')}] MODEL: Inference idle."""

st.markdown(f"""
    <div class="mt-3 bg-white p-3 rounded text-sm font-mono whitespace-pre-wrap h-32 overflow-y-auto border border-gray-500" style="background-color: {BG_DARK}; color: {TEXT_LIGHT}; border-color: {ACCENT_PRIMARY_NEON} !important; box-shadow: 0 0 8px rgba(77, 255, 255, 0.4);">
        {log_content}
    </div>
    """, unsafe_allow_html=True)

# BARIS 3: TABEL DETAIL ASET TERDETEKSI
st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Tabel Detail Aset Terdeteksi</h3>', unsafe_allow_html=True)

if st.session_state.analysis_results:
    df = pd.DataFrame(st.session_state.analysis_results['detections'])
    df = df.rename(columns={
        'asset_id': 'Asset ID', 
        'confidence_score': 'Confidence Score (%)', 
        'classification_tag': 'Classification Tag',
        'normalized_coordinates': 'Normalized Coordinates (x, y, w, h)'
    })
    df['Confidence Score (%)'] = (df['Confidence Score (%)'] * 100).round(2).astype(str) + '%'
    
    # Styling Streamlit DataFrame untuk mencocokkan tema gelap
    st.dataframe(
        df, 
        use_container_width=True,
        # Menggunakan custom styling untuk DataFrame
        height=250 
    )

else:
    st.info("Tabel detail akan muncul setelah analisis berhasil dijalankan.")

st.markdown('</div>', unsafe_allow_html=True)
