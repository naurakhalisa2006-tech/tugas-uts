import streamlit as st
import random
import time
import json
import io
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pandas as pd
import numpy as np
from math import floor

# --- 1. Import Pustaka Machine Learning (Hanya akan berfungsi jika diinstal) ---
try:
    from ultralytics import YOLO
    # Peringatan: TensorFlow seringkali harus dimuat secara penuh
    import tensorflow as tf 
    from tensorflow.keras.models import load_model 
    
    ML_LIBRARIES_LOADED = True
except ImportError:
    # st.warning("Pustaka 'ultralytics' atau 'tensorflow' tidak ditemukan. Aplikasi akan berjalan dalam mode SIMULASI yang ditingkatkan.")
    ML_LIBRARIES_LOADED = False

# --- 2. Konfigurasi dan Styling (Tema Cyber Pastel / Vaporwave) ---

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
NEON_CYAN = "#00FFFF"            # Neon Cyan (CLEAN / RAPI)
NEON_MAGENTA = "#FF00FF"         # Neon Magenta (MESSY / BERANTAKAN)

# Warna Teks Kontras (Di atas card gelap)
TEXT_CLEAN_LIGHT = NEON_CYAN     # Cyan
TEXT_MESSY_LIGHT = NEON_MAGENTA  # Magenta

# Tombol Neon (Sedikit lebih jenuh untuk kontras)
BUTTON_COLOR_NEON = "#3498DB"

# CSS Kustom untuk menyesuaikan tema Streamlit ke Cyber Pastel Dynamic
custom_css = f"""
<style>
    /* Definisi Keyframe untuk Efek Neon Pulse */
    @keyframes neon-pulse-cyan {{
        0% {{ box-shadow: 0 0 5px {NEON_CYAN}, 0 0 10px {NEON_CYAN}; }}
        50% {{ box-shadow: 0 0 10px {NEON_CYAN}, 0 0 20px {NEON_CYAN}; }}
        100% {{ box-shadow: 0 0 5px {NEON_CYAN}, 0 0 10px {NEON_CYAN}; }}
    }}
    @keyframes neon-pulse-magenta {{
        0% {{ box-shadow: 0 0 5px {NEON_MAGENTA}, 0 0 10px {NEON_MAGENTA}; }}
        50% {{ box-shadow: 0 0 10px {NEON_MAGENTA}, 0 0 20px {NEON_MAGENTA}; }}
        100% {{ box-shadow: 0 0 5px {NEON_MAGENTA}, 0 0 10px {NEON_MAGENTA}; }}
    }}

    /* Global Styling */
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_LIGHT};
        font-family: 'Consolas', monospace; /* Font ala Terminal/Cyber */
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {ACCENT_PRIMARY_NEON};
        text-shadow: 0 0 3px {ACCENT_PRIMARY_NEON};
    }}
    
    /* Card/Container Styling */
    .stCard, .stAlert, .log-container {{
        background-color: {CARD_BG};
        border-radius: 12px;
        border: 2px solid {ACCENT_PRIMARY_NEON};
        padding: 20px;
        box-shadow: 0 0 10px rgba(77, 255, 255, 0.5);
        margin-bottom: 20px;
    }}
    .log-container {{
        max-height: 200px;
        overflow-y: auto;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid #34495E;
    }}

    /* Status Card Styling (CLEAN) */
    .clean-status {{
        background-color: #1A344E; /* Darker Blue */
        border-color: {NEON_CYAN};
        animation: neon-pulse-cyan 3s infinite alternate;
        color: {TEXT_CLEAN_LIGHT} !important;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
    }}
    /* Status Card Styling (MESSY) */
    .messy-status {{
        background-color: #4E1A34; /* Darker Magenta */
        border-color: {NEON_MAGENTA};
        animation: neon-pulse-magenta 3s infinite alternate;
        color: {TEXT_MESSY_LIGHT} !important;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
    }}
    
    /* Tombol Upload & Proses */
    div.stButton > button:first-child {{
        background-color: {BUTTON_COLOR_NEON};
        color: {BG_DARK};
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #4DFFFF;
        color: {BG_DARK};
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(77, 255, 255, 0.7);
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. Inisialisasi State dan Model ---

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'execution_log_data' not in st.session_state:
    st.session_state.execution_log_data = f"""[{time.strftime('%H:%M:%S')}] INFO: System Initialized. Awaiting Input Payload.<br>
[{time.strftime('%H:%M:%S')}] DATA: No active payload detected. <br>
[{time.strftime('%H:%M:%S')}] MODEL: Detection Model (<b>Siti Naura Khalisa_Laporan 4.pt</b>) and Classification Model (<b>SitiNauraKhalisa_Laporan2.h5</b>) are idle."""

# Penamaan Model
YOLO_MODEL_NAME = "model/Siti Naura Khalisa_Laporan 4.pt"
CLASSIFICATION_MODEL_NAME = "model/SitiNauraKhalisa_Laporan2.h5"

# Placeholder untuk model yang dimuat
if ML_LIBRARIES_LOADED and 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None
    st.session_state.keras_model = None
    try:
        st.session_state.yolo_model = YOLO(YOLO_MODEL_NAME)
        st.session_state.keras_model = load_model(CLASSIFICATION_MODEL_NAME)
        # Menyesuaikan ukuran input yang diharapkan oleh model Keras
        st.session_state.keras_input_size = st.session_state.keras_model.input_shape[1:3] 
        st.session_state.execution_log_data += f"<br>[{time.strftime('%H:%M:%S')}] MODEL: Loaded {YOLO_MODEL_NAME} and {CLASSIFICATION_MODEL_NAME}."
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        ML_LIBRARIES_LOADED = False

# --- 4. Fungsi Analisis Utama (SIMULASI/NYATA) ---

def run_analysis(image_file):
    
    log = []
    log.append(f"[{time.strftime('%H:%M:%S')}] DATA: Input file received: {image_file.name}")
    
    results = {
        "detection_model": YOLO_MODEL_NAME,
        "classification_model": CLASSIFICATION_MODEL_NAME,
        "detections": [],
        "overall_room_tag": "N/A", # Tag klasifikasi ruangan RAPI/BERANTAKAN
        "processed_image": None
    }
    
    img = Image.open(image_file)
    width, height = img.size
    log.append(f"[{time.strftime('%H:%M:%S')}] PROC: Image size detected: {width}x{height} pixels.")
    
    # --- Mode SIMULASI yang Ditingkatkan (Memastikan Logika Klasifikasi Ruangan Terpisah) ---
    if not ML_LIBRARIES_LOADED:
        log.append(f"[{time.strftime('%H:%M:%S')}] MODEL: Running in SIMULATION Mode.")
        
        # SIMULASI Deteksi Bounding Box (YOLO)
        num_detections = random.choice([0, 1, 2, 3])
        if num_detections > 0:
            log.append(f"[{time.strftime('%H:%M:%S')}] DETECT: {num_detections} assets simulated.")
            draw = ImageDraw.Draw(img)
            sim_classes = ["kursi", "meja", "buku", "pakaian"]
            
            for i in range(num_detections):
                # Koordinat normalisasi acak (x, y, w, h)
                nx = random.uniform(0.1, 0.8)
                ny = random.uniform(0.1, 0.8)
                nw = random.uniform(0.05, 0.2)
                nh = random.uniform(0.05, 0.2)
                
                # Konversi ke koordinat piksel
                x_min = floor(nx * width)
                y_min = floor(ny * height)
                x_max = floor((nx + nw) * width)
                y_max = floor((ny + nh) * height)
                
                # Gambar Bounding Box (Simulasi)
                draw.rectangle([x_min, y_min, x_max, y_max], outline=ACCENT_PRIMARY_NEON, width=2)
                
                results["detections"].append({
                    'asset_id': f"{random.choice(sim_classes)}-{i+1}", 
                    'confidence_score': round(random.uniform(0.7, 0.99), 4),
                    # *** PENTING: classification_tag INDIVIDU DIBIARKAN N/A SESUAI PERMINTAAN USER ***
                    'classification_tag': 'N/A', 
                    'normalized_coordinates': f"({round(nx, 2)}, {round(ny, 2)}, {round(nw, 2)}, {round(nh, 2)})"
                })
        else:
             log.append(f"[{time.strftime('%H:%M:%S')}] DETECT: No assets detected by YOLO simulation.")
        
        # SIMULASI Klasifikasi Ruangan Keseluruhan (Keras/H5)
        # Klasifikasi ini diterapkan ke SELURUH gambar, bukan bounding box.
        
        # LOG KHUSUS: Menanggapi permintaan user, memastikan klasifikasi global dijalankan
        if not results["detections"]:
             log.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY: YOLO simulasi menemukan 0 objek. Klasifikasi Keras/H5 dijalankan pada KESELURUHAN GAMBAR (area non-bounding box) untuk menentukan Status Ruangan.")
             
        random_tag = random.choice(["RAPI", "BERANTAKAN"])
        random_conf = round(random.uniform(0.8, 0.99), 4)
        results["overall_room_tag"] = random_tag
        log.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY: Status Ruangan Klasifikasi Keras/H5: '{random_tag}' (Conf: {random_conf*100:.2f}%)")

        results["processed_image"] = img # Gambar dengan bounding box simulasi
    
    # --- Mode NYATA (Logika ML Sebenarnya) ---
    else:
        log.append(f"[{time.strftime('%H:%M:%S')}] MODEL: Running in PRODUCTION Mode.")
        
        # --- 1. DETEKSI OBJEK (YOLO) ---
        yolo_results = st.session_state.yolo_model(img, verbose=False)
        detection_data = []
        draw = ImageDraw.Draw(img)
        
        if yolo_results and len(yolo_results[0].boxes) > 0:
            log.append(f"[{time.strftime('%H:%M:%S')}] DETECT: {len(yolo_results[0].boxes)} assets found.")
            
            for i, box in enumerate(yolo_results[0].boxes):
                x_min, y_min, x_max, y_max = [int(val) for val in box.xyxy[0].tolist()]
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                asset_class = st.session_state.yolo_model.names[cls_id]
                
                # Gambar Bounding Box (Nyata)
                draw.rectangle([x_min, y_min, x_max, y_max], outline=ACCENT_PRIMARY_NEON, width=2)

                # Hitung Koordinat Normalisasi
                nx = round(x_min / width, 2)
                ny = round(y_min / height, 2)
                nw = round((x_max - x_min) / width, 2)
                nh = round((y_max - y_min) / height, 2)
                
                results["detections"].append({
                    'asset_id': f"{asset_class}-{i+1}", 
                    'confidence_score': round(conf, 4),
                    # *** PENTING: classification_tag INDIVIDU DIBIARKAN N/A SESUAI PERMINTAAN USER ***
                    'classification_tag': 'N/A',
                    'normalized_coordinates': f"({nx}, {ny}, {nw}, {nh})"
                })
        else:
            log.append(f"[{time.strftime('%H:%M:%S')}] DETECT: No assets detected by YOLO.")


        # --- 2. KLASIFIKASI RUANGAN KESELURUHAN (Keras/H5) ---
        # Klasifikasi ini diterapkan ke SELURUH gambar, terlepas dari hasil YOLO.
        
        # LOG KHUSUS: Menanggapi permintaan user, memastikan klasifikasi global dijalankan
        if not results["detections"]:
             log.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY: YOLO tidak menemukan objek. Klasifikasi Keras/H5 dijalankan pada KESELURUHAN GAMBAR (area non-bounding box) untuk menentukan Status Ruangan.")

        keras_size = st.session_state.keras_input_size
        img_resized = img.resize(keras_size)
        img_array = np.array(img_resized) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0) # Tambahkan batch dimension

        log.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY: Preprocessing image for Keras ({keras_size}).")
        
        try:
            predictions = st.session_state.keras_model.predict(img_array, verbose=0)
            
            # Asumsi model Keras output 2 kelas (0: RAPI, 1: BERANTAKAN)
            is_messy = predictions[0][0]
            
            if is_messy > 0.5:
                tag = "BERANTAKAN"
                conf = is_messy
            else:
                tag = "RAPI"
                conf = 1.0 - is_messy

            results["overall_room_tag"] = tag
            log.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY: Status Ruangan Klasifikasi Keras/H5: '{tag}' (Conf: {conf*100:.2f}%)")
        
        except Exception as e:
            log.append(f"[{time.strftime('%H:%M:%S')}] ERROR: Keras classification failed: {e}")
            results["overall_room_tag"] = "ERROR"

        results["processed_image"] = img # Gambar dengan bounding box dari YOLO

    # Simpan Log dan Hasil
    st.session_state.execution_log_data = "<br>".join(log)
    st.session_state.analysis_results = results
    
    return results

# --- 5. Tampilan Utama Aplikasi Streamlit ---

st.title("ROOM INSIGHT")
st.markdown(f"**Sistem Analisis Deteksi Aset dan Klasifikasi Kerapihan Ruangan**", unsafe_allow_html=True)

# BARIS 1: UPLOAD DAN PROSES
col1, col2 = st.columns([1, 3])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Gambar Ruangan (.jpg, .png)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    
    if uploaded_file is not None:
        if st.button("PROSES ANALISIS", key="process_button"):
            with st.spinner('Menganalisis gambar...'):
                results = run_analysis(uploaded_file)
            st.success("Analisis Selesai! Lihat hasilnya di bawah.")
        else:
            results = st.session_state.analysis_results
    else:
        results = st.session_state.analysis_results


# BARIS 2: RINGKASAN ANALISIS DAN LOG
if results:
    
    # Ambil tag klasifikasi ruangan yang sudah pasti diproses
    room_tag = results.get("overall_room_tag", "N/A")
    
    # Tentukan gaya CSS berdasarkan tag
    status_class = "clean-status" if room_tag == "RAPI" else "messy-status"
    status_text = f"STATUS RUANGAN: {room_tag}"

    with col2:
        st.markdown(f'<div class="{status_class} stCard" style="padding: 15px; margin-top: 25px;">{status_text}</div>', unsafe_allow_html=True)
        
        # Display Processed Image
        if results["processed_image"]:
            st.image(results["processed_image"], caption="Gambar dengan Hasil Deteksi", use_column_width=True)


# BARIS 3: LOG EKSEKUSI
st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Log Eksekusi</h3>', unsafe_allow_html=True)

log_content = st.session_state.execution_log_data

st.markdown(f"""
    <div class="log-container">
        {log_content}
    </div>
    """, unsafe_allow_html=True)


# BARIS 4: TABEL DETAIL ASET TERDETEKSI
st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Tabel Detail Aset Terdeteksi ({YOLO_MODEL_NAME})</h3>', unsafe_allow_html=True)

if st.session_state.analysis_results and st.session_state.analysis_results['detections']:
    df = pd.DataFrame(st.session_state.analysis_results['detections'])
    df = df.rename(columns={
        'asset_id': 'Asset ID (Deteksi)', 
        'confidence_score': 'Conf. Deteksi (%)', 
        'classification_tag': 'Tag Kerapihan Individual', # Diubah namanya agar tidak ambigu
        'normalized_coordinates': 'Koordinat Norm. (x, y, w, h)'
    })
    
    # Format persentase
    df['Conf. Deteksi (%)'] = (df['Conf. Deteksi (%)'] * 100).round(2).astype(str) + ' %'
    
    st.dataframe(df, use_container_width=True)
    
    # Menampilkan Keterangan Klasifikasi Ruangan di bawah tabel
    st.markdown(f"""
        <p style="font-size: 14px; margin-top: 10px; color: {ACCENT_PRIMARY_NEON};">
            <strong>Catatan Penting:</strong> Tag Kerapihan Individual ('N/A') disengaja. Klasifikasi RAPI/BERANTAKAN 
            diterapkan HANYA pada keseluruhan ruangan (Status Ruangan di atas), 
            bukan pada setiap objek yang terdeteksi.
        </p>
    """, unsafe_allow_html=True)
    
else:
    st.info(f"Tidak ada aset terdeteksi oleh {YOLO_MODEL_NAME}, atau belum ada file yang dianalisis.")

eof
