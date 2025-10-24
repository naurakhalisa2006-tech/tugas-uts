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
TEXT_ERROR = "#FF6347" # Tomat Neon Merah

# CSS Kustom untuk menyesuaikan tema Streamlit ke Cyber Pastel Dynamic
custom_css = f"""
<style>
    /* Definisi Keyframe untuk Efek Neon Pulse pada tombol */
    @keyframes neon-pulse {{
        0% {{ box-shadow: 0 0 5px {BUTTON_COLOR_NEON}, 0 0 10px {BUTTON_COLOR_NEON}; }}
        50% {{ box-shadow: 0 0 15px {ACCENT_PRIMARY_NEON}, 0 0 20px {ACCENT_PRIMARY_NEON}; }}
        100% {{ box-shadow: 0 0 5px {BUTTON_COLOR_NEON}, 0 0 10px {BUTTON_COLOR_NEON}; }}
    }}

    /* CSS Global */
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_LIGHT};
        font-family: 'Inter', sans-serif;
    }}
    .stButton>button {{
        background-color: {BUTTON_COLOR_NEON};
        color: {BG_DARK};
        border-radius: 8px;
        border: 2px solid {ACCENT_PRIMARY_NEON};
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        animation: neon-pulse 2s infinite alternate; /* Menerapkan animasi */
    }}
    .stButton>button:hover {{
        background-color: {ACCENT_PRIMARY_NEON};
        color: {BG_DARK};
        box-shadow: 0 0 20px {NEON_CYAN}, 0 0 30px {NEON_CYAN};
        border-color: {NEON_CYAN};
        transform: translateY(-2px);
    }}
    
    /* Styling untuk Kartu Hasil */
    .result-card {{
        background-color: {CARD_BG};
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid {ACCENT_PRIMARY_NEON};
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }}

    /* Styling untuk Log */
    .log-container {{
        background-color: #0F0F1A; /* Warna lebih gelap untuk kontras log */
        border: 1px solid #34495E;
        border-radius: 8px;
        padding: 15px;
        max-height: 250px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        color: #A9A9A9;
    }}

    /* Header Styling */
    .header-text {{
        font-size: 40px;
        font-weight: 800;
        color: {ACCENT_PRIMARY_NEON};
        text-shadow: 0 0 5px {ACCENT_PRIMARY_NEON}, 0 0 10px rgba(77, 255, 255, 0.5);
        margin-bottom: 20px;
    }}
    
    /* Konten Dinamis - Status CLEAN */
    .status-clean {{
        color: {TEXT_CLEAN_LIGHT};
        font-size: 24px;
        font-weight: bold;
        text-shadow: 0 0 8px {NEON_CYAN};
    }}

    /* Konten Dinamis - Status MESSY */
    .status-messy {{
        color: {TEXT_MESSY_LIGHT};
        font-size: 24px;
        font-weight: bold;
        text-shadow: 0 0 8px {NEON_MAGENTA};
    }}

    /* Tabel Styling */
    .stDataFrame {{
        border: 1px solid {ACCENT_PRIMARY_NEON} !important;
        border-radius: 8px !important;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. Variabel Global dan State ---

# Tag Klasifikasi
CLASSIFICATION_TAGS = ['CLEAN', 'MESSY']
# Tag Aset (Untuk Simulasi Deteksi)
ASSET_TAGS = ['Laptop', 'Buku', 'Cangkir', 'Pakaian', 'Alat Tulis']

# Inisialisasi State Session
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'execution_log_data' not in st.session_state:
    st.session_state.execution_log_data = ""
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None


# --- 4. Fungsi Utama Simulasi Analisis (Telah diperbarui) ---

def run_analysis_simulation(image_file):
    """
    Simulasi proses deteksi dan klasifikasi objek.
    
    Catatan PENTING: Dalam implementasi riil, ini akan digantikan 
    dengan panggilan ke model ML yang sebenarnya.
    """
    
    log_data = f"[{time.strftime('%H:%M:%S')}] INFO: Memulai Analisis Citra...\n"
    
    if not image_file:
        log_data += f"[{time.strftime('%H:%M:%S')}] ERROR: Tidak ada citra masukan yang ditemukan.\n"
        st.session_state.execution_log_data = log_data
        st.session_state.analysis_results = None
        return
        
    log_data += f"[{time.strftime('%H:%M:%S')}] DATA: Citra diterima ({image_file.name}).\n"
    
    # --- SIMULASI MODEL ---
    
    # STEP 1: SIMULASI DETEKSI (YOLO)
    # Tentukan jumlah aset yang terdeteksi secara acak (misalnya 4 hingga 8 objek)
    num_detections = random.randint(4, 8)
    detections = []
    
    image = Image.open(image_file)
    img_width, img_height = image.size
    
    log_data += f"[{time.strftime('%H:%M:%S')}] MODEL: Melakukan deteksi {num_detections} aset...\n"
    
    # Tentukan dulu Klasifikasi Keseluruhan untuk membuat simulasi lebih koheren
    # 60% peluang untuk mendapatkan MESSY, 40% untuk CLEAN
    target_overall_classification = random.choices(CLASSIFICATION_TAGS, weights=[0.4, 0.6], k=1)[0]
    log_data += f"[{time.strftime('%H:%M:%S')}] SIMULASI: Target Klasifikasi Keseluruhan: {target_overall_classification}.\n"

    # Hitung jumlah tag 'MESSY' dan 'CLEAN' yang harus dihasilkan untuk mencapai target
    target_messy_count = 0
    if target_overall_classification == 'MESSY':
        # Jika target MESSY, pastikan MESSY menang (misal 60% dari total)
        target_messy_count = int(num_detections * random.uniform(0.55, 0.75)) # > 50%
    else:
        # Jika target CLEAN, pastikan CLEAN menang
        target_messy_count = int(num_detections * random.uniform(0.25, 0.45)) # < 50%

    # Jika ganjil, pastikan jumlahnya masih mayoritas
    if target_overall_classification == 'MESSY' and target_messy_count <= num_detections / 2:
         target_messy_count = int(num_detections / 2) + 1
    elif target_overall_classification == 'CLEAN' and target_messy_count >= num_detections / 2:
         target_messy_count = int(num_detections / 2) - 1

    # Memastikan tidak kurang dari 0 atau lebih dari num_detections
    target_messy_count = max(0, min(num_detections, target_messy_count))

    tag_list = (['MESSY'] * target_messy_count) + (['CLEAN'] * (num_detections - target_messy_count))
    random.shuffle(tag_list)
    
    for i in range(num_detections):
        # Simulasi bounding box (x_min, y_min, x_max, y_max)
        x_min = random.randint(0, img_width - 100)
        y_min = random.randint(0, img_height - 100)
        x_max = random.randint(x_min + 50, img_width)
        y_max = random.randint(y_min + 50, img_height)
        
        # Batasan agar tidak melebihi ukuran gambar
        x_max = min(x_max, img_width)
        y_max = min(y_max, img_height)

        bbox = (x_min, y_min, x_max, y_max)
        conf = random.uniform(0.70, 0.99) # Skor Deteksi
        
        # Hitung koordinat normalisasi (0-1)
        x_norm = x_min / img_width
        y_norm = y_min / img_height
        w_norm = (x_max - x_min) / img_width
        h_norm = (y_max - y_min) / img_height

        # STEP 2: KLASIFIKASI PER ASET DITENTUKAN DARI TAG_LIST
        classification_tag = tag_list.pop()
        
        detections.append({
            'asset_id': random.choice(ASSET_TAGS),
            'confidence_score': conf,
            'bbox': bbox,
            'classification_tag': classification_tag, 
            'normalized_coordinates': (x_norm, y_norm, w_norm, h_norm)
        })
        log_data += f"[{time.strftime('%H:%M:%S')}] KLASIFIKASI: Aset {i+1} diklasifikasikan sebagai '{classification_tag}'.\n"

    # STEP 3: TENTUKAN KLASIFIKASI KESELURUHAN BERDASARKAN HASIL AGREGASI
    # Klasifikasi keseluruhan didasarkan pada perbandingan jumlah tag MESSY vs CLEAN
    messy_count = sum(1 for d in detections if d['classification_tag'] == 'MESSY')
    clean_count = num_detections - messy_count
    
    if messy_count > clean_count:
        overall_classification = 'MESSY'
    else:
        overall_classification = 'CLEAN'
        
    log_data += f"[{time.strftime('%H:%M:%S')}] HASIL: Ditemukan {messy_count} MESSY dan {clean_count} CLEAN. Kesimpulan Akhir: {overall_classification}.\n"
    
    st.session_state.execution_log_data = log_data

    # --- SIMULASI VISUALISASI BOUNDING BOX (BOXES + TAG) ---
    draw = ImageDraw.Draw(image)
    
    # Pilih font yang tersedia di PIL, atau gunakan default
    try:
        font = ImageFont.truetype("Arial.ttf", size=18)
    except IOError:
        font = ImageFont.load_default()

    for d in detections:
        bbox = d['bbox']
        tag = d['classification_tag']
        conf = d['confidence_score']
        
        # Pilih warna berdasarkan tag
        if tag == 'CLEAN':
            color = NEON_CYAN
        else:
            color = NEON_MAGENTA
            
        # Gambar Bounding Box
        draw.rectangle(bbox, outline=color, width=3)
        
        # Gambar Label Background
        label_text = f"{d['asset_id']} | {tag} ({conf*100:.1f}%)"
        text_bbox = draw.textbbox((bbox[0], bbox[1]), label_text, font=font)
        
        # Tambahkan sedikit padding untuk latar belakang teks
        text_x_start = bbox[0]
        text_y_start = bbox[1] - (text_bbox[3] - text_bbox[1]) - 5
        text_x_end = text_x_start + (text_bbox[2] - text_bbox[0]) + 10
        text_y_end = bbox[1]

        # Pastikan label tidak keluar dari batas atas gambar
        if text_y_start < 0:
            # Jika keluar, gambar label di bawah kotak
            text_y_start = bbox[3]
            text_y_end = bbox[3] + (text_bbox[3] - text_bbox[1]) + 5
        
        # Gambar background label
        draw.rectangle([(text_x_start, text_y_start), (text_x_end, text_y_end)], fill=color)
        
        # Gambar Teks Label
        draw.text((text_x_start + 5, text_y_start + 2), label_text, fill=BG_DARK, font=font)


    # --- FINAL RESULTS OBJECT ---
    results = {
        'overall_classification': overall_classification,
        'detection_model': 'YOLOv8s-Custom (Siti Naura Khalisa_Laporan 4.pt)',
        'classification_model': 'CNN-Custom (SitiNauraKhalisa_Laporan2.h5)',
        'messy_count': messy_count,
        'clean_count': clean_count,
        'detections': detections,
        'annotated_image': image,
    }
    
    st.session_state.analysis_results = results
    
    return results

# --- 5. Tampilan Streamlit ---

# Header Utama
st.markdown(f'<p class="header-text">ROOM INSIGHT - ANÁLISIS KERAPIHAN</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color: #A9A9A9; font-size: 16px; margin-top: -15px; margin-bottom: 25px;">Sistem Deteksi Aset & Klasifikasi Kerapihan Ruangan Berbasis Visi Komputer</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    # Uploader Citra
    uploaded_file = st.file_uploader(
        "Unggah Citra Ruangan (JPG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file
        
        # Tampilkan Tombol Analisis jika file diunggah
        if st.button("JALANKAN ANALISIS (RUN)", use_container_width=True):
            # Analisis akan disimpan ke st.session_state
            with st.spinner("Menganalisis citra..."):
                 run_analysis_simulation(st.session_state.uploaded_image)
            st.rerun() # Refresh untuk menampilkan hasil

    # Jika ada hasil analisis, tampilkan metrik utama
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Tentukan styling berdasarkan hasil
        status = results['overall_classification']
        status_style_class = "status-messy" if status == 'MESSY' else "status-clean"
        status_color = TEXT_MESSY_LIGHT if status == 'MESSY' else TEXT_CLEAN_LIGHT
        
        st.markdown(f"""
            <div class="result-card">
                <p style="font-size: 14px; color: {TEXT_LIGHT}; margin-bottom: 5px;">KLASIFIKASI KESELURUHAN:</p>
                <p class="{status_style_class}">{status}</p>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px dashed {status_color};">
                    <p style="font-size: 14px; color: {TEXT_LIGHT};">Ringkasan Aset:</p>
                    <p style="font-size: 16px; color: {TEXT_CLEAN_LIGHT}; margin: 0;">&bull; CLEAN: {results['clean_count']} Aset</p>
                    <p style="font-size: 16px; color: {TEXT_MESSY_LIGHT}; margin: 0;">&bull; MESSY: {results['messy_count']} Aset</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

with col2:
    # Tampilan Citra
    image_to_display = None
    if st.session_state.analysis_results:
        # Tampilkan citra yang sudah diberi anotasi (bounding box)
        image_to_display = st.session_state.analysis_results['annotated_image']
        st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-bottom: 10px;">Citra dengan Anotasi</h3>', unsafe_allow_html=True)
    elif st.session_state.uploaded_image:
        # Tampilkan citra asli
        image_to_display = Image.open(st.session_state.uploaded_image)
        st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-bottom: 10px;">Citra Asli</h3>', unsafe_allow_html=True)
    else:
         # Tampilkan placeholder jika tidak ada gambar
        st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-bottom: 10px;">Unggah Citra di Panel Kiri</h3>', unsafe_allow_html=True)
        # Buat placeholder visual
        placeholder_img = Image.new('RGB', (800, 600), color = BG_DARK)
        d = ImageDraw.Draw(placeholder_img)
        d.text((400,300), "Tunggu Citra Diunggah...", fill=ACCENT_PRIMARY_NEON, anchor="mm")
        image_to_display = placeholder_img
        
    st.image(image_to_display, use_column_width=True)

# BARIS 2: KONSOL LOG
results = st.session_state.analysis_results

st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Konsol Log</h3>', unsafe_allow_html=True)

log_content = ""
if results:
    log_content = st.session_state.execution_log_data
else:
    log_content = f"""[{time.strftime('%H:%M:%S')}] INFO: System Initialized. Awaiting Input Payload.<br>
[{time.strftime('%H:%M:%S')}] DATA: No active payload detected. <br>
[{time.strftime('%H:%M:%S')}] MODEL: Detection Model (<b>Siti Naura Khalisa_Laporan 4.pt</b>) and Classification Model (<b>SitiNauraKhalisa_Laporan2.h5</b>) are idle."""

# Mengganti baris baru \n dengan <br> untuk HTML
html_log_content = log_content.replace('\n', '<br>')

st.markdown(f"""
    <div class="log-container">
        {html_log_content}
    </div>
    """, unsafe_allow_html=True)


# BARIS 3: TABEL DETAIL ASET TERDETEKSI
st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Tabel Detail Aset Terdeteksi ({results["detection_model"] if results else "..."})</h3>', unsafe_allow_html=True)

if st.session_state.analysis_results and st.session_state.analysis_results['detections']:
    df = pd.DataFrame(st.session_state.analysis_results['detections'])
    df = df.rename(columns={
        'asset_id': 'Asset ID (Deteksi)', 
        'confidence_score': 'Conf. Deteksi (%)', 
        'classification_tag': 'Tag Kerapihan',
        'normalized_coordinates': 'Koordinat Norm. (x, y, w, h)'
    })
    # Format persentase dan koordinat untuk tampilan yang lebih rapi
    df['Conf. Deteksi (%)'] = (df['Conf. Deteksi (%)'] * 100).round(2).astype(str) + ' %'
    df['Koordinat Norm. (x, y, w, h)'] = df['Koordinat Norm. (x, y, w, h)'].apply(lambda x: f"({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}, {x[3]:.2f})")
    
    # Hilangkan kolom 'bbox' karena berisi pixel values
    df = df.drop(columns=['bbox'])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Unggah citra dan jalankan analisis untuk melihat detail aset.")

# Info tambahan (Footer)
if not ML_LIBRARIES_LOADED:
    st.markdown(f"""
        <p style='color: {TEXT_ERROR}; font-size: 14px; margin-top: 20px;'>
            *CATATAN: Aplikasi berjalan dalam mode SIMULASI karena pustaka ML (YOLO/TensorFlow) tidak ditemukan. Hasil klasifikasi acak namun logikanya sudah disesuaikan dengan mayoritas deteksi per aset.*
        </p>
    """, unsafe_allow_html=True)
