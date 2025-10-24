import streamlit as st
import random
import time
import json
import io
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pandas as pd

# --- 1. Konfigurasi dan Styling (Tema Cyber Pastel / Vaporwave) ---
# Diperbarui: Mengubah judul halaman browser
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
    
    /* Gaya Monospace untuk Log */
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

# Inisialisasi State Session (Tetap Sama)
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'execution_log_data' not in st.session_state:
    st.session_state.execution_log_data = []

# --- 2. Fungsi Simulasi Analisis Model Khusus ---
def generate_mock_detections(is_clean):
    # Items yang terdeteksi oleh Model Deteksi (Siti Naura Khalisa_Laporan 4.pt - YOLO)
    clean_items = ["Desk_Unit", "Chair_Ergonomic", "Bookshelf_Structured", "Monitor_System", "Rug_Area", "Wall_Lamp", "PC_Tower"]
    messy_items = ["Scattered_Clothes", "Loose_Cables", "Unsorted_Papers", "Trash_Object", "Empty_Bottles", "Food_Wrapper"]
    
    # Jumlah item terdeteksi, lebih banyak jika berantakan
    num_items = random.randint(3, 7) + (0 if is_clean else random.randint(3, 5))
    detections = []
    
    for i in range(num_items):
        is_messy_item = False
        if is_clean:
            # Pada mode CLEAN, 90% objek adalah terstruktur
            label = random.choice(clean_items) if random.random() < 0.9 else random.choice(messy_items)
            confidence = round(random.uniform(0.75, 0.95), 4)
            is_messy_item = label in messy_items
        else:
            # Pada mode MESSY, 60% - 70% objek adalah unoptimized
            if random.random() < 0.7:
                label = random.choice(messy_items)
                confidence = round(random.uniform(0.65, 0.90), 4)
                is_messy_item = True
            else:
                label = random.choice(clean_items)
                confidence = round(random.uniform(0.55, 0.80), 4)
        
        # Penentuan Classification Tag berdasarkan label
        classification_tag = 'UNOPTIMIZED' if is_messy_item else 'STRUCTURED'
        
        # Koordinat Bounding Box (Normalized)
        x_norm = random.uniform(0.05, 0.7)
        y_norm = random.uniform(0.05, 0.7)
        w_norm = random.uniform(0.1, 0.3)
        h_norm = random.uniform(0.1, 0.3)
        
        detections.append({
            "asset_id": label.upper().replace('_', '-'),
            "confidence_score": confidence,
            "classification_tag": classification_tag,
            "normalized_coordinates": [
                round(x_norm, 4), 
                round(y_norm, 4),
                round(w_norm, 4),
                round(h_norm, 4)
            ]
        })
    
    return sorted(detections, key=lambda x: x['confidence_score'], reverse=True)

def draw_boxes_on_image(image_bytes, detections):
    """Menggambar Bounding Box Neon pada Gambar"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    
    # Menggunakan warna Neon
    CLEAN_RGB = ImageColor.getrgb(NEON_CYAN)
    MESSY_RGB = ImageColor.getrgb(NEON_MAGENTA)
    TEXT_RGB = ImageColor.getrgb(BG_DARK) # Teks gelap pada latar belakang kotak neon
    
    try:
        font_size = max(15, min(image_width // 40, 30))
        # PIL menggunakan font default, tidak perlu path
        font = ImageFont.load_default(size=font_size) 
    except IOError:
        font = ImageFont.load_default()
        
    for det in detections:
        x_norm, y_norm, w_norm, h_norm = det['normalized_coordinates']
        
        # Konversi koordinat normalized ke koordinat piksel
        x_min = int(x_norm * image_width)
        y_min = int(y_norm * image_height)
        x_max = int((x_norm + w_norm) * image_width)
        y_max = int((y_norm + h_norm) * image_height)
        
        # Clamp koordinat agar tidak melebihi batas gambar
        x_max = min(x_max, image_width - 1)
        y_max = min(y_max, image_height - 1)
        
        label = det['asset_id']
        confidence = det['confidence_score']
        
        # Warna box berdasarkan status
        box_rgb = CLEAN_RGB if det['classification_tag'] == 'STRUCTURED' else MESSY_RGB
        
        # Gambar kotak dengan warna neon yang cerah
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_rgb, width=4) 
        
        text_content = f"{label} [{int(confidence * 100)}%]"
        
        try:
            # Gunakan draw.textbbox untuk menghitung ukuran teks
            text_bbox = draw.textbbox((0, 0), text_content, font=font) 
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text_content, font=font)
        
        # Posisi Teks: Di atas kotak
        text_x = x_min
        text_y = y_min - text_height - 5 
        
        # Jika teks terlalu tinggi, letakkan di bawah
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
    """Membuat format log tekstual yang menyerupai output konsol"""
    log_lines = []
    
    # 1. Start Log
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] INFO: System Initialized.")
    
    # 2. Data Payload
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 4))}] DATA: Payload Acquired: {uploaded_file_name}.")

    # 3. Model Loading: Model Deteksi
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 3))}] MODEL-DETECTION: Loading <b>{results['detection_model']}</b> (YOLO V8 Architecture).")
    
    # 4. Model Inference: Deteksi
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 2))}] INFERENCE-YOLO: Detecting {len(results['detections'])} Assets. Messy Count: {results['messy_count']}.")
    
    # 5. Model Loading: Model Klasifikasi
    log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime(time.time() - 1))}] MODEL-CLASSIFICATION: Loading <b>{results['classification_model']}</b> (Keras/CNN).")
    
    # 6. Classification
    tag_color = NEON_CYAN if results['is_clean'] else NEON_MAGENTA
    final_status = results['final_status'].split(': ')[1]
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] CLASSIFY-CNN: Final Classification Complete. Input: Detection Data Stream.")
    
    # 7. Final Report
    log_lines.append(f"[{time.strftime('%H:%M:%S')}] REPORT: Final Status: <span style='color:{tag_color};'><b>{final_status}</b></span> (Clean Conf: {results['conf_clean']}%, Messy Conf: {results['conf_messy']}%).")
    
    # Gabungkan dengan <br> untuk markdown
    return '<br>'.join(log_lines)

def simulate_yolo_analysis():
    if st.session_state.uploaded_file is None:
        st.error("Sila muat naik imej ruangan dahulu.")
        return
    st.session_state.analysis_results = None
    st.session_state.processed_image = None
    
    # Membersihkan log sebelumnya dan log awal
    st.session_state.execution_log_data = [
        f"[{time.strftime('%H:%M:%S')}] INFO: System Initialized. Awaiting user action."
    ]
    
    # Placeholder log loading
    log_placeholder = st.empty()
    log_placeholder.markdown(f'<p style="color: {TEXT_LIGHT};">SYSTEM> Initiating inference. Loading model <b>Siti Naura Khalisa_Laporan 4.pt</b>...</p>', unsafe_allow_html=True)
    
    # Progress Bar Simulasi
    progress_bar = st.progress(0, text="Loading Tensor Core & Running Inference...")
    for percent_complete in range(100):
        time.sleep(0.01) # Mengurangi sleep time agar lebih cepat
        progress_bar.progress(percent_complete + 1, text=f"Processing... {percent_complete+1}%")
    progress_bar.empty()
    
    # --- SIMULASI INFERENCE DUA MODEL ---
    
    image_data = st.session_state.uploaded_file.getvalue()
    
    # Tentukan apakah hasilnya akan cenderung bersih atau berantakan (55% clean chance)
    is_clean_outcome = random.random() < 0.55
    
    # Model 1: Deteksi Objek (Simulasi Laporan 4.pt - YOLO)
    mock_detections = generate_mock_detections(is_clean_outcome) 
    messy_count = sum(1 for d in mock_detections if d['classification_tag'] == 'UNOPTIMIZED')
    
    # Model 2: Klasifikasi Akhir (Simulasi Laporan 2.h5 - CNN)
    
    # Logic untuk menentukan status akhir berdasarkan jumlah item berantakan
    if messy_count <= 2:
        conf_clean = random.uniform(0.90, 0.98) # Confidence tinggi untuk Clean
        conf_messy = 1.0 - conf_clean
        final_status = "STATUS: ROOM CLEAN - OPTIMAL"
        final_message = "System Integrity Check: GREEN (CYAN NEON). Minimum clutter detected. Excellent organization."
    else:
        conf_messy = random.uniform(0.80, 0.95) # Confidence tinggi untuk Messy
        conf_clean = 1.0 - conf_messy
        final_status = "STATUS: ROOM MESSY - ALERT"
        final_message = "System Integrity Check: RED (MAGENTA NEON). High probability of unoptimized state. Clutter detected. Recommendation: De-clutter immediately."

    # Gambar Bounding Box (Visualisasi output Model 1)
    processed_image_bytes = draw_boxes_on_image(image_data, mock_detections)
    
    results = {
        "final_status": final_status,
        "is_clean": final_status == "STATUS: ROOM CLEAN - OPTIMAL",
        "conf_clean": round(conf_clean * 100, 2),
        "conf_messy": round(conf_messy * 100, 2),
        "messy_count": messy_count, 
        "detection_model": "Siti Naura Khalisa_Laporan 4.pt", # Integrasi Model 1
        "classification_model": "SitiNauraKhalisa_Laporan2.h5", # Integrasi Model 2
        "detections": mock_detections,
        "final_message": final_message
    }
    
    st.session_state.processed_image = processed_image_bytes
    st.session_state.analysis_results = results
    
    # Generate the log data for display
    st.session_state.execution_log_data = format_execution_log(results, st.session_state.uploaded_file.name)
    
    log_placeholder.empty()
    st.success("SYSTEM> Analysis Completed. Report Generated and Visualization Rendered.")

# --- 3. Tata Letak Streamlit ---

# Diperbarui: Mengubah Judul dan Subjudul
st.markdown(f"""
    <header>
        <h1>ROOM INSIGHT <span style="font-size: 18px; margin-left: 15px; color: {ACCENT_PRIMARY_NEON};">CLEAN OR MESSY?</span></h1>
        <p style="color: {TEXT_LIGHT}; font-size: 14px;">Klasifikasikan kerapihan ruangan Anda menggunakan arsitektur model ganda (Deteksi + Klasifikasi).</p>
    </header>
    <div style="margin-bottom: 20px;"></div>
    """, unsafe_allow_html=True)


# --- TAMPILAN 1: TATA LETAK DUA KOLOM UTAMA (ATAS) ---
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
    
    if st.button("⚡ INITIATE DUAL-MODEL ANALYSIS", disabled=button_disabled, use_container_width=True):
        simulate_yolo_analysis()
        
    st.markdown('</div>', unsafe_allow_html=True)


# Kolom 2: Live Detection Grid
with col_detection:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color: {ACCENT_PRIMARY_NEON};">2. Live Detection Grid </h2>', unsafe_allow_html=True)

    # Tentukan warna border dinamis
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

    # Container untuk menampung gambar dengan border dinamis
    st.markdown(f"""
        <div style="border: 4px solid #34495E; border-radius: 10px; padding: 5px; background-color: {BG_DARK};" class="{border_class}">
        """, unsafe_allow_html=True)

    if st.session_state.uploaded_file:
        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, caption='VISUALIZATION: Live Detection Grid (YOLO v8 Output)', use_container_width=True)
        else:
            # Tampilkan gambar asli jika belum diproses
            image_data = st.session_state.uploaded_file.getvalue()
            st.image(image_data, caption='Image Data Stream (Original)', use_container_width=True)
            
        st.markdown(f'<p style="text-align: center; color: {caption_style}; font-weight: bold; margin-top: 10px; text-shadow: 0 0 3px {caption_style};">'
                    f'{caption_text}</p>', unsafe_allow_html=True)
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
                <p style="color: {TEXT_LIGHT}; font-size: 10px; margin-top: 5px; opacity: 0.6;">(From Model {results['classification_model']})</p>
            </div>
            """, unsafe_allow_html=True)
        
    # Kolom 3 (1 Unit Lebar): Confidence: Messy
    with col_messy_conf:
        # Card Neon Magenta
        st.markdown(f"""
            <div class="status-metric-card" style="height: 100%; border-color: {NEON_MAGENTA}; background-color: {CARD_BG}; box-shadow: 0 0 8px {NEON_MAGENTA};">
                <p style="color: {NEON_MAGENTA}; font-size: 12px; margin-bottom: 5px; font-weight: bold;">CONFIDENCE: MESSY</p>
                <p style="color: {TEXT_LIGHT}; font-size: 28px; font-weight: bold;">{results["conf_messy"]}%</p>
                <p style="color: {TEXT_LIGHT}; font-size: 10px; margin-top: 5px; opacity: 0.6;">(From Model {results['classification_model']})</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Placeholder saat belum ada hasil
    st.markdown(f"""
        <div style="text-align: center; padding: 40px; border: 2px dashed {ACCENT_PRIMARY_NEON}; border-radius: 10px; background-color: {CARD_BG};">
            <h3 style="font-size: 24px; color:{ACCENT_PRIMARY_NEON};">METRICS AWAITING INFERENCE</h3>
            <p style="font-size: 16px; color: {TEXT_LIGHT};">Upload image and click 'INITIATE DUAL-MODEL ANALYSIS' to generate report.</p>
        </div>
    """, unsafe_allow_html=True)


# BARIS 2: LOG OBJEK TERDETEKSI (Execution Log)
st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Log Eksekusi Model Ganda (Dual-Model Execution Log)</h3>', unsafe_allow_html=True)

# Container untuk Log yang bergulir
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

# BARIS 3: TABEL DETAIL ASET TERDETEKSI
st.markdown(f'<h3 style="color: {ACCENT_PRIMARY_NEON}; font-size: 20px; margin-top: 25px; border-bottom: 1px solid #34495E; padding-bottom: 5px;">Tabel Detail Aset Terdeteksi ({results["detection_model"]})</h3>', unsafe_allow_html=True)

if st.session_state.analysis_results:
    df = pd.DataFrame(st.session_state.analysis_results['detections'])
    df = df.rename(columns={
        'asset_id': 'Asset ID (Deteksi)', 
        'confidence_score': 'Conf. Deteksi (%)', 
        'classification_tag': 'Tag Kerapihan',
        'normalized_coordinates': 'Koordinat Norm. (x, y, w, h)'
    })
    df['Conf. Deteksi (%)'] = (df['Conf. Deteksi (%)'] * 100).round(2).astype(str) + '%'
    
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
