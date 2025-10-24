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
    ML_LIBRARIES_LOADED = False

# --- 2. Konfigurasi dan Styling (Tema Cute Vision AI / Girly Pastel) ---
# CATATAN: Semua kustomisasi HTML/CSS DIBUANG. Hanya menggunakan fitur Streamlit bawaan.

st.set_page_config(
    page_title="ROOM INSIGHT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Palet Warna Cute Vision AI (Digunakan untuk st.info/st.error/warna teks dinamis)
# Warna ini sekarang hanya berfungsi sebagai referensi/nilai jika bisa diimplementasikan.
BG_LIGHT = "#F0F0FF"            # Very Light Lavender/White
CARD_BG = "#FFFFFF"             # Soft White Card Background
TEXT_DARK = "#333333"           # Main text color (Dark)
ACCENT_PRIMARY_PINK = "#FF99C8" # Soft Pink (Main Accent)
ACCENT_BLUE = "#93CCFF"         # Light Blue (Secondary accent)
ACCENT_NEON_CYAN = "#00FFFF"    # Neon Cyan (Clean Status)  
ACCENT_PINK_MESSY = "#FF3366"   # Hot Pink/Fuschia (Messy Status)

TEXT_CLEAN_STATUS = ACCENT_NEON_CYAN
TEXT_MESSY_STATUS = ACCENT_PINK_MESSY

# Inisialisasi State Session
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'execution_log_data' not in st.session_state:
    st.session_state.execution_log_data = "Log display is disabled."
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'UPLOAD' # Mengelola navigasi

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
        cnn_model = load_model(CNN_MODEL_PATH)
        
        return yolo_model, cnn_model
    except Exception as e:
        st.error(f"FATAL ERROR: Gagal memuat file model dari path. Pastikan file berada di folder 'model/' dan namanya benar: {e}")
        return None, None

# --- 4. Fungsi Real Inference dan Pemrosesan Data (TETAP SAMA) ---

def run_yolo_detection(yolo_model, image_path):
    """Menjalankan inferensi YOLOv8 dan memproses hasilnya."""
    
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
    
    MESSY_CLASS_NAMES = ["Scattered_Clothes", "Loose_Cables", "Unsorted_Papers", "Trash_Object", "Empty_Bottles", "Food_Wrapper"]
    class_names = result.names # Peta ID ke nama kelas
    
    
    for box in result.boxes:
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
    
    target_size = (128, 128) 
    image_resized = image.resize(target_size)
    
    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 
    
    predictions = cnn_model.predict(img_array, verbose=0)[0]
    
    conf_clean = predictions[0]  
    conf_messy = predictions[1]  
    
    return conf_clean, conf_messy

# --- 5. Fungsi Utilitas Visualisasi (TETAP SAMA) ---

def draw_boxes_on_image(image_bytes, detections):
    """Menggambar Bounding Box Neon pada Gambar dari hasil deteksi."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    
    CLEAN_RGB = ImageColor.getrgb(ACCENT_NEON_CYAN) 
    MESSY_RGB = ImageColor.getrgb(ACCENT_PINK_MESSY)
    TEXT_BG_RGB = ImageColor.getrgb(CARD_BG)
    
    try:
        font_size = max(15, min(image_width // 40, 30))
        font = ImageFont.load_default(size=font_size) 
    except IOError:
        font = ImageFont.load_default()
        
    for det in detections:
        x_norm, y_norm, w_norm, h_norm = det['normalized_coordinates']
        
        x_min = int((x_norm - w_norm/2) * image_width)
        y_min = int((y_norm - h_norm/2) * image_height)
        x_max = int((x_norm + w_norm/2) * image_width)
        y_max = int((y_norm + h_norm/2) * image_height)
        
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

# --- 6. Fungsi Utama Alur Kerja ML (Disesuaikan untuk navigasi) ---

def run_ml_analysis():
    """Menjalankan analisis ML nyata dan beralih ke halaman report."""
    
    if not ML_LIBRARIES_LOADED:
        st.error("Analisis dibatalkan: Pustaka Machine Learning gagal dimuat saat startup.")
        return

    if st.session_state.uploaded_file is None:
        st.error("Sila muat naik imej ruangan dahulu.")
        return
    
    yolo_model, cnn_model = load_ml_model()
    
    if yolo_model is None or cnn_model is None:
        st.error("Analisis dibatalkan: File Model gagal dimuat dari direktori.")
        return

    st.session_state.analysis_results = None
    st.session_state.processed_image = None
    st.session_state.execution_log_data = "Log display is disabled."
    
    
    log_placeholder = st.empty()
    log_placeholder.info("SYSTEM> Initiating inference. Loading models...")
    
    progress_bar = st.progress(0, text="Loading Tensor Core & Running Inference...")
    
    image_bytes = st.session_state.uploaded_file.getvalue()
    
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
        st.error(f"Error saat menjalankan Klasifikasi CNN: {e}")
        progress_bar.empty()
        return
    
    progress_bar.progress(90, text="[Step 2/2] Post-processing results...")
    
    # --- PENENTUAN STATUS AKHIR (Berdasarkan CNN + Hybrid Rule) ---
    
    MESSY_DETECTION_THRESHOLD = 3 
    
    is_clean = conf_clean > conf_messy
        
    is_overridden = False
    if is_clean and messy_count >= MESSY_DETECTION_THRESHOLD:
        is_clean = False # Override status
        is_overridden = True
    
    if is_clean:
        final_status = "STATUS: CLEAN ROOM - OPTIMAL" 
        final_message = f"WAH KEREN KAMU MENJAGA KEBERSIHAN KAMAR"
    else:
        final_status = "STATUS: RUANGAN BERANTAKAN - PERINGATAN"
        
        if is_overridden:
            final_message = f"HYBRID OVERRIDE: YOLO mendeteksi {messy_count} aset TIDAK OPTIMAL. Status Akhir: PERINGATAN."
        else:
            final_message = f"Pemeriksaan Integritas Sistem: MERAH. Kepercayaan berantakan {round(conf_messy * 100, 2)}%. Kekacauan terdeteksi. Rekomendasi: Segera Rapikan."
    
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
    
    progress_bar.progress(100, text="Analysis Complete. Generating Report.")
    progress_bar.empty()

    try:
        os.remove(temp_path)
    except OSError:
        pass

    st.session_state.processed_image = processed_image_bytes
    st.session_state.analysis_results = results
    
    log_placeholder.empty()
    
    # --- BERALIH KE HALAMAN REPORT ---
    st.session_state.app_state = 'REPORT'
    st.rerun() 

# --- 7. Fungsi Utility Tips/Apresiasi ---

def get_tips_and_appreciation(is_clean, messy_count, is_overridden):
    """Menghasilkan konten untuk tips atau apresiasi (Pure Streamlit markdown)."""
    if is_clean:
        return {
            "title": "✅ STATUS OPTIMAL: APRESIASI",
            "emoji": "✨",
            "type": "success",
            "content": f"""
Selamat! Ruangan Anda menunjukkan tingkat kerapihan yang luar biasa. Sistem kami mendeteksi sedikit atau tidak ada **ASET TIDAK OPTIMAL** (jumlah: {messy_count}).

**Tips Maintenance:**
* Lanjutkan dengan prinsip 'Less is More': Pastikan setiap barang memiliki tempatnya yang spesifik.
* Audit Digital: Jika ini ruang kerja, pertimbangkan untuk merapikan file digital secara berkala, seperti yang Anda lakukan pada aset fisik.
* Sistem 5 Menit: Lakukan audit kerapihan cepat 5 menit setiap hari untuk mencegah penumpukan.
            """
        }
    else: 
        override_note = ""
        if is_overridden:
            override_note = f"\n\n**CATATAN SISTEM:** Meskipun klasifikasi CNN awal mungkin 'Rapi', YOLOv8 mendeteksi {messy_count} item tidak optimal yang signifikan, memicu Aturan Hibrida OVERRIDE ke status PERINGATAN.\n"

        return {
            "title": "🚨 STATUS PERINGATAN: SARAN OPTIMASI RUANGAN",
            "emoji": "🧹",
            "type": "warning",
            "content": f"""
{override_note}
Ruangan Anda teridentifikasi sebagai **TIDAK OPTIMAL / BERANTAKAN**. Ini menunjukkan adanya aset-aset yang perlu dikelola ulang. Model Deteksi kami mengidentifikasi **{messy_count} item TIDAK OPTIMAL**.

**Rekomendasi Tindakan (De-Clutter Protocol):**
* Fokus pada Aset Berisiko: Prioritaskan merapikan item yang terdeteksi (seperti **Pakaian Berserakan** atau **Kertas Tidak Teratur**).
* Prinsip 4 Kotak: Gunakan 4 kotak: Sampah, Donasi, Simpan (jauh), dan Simpan (di sini). Segera distribusikan aset berdasarkan kategori ini.
* Re-Scan: Setelah merapikan, muat ulang gambar ruangan Anda dan jalankan analisis kembali untuk memverifikasi Status Optimal.
            """
        }

# --- 8. Fungsi Render Halaman (Pemisahan UI - PURE STREAMLIT) ---

def render_upload_page():
    """Halaman 1: Upload Gambar Saja (Pure Streamlit)."""
    
    st.title("ROOM INSIGHT")
    st.header("Clean or Messy?")
    st.write("Klasifikasikan Keadaan Ruangan Anda!")

    # Menggantikan modern-card dengan st.container
    with st.container(border=True): 
        st.subheader("Upload Foto")

        if not ML_LIBRARIES_LOADED:
            st.error("GAGAL PENTING: Pustaka 'ultralytics' atau 'tensorflow' tidak ditemukan. Analisis ML TIDAK DAPAT DILANJUTKAN.")
            
        uploaded_file = st.file_uploader(
            "Upload Image File (JPG/PNG)", 
            type=["jpg", "jpeg", "png"],
            key="uploader_main",
            help="Unggah file gambar ruangan untuk dianalisis."
        )

        st.session_state.uploaded_file = uploaded_file

        if uploaded_file is not None:
            st.image(uploaded_file, caption='Image Preview', use_column_width=True)

    # Tombol dinonaktifkan jika tidak ada file yang diunggah ATAU jika pustaka ML gagal dimuat
    button_disabled = st.session_state.uploaded_file is None or not ML_LIBRARIES_LOADED
    
    # Menggunakan st.button biasa
    if st.button("💖 INITIATE DUAL-MODEL ANALYSIS", disabled=button_disabled, use_container_width=True, type="primary"):
        with st.spinner('Running Dual-Model Analysis...'):
            run_ml_analysis() 
        
    if st.session_state.uploaded_file and not ML_LIBRARIES_LOADED:
        st.warning("Analisis dinonaktifkan karena pustaka Machine Learning tidak tersedia. Harap instal 'ultralytics' dan 'tensorflow' untuk fungsi penuh.")

def render_report_page():
    """Halaman 2: Tampilan Laporan Analisis (Pure Streamlit)."""

    results = st.session_state.analysis_results
    
    if not results or not st.session_state.uploaded_file:
        st.error("Sesi analisis tidak valid. Kembali ke halaman utama.")
        st.session_state.app_state = 'UPLOAD'
        st.rerun()
        return

    st.title(f"ANALYSIS REPORT: {st.session_state.uploaded_file.name.upper()}")
    st.caption("Laporan lengkap hasil deteksi objek dan klasifikasi objek.")

    st.header("Deteksi Objek")

    # Menggunakan st.image dengan container untuk border
    with st.container(border=True):
        st.subheader('VISUALIZATION: Live Detection Grid (YOLO v8 Output)')
        if st.session_state.processed_image:
            st.image(st.session_state.processed_image, caption='Processed Image', use_container_width=True)
        else:
            image_data = st.session_state.uploaded_file.getvalue()
            st.image(image_data, caption='Original Image', use_container_width=True)
        
    st.divider()

    # --- 2. KLASIFIKASI FINAL STATUS (UTAMA) ---
    st.header("Classification Status")

    col_report, col_clean_conf, col_messy_conf = st.columns([2, 1, 1])

    status_main_text = results['final_status'].split(': ')[1]
    message = results['final_message']
    
    with col_report:
        # Menggunakan st.info/st.success/st.error sebagai pengganti card kustom
        if results['is_clean']:
            st.success(f"**FINAL STATUS: {status_main_text}**\n\n{message}")
        else:
            st.error(f"**FINAL STATUS: {status_main_text}**\n\n{message}")
            
    with col_clean_conf:
        # Menggunakan st.metric
        st.metric(label=f"CONFIDENCE: CLEAN (Model: {results['classification_model']})", 
                  value=f"{results['conf_clean']}%", 
                  delta=None)
            
    with col_messy_conf:
        # Menggunakan st.metric
        st.metric(label=f"CONFIDENCE: MESSY (Model: {results['classification_model']})", 
                  value=f"{results['conf_messy']}%", 
                  delta=None)

    st.divider()
    
    # --- 3. TIPS / APRESIASI ---
    st.header("Rekomendasi")
    
    tips = get_tips_and_appreciation(results['is_clean'], results['messy_count'], results.get('is_overridden', False))
    
    # Mengganti tips-box dengan container dan notifikasi Streamlit
    with st.container(border=True):
        st.subheader(f"{tips['emoji']} {tips['title']}")
        if tips['type'] == 'success':
             st.success(tips['content'])
        elif tips['type'] == 'warning':
             st.warning(tips['content'])
        else:
             st.info(tips['content'])


    # Tombol untuk kembali
    st.button("↩ BACK", 
              on_click=lambda: st.session_state.update(app_state='UPLOAD', analysis_results=None, processed_image=None), 
              use_container_width=False)

# --- 9. FUNGSI UTAMA APP CONTROLLER ---
if st.session_state.app_state == 'UPLOAD':
    render_upload_page()
elif st.session_state.app_state == 'REPORT':
    render_report_page()
else:
    st.session_state.app_state = 'UPLOAD'
    st.rerun()
