import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import time

# --- Konfigurasi Halaman dan Styling UNIK & KREATIF ---
st.set_page_config(
    page_title="Room Genius: Analisis Clean/Messy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan berwarna, modern, dan unik
st.markdown("""
    <style>
    /* Global Background and Font */
    .stApp {
        background-color: #f0f8ff; /* Light, calming blue background */
        font-family: 'Inter', sans-serif;
    }
    /* Main Header Styling */
    h1 {
        color: #0077b6; /* Deep blue title */
        font-weight: 900;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff; /* White sidebar */
        border-right: 3px solid #0077b6;
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
    }
    /* Metric Box Styling (Clean/Messy results) */
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .clean-result {
        background-image: linear-gradient(to right, #d4f7d4, #a8e6a8); /* Light green gradient */
        border: 2px solid #00a800;
        color: #005a00;
    }
    .messy-result {
        background-image: linear-gradient(to right, #ffdddd, #ffaaaa); /* Light red gradient */
        border: 2px solid #cc0000;
        color: #800000;
    }
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .stProgress > div > div > div > div {
        background-color: #0077b6; /* Blue progress bar */
    }
    </style>
""", unsafe_allow_html=True)

# --- Simulasi Data Model (Inti Logika) ---

# Simulasi Inferensi untuk Klasifikasi Ruangan (Messy/Clean) - Model .h5
def simulate_classification_inference(file_name):
    """Mensimulasikan output klasifikasi Clean Room / Messy Room."""
    # Menghitung Chaos Index dan Tidy Quotient (Metrik Kreatif)
    if 'rapi' in file_name.lower() or 'bersih' in file_name.lower():
        tidy_quotient = np.random.randint(85, 99)
    elif 'berantakan' in file_name.lower() or 'kotor' in file_name.lower() or 'messy' in file_name.lower():
        tidy_quotient = np.random.randint(10, 35)
    else:
        tidy_quotient = np.random.randint(40, 75)
    
    chaos_index = 100 - tidy_quotient
    
    # Klasifikasi Akhir
    if tidy_quotient > 70:
        status = "CLEAN ROOM"
    elif tidy_quotient < 40:
        status = "MESSY ROOM"
    else:
        status = "NEUTRAL (Ambang Batas)"
        
    confidence = tidy_quotient / 100 # Menggunakan Tidy Quotient sebagai Confidence
    
    return status, confidence, tidy_quotient, chaos_index

# Simulasi Inferensi untuk Deteksi Objek (YOLO) - Model .pt
def simulate_detection_inference(file_name):
    """Mensimulasikan output deteksi objek dan menghitung metrik kekacauan."""
    
    if 'rapi' in file_name.lower() or 'bersih' in file_name.lower():
        misplaced_items = np.random.randint(1, 4)
        clothing_count = np.random.randint(0, 2)
        
    elif 'berantakan' in file_name.lower() or 'kotor' in file_name.lower() or 'messy' in file_name.lower():
        misplaced_items = np.random.randint(15, 30)
        clothing_count = np.random.randint(8, 20)
        
    else:
        misplaced_items = np.random.randint(5, 15)
        clothing_count = np.random.randint(3, 8)

    # Data simulasi deteksi dengan bounding box sederhana (hanya metrik)
    return {
        "misplaced_items": int(misplaced_items),
        "clothing_count": int(clothing_count),
        "book_piles": np.random.randint(0, 3)
    }

# --- Fungsi Utility untuk Visualisasi Simulasi Bounding Box ---

def draw_simulated_detection(image_bytes, detection_data):
    """
    Mensimulasikan gambar hasil deteksi objek (yaitu gambar dengan bounding box).
    Fungsi ini hanya memberikan gambar asli untuk kesederhanaan.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except:
        return None

    return img

# --- MODUL DETEKSI OBJEK (Menu 1) ---
def detection_page(uploaded_file):
    st.title("🔎 Modul Deteksi Objek (Model YOLO)")
    st.markdown("Identifikasi objek individual (pakaian, buku, sampah) dan hitung metrik kekacauan.")

    if uploaded_file is not None:
        with st.spinner('⏳ Model YOLO sedang menganalisis objek di ruangan Anda...'):
            time.sleep(2)
        
        # Jalankan Simulasi Deteksi
        detection_results = simulate_detection_inference(uploaded_file.name)
        
        col_img, col_metrics = st.columns([2, 1])

        with col_img:
            image_bytes = uploaded_file.read()
            # Tampilkan gambar hasil deteksi simulasi
            simulated_img = draw_simulated_detection(image_bytes, detection_results)
            st.image(simulated_img, caption="Hasil Deteksi Objek (Simulasi Bounding Box)", use_column_width=True)
            uploaded_file.seek(0) # Reset pointer file
        
        with col_metrics:
            st.subheader("Metrik Kuantitatif Kekacauan")
            
            st.metric(
                label="Total Objek Salah Tempat",
                value=f"{detection_results['misplaced_items']} Item",
                delta="Item yang tidak berada di lokasi seharusnya.",
                delta_color="off"
            )
            st.metric(
                label="Pakaian Berserakan di Lantai",
                value=f"{detection_results['clothing_count']} Item",
                delta="Kontributor utama Chaos Index.",
                delta_color="off"
            )
            st.metric(
                label="Tumpukan Buku/Kertas Liar",
                value=f"{detection_results['book_piles']} Tumpukan",
                delta="Bukan rak yang tersusun rapi.",
                delta_color="off"
            )
            
            st.markdown("---")
            st.caption("Data deteksi ini diserahkan ke Modul Klasifikasi.")
            
    else:
        st.warning("Silakan unggah gambar di sidebar untuk memulai deteksi.")

# --- MODUL KLASIFIKASI GAMBAR (Menu 2) ---
def classification_page(uploaded_file):
    st.title("⭐ Modul Klasifikasi Gambar (Model Keras)")
    st.markdown("Klasifikasi akhir ruangan: **CLEAN ROOM** atau **MESSY ROOM**.")

    if uploaded_file is not None:
        with st.spinner('⏳ Model Keras sedang mengklasifikasi status ruangan...'):
            time.sleep(2)
            
        # Jalankan Simulasi Klasifikasi
        status, confidence, tidy_quotient, chaos_index = simulate_classification_inference(uploaded_file.name)
        
        image_bytes = uploaded_file.read()
        uploaded_file.seek(0) # Reset pointer file
        
        col_img, col_result = st.columns([2, 1])
        
        with col_img:
            st.image(
                Image.open(io.BytesIO(image_bytes)), 
                caption="Gambar Ruangan yang Dianalisis", 
                use_column_width=True
            )

        with col_result:
            st.subheader("Hasil Akhir dan Metrik Kreatif")
            
            # Tentukan style box
            box_style = "clean-result" if status == "CLEAN ROOM" else ("messy-result" if status == "MESSY ROOM" else "result-box")
            color = "#00a800" if status == "CLEAN ROOM" else ("#cc0000" if status == "MESSY ROOM" else "#ff8c00")
            
            st.markdown(f"""
                <div class='result-box {box_style}'>
                    <p style='margin: 0; font-size: 1.2rem; color: #555;'>STATUS KLASIFIKASI</p>
                    <h2 style='margin: 5px 0 0; color: {color}; font-size: 3rem; font-weight: 900;'>{status}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrik Kreatif
            st.metric(
                label="🌟 Tidy Quotient (Kuosien Kerapihan)",
                value=f"{tidy_quotient}%",
                delta="Di atas 70% dianggap 'CLEAN'.",
                delta_color="off"
            )
            
            st.metric(
                label="🔥 Chaos Index (Indeks Kekacauan)",
                value=f"{chaos_index}%",
                delta="Di bawah 30% dianggap 'CLEAN'.",
                delta_color="off"
            )
            
            st.markdown("---")
            
            # Visualisasi Progress Bar Tingkat Kerapihan
            st.markdown(f"**Tingkat Kerapihan ({tidy_quotient}%)**")
            st.progress(tidy_quotient / 100)
            
            # Saran berwarna
            if status == "CLEAN ROOM":
                st.success("🎉 Hasil Luar Biasa! Ruangan ini memenuhi standar CLEAN ROOM.")
            elif status == "MESSY ROOM":
                st.error("❌ Peringatan! Ruangan ini diklasifikasikan sebagai MESSY ROOM.")
            else:
                st.warning("🔶 Ambang Batas: Diperlukan sedikit usaha untuk mencapai CLEAN ROOM.")

    else:
        st.warning("Silakan unggah gambar di sidebar untuk memulai klasifikasi.")

# --- MAIN APP LOGIC ---

st.markdown("<h1>ROOM GENIUS: Analisis Kebersihan AI 🤖</h1>", unsafe_allow_html=True)

# Sidebar untuk navigasi menu dan upload file
st.sidebar.title("Kontrol Analisis")
menu = st.sidebar.radio(
    "Pilih Modul",
    ("Deteksi Objek", "Klasifikasi Gambar")
)

st.sidebar.markdown("---")
# Pindahkan uploader ke sidebar agar tersedia di kedua menu
sidebar_uploaded_file = st.sidebar.file_uploader(
    "🖼️ Unggah Foto Ruangan (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

# Render halaman berdasarkan pilihan menu
if menu == "Deteksi Objek":
    detection_page(sidebar_uploaded_file)
elif menu == "Klasifikasi Gambar":
    classification_page(sidebar_uploaded_file)

st.sidebar.markdown("---")
st.sidebar.caption("Aplikasi ini dibuat dengan Streamlit untuk mensimulasikan fungsionalitas model AI Anda (.pt dan .h5).")
