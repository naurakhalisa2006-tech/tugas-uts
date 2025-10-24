import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import time # Untuk simulasi proses loading model

# --- Konfigurasi Halaman dan Styling ---
st.set_page_config(
    page_title="The Room Whisperer (Pembisik Ruangan)",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan keren dan unik
st.markdown("""
    <style>
    .stApp {
        background-color: #f7f9fc;
    }
    .header-text {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        transition: all 0.3s;
        box-shadow: 0 4px #2980b9;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 2px #1f618d;
        transform: translateY(2px);
    }
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        color: #2c3e50;
    }
    .clean-result {
        background-color: #e6f7ff;
        border-left: 5px solid #00c763;
    }
    .messy-result {
        background-color: #ffe6e6;
        border-left: 5px solid #e74c3c;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Simulasi Data Model (Menggantikan Model .pt dan .h5) ---

# Simulasi Kelas Objek yang dideteksi model YOLO (.pt)
OBJECT_CLASSES = {
    'Messy': ['pakaian_berserakan', 'sampah_terbuka', 'piring_kotor', 'kabel_kusut'],
    'Neutral': ['meja', 'kursi', 'lampu', 'jendela'],
    'Tidy_Aid': ['kotak_penyimpanan', 'rak_buku_terisi_rapi', 'tempat_sampah_tertutup']
}

# Simulasi Inferensi Model: Menghasilkan metrik berdasarkan nama file
def simulate_model_inference(file_name):
    """
    Mensimulasikan output dari pipeline model Deteksi (.pt) dan Klasifikasi (.h5).
    Metrik dihasilkan berdasarkan kriteria unik yang telah ditentukan.
    """
    
    # Kriteria simulasi untuk menghasilkan output yang bervariasi
    if 'rapi' in file_name.lower() or 'bersih' in file_name.lower():
        # Clean Room State
        chaos_index = np.random.randint(5, 20) 
        tidy_quotient = np.random.randint(80, 99)
        misplaced_items = np.random.randint(1, 4)
        
    elif 'berantakan' in file_name.lower() or 'kotor' in file_name.lower() or 'messy' in file_name.lower():
        # Messy Room State
        chaos_index = np.random.randint(70, 95)
        tidy_quotient = np.random.randint(5, 30)
        misplaced_items = np.random.randint(15, 30)
        
    else:
        # Neutral/Random State
        chaos_index = np.random.randint(30, 65)
        tidy_quotient = 100 - chaos_index
        misplaced_items = np.random.randint(5, 15)

    # Klasifikasi Akhir (Simulasi Model .h5)
    if tidy_quotient > 75:
        room_status = "CLEAN ROOM"
    elif tidy_quotient < 40:
        room_status = "MESSY ROOM"
    else:
        room_status = "NEUTRAL (Perlu Sedikit Perhatian)"
        
    return {
        "status": room_status,
        "chaos_index": chaos_index, # Indeks Kekacauan (0=Bersih, 100=Chaos Total)
        "tidy_quotient": tidy_quotient, # Kuosien Kerapihan (100=Sangat Rapi)
        "misplaced_items": misplaced_items, # Jumlah Objek yang Salah Tempat
        "detected_classes": {
            'pakaian_berserakan': np.random.randint(misplaced_items / 2, misplaced_items),
            'tumpukan_buku_liar': np.random.randint(0, 5),
            'rak_buku_terisi_rapi': np.random.randint(1, 4),
            'sampah_terbuka': np.random.randint(0, 3)
        }
    }

# --- Fungsi Utility untuk Visualisasi Simulasi Heatmap ---

def draw_simulated_heatmap(image_bytes, chaos_index):
    """
    Mensimulasikan visualisasi Heatmap Kekacauan pada gambar.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except:
        return None

    # Simulasikan Heatmap sebagai overlay transparan merah
    overlay_color = (255, 0, 0)  # Merah
    # Intensitas transparansi berdasarkan Chaos Index
    alpha = int((chaos_index / 100) * 128) # Max 128 (setengah transparan)
    
    # Buat layer overlay merah dengan alpha yang sesuai
    overlay = Image.new('RGBA', img.size, overlay_color + (alpha,))
    
    # Gabungkan gambar asli dengan overlay
    combined = Image.alpha_composite(img.convert('RGBA'), overlay)
    
    return combined.convert("RGB")

# --- JUDUL UTAMA ---
st.markdown("<h1 class='header-text'>The Room Whisperer ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='header-text' style='font-weight: 300; font-size: 1.5rem; color: #7f8c8d;'>Klasifikasi Ruangan: Messy atau Clean? - Analisis Berbasis AI</h3>", unsafe_allow_html=True)

st.sidebar.title("Instruksi Aplikasi")
st.sidebar.info("""
    1. **Unggah Foto:** Ambil atau unggah foto ruangan Anda.
    2. **Analisis Model:** Aplikasi akan memproses foto (menggunakan simulasi model `YOLO` dan `Keras` Anda).
    3. **Lihat Hasil:** Dapatkan skor metrik unik seperti **Tidy Quotient** dan saran spesifik.
""")

# --- UPLOAD GAMBAR ---
uploaded_file = st.file_uploader(
    "🖼️ Unggah Foto Ruangan (Format JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # --- PROSES ANALISIS ---
    
    st.markdown("---")
    
    # Display loading state
    with st.spinner('⏳ Model Deteksi & Klasifikasi sedang bekerja... Analisis Chaos Index...'):
        time.sleep(2.5) # Simulasi waktu loading model
    
    # Jalankan Simulasi Inferensi
    results = simulate_model_inference(uploaded_file.name)
    
    # Tentukan Style Box berdasarkan status
    box_style = "clean-result" if results['status'] == "CLEAN ROOM" else ("messy-result" if results['status'] == "MESSY ROOM" else "")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("💡 Hasil Klasifikasi")
        
        # Tampilkan Status Ruangan
        st.markdown(f"""
            <div class='metric-box {box_style}'>
                <p style='font-size: 1.2rem; margin-bottom: 5px;'>STATUS RUANGAN</p>
                <h2 style='font-size: 2.5rem; font-weight: 700; color: {'#00c763' if results['status'] == 'CLEAN ROOM' else ('#e74c3c' if results['status'] == 'MESSY ROOM' else '#f39c12')};'>{results['status']}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Metrik Bervariasi (Tidy Quotient & Chaos Index)
        st.subheader("📊 Metrik Kerapihan Unik")

        # Tidy Quotient (Kuosien Kerapihan)
        st.metric(
            label="✨ Tidy Quotient (100 = Sempurna)",
            value=f"{results['tidy_quotient']}%",
            delta=f"Ideal > 75%",
            delta_color="off"
        )
        
        # Chaos Index (Indeks Kekacauan)
        st.metric(
            label="⚠️ Chaos Index (0 = Minimal)",
            value=f"{results['chaos_index']}/100",
            delta=f"Berantakan > 60",
            delta_color="off"
        )
        
        # Jumlah Item Misplaced
        st.metric(
            label="🗑️ Objek Salah Tempat (Deteksi)",
            value=f"{results['misplaced_items']} Item",
            delta="Fokus di sini!",
            delta_color="off"
        )


    with col2:
        st.subheader("📍 Visualisasi & Heatmap Kekacauan")
        
        # Simulasi Heatmap
        image_bytes = uploaded_file.read()
        simulated_img = draw_simulated_heatmap(image_bytes, results['chaos_index'])
        
        if simulated_img:
            st.image(
                simulated_img, 
                caption=f"Visualisasi Heatmap (Warna merah menunjukkan area kepadatan kekacauan tinggi)", 
                use_column_width=True
            )
        else:
            st.warning("Gagal memuat visualisasi gambar.")
            
    st.markdown("---")
    
    # --- SARAN TINDAKAN KREATIF ---
    st.subheader("🎯 Saran Tindakan Terarah (Actionable Insights)")
    
    if results['status'] == "CLEAN ROOM":
        st.balloons()
        st.success("🎉 SELAMAT! Ruangan Anda memiliki Tidy Quotient yang tinggi. Pertahankan Kerapihan ini!")
        st.info("Saran Lanjutan: Ulangi analisis ini seminggu sekali untuk memantau konsistensi kebersihan Anda.")
        
    elif results['status'] == "MESSY ROOM":
        st.error("🚨 PERINGATAN! Chaos Index tinggi. Ruangan ini membutuhkan 'Intervensi Kerapihan Cepat'.")
        
        st.markdown("Berikut adalah **Misi Kerapihan** Anda:")
        
        # Membuat saran dinamis berdasarkan hasil deteksi
        detected = results['detected_classes']
        
        st.markdown("""
        <style>
            .mission-item {
                background-color: #fcebe8; 
                padding: 10px; 
                border-radius: 8px; 
                margin-bottom: 8px; 
                border-left: 4px solid #e74c3c;
            }
        </style>
        """, unsafe_allow_html=True)
        
        if detected.get('pakaian_berserakan', 0) > 5:
            st.markdown(f"<div class='mission-item'>**Tugas Utama:** Deteksi menemukan **{detected['pakaian_berserakan']} item pakaian** di luar tempatnya. Selesaikan 'Misi Keranjang Kotor' segera!</div>", unsafe_allow_html=True)
            
        if detected.get('tumpukan_buku_liar', 0) > 0:
            st.markdown(f"<div class='mission-item'>**Tugas Tambahan:** Ada **{detected['tumpukan_buku_liar']} tumpukan buku/dokumen** yang tidak stabil. Lakukan 'Misi Stabilisasi Rak'.</div>", unsafe_allow_html=True)

        if detected.get('sampah_terbuka', 0) > 0:
            st.markdown(f"<div class='mission-item'>**Misi Kebersihan Cepat:** Terdeteksi adanya sampah di luar wadah. Selesaikan 'Misi Zero-Waste'!</div>", unsafe_allow_html=True)

        if results['misplaced_items'] > 20:
            st.markdown(f"<div class='mission-item'>**Level Darurat:** {results['misplaced_items']} item salah tempat! Mulai dari satu area kecil saja (misal: area meja).</div>", unsafe_allow_html=True)

        
    else: # Neutral
        st.warning("⚠️ Status Netral. Ruangan Anda berada di perbatasan Clean/Messy. Perlu sedikit 'dorongan' kerapihan.")
        st.info("Saran: Fokus pada objek salah tempat dan naikkan Tidy Quotient Anda di atas 75%!")

else:
    st.info("Ayo coba! Unggah foto ruangan Anda dan biarkan The Room Whisperer menganalisis Chaos Index-nya!")

st.sidebar.markdown("---")
st.sidebar.caption("Simulasi dibangun dengan Streamlit dan mensimulasikan inferensi dari model YOLO (.pt) dan Keras (.h5) yang Anda sediakan.")
