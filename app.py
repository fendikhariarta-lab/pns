# engine.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# =====================================================================
# LANGKAH 1: PERSIAPAN MODEL DAN BASELINE (SESUAI MATERI KULIAH)
# =====================================================================

# 1. Menyiapkan data historis sederhana untuk pelatihan model
# Fitur yang digunakan: [Anggaran Iklan (Juta), Besaran Diskon (%)]
X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])

# Target operasional: Keuntungan Toko (Juta)
y_train = np.array([50, 80, 110, 90, 150])

# 2. Melatih model regresi linear sebagai mesin replika (Digital Twin)
model = LinearRegression().fit(X_train, y_train)

# 3. Menetapkan skenario dasar (Baseline) sebagai pembanding performa
# Kondisi operasional saat ini: Anggaran Iklan 10 Juta, Besaran Diskon 10%
baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]


# =====================================================================
# LANGKAH 2: LOGIKA SIMULATOR (ANALISIS WHAT-IF)
# =====================================================================

def run_simulation(new_iklan, new_diskon):
    """
    Menjalankan simulasi kebijakan intervensi baru dan membandingkannya
    dengan kondisi dasar (baseline).
    
    Argument:
        new_iklan (int/float): Anggaran iklan baru hasil intervensi user.
        new_diskon (int/float): Besaran diskon baru hasil intervensi user.
        
    Return:
        prediction (float): Estimasi keuntungan baru.
        delta_y (float): Selisih keuntungan baru terhadap baseline.
    """
    # Mengubah masukan intervensi pengguna menjadi format vektor fitur
    intervention_input = np.array([[new_iklan, new_diskon]])
    
    # Melakukan prediksi dinamis menggunakan Digital Twin
    prediction = model.predict(intervention_input)[0]
    
    # Menghitung selisih dampak kebijakan (Delta Analysis)
    delta_y = prediction - baseline_pred
    
    return prediction, delta_y




import streamlit as st 
 
st.title("🚀🚀 Simulator Kebijakan Keuntungan Toko") 
st.write("Gunakan slider untuk menguji skenario 'What-If'.") 
 
# --- SIDEBAR: Variabel Kontrol --- 
st.sidebar.header("Tuas Kebijakan (Intervensi)") 
iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta)", 0, 50, 10) 
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10) 
 
# --- ENGINE: Jalankan Simulasi --- 
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider) 
 
# --- UI: Tampilkan Hasil --- 
col1, col2 = st.columns(2) 
col1.metric("Prediksi Keuntungan", f"Rp {hasil_pred:.2f} Jt", f"{delta:.2f} Jt") 
col2.write(f"Skenario ini menghasilkan perubahan sebesar {delta:.2f} Juta dibandingkan kondisi baseline.") 
 
# Visualisasi Perbandingan 
data_plot = pd.DataFrame({ 
    'Skenario': ['Baseline', 'Intervensi'], 
    'Keuntungan': [baseline_pred, hasil_pred] 
}) 
st.bar_chart(data=data_plot, x='Skenario', y='Keuntungan') 