import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Klasifikasi Kurma",
	page_icon = "🌴"
)

# Memuat model machine learning kurma
model = joblib.load("model_klasifikasi_kurma.joblib")

st.title("🌴 Aplikasi Klasifikasi Kurma")
st.markdown("Masukkan parameter kurma untuk memprediksi *stage* atau kualitasnya.")

# Input slider disesuaikan dengan kolom numerik di dataset kamu
ukuran_cm = st.slider("Ukuran (cm)", 1.0, 10.0, 3.5)
berat_g = st.slider("Berat (g)", 1.0, 50.0, 10.0)
kekerasan_N = st.slider("Kekerasan (N)", 1.0, 100.0, 15.0)
kadar_gula_brix = st.slider("Kadar Gula (brix)", 10.0, 80.0, 60.0)
kadar_air_pct = st.slider("Kadar Air (%)", 5.0, 80.0, 15.0)

if st.button("Prediksi", type="primary"):
	# Nama kolom DISAMAKAN PERSIS dengan header tabel dataset_kurma.csv
	data_baru = pd.DataFrame(
		[[ukuran_cm, berat_g, kekerasan_N, kadar_gula_brix, kadar_air_pct]], 
		columns=["ukuran_cm", "berat_g", "kekerasan_N", "kadar_gula_brix", "kadar_air_pct"]
	)
	
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi kurma **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan 🌴 oleh **Jerry**")