import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
import shutil

# Coba Import YOLO

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
st.set_page_config(page_title="Pengenalan Gambar Penyakit Kentang")

# Periksa apakah library YOLO tersedia
def cek_library():
    if not YOLO_AVAILABLE:
        st.error("Ultralytics tidak terpasang. Silakan instal dengan perintah berikut:")
        st.code("pip install ultralytics")
        return False
    return True

# Correct file path
image_path = "SistaR.jpg"

#st.markdown("<h1 style='text-align: center;'>Mineral Image</h1>", unsafe_allow_html=True)
#st.image(image_path, width=500, use_container_width=False)
#col1, col2, col3 = st.columns([1, 2, 1])
# Adjust column widths (reduce side columns for more space)
col1, col2, col3 = st.columns([0.5, 3, 0.5])  

with col2:
    st.image(image_path, width=800)  # Increase width for a larger display

st.markdown("""
<div style="background-color: white; border: 5px solid blue; padding: 25px; text-align: center; 
            width: 200%; max-width: 1200px; position: relative; left: 50%; transform: translateX(-50%);">
    <h1 style="color: black;"> Program Pengenalan Penyakit Kentang </h1>
    <h3 style="color: black;"> by: SisTa-R </h3>
    <h5 style="color: black;">Deteksi Gambar Penyakit Kentang</h5>
</div>
""", unsafe_allow_html=True)


# Pastikan library sudah terpasang sebelum melanjutkan
if cek_library():
    # upload gambar
     uploaded_file = st.file_uploader("upload gambar penyakit kentang", type=['jpg', 'jpeg', 'png'])

     if uploaded_file:
        # Simpan sementara
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "gambar.jpg")
        image = Image.open(uploaded_file)
        
        #Ubah Ukuran Gambar
        image = image.resize((300,300))
        image.save(temp_file)

        #Tampilkan gamabar
        st.markdown("<div style='text-align: center;'>",unsafe_allow_html=True)
        st.image(image, caption="Gambar yang diupload")
        st.markdown("</div>", unsafe_allow_html=True)
        
        #Deteksi Gambar
        if st.button("Deteksi Gambar"):
          with st.spinner("Sek...sek...sekk....sabar yo......lagi proses identifikasi"):
              try:
                  model = YOLO('best.pt')
                  hasil = model(temp_file)
                  
                  #Ambil Hasil Prediksi
                  nama_objek = hasil[0].names
                  nilai_prediksi= hasil[0].probs.data.numpy().tolist()
                  objek_terdeteksi = nama_objek[np.argmax(nilai_prediksi)]
                  
                  #buat grafik
                  grafik = go.Figure([go.Bar(x=list(nama_objek.values()), y=nilai_prediksi)])
                  grafik.update_layout(title='Tingkat Keyakinan Prediksi', xaxis_title='Penyakit kentang',
                  yaxis_title='Keyakinan')
                  
                  #Tampilkan hasil
                  st.write(f"penyakit kentang terdeteksi:{objek_terdeteksi}")
                  st.plotly_chart(grafik)
                
              except Exception as e :
                  st.error("Gambar tidak dapat terdeteksi")
                  st.error(f"Error:{e}")
                  
              #Hapus file sementara
              shutil.rmtree(temp_dir,ignore_errors=True)

st.markdown(
"<div style='text-align: center;' class='footer'>Program Aplikasi deteksi penyakit kentang by SisTa-R @2025</div>",
unsafe_allow_html=True
)

                 
