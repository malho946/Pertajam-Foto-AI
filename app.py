
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Pertajam Foto", layout="centered")
st.title("📸 Pertajam Foto Buram (Versi Ringan)")
st.write("Aplikasi ini mempertajam foto buram menggunakan OpenCV filter.")

uploaded_file = st.file_uploader("Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Gambar Asli", use_column_width=True)

    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])

    sharpened = cv2.filter2D(img_np, -1, kernel)

    st.image(sharpened, caption="Gambar Setelah Dipertajam", use_column_width=True)

    # Konversi untuk download
    result_img = Image.fromarray(sharpened)
    img_bytes = io.BytesIO()
    result_img.save(img_bytes, format="PNG")
    st.download_button("📥 Unduh Gambar", data=img_bytes.getvalue(), file_name="sharpened.png", mime="image/png")
