
import streamlit as st
from PIL import Image
import torch
from realesrgan import RealESRGAN
import io

st.set_page_config(page_title="Pertajam Foto AI", layout="centered")

st.title("📸 Pertajam Foto dengan AI")
st.write("Upload gambar buram atau resolusi rendah, dan AI akan mempertajamnya menggunakan Real-ESRGAN.")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar asli
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Asli", use_column_width=True)

    # Proses dengan AI
    with st.spinner("Memproses dengan AI..."):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4.pth')
        enhanced = model.predict(image)

    # Tampilkan hasil
    st.image(enhanced, caption="Gambar Setelah Ditingkatkan", use_column_width=True)

    # Download hasil
    img_bytes = io.BytesIO()
    enhanced.save(img_bytes, format="PNG")
    st.download_button("📥 Unduh Gambar Hasil", data=img_bytes.getvalue(), file_name="enhanced_image.png", mime="image/png")
