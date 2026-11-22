import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path("pilahin_mobilenetv2.pt")
CONFIDENCE_MIN = 0.6

INFO = {
    "kertas": ("Anorganik, bisa didaur ulang",
               "Pastikan kering dan tidak berminyak. Kertas dan kardus bisa disetor ke bank sampah."),
    "plastik": ("Anorganik, bisa didaur ulang",
                "Kosongkan isinya dan bilas sebentar. Pisahkan tutup dan label kalau bisa."),
    "logam": ("Anorganik, bisa didaur ulang",
              "Kaleng dan logam bernilai jual di bank sampah. Bilas dan pipihkan supaya hemat tempat."),
    "kaca": ("Anorganik, bisa didaur ulang",
             "Bilas dulu. Kalau pecah, bungkus dengan kertas tebal dan beri tanda supaya petugas tidak terluka."),
}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def predict(image: Image.Image):
    model, classes = load_model()
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return {cls: float(p) for cls, p in zip(classes, probs)}


st.set_page_config(page_title="Pilah.in", page_icon="♻️")
st.title("Pilah.in")
st.write("Foto sampahmu, lalu cari tahu jenis dan cara menanganinya.")

tab_upload, tab_camera = st.tabs(["Upload foto", "Ambil foto"])
with tab_upload:
    uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])
with tab_camera:
    camera = st.camera_input("Arahkan kamera ke sampah")

source = uploaded or camera
if source is not None:
    image = Image.open(source)
    st.image(image, use_container_width=True)

    scores = predict(image)
    best_class = max(scores, key=scores.get)
    confidence = scores[best_class]

    if confidence < CONFIDENCE_MIN:
        st.warning(
            f"Model kurang yakin (kemungkinan terbesar: {best_class}, {confidence:.0%}). "
            "Coba foto lagi dengan satu benda, latar polos, dan cahaya cukup."
        )
    else:
        kategori, saran = INFO.get(best_class, ("", ""))
        st.success(f"Terdeteksi: {best_class} ({confidence:.0%})")
        st.write(f"**{kategori}**")
        st.write(saran)

    st.subheader("Skor semua kelas")
    for cls, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        st.progress(score, text=f"{cls}: {score:.0%}")

st.caption(
    "Model dilatih dari dataset TrashNet dengan 4 kelas (kertas, plastik, logam, kaca). "
    "Hasil bisa meleset, terutama untuk foto yang berbeda dari data latih."
)
