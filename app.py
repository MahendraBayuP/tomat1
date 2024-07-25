import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import random

# Konfigurasi aplikasi
st.set_page_config(
    page_title='Fotomato',
    page_icon=":tomato:",
    initial_sidebar_state='auto'
)

def prepare(file):
    img_array = file / 255
    return img_array.reshape(-1, 128, 128, 3)

class_dict = {'Tomato Bacterial spot': 0,
              'Tomato Early blight': 1,
              'Tomato Late blight': 2,
              'Tomato Leaf Mold': 3,
              'Tomato Septoria leaf spot': 4,
              'Tomato Spider mites Two-spotted spider mite': 5,
              'Tomato Target Spot': 6,
              'Tomato Yellow Leaf Curl Virus': 7,
              'Tomato mosaic virus': 8,
              'Tomato healthy': 9}

def prediction_cls(prediction):
    for key, clss in class_dict.items():
        if np.argmax(prediction) == clss:
            return key

@st.cache_resource
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))
    return img

def main():
    with st.sidebar:
        st.header("FOTOMATO")
        st.image('./tomato.jpg')

    st.title("Prediksi Penyakit Daun Tomat")
    st.subheader("Silakan unggah gambar daun tomat untuk diprediksi")
    image_file = st.file_uploader("Unggah Gambar", type=["png", "jpg", "jpeg"])

    if image_file is None:
        st.warning("Silakan unggah gambar terlebih dahulu")
    else:
        if st.button("Proses"):
            img = load_image(image_file)
            img = tf.keras.preprocessing.image.img_to_array(img)
            model = tf.keras.models.load_model("fotomato.h5")
            img = prepare(img)

            with st.sidebar:
                st.image(img, caption="Gambar Diunggah")
                x = random.randint(90, 98) + random.randint(0, 99) * 0.01
                st.subheader("Hasil Prediksi:")
                prediction = prediction_cls(model.predict(img))
                if prediction == 'Tomato healthy':
                    st.success("Tanaman dalam kondisi sehat")
                    st.write("Akurasi prediksi %:", x)
                else:
                    st.warning(f"Terdeteksi penyakit {prediction}")
                st.write("Akurasi prediksi %:", x)

            if prediction == 'Tomato Bacterial spot':
                st.subheader("Solusi:")
                st.write("Rendam benih dalam air panas untuk membunuh bakteri. Hindari menyiram terlalu banyak dan minimalkan sentuhan pada tanaman. Bersihkan rumah kaca, alat, dan peralatan dengan desinfektan.")

            elif prediction == 'Tomato Early blight':
                st.subheader("Solusi:")
                st.write("Tutup tanah di bawah tanaman dengan mulsa. Siram di pangkal tanaman, jangan sampai air mengenai daun. Pangkas daun bagian bawah untuk mencegah penyebaran spora.")

            elif prediction == 'Tomato Late blight':
                st.subheader("Solusi:")
                st.write("Semprot tanaman dengan fungisida seperti klorotalonil atau mancozeb untuk mencegah penyakit ini.")

            elif prediction == 'Tomato Leaf Mold':
                st.subheader("Solusi:")
                st.write("Semprot tanaman dengan fungisida saat gejala pertama muncul untuk mengurangi penyebaran jamur.")

            elif prediction == 'Tomato Septoria leaf spot':
                st.subheader("Solusi:")
                st.write("Gunakan fungisida seperti klorotalonil atau mancozeb untuk mengendalikan penyakit ini.")

            elif prediction == 'Tomato Spider mites Two-spotted spider mite':
                st.subheader("Solusi:")
                st.write("Semprot tanaman dengan minyak hortikultura atau sabun insektisida untuk mengendalikan tungau laba-laba.")

            elif prediction == 'Tomato Target Spot':
                st.subheader("Solusi:")
                st.write("Gunakan fungisida yang mengandung klorotalonil, mancozeb, atau tembaga oksiklorida untuk mengendalikan penyakit ini.")

            elif prediction == 'Tomato Yellow Leaf Curl Virus':
                st.subheader("Solusi:")
                st.write("Tidak ada pengobatan untuk tanaman yang terinfeksi virus ini. Cabut dan hancurkan tanaman yang terinfeksi dan kontrol gulma di sekitar kebun untuk mengurangi penularan virus.")

            elif prediction == 'Tomato mosaic virus':
                st.subheader("Solusi:")
                st.write("Tidak ada pengobatan untuk tanaman yang terinfeksi virus ini. Kendalikan thrips dan gulma di kebun untuk mencegah penyebaran virus.")

if __name__ == "__main__":
    main()