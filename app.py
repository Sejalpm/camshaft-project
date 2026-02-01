import streamlit as st
import cv2
import easyocr
import numpy as np

st.set_page_config(page_title="Camshaft OCR", layout="wide")
st.title("🔍 Camshaft Character Detection System")

# Cache EasyOCR model (IMPORTANT)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

uploaded_file = st.file_uploader(
    "Upload Camshaft Image",
    type=["jpg", "jpeg", "png"]
)

format_choice = st.selectbox(
    "Select Character Format",
    ["Alphanumeric", "Only Numbers", "Only Letters"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("📷 Uploaded Image")
    st.image(image, channels="BGR")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    st.subheader("🛠️ Processed Image")
    st.image(processed, channels="GRAY")

    results = reader.readtext(processed)

    extracted_text = " ".join([r[1] for r in results])

    if format_choice == "Only Numbers":
        extracted_text = "".join(filter(str.isdigit, extracted_text))
    elif format_choice == "Only Letters":
        extracted_text = "".join(filter(str.isalpha, extracted_text))

    st.subheader("🔤 Extracted Characters")
    st.text(extracted_text if extracted_text else "No text detected")
